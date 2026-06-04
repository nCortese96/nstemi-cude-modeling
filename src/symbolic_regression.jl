"""
symbolic_regression.jl

Reusable non-plotting helpers for workflow step 04a.

Sections:
- Teacher Dataset: deterministic synthetic grids and NN teacher evaluation.
- Symbolic Regression: bounded loss, options, search, and Pareto selection.
- Evaluation Tables: stable teacher, frontier, and selected-model summaries.
"""

using DataFrames: DataFrame
using Statistics: mean
using SymbolicRegression
import SymbolicRegression: string_tree, compute_complexity

# =============================================================================
# Teacher Dataset
# =============================================================================

"""
    build_symbolic_teacher_grid(t_grid, beta_grid; t_scale)

Build the deterministic cUDE symbolic-regression teacher grid.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function build_symbolic_teacher_grid(t_grid, beta_grid; t_scale::Real)
    patient_id = String[]
    t_h = Float64[]
    t_norm = Float64[]
    beta = Float64[]

    for (beta_idx, beta_value) in enumerate(beta_grid)
        for time_h in t_grid
            push!(patient_id, "synth$(beta_idx)")
            push!(t_h, Float64(time_h))
            push!(t_norm, Float64(time_h / t_scale))
            push!(beta, Float64(beta_value))
        end
    end

    X = [t_norm'; beta']
    size(X) == (2, length(t_norm)) ||
        error("Invalid symbolic teacher matrix size: $(size(X))")

    return (
        patient_id=patient_id,
        t_h=t_h,
        t_norm=t_norm,
        beta=beta,
        X=X,
        t_grid=Float64.(collect(t_grid)),
        beta_grid=Float64.(collect(beta_grid)),
    )
end

"""
    evaluate_symbolic_nn_teacher(chain, neural_params, X)

Evaluate the selected cUDE neural correction on a `2 x N` teacher grid.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function evaluate_symbolic_nn_teacher(chain, neural_params, X)
    y = Vector{Float64}(undef, size(X, 2))

    for idx in axes(X, 2)
        y[idx] = chain((@view X[:, idx]), neural_params)[1]
    end

    all(isfinite, y) || error("Symbolic teacher target contains non-finite values.")
    return y
end

"""
    symbolic_teacher_dataframe(grid, y)

Build the canonical symbolic-regression teacher dataset table.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_teacher_dataframe(grid, y)
    length(y) == length(grid.t_norm) ||
        error("Teacher target length $(length(y)) does not match grid length $(length(grid.t_norm)).")

    return DataFrame(
        patient_id=grid.patient_id,
        t_h=grid.t_h,
        t_norm=grid.t_norm,
        beta=grid.beta,
        y_nn=y,
    )
end

# =============================================================================
# Symbolic Regression
# =============================================================================

"""
    smooth_relu_fast(x; eps_value=1e-5)

Smooth positive-part approximation used by the legacy SR bounded loss.
"""
# Used by: src/symbolic_regression.jl (build_symbolic_regression_loss).
smooth_relu_fast(x; eps_value::Real=1e-5) = 0.5 * (x + sqrt(x * x + eps_value * eps_value))

"""
    build_symbolic_regression_loss(settings)

Return the legacy bounded elementwise loss used during symbolic regression.
"""
# Used by: src/symbolic_regression.jl (build_symbolic_regression_options).
function build_symbolic_regression_loss(settings)
    lambda_negative = settings.lambda_negative
    lambda_high = settings.lambda_high
    smooth_eps = settings.smooth_eps

    return (y_pred, y_true) ->
        (y_pred - y_true)^2 +
        lambda_negative * smooth_relu_fast(-y_pred; eps_value=smooth_eps)^2 +
        lambda_high * smooth_relu_fast(y_pred - 1.0; eps_value=smooth_eps)^2
end

"""
    build_symbolic_regression_options(settings, output_directory)

Create `SymbolicRegression.Options` from workflow config.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function build_symbolic_regression_options(settings, output_directory::AbstractString)
    return Options(
        binary_operators=settings.binary_operators,
        unary_operators=settings.unary_operators,
        maxsize=settings.maxsize,
        populations=settings.populations,
        parsimony=settings.parsimony,
        complexity_of_constants=settings.complexity_of_constants,
        batching=settings.batching,
        batch_size=settings.batch_size,
        should_optimize_constants=settings.should_optimize_constants,
        elementwise_loss=build_symbolic_regression_loss(settings),
        output_directory=output_directory,
        save_to_file=true,
        seed=settings.seed,
    )
end

"""
    run_symbolic_regression_search(X, y, settings, options)

Run the warm-up and main symbolic-regression searches, returning the main hall
of fame object.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function run_symbolic_regression_search(X, y, settings, options)
    variable_names = collect(settings.variable_names)

    if settings.niterations_warmup > 0
        @info "Running symbolic-regression warm-up." niterations=settings.niterations_warmup
        equation_search(
            X,
            y;
            niterations=settings.niterations_warmup,
            options=options,
            parallelism=:multithreading,
            progress=settings.progress_bars,
            variable_names=variable_names,
        )
    end

    @info "Running main symbolic-regression search." niterations=settings.niterations_main
    return equation_search(
        X,
        y;
        niterations=settings.niterations_main,
        options=options,
        parallelism=:multithreading,
        progress=settings.progress_bars,
        variable_names=variable_names,
    )
end

"""
    symbolic_sr_eval(tree, X)

Evaluate one SymbolicRegression tree and replace non-finite outputs with the
legacy large penalty value.
"""
# Used by: scripts/04a_run_symbolic_regression.jl and src/symbolic_regression.jl.
function symbolic_sr_eval(tree, X)
    out = eval_tree_array(tree, X)
    vals = out isa Tuple ? collect(out[1]) : collect(out)
    y = Float64.(vals)

    for idx in eachindex(y)
        if !isfinite(y[idx])
            y[idx] = 1e6
        end
    end

    return y
end

"""
    select_symbolic_regression_model(hof, X_teacher, y_teacher, settings, options)

Select the simplest Pareto member whose teacher-grid MSE is within the
configured tolerance of the minimum teacher-grid MSE.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function select_symbolic_regression_model(hof, X_teacher, y_teacher, settings, options)
    frontier = calculate_pareto_frontier(hof)
    isempty(frontier) && error("Symbolic regression produced an empty Pareto frontier.")

    teacher_mses = Float64[]
    complexities = Int[]

    for member in frontier
        y_hat_teacher = symbolic_sr_eval(member.tree, X_teacher)
        push!(teacher_mses, mean((y_teacher .- y_hat_teacher) .^ 2))
        push!(complexities, compute_complexity(member, options))
    end

    best_idx = select_symbolic_regression_index(
        teacher_mses,
        complexities;
        tolerance=settings.teacher_mse_tolerance,
    )

    best = frontier[best_idx]
    equation = string_tree(best.tree; variable_names=collect(settings.variable_names))

    return (
        frontier=frontier,
        best=best,
        best_idx=best_idx,
        equation=equation,
        teacher_mse=teacher_mses[best_idx],
        complexity=complexities[best_idx],
        teacher_mses=teacher_mses,
        complexities=complexities,
    )
end

"""
    select_symbolic_regression_index(teacher_mses, complexities; tolerance)

Return the least-complex candidate whose teacher-grid MSE is within `tolerance`
of the minimum MSE.
"""
# Used by: src/symbolic_regression.jl.
function select_symbolic_regression_index(teacher_mses, complexities; tolerance::Real)
    isempty(teacher_mses) && error("Cannot select a symbolic surrogate from an empty candidate set.")
    length(teacher_mses) == length(complexities) ||
        error("Symbolic surrogate MSE and complexity vectors must have the same length.")

    minimum_mse_idx = argmin(teacher_mses)
    near_best = findall(teacher_mses .<= tolerance * teacher_mses[minimum_mse_idx])
    return near_best[argmin(complexities[near_best])]
end

"""
    symbolic_equation_eval(equation, X)

Evaluate a trusted symbolic equation on a `2 x N` teacher grid.
"""
# Used by: src/symbolic_regression.jl (report-only symbolic selection).
function symbolic_equation_eval(equation::AbstractString, X)
    body = Meta.parse(equation; raise=true)
    correction = Core.eval(@__MODULE__, :((t_norm, β) -> $body))
    # Report mode evaluates a newly compiled callable immediately. invokelatest
    # is intentionally confined here: candidate equations are never injected
    # into the step 04b ODE solve.
    return [Float64(Base.invokelatest(correction, X[1, idx], X[2, idx])) for idx in axes(X, 2)]
end

"""
    symbolic_teacher_arrays(teacher_table; t_scale)

Reconstruct the teacher matrix and target vector from the stable step 04a CSV.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_teacher_arrays(teacher_table::DataFrame; t_scale::Real)
    required = (:t_h, :t_norm, :beta, :y_nn)
    missing_columns = setdiff(required, propertynames(teacher_table))
    isempty(missing_columns) ||
        error("Symbolic teacher table is missing columns: $(join(missing_columns, ", ")).")

    t_h = Float64.(teacher_table.t_h)
    t_norm = Float64.(teacher_table.t_norm)
    beta = Float64.(teacher_table.beta)
    y = Float64.(teacher_table.y_nn)

    all(isapprox.(t_norm, t_h ./ t_scale)) ||
        error("Symbolic teacher table is inconsistent with t_scale=$(t_scale).")

    return (
        X=[t_norm'; beta'],
        y=y,
        t_grid=unique(t_h),
        beta_grid=unique(beta),
        training_points=length(y),
    )
end

"""
    select_symbolic_regression_model(frontier_table, X_teacher, y_teacher, settings)

Select a trusted equation from the stable Pareto-frontier CSV without rerunning
symbolic regression.
"""
# Used by: scripts/04a_run_symbolic_regression.jl (`report` mode).
function select_symbolic_regression_model(frontier_table::DataFrame, X_teacher, y_teacher, settings)
    required = (:idx, :complexity, :equation)
    missing_columns = setdiff(required, propertynames(frontier_table))
    isempty(missing_columns) ||
        error("Symbolic frontier table is missing columns: $(join(missing_columns, ", ")).")
    isempty(frontier_table.idx) && error("Symbolic frontier table is empty.")

    teacher_mses = [
        mean((y_teacher .- symbolic_equation_eval(String(equation), X_teacher)) .^ 2)
        for equation in frontier_table.equation
    ]
    complexities = Int.(frontier_table.complexity)
    position = select_symbolic_regression_index(
        teacher_mses,
        complexities;
        tolerance=settings.teacher_mse_tolerance,
    )
    symbolic_target = symbolic_equation_eval(String(frontier_table.equation[position]), X_teacher)

    return (
        best_idx=Int(frontier_table.idx[position]),
        equation=String(frontier_table.equation[position]),
        teacher_mse=teacher_mses[position],
        complexity=complexities[position],
        teacher_mses=teacher_mses,
        complexities=complexities,
        symbolic_target=symbolic_target,
    )
end

# =============================================================================
# Evaluation Tables
# =============================================================================

"""
    symbolic_frontier_dataframe(frontier, options, variable_names)

Build the canonical symbolic-regression Pareto frontier table.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_frontier_dataframe(frontier, options, variable_names)
    return DataFrame(
        idx=collect(1:length(frontier)),
        complexity=[compute_complexity(member, options) for member in frontier],
        loss=[member.loss for member in frontier],
        equation=[string_tree(member.tree; variable_names=collect(variable_names)) for member in frontier],
    )
end

"""
    symbolic_grid_metrics(y_true, y_pred)

Return MSE, MAE, and R2 for symbolic surrogate predictions on a grid.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_grid_metrics(y_true, y_pred)
    mse = mean((y_true .- y_pred) .^ 2)
    mae = mean(abs.(y_true .- y_pred))
    r2 = 1 - sum((y_true .- y_pred) .^ 2) / sum((y_true .- mean(y_true)) .^ 2)
    return (mse=mse, mae=mae, r2=r2)
end

"""
    build_symbolic_plot_curves(chain, neural_params, tree, t_grid, beta_values; t_scale)

Build NN and symbolic surrogate curves used by step 04a comparison plots.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function build_symbolic_plot_curves(chain, neural_params, tree, t_grid, beta_values; t_scale::Real)
    records = Vector{Any}()
    t_values = Float64.(collect(t_grid))

    for beta_value in beta_values
        beta_float = Float64(beta_value)
        y_nn = [chain([time_h / t_scale, beta_float], neural_params)[1] for time_h in t_values]
        X_tmp = hcat([[time_h / t_scale, beta_float] for time_h in t_values]...)
        y_sr = symbolic_sr_eval(tree, X_tmp)

        push!(records, (
            beta=beta_float,
            t_h=t_values,
            y_nn=Float64.(y_nn),
            y_sr=y_sr,
        ))
    end

    return records
end

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
    select_symbolic_regression_model(hof, X_val, y_val, settings, options)

Select the Pareto member with minimum validation loss, tie-breaking among
near-identical losses by lower complexity.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function select_symbolic_regression_model(hof, X_val, y_val, settings, options)
    frontier = calculate_pareto_frontier(hof)
    isempty(frontier) && error("Symbolic regression produced an empty Pareto frontier.")

    val_losses = Float64[]
    complexities = Int[]

    for member in frontier
        y_hat_val = symbolic_sr_eval(member.tree, X_val)
        push!(val_losses, mean((y_val .- y_hat_val) .^ 2))
        push!(complexities, compute_complexity(member, options))
    end

    best_idx = argmin(val_losses)
    near_best = findall(val_losses .<= settings.validation_loss_tolerance * val_losses[best_idx])

    if !isempty(near_best)
        best_idx = near_best[argmin(complexities[near_best])]
    end

    best = frontier[best_idx]
    equation = string_tree(best.tree; variable_names=collect(settings.variable_names))

    return (
        frontier=frontier,
        best=best,
        best_idx=best_idx,
        equation=equation,
        validation_loss=val_losses[best_idx],
        complexity=complexities[best_idx],
        val_losses=val_losses,
        complexities=complexities,
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

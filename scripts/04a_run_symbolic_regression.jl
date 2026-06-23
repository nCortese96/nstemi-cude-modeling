"""
04a_run_symbolic_regression.jl

Train a symbolic surrogate for the selected cUDE neural correction function.

Pipeline:
1. Read workflow settings from `config/workflow_config.jl`.
2. Load the selected cUDE model from step 02c.
3. Load the trained cUDE neural-network weights from step 02a.
4. Build the deterministic synthetic teacher grid.
5. Run warm-up and main symbolic-regression searches.
6. Select the best Pareto member and save stable CSV/report/plot artifacts.

The selected equation is a candidate for manual promotion. Inspect and simplify
it into a numerically stable form, then update the dedicated section at the end
of `src/models.jl` before running step 04b.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/04a_run_symbolic_regression.jl
    julia --project=. scripts/04a_run_symbolic_regression.jl report
    julia --project=. scripts/04a_run_symbolic_regression.jl inspection [tmax_h] [n_beta]

Use `report` to regenerate only the human-readable report from the stable
teacher and Pareto-frontier CSVs.

Use `inspection` to plot the selected trained cUDE neural correction function
without generating the symbolic-regression teacher dataset, CSVs, report, or
SymbolicRegression outputs. With no extra arguments, `inspection` uses
`t_grid` and `plot_beta_grid` from config. Optional arguments override the
inspection time horizon and number of plotted beta values.
"""

using Base.Threads: nthreads
using Dates
using Logging

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "symbolic_regression.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings are defined in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.symbolic_regression
dataset_config = getproperty(config.datasets, settings.model_selection_dataset_key)

function parse_positive_float_arg(value::AbstractString, name::AbstractString)
    parsed = try
        parse(Float64, value)
    catch err
        error("Invalid $(name): $(value). Expected a positive numeric value.")
    end
    parsed > 0 || error("Invalid $(name): $(value). Expected a positive numeric value.")
    return parsed
end

function parse_positive_int_arg(value::AbstractString, name::AbstractString)
    parsed = try
        parse(Int, value)
    catch err
        error("Invalid $(name): $(value). Expected a positive integer value.")
    end
    parsed > 0 || error("Invalid $(name): $(value). Expected a positive integer value.")
    return parsed
end

if isempty(ARGS)
    execution_mode = :run
    inspection_tmax_h = nothing
    inspection_n_beta = nothing
elseif length(ARGS) == 1 && lowercase(strip(ARGS[1])) == "report"
    execution_mode = :report
    inspection_tmax_h = nothing
    inspection_n_beta = nothing
elseif !isempty(ARGS) && lowercase(strip(ARGS[1])) == "inspection"
    length(ARGS) <= 3 ||
        error("Usage: julia --project=. scripts/04a_run_symbolic_regression.jl inspection [tmax_h] [n_beta]")
    execution_mode = :inspection
    inspection_tmax_h = length(ARGS) >= 2 ? parse_positive_float_arg(ARGS[2], "inspection tmax_h") : nothing
    inspection_n_beta = length(ARGS) == 3 ? parse_positive_int_arg(ARGS[3], "inspection n_beta") : nothing
else
    error("Usage: julia --project=. scripts/04a_run_symbolic_regression.jl [report | inspection [tmax_h] [n_beta]]")
end

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run.
# =============================================================================

selection_paths = cude_model_selection_output_paths(settings.model_selection_dir, dataset_config.dataset_name)
selected_model_path = settings.selected_model_path === nothing ? selection_paths.selected_model : settings.selected_model_path

# =============================================================================
# OUTPUT PATHS
# Files and folders produced by this run.
# =============================================================================

output_paths = symbolic_regression_output_paths(settings.output_dir)

# =============================================================================
# DERIVED SETTINGS
# Values derived from settings and paths. No heavy side effects here.
# =============================================================================

settings.t_scale == T_SCALE ||
    error("Symbolic-regression t_scale ($(settings.t_scale)) must match model T_SCALE ($(T_SCALE)).")

# =============================================================================
# PIPELINE
# Main readable execution flow.
# =============================================================================

log_workflow_context(
    config;
    script_name=basename(@__FILE__),
    output_paths=(
        output_dir=output_paths.output_dir,
        selected_model=selected_model_path,
    ),
)

if execution_mode === :report
    ensure_output_dirs!(output_paths.output_dir; header="Ensured symbolic-regression output directory")
    log_output_paths(
        (
            teacher_dataset=output_paths.teacher_dataset,
            pareto_frontier=output_paths.pareto_frontier,
            selected_symbolic_model=output_paths.selected_model,
        );
        header="Symbolic-regression report paths",
    )

    @info "Starting symbolic-regression report-only workflow." dataset=dataset_config.dataset_name
    @warn "Report mode assumes that the stable teacher CSV was generated from the currently selected cUDE model."

    selected_model = load_selected_cude_model(selected_model_path)
    tables = load_symbolic_regression_tables(output_paths)
    teacher_summary = symbolic_teacher_arrays(tables.teacher; t_scale=settings.t_scale)
    selection = select_symbolic_regression_model(tables.frontier, teacher_summary.X, teacher_summary.y, settings)
    metrics = symbolic_grid_metrics(teacher_summary.y, selection.symbolic_target)

    write_symbolic_regression_report(
        output_paths.selected_model,
        selection,
        metrics;
        selected_model=selected_model,
        settings=settings,
        output_paths=output_paths,
        teacher_summary=teacher_summary,
    )
    @info "Regenerated symbolic-regression report." frontier_idx=selection.best_idx complexity=selection.complexity teacher_mse=selection.teacher_mse
elseif execution_mode === :inspection
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    inspection_dir = joinpath(output_paths.fig_dir, "inspection_$(timestamp)")
    ensure_output_dirs!(inspection_dir; header="Ensured symbolic-regression inspection directory")

    selected_model = load_selected_cude_model(selected_model_path)
    training_artifacts = load_cude_training_artifacts(settings.cude_training_input_dir, selected_model.nn_width)

    1 <= selected_model.model_idx <= length(training_artifacts.neural_network_parameters) ||
        error(
            "Selected model index $(selected_model.model_idx) is invalid for width $(selected_model.nn_width); " *
            "available models: $(length(training_artifacts.neural_network_parameters)).",
        )

    chain = neural_network_model(selected_model.nn_depth, selected_model.nn_width; input_dims=settings.input_dim)
    selected_neural_params = training_artifacts.neural_network_parameters[selected_model.model_idx]

    inspection_t_grid = if inspection_tmax_h === nothing
        Float64.(collect(settings.t_grid))
    else
        t_start = first(Float64.(collect(settings.t_grid)))
        inspection_tmax_h > t_start ||
            error("inspection tmax_h must be greater than the first configured time point ($(t_start)).")
        n_time = length(settings.t_grid)
        n_time > 1 || error("Config t_grid must contain at least two points for inspection.")
        collect(range(t_start, Float64(inspection_tmax_h); length=n_time))
    end

    config_beta_grid = Float64.(collect(settings.plot_beta_grid))
    inspection_beta_grid = if inspection_n_beta === nothing
        config_beta_grid
    else
        collect(range(minimum(config_beta_grid), maximum(config_beta_grid); length=inspection_n_beta))
    end

    png_path = joinpath(inspection_dir, "cude_nn_correction_inspection.png")
    svg_path = joinpath(inspection_dir, "cude_nn_correction_inspection.svg")
    settings_path = joinpath(inspection_dir, "inspection_settings.txt")

    @info "Starting selected cUDE NN inspection." model_id=selected_model.model_id width=selected_model.nn_width model_idx=selected_model.model_idx
    @info "Inspection grid." time_points=length(inspection_t_grid) tmax_h=maximum(inspection_t_grid) beta_points=length(inspection_beta_grid)

    save_cude_correction_function_plot(
        png_path,
        chain,
        selected_neural_params;
        t_scale=settings.t_scale,
        t_grid=inspection_t_grid,
        beta_values=inspection_beta_grid,
        plotting=true,
        display_plot=settings.display_plots,
    )
    save_cude_correction_function_plot(
        svg_path,
        chain,
        selected_neural_params;
        t_scale=settings.t_scale,
        t_grid=inspection_t_grid,
        beta_values=inspection_beta_grid,
        plotting=true,
        display_plot=false,
    )

    open(settings_path, "w") do io
        println(io, "Symbolic-regression inspection mode")
        println(io, "timestamp: $(timestamp)")
        println(io, "command: julia --project=. scripts/04a_run_symbolic_regression.jl $(join(ARGS, ' '))")
        println(io, "selected_model_id: $(selected_model.model_id)")
        println(io, "nn_width: $(selected_model.nn_width)")
        println(io, "model_idx: $(selected_model.model_idx)")
        println(io, "time_grid_source: $(inspection_tmax_h === nothing ? "config.t_grid" : "cli_tmax_h")")
        println(io, "tmax_h: $(maximum(inspection_t_grid))")
        println(io, "time_points: $(length(inspection_t_grid))")
        println(io, "beta_grid_source: $(inspection_n_beta === nothing ? "config.plot_beta_grid" : "cli_n_beta")")
        println(io, "beta_points: $(length(inspection_beta_grid))")
        println(io, "beta_min: $(minimum(inspection_beta_grid))")
        println(io, "beta_max: $(maximum(inspection_beta_grid))")
        println(io, "output_png: $(abspath(png_path))")
        println(io, "output_svg: $(abspath(svg_path))")
        println(io)
        println(io, "Inspection mode does not generate the symbolic-regression teacher dataset.")
        println(io, "Run this script without arguments to build the official teacher dataset and run symbolic regression.")
    end

    log_output_paths(
        (
            inspection_dir=inspection_dir,
            correction_png=png_path,
            correction_svg=svg_path,
            inspection_settings=settings_path,
        );
        header="Symbolic-regression inspection outputs",
    )
else
    ensure_output_dirs!(
        (
            output=output_paths.output_dir,
            figures=output_paths.fig_dir,
            sr_outputs=output_paths.sr_outputs_dir,
        );
        header="Ensured symbolic-regression output directories",
    )
    log_output_paths(
        (
            teacher_dataset=output_paths.teacher_dataset,
            pareto_frontier=output_paths.pareto_frontier,
            selected_symbolic_model=output_paths.selected_model,
            teacher_plot=output_paths.teacher_plot,
            nn_vs_sr_plot=output_paths.nn_vs_sr_plot,
            sr_plot=output_paths.sr_plot,
        );
        header="Symbolic-regression save paths",
    )

    @info "Starting symbolic-regression workflow." dataset=dataset_config.dataset_name threads=nthreads()

    selected_model = load_selected_cude_model(selected_model_path)
    training_artifacts = load_cude_training_artifacts(settings.cude_training_input_dir, selected_model.nn_width)

    1 <= selected_model.model_idx <= length(training_artifacts.neural_network_parameters) ||
        error(
            "Selected model index $(selected_model.model_idx) is invalid for width $(selected_model.nn_width); " *
            "available models: $(length(training_artifacts.neural_network_parameters)).",
        )

    chain = neural_network_model(selected_model.nn_depth, selected_model.nn_width; input_dims=settings.input_dim)
    selected_neural_params = training_artifacts.neural_network_parameters[selected_model.model_idx]

    @info "Loaded selected cUDE teacher model." model_id=selected_model.model_id width=selected_model.nn_width model_idx=selected_model.model_idx

    teacher_grid = build_symbolic_teacher_grid(settings.t_grid, settings.beta_grid; t_scale=settings.t_scale)
    teacher_target = evaluate_symbolic_nn_teacher(chain, selected_neural_params, teacher_grid.X)
    teacher_table = symbolic_teacher_dataframe(teacher_grid, teacher_target)
    teacher_summary = symbolic_teacher_arrays(teacher_table; t_scale=settings.t_scale)

    @info "Built symbolic-regression teacher dataset." points=length(teacher_target) time_points=length(settings.t_grid) beta_points=length(settings.beta_grid)

    if settings.plotting
        save_symbolic_teacher_plot(
            teacher_grid.t_grid,
            teacher_grid.beta_grid,
            teacher_target,
            output_paths.teacher_plot;
            display_plots=settings.display_plots,
        )
    end

    options = build_symbolic_regression_options(settings, output_paths.sr_outputs_dir)
    hall_of_fame = run_symbolic_regression_search(teacher_grid.X, teacher_target, settings, options)

    selection = select_symbolic_regression_model(hall_of_fame, teacher_grid.X, teacher_target, settings, options)
    frontier_table = symbolic_frontier_dataframe(selection.frontier, options, collect(settings.variable_names))
    symbolic_target = symbolic_sr_eval(selection.best.tree, teacher_grid.X)
    metrics = symbolic_grid_metrics(teacher_target, symbolic_target)

    save_symbolic_regression_tables(output_paths; teacher=teacher_table, frontier=frontier_table)
    write_symbolic_regression_report(
        output_paths.selected_model,
        selection,
        metrics;
        selected_model=selected_model,
        settings=settings,
        output_paths=output_paths,
        teacher_summary=teacher_summary,
    )
    @info "Selected symbolic surrogate." frontier_idx=selection.best_idx complexity=selection.complexity teacher_mse=selection.teacher_mse
    @info "Symbolic surrogate metrics on teacher grid." mse=metrics.mse mae=metrics.mae r2=metrics.r2

    if settings.plotting
        plot_records = build_symbolic_plot_curves(
            chain,
            selected_neural_params,
            selection.best.tree,
            settings.t_grid,
            Float64.(collect(settings.plot_beta_grid));
            t_scale=settings.t_scale,
        )

        save_symbolic_nn_vs_sr_plot(
            plot_records,
            output_paths.nn_vs_sr_plot;
            display_plots=settings.display_plots,
        )

        save_symbolic_sr_plot(
            plot_records,
            output_paths.sr_plot;
            display_plots=settings.display_plots,
        )
    end
end

@info "Symbolic-regression workflow completed." output_dir=abspath(output_paths.output_dir)

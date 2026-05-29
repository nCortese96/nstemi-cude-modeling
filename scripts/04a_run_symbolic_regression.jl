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

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/04a_run_symbolic_regression.jl
"""

using Base.Threads: nthreads
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

plot_beta_grid = Float64.(collect(settings.plot_beta_grid))
variable_names = collect(settings.variable_names)

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

@info "Built symbolic-regression teacher dataset." points=length(teacher_target) time_points=length(settings.t_grid) beta_points=length(settings.beta_grid)

if settings.use_validation
    validation_grid = build_symbolic_teacher_grid(settings.t_validation_grid, settings.beta_validation_grid; t_scale=settings.t_scale)
    validation_target = evaluate_symbolic_nn_teacher(chain, selected_neural_params, validation_grid.X)
    validation_X = validation_grid.X
    @info "Built symbolic-regression validation dataset." points=length(validation_target)
else
    validation_target = teacher_target
    validation_X = teacher_grid.X
    @info "Validation disabled: symbolic model selection uses the teacher grid."
end

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

selection = select_symbolic_regression_model(hall_of_fame, validation_X, validation_target, settings, options)
frontier_table = symbolic_frontier_dataframe(selection.frontier, options, variable_names)

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
)

@info "Selected symbolic surrogate." frontier_idx=selection.best_idx complexity=selection.complexity validation_loss=selection.validation_loss
@info "Symbolic surrogate metrics on teacher grid." mse=metrics.mse mae=metrics.mae r2=metrics.r2

if settings.plotting
    plot_records = build_symbolic_plot_curves(
        chain,
        selected_neural_params,
        selection.best.tree,
        settings.t_grid,
        plot_beta_grid;
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

@info "Symbolic-regression workflow completed." output_dir=abspath(output_paths.output_dir)

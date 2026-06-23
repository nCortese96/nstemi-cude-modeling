"""
03b_run_profile_likelihood.jl

Run profile likelihood analysis for the ODE baseline and selected cUDE model.

Pipeline:
1. Load shared helpers and workflow configuration.
2. Resolve requested PLA targets and canonical input/output paths.
3. Load step 00 cohorts, step 01 ODE fits, step 02a cUDE weights, step 02b/02d
   cUDE patient fits, and the step 02c selected cUDE model.
4. Compute profile likelihood CSV artifacts when enabled.
5. Save patient-level and aggregate PLA plots when enabled.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/03b_run_profile_likelihood.jl [target] [mode]

Valid targets:
    all
    cude_mimic
    cude_umg
    ode_mimic
    ode_umg

Plot-only modes:
    plots             # regenerate patient and aggregate plots from existing CSVs
    plots_patients    # regenerate only patient-level plots
    plots_aggregate   # regenerate only aggregate plots

No arguments means full `all`: compute profiles and overwrite plots. Target
and mode can be provided in either order, for example `cude_mimic plots`.
Use `config/workflow_config.jl` to switch between `results/` and
`results_test/`, change PLA profiler settings, or disable progress bars.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using Dates
using Logging
using Base.Threads: nthreads

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "profile_likelihood.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings live in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.profile_likelihood
cli = parse_profile_likelihood_cli(ARGS, settings)
target_keys = resolve_profile_likelihood_targets(settings, cli.target)
selection_dataset_config = config.datasets[settings.model_selection_dataset_key]

# =============================================================================
# INPUT PATHS
# Step 03b consumes step 00 cohorts, step 01 ODE parameters, step 02a cUDE
# weights, step 02b/02d cUDE patient parameters, and the selected-model CSV.
# =============================================================================

selection_paths = cude_model_selection_output_paths(settings.model_selection_dir, selection_dataset_config.dataset_name)
selected_model_path = settings.selected_model_path === nothing ?
                      selection_paths.selected_model :
                      settings.selected_model_path

# =============================================================================
# OUTPUT PATHS
# Each target writes a stable tree under `03_comparison_analyses/03b_pla`.
# =============================================================================

output_root = settings.output_dir

# =============================================================================
# DERIVED SETTINGS
# Values derived from config and CLI arguments. No heavy side effects here.
# =============================================================================

config.model.t_scale == T_SCALE ||
    error("Profile likelihood requires config.model.t_scale=$(config.model.t_scale) to match models.jl T_SCALE=$(T_SCALE).")

target_specs = profile_likelihood_target_specs(config, settings)
threshold = profile_likelihood_threshold()

# =============================================================================
# PIPELINE
# =============================================================================

@info "Profile likelihood workflow started at $(now())."
log_workflow_context(
    config;
    script_name=basename(@__FILE__),
    output_paths=(
        cohort_dir=settings.cohort_dir,
        ode_input_dir=settings.ode_input_dir,
        cude_training_input_dir=settings.cude_training_input_dir,
        cude_evaluation_input_dir=settings.cude_evaluation_input_dir,
        cude_external_test_input_dir=settings.cude_external_test_input_dir,
        selected_model=selected_model_path,
        output_root=output_root,
    ),
)

@info "Requested PLA target: $(cli.target)"
@info "Requested PLA mode: $(cli.mode)"
@info "Resolved PLA targets: $(target_keys)"
@info "Run compute: $(cli.run_compute)"
@info "Run patient plots: $(cli.run_plot_patients)"
@info "Run aggregate plots: $(cli.run_plot_aggregate)"
@info "Julia threads: $(nthreads())"

@info "Loading selected cUDE model from step 02c."
selected_model = load_selected_cude_model(selected_model_path)
@info "Selected cUDE model: $(selected_model.model_id) | width=$(selected_model.nn_width) | model_idx=$(selected_model.model_idx)"

for target_key in target_keys
    spec = target_specs[target_key]
    paths = profile_likelihood_output_paths(output_root, spec.target_name)

    @info "Starting PLA target $(spec.target_name)." model=spec.model_kind dataset=spec.dataset_name
    log_output_paths(
        (
            target_dir=paths.target_dir,
            per_patient=paths.per_patient_dir,
            composite_figures=paths.composite_fig_dir,
            aggregate_figures=paths.aggregate_fig_dir,
            profiles_long=paths.profiles_long,
            profiles_summary=paths.profiles_summary,
        );
        header="PLA output paths for $(spec.target_name)",
    )

    @info "Loading PLA target inputs."
    target_inputs = load_profile_likelihood_target_inputs(config, settings, spec, selected_model)
    @info "Loaded $(length(target_inputs.patients)) patients for $(spec.target_name)."
    @info "Loaded parameter starts matrix: $(size(target_inputs.reshaped_params))."

    reset_profile_likelihood_output!(
        paths;
        root=output_root,
        compute=cli.run_compute,
        plot_patients=cli.run_plot_patients,
        plot_aggregate=cli.run_plot_aggregate,
    )

    if cli.run_compute
        @info "Computing PLA profiles for $(spec.target_name)."
        compute_profile_likelihood_target(
            target_inputs.patients,
            target_inputs.reshaped_params,
            spec,
            settings,
            paths;
            chain=target_inputs.chain,
            neural_params=target_inputs.neural_params,
        )
        @info "Saved PLA CSV artifacts for $(spec.target_name)."
    end

    if cli.run_plot_patients
        @info "Saving patient-level PLA plots for $(spec.target_name)."
        save_profile_likelihood_patient_plots(
            paths,
            target_inputs.patients,
            spec.dataset_name;
            param_names=spec.param_names,
            pnames_plot=spec.pnames_plot,
            plotting=true,
            show_progress=settings.progress_bars,
        )
    end

    if cli.run_plot_aggregate
        @info "Saving aggregate PLA plots for $(spec.target_name)."
        save_profile_likelihood_aggregate_plots(
            paths,
            spec.dataset_name;
            param_names=spec.param_names,
            pnames_plot=spec.pnames_plot,
            threshold=threshold,
            plotting=true,
            style=settings.aggregate_plot_style,
        )
    end

    @info "Completed PLA target $(spec.target_name)."
end

@info "Profile likelihood workflow completed at $(now())."

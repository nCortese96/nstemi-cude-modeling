"""
01_run_ode_tdsigmoid_fit.jl

Fit the mechanistic Td-sigmoid ODE model on the cohorts produced by workflow
step 00.

Pipeline:
1. Read workflow and ODE settings from `config/workflow_config.jl`.
2. Resolve step 00 cohort inputs and step 01 output paths.
3. Load ordered patient cohorts from JLD2/CSV artifacts.
4. Fit one mechanistic ODE per patient with multi-start optimization.
5. Save fitted parameters, validation subset parameters, and patient plots.
"""

using Dates
using DataFrames: nrow
using Logging

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "preprocessing.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "fitting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings are defined in `config/workflow_config.jl` under
# `WORKFLOW_CONFIG.ode_tdsigmoid`.
# =============================================================================

# =============================================================================
# INPUT PATHS
# Step 01 consumes only step 00 artifacts from `WORKFLOW_CONFIG.outputs.cohorts`.
# =============================================================================

# =============================================================================
# OUTPUT PATHS
# Step 01 writes to `WORKFLOW_CONFIG.ode_tdsigmoid.output_dir`.
# =============================================================================

# =============================================================================
# HELPERS
# Small script-local helpers for config traversal only.
# =============================================================================

"""
    configured_ode_datasets(config, settings)

Return dataset configs selected by `settings.dataset_keys`.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function configured_ode_datasets(config, settings)
    return [getproperty(config.datasets, key) for key in settings.dataset_keys]
end

# =============================================================================
# PIPELINE
# =============================================================================

"""
    main(config=WORKFLOW_CONFIG)

Run the full step 01 Td-sigmoid ODE fitting pipeline.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function main(config=WORKFLOW_CONFIG)
    settings = config.ode_tdsigmoid
    cohort_dir = config.outputs.cohorts
    output_root = settings.output_dir

    log_workflow_context(
        config;
        script_name=basename(@__FILE__),
        output_paths=(
            cohorts=cohort_dir,
            ode_evaluation=output_root,
        ),
    )

    ensure_output_dirs!((ode_evaluation=output_root,); header="Ensured step 01 output root")
    @info "ODE Td-sigmoid fitting started at $(now())"

    for dataset in configured_ode_datasets(config, settings)
        dataset_name = dataset.dataset_name
        paths = ode_dataset_output_paths(output_root, dataset_name)
        ensure_output_dirs!(
            (
                dataset_dir=paths.dataset_dir,
                figures=paths.fig_dir,
            );
            header="Ensured step 01 output directories for $(dataset_name)",
        )

        output_paths = (
            dataset_dir=paths.dataset_dir,
            figures=paths.fig_dir,
            params=paths.params_csv,
        )
        if dataset_name == "MIMIC-IV"
            output_paths = merge(output_paths, (params_validation=paths.params_val_csv,))
        end

        log_output_paths(
            output_paths;
            header="Step 01 output paths for $(dataset_name)",
        )

        cohort = load_preprocessed_cohort(dataset_name, cohort_dir)
        @info "Loaded $(dataset_name) cohort: $(length(cohort.patients)) patients from $(cohort_dir)"

        fit_results = fit_ode_dataset(
            cohort.patients,
            settings;
            dataset_name=dataset_name,
            fig_dir=paths.fig_dir,
        )

        validation_ids = dataset_name == "MIMIC-IV" ? cohort.validation_ids : String[]
        saved = save_ode_fit_results(paths, fit_results; validation_ids=validation_ids)

        @info "Saved $(dataset_name) ODE parameters: $(nrow(saved.all)) rows"
        if saved.validation !== nothing
            @info "Saved $(dataset_name) validation subset: $(nrow(saved.validation)) rows"
        end
    end

    @info "ODE Td-sigmoid fitting completed at $(now())"
end

main()

"""
04b_evaluate_symbolic_formula.jl

Evaluate the fixed official symbolic surrogate formula on the canonical step 00
test cohorts.

Pipeline:
1. Read workflow settings from `config/workflow_config.jl`.
2. Load the configured preprocessed test cohorts from step 00.
3. Fit patient-level ODE parameters for the fixed symbolic formula.
4. Save optimized parameters, metrics, residuals, and patient profile plots.
5. Save the symbolic correction plots used to document the surrogate.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/04b_evaluate_symbolic_formula.jl

Use `config/workflow_config.jl` to switch between `results/` and
`results_test/`, change optimizer settings, or disable plots/progress bars.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using Base.Threads: nthreads
using CSV
using Dates
using Logging

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "fitting.jl"))
include(joinpath(@__DIR__, "..", "src", "diagnostics.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings are defined in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.symbolic_formula_evaluation
dataset_configs = resolve_dataset_configs(config, settings.dataset_keys)

# =============================================================================
# INPUT PATHS
# Step 04b consumes only step 00 preprocessed cohorts.
# =============================================================================

cohort_dir = settings.cohort_dir

# =============================================================================
# OUTPUT PATHS
# Step 04b writes to the canonical symbolic-surrogate evaluation tree.
# =============================================================================

output_root = settings.output_dir

# =============================================================================
# DERIVED SETTINGS
# Values derived from settings and paths. No heavy side effects here.
# =============================================================================

config.model.t_scale == T_SCALE ||
    error("Symbolic formula evaluation requires config.model.t_scale=$(config.model.t_scale) to match models.jl T_SCALE=$(T_SCALE).")

workflow_output_paths = (
    cohorts=cohort_dir,
    symbolic_formula_evaluation=output_root,
)

# =============================================================================
# PIPELINE
# Linear workflow execution.
# =============================================================================

@info "Symbolic formula evaluation workflow started at $(now())."
log_workflow_context(
    config;
    script_name=basename(@__FILE__),
    output_paths=workflow_output_paths,
)

@info "Configured datasets: $([dataset.dataset_name for dataset in dataset_configs])"
@info "Julia threads: $(nthreads())"

ensure_output_dirs!(output_root; header="Ensured step 04b output root")

for dataset_config in dataset_configs
    dataset_name = dataset_config.dataset_name
    paths = symbolic_formula_output_paths(output_root, dataset_name)
    dataset_label = "$(dataset_name)_FORMULA"

    @info "Starting symbolic formula evaluation for $(dataset_name)."
    ensure_output_dirs!(
        (
            dataset=paths.dataset_dir,
            residuals=paths.residuals_dir,
            profiles=paths.profiles_dir,
        );
        header="Ensured symbolic formula output directories for $(dataset_name)",
    )
    log_output_paths(
        (
            best_params=paths.best_params,
            patients_metrics=paths.patients_metrics,
            patients_params=paths.patients_params,
            residuals=paths.residuals_csv,
            parameter_boxplot=paths.parameter_boxplot,
            profiles=paths.profiles_dir,
        );
        header="Symbolic formula output files for $(dataset_name)",
    )

    @info "Loading preprocessed $(dataset_name) cohort from step 00."
    cohort = load_preprocessed_cohort(dataset_name, cohort_dir)
    test_dataset = cohort.test

    @info "Loaded $(length(test_dataset)) test patients for $(dataset_name)."
    @info "Fitting symbolic formula for $(dataset_name)." n_multistart=settings.n_multistart maxiters=settings.maxiters maxtime=settings.maxtime

    evaluation = evaluate_symbolic_formula_dataset(
        test_dataset,
        settings;
        dataset_name=dataset_name,
    )

    isempty(evaluation.successful_patients) &&
        error("No successful symbolic formula patients for $(dataset_name).")

    for (patient, result) in zip(evaluation.successful_patients, evaluation.results)
        save_symbolic_formula_patient_plots(
            result.sol,
            patient,
            dataset_name,
            paths.profiles_dir;
            plotting=settings.plotting,
            display_plots=settings.display_plots,
        )
    end

    saved = save_symbolic_formula_artifacts(
        paths;
        patient_ids=evaluation.patient_ids,
        flat_log_params=evaluation.params_list_flat,
        smapes=evaluation.smape_values,
        rmsles=evaluation.rmsle_values,
        losses=evaluation.loss_values,
        n_params=settings.n_params,
    )

    residuals = compute_symbolic_formula_residuals(
        evaluation.successful_patients,
        evaluation.params_list;
        edges=EDGES,
        n_params=settings.n_params,
    )
    CSV.write(paths.residuals_csv, residuals)

    save_symbolic_formula_parameter_boxplot(
        paths.parameter_boxplot,
        evaluation.params_list_flat;
        n_params=settings.n_params,
        dataset_label=dataset_label,
        plotting=settings.plotting,
        display_plots=settings.display_plots,
    )

    save_symbolic_formula_residual_plots(
        paths,
        residuals;
        dataset_label=dataset_label,
        edges=EDGES,
        tmax=config.model.t_scale,
        plotting=settings.plotting,
        display_plots=settings.display_plots,
    )

    save_symbolic_formula_correction_plots(
        paths;
        t_grid=settings.correction_t_grid,
        beta_values=settings.correction_beta_grid,
        plotting=settings.plotting,
        display_plots=settings.display_plots,
    )

    @info "Saved symbolic formula metrics for $(dataset_name): $(length(evaluation.patient_ids)) patient rows."
    @info "Saved symbolic formula parameter rows for $(dataset_name): $(length(saved.params.patient_id))."
end

@info "Symbolic formula evaluation workflow completed at $(now())."

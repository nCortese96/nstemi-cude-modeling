"""
04b_evaluate_symbolic_formula.jl

Evaluate the manually promoted symbolic surrogate formula on the canonical
step 00 test cohorts.

Pipeline:
1. Read workflow settings from `config/workflow_config.jl`.
2. Load the configured preprocessed test cohorts from step 00.
3. Fit patient-level ODE parameters for the promoted symbolic formula.
4. Save optimized parameters, metrics, residuals, and patient profile plots.
5. Save beta-labelled and T_eff-labelled correction plots.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/04b_evaluate_symbolic_formula.jl
    julia --project=. scripts/04b_evaluate_symbolic_formula.jl plots

The optional `plots` mode regenerates patient profile and correction figures
from existing step 04b artifacts. It does not refit patients and does not write
CSV/JLD2 numerical outputs.

Use `config/workflow_config.jl` to switch between `results/` and
`results_test/`, change optimizer settings, or disable plots/progress bars.

Before running:
1. Run `scripts/04a_run_symbolic_regression.jl`.
2. Inspect the selected equation, Pareto frontier, and NN-vs-SR figures from
   `results*/04_symbolic_surrogate/04a_symbolic_regression/`.
3. Simplify the selected equation into an ODE-stable expression, especially
   near `t_norm = 0`.
4. Promote the formula manually in the final
   `USER-EDITABLE PROMOTED SYMBOLIC FORMULA` section of `src/models.jl`.

This script does not read step 04a outputs at runtime. It fits patient-level
parameters using only the promoted formula currently defined in `src/models.jl`.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using Base.Threads: nthreads
using CSV
using DataFrames: DataFrame
using Dates
using JLD2
using Logging
using OrdinaryDiffEq: Tsit5
using SciMLBase: solve, successful_retcode

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
execution_mode = isempty(ARGS) ? :run :
                 length(ARGS) == 1 && lowercase(strip(ARGS[1])) == "plots" ? :plots :
                 error("Usage: julia --project=. scripts/04b_evaluate_symbolic_formula.jl [plots]")

# =============================================================================
# INPUT PATHS
# Step 04b consumes step 00 cohorts.
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
@info "Execution mode: $(execution_mode)"

ensure_output_dirs!(output_root; header="Ensured step 04b output root")
@info "Using manually promoted symbolic formula from src/models.jl."

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
            correction_surrogate_beta=paths.correction_surrogate_beta,
            correction_surrogate_beta_with_title=paths.correction_surrogate_beta_with_title,
            correction_surrogate_teff=paths.correction_surrogate_teff,
            correction_surrogate_teff_with_title=paths.correction_surrogate_teff_with_title,
        );
        header="Symbolic formula output files for $(dataset_name)",
    )

    @info "Loading preprocessed $(dataset_name) cohort from step 00."
    cohort = load_preprocessed_cohort(dataset_name, cohort_dir)
    test_dataset = cohort.test

    @info "Loaded $(length(test_dataset)) test patients for $(dataset_name)."

    if execution_mode === :plots
        validate_existing_paths(
            (
                best_params=paths.best_params,
                patients_params=paths.patients_params,
            );
            header="Required step 04b plot artifacts for $(dataset_name)",
        )

        params_df = CSV.read(paths.patients_params, DataFrame)
        patient_lookup = Dict(string(patient.id) => patient for patient in test_dataset)
        patient_ids = string.(params_df.patient_id)
        successful_patients = [get(patient_lookup, patient_id, nothing) for patient_id in patient_ids]

        missing_ids = patient_ids[isnothing.(successful_patients)]
        isempty(missing_ids) ||
            error("Cannot regenerate symbolic formula profiles for $(dataset_name): patient IDs missing from step 00 cohort: $(missing_ids)")

        patients_for_plots = PatientData[patient for patient in successful_patients]

        flat_log_params = JLD2.load(paths.best_params, "params_list_flat")
        expected = length(patients_for_plots) * settings.n_params
        length(flat_log_params) == expected ||
            error("Unexpected parameter length in $(paths.best_params): got $(length(flat_log_params)), expected $(expected).")

        for (i, patient) in enumerate(patients_for_plots)
            idx1 = settings.n_params * (i - 1) + 1
            idx2 = settings.n_params * i
            prob = symbolic_formula_problem(flat_log_params[idx1:idx2], patient)
            sol = solve(prob, Tsit5(); saveat=1.0, abstol=1e-8, reltol=1e-6)

            if successful_retcode(sol)
                save_symbolic_formula_patient_plots(
                    sol,
                    patient,
                    dataset_name,
                    paths.profiles_dir;
                    plotting=settings.plotting,
                    display_plots=settings.display_plots,
                )
            else
                @warn "Skipping symbolic formula plot for $(dataset_name) patient $(patient.id): solver retcode $(sol.retcode)"
            end
        end

        save_symbolic_formula_correction_plots(
            paths,
            t_grid=settings.correction_t_grid,
            beta_values=settings.correction_beta_grid,
            plotting=settings.plotting,
            display_plots=settings.display_plots,
        )

        @info "Regenerated symbolic formula plots without refitting for $(dataset_name)."
        continue
    end

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
        paths,
        t_grid=settings.correction_t_grid,
        beta_values=settings.correction_beta_grid,
        plotting=settings.plotting,
        display_plots=settings.display_plots,
    )

    @info "Saved symbolic formula metrics for $(dataset_name): $(length(evaluation.patient_ids)) patient rows."
    @info "Saved symbolic formula parameter rows for $(dataset_name): $(length(saved.params.patient_id))."
end

@info "Symbolic formula evaluation workflow completed at $(now())."

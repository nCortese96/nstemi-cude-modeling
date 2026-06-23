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

Command-line usage:
  julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl
  julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl plots

Threaded execution:
  JULIA_NUM_THREADS=auto julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl
  JULIA_NUM_THREADS=8 julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl

Julia threads are selected when Julia starts. For REPL work, start the REPL with
`JULIA_NUM_THREADS=<N> julia --project=.` before including this script. Progress
bars for multi-start fitting are controlled by `WORKFLOW_CONFIG.run.progress_bars`
and the step-level `WORKFLOW_CONFIG.ode_tdsigmoid.progress_bars`.

Use `plots` to regenerate patient profile SVG/PNG files from existing step 01
CSV parameters and step 00 cohorts without refitting or modifying CSV outputs.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using CSV
using Dates
using DataFrames: DataFrame, nrow
using Logging
using OrdinaryDiffEq: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "fitting.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings are defined in `config/workflow_config.jl` under
# `WORKFLOW_CONFIG.ode_tdsigmoid`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.ode_tdsigmoid
datasets = resolve_dataset_configs(config, settings.dataset_keys)
execution_mode = isempty(ARGS) ? :run :
                 length(ARGS) == 1 && lowercase(strip(ARGS[1])) == "plots" ? :plots :
                 error("Usage: julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl [plots]")

# =============================================================================
# INPUT PATHS
# Step 01 consumes only step 00 artifacts from `WORKFLOW_CONFIG.outputs.cohorts`.
# =============================================================================

cohort_dir = config.outputs.cohorts

# =============================================================================
# OUTPUT PATHS
# Step 01 writes to `WORKFLOW_CONFIG.ode_tdsigmoid.output_dir`.
# =============================================================================

output_root = settings.output_dir

# =============================================================================
# DERIVED SETTINGS
# Values derived from settings and paths. No heavy side effects here.
# =============================================================================

workflow_output_paths = (
    cohorts=cohort_dir,
    ode_evaluation=output_root,
)

# =============================================================================
# PIPELINE
# =============================================================================

log_workflow_context(
    config;
    script_name=basename(@__FILE__),
    output_paths=workflow_output_paths,
)

ensure_output_dirs!((ode_evaluation=output_root,); header="Ensured step 01 output root")
@info "ODE Td-sigmoid fitting started at $(now())"
@info "Execution mode: $(execution_mode)"

for dataset in datasets
    dataset_name = dataset.dataset_name
    @info "Starting ODE Td-sigmoid dataset: $(dataset_name)"

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

    if execution_mode === :plots
        validate_existing_paths(
            (params=paths.params_csv,);
            header="Required step 01 plot artifacts for $(dataset_name)",
        )

        params_df = CSV.read(paths.params_csv, DataFrame)
        required_columns = [:patient, :p1, :p2, :p3, :p4, :p5]
        missing_columns = setdiff(required_columns, Symbol.(names(params_df)))
        isempty(missing_columns) ||
            error("Missing required columns in $(paths.params_csv): $(missing_columns)")

        patient_lookup = Dict(String(patient.id) => patient for patient in cohort.patients)
        for row in eachrow(params_df)
            patient_id = String(row.patient)
            haskey(patient_lookup, patient_id) ||
                error("Patient $(patient_id) from $(paths.params_csv) was not found in the $(dataset_name) cohort.")

            patient = patient_lookup[patient_id]
            params_log = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]
            u0 = initial_conditions_from_log_params(params_log)
            tmax = maximum(patient.timepoints) + 10.0
            problem = ODEProblem(troponin_ode!, u0, (0.0, tmax), params_log)
            sol = solve(problem, Tsit5(); saveat=1.0, abstol=1e-8, reltol=1e-6)
            successful_retcode(sol) ||
                @warn "ODE plot-only solve failed." dataset=dataset_name patient=patient_id

            save_ode_patient_plots(sol, patient, dataset_name, paths.fig_dir; plotting=settings.plotting)
        end

        @info "Regenerated ODE patient plots without refitting." dataset=dataset_name patients=nrow(params_df)
        @info "Completed ODE Td-sigmoid dataset: $(dataset_name)"
        continue
    end

    fit_results = fit_ode_dataset(
        cohort.patients,
        settings;
        dataset_name=dataset_name,
    )

    for (patient, result) in zip(cohort.patients, fit_results)
        save_ode_patient_plots(result.sol, patient, dataset_name, paths.fig_dir; plotting=settings.plotting)
    end

    validation_ids = dataset_name == "MIMIC-IV" ? cohort.validation_ids : String[]
    saved = save_ode_fit_results(paths, fit_results; validation_ids=validation_ids)

    @info "Saved $(dataset_name) ODE parameters: $(nrow(saved.all)) rows"
    if saved.validation !== nothing
        @info "Saved $(dataset_name) validation subset: $(nrow(saved.validation)) rows"
    end

    @info "Completed ODE Td-sigmoid dataset: $(dataset_name)"
end

@info "ODE Td-sigmoid fitting completed at $(now())"

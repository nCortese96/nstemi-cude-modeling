"""
02a_run_cude_training.jl

Train cUDE neural correction models on the preprocessed MIMIC-IV training
cohort produced by step 00.

Pipeline:
1. Load shared helpers and workflow configuration.
2. Resolve step 00 cohort inputs and step 02a output paths.
3. Load the MIMIC-IV training cohort without re-running preprocessing.
4. Train cUDE models for each configured neural-network width.
5. Save canonical training artifacts, loss plots, and reports.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/02a_run_cude_training.jl

Use `config/workflow_config.jl` to switch between `results/` and
`results_test/`, change widths, disable progress bars, or reuse existing
`init_params.jld2` files as selected training starts.
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
include(joinpath(@__DIR__, "..", "src", "fitting.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings live in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.cude_training
dataset_config = config.datasets[settings.dataset_key]

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run.
# =============================================================================

cohort_dir = config.outputs.cohorts
dataset_name = dataset_config.dataset_name

# =============================================================================
# OUTPUT PATHS
# Files and folders produced by this run.
# =============================================================================

output_root = settings.output_dir

# =============================================================================
# DERIVED SETTINGS
# Values derived from config and loaded artifacts. No heavy side effects here.
# =============================================================================

config.model.t_scale == T_SCALE ||
    error("cUDE training currently requires config.model.t_scale=$(config.model.t_scale) to match models.jl T_SCALE=$(T_SCALE).")

widths = settings.widths

# =============================================================================
# PIPELINE
# =============================================================================

@info "cUDE training workflow started at $(now())."
log_workflow_context(
    config;
    script_name="02a_run_cude_training.jl",
    output_paths=(cohort_dir=cohort_dir, output_root=output_root),
)

@info "Loading preprocessed $(dataset_name) cohort from step 00."
cohort = load_preprocessed_cohort(dataset_name, cohort_dir)
training_dataset = cohort.training

@info "Dataset: $(dataset_name)"
@info "Training patients: $(length(training_dataset))"
@info "Configured widths: $(collect(widths))"
@info "Julia threads: $(nthreads())"

ensure_output_dirs!(output_root; header="Ensured step 02a output root")

for width in widths
    @info "Starting cUDE training for $(dataset_name), width=$(width)."

    paths = cude_training_output_paths(output_root, width)
    ensure_output_dirs!(
        (width_dir=paths.width_dir, fig_dir=paths.fig_dir);
        header="Ensured cUDE training output directories for width $(width)",
    )
    log_output_paths(
        (
            init_params=paths.init_params,
            losses=paths.losses,
            nn_weights=paths.nn_weights,
            train_params=paths.train_params,
            report=paths.report,
        );
        header="cUDE training output files for width $(width)",
    )

    initial_parameters = load_existing_cude_initial_parameters(
        paths;
        enabled=settings.reuse_existing_initials,
    )

    training_result = train_cude_width(
        training_dataset,
        settings;
        width=width,
        initial_parameters=initial_parameters,
        initial_parameters_source=initial_parameters === nothing ? nothing : paths.init_params,
    )
    save_cude_training_artifacts(paths, training_result)
    save_cude_training_loss_plots(
        paths,
        training_result.losses_per_model;
        adam_maxiters=settings.adam_maxiters,
        plotting=settings.plotting,
        show_progress=settings.progress_bars,
    )

    if settings.write_report
        write_cude_training_report(
            paths.report,
            training_result,
            settings;
            dataset_name=dataset_name,
            width=width,
            n_patients=length(training_dataset),
            n_threads=nthreads(),
            t_scale=config.model.t_scale,
            paths=paths,
        )
    end

    @info "Completed cUDE training for $(dataset_name), width=$(width)."
end

@info "cUDE training workflow ended at $(now())."

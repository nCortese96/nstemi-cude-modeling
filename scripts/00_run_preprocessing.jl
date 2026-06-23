"""
00_run_preprocessing.jl

Run the shared dataset preprocessing/report pipeline and save reusable patient
sets for downstream training and evaluation scripts.

Pipeline:
1. Read preprocessing configuration from `config/workflow_config.jl`.
2. Resolve input/output paths.
3. Load raw Excel datasets through dedicated preprocessing helpers.
4. Collapse duplicate timepoints, trim, filter anomalies, and report each step.
5. Save all-eligible IDs, validation IDs, gold-standard IDs, and JLD2 artifacts.

Command-line usage:
  julia --project=. scripts/00_run_preprocessing.jl

Workflow-consistent threaded execution:
  JULIA_NUM_THREADS=auto julia --project=. scripts/00_run_preprocessing.jl
  JULIA_NUM_THREADS=8 julia --project=. scripts/00_run_preprocessing.jl

Step 00 also writes the gold-standard cohort artifacts consumed by the
systematic truncation workflow.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using Dates
using Logging

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "preprocessing.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable preprocessing settings live in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
preprocessing = config.preprocessing
filters = preprocessing.filters
gold_standard = preprocessing.gold_standard
datasets = resolve_dataset_configs(config, preprocessing.dataset_keys)

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run.
# =============================================================================

data_root = config.paths.data_root

# =============================================================================
# OUTPUT PATHS
# Files and folders produced by this run.
# =============================================================================

preprocessing_output_dirs = (cohorts=preprocessing.output_dir,)

# =============================================================================
# DERIVED SETTINGS
# Values derived from settings and paths. No heavy side effects here.
# =============================================================================

train_percent_label = round(Int, preprocessing.train_fraction * 100)
test_percent_label = 100 - train_percent_label

# =============================================================================
# PIPELINE
# =============================================================================

@info "═══ Preprocessing pipeline started $(now()) ═══"
log_workflow_context(config;
    script_name=basename(@__FILE__),
    output_paths=preprocessing_output_dirs,
)
ensure_output_dirs!(preprocessing_output_dirs; header="Created or verified preprocessing output directories")
@info "MIMIC-IV split: $(train_percent_label)% train / $(test_percent_label)% test"

results = Dict{String,Any}()

for dataset in datasets
    @info "Processing $(dataset.dataset_name)..."

    dataset_result = run_dataset_report(;
        dataset_name=dataset.dataset_name,
        dataset_path=dataset.dataset_path,
        column_letter=dataset.column_letter,
        T_SCALE=config.model.t_scale,
        meas_min_number=filters.meas_min_number,
        min_acq_time_before=filters.min_acq_time_before,
        min_acq_n_before=filters.min_acq_n_before,
        min_acq_time_after=filters.min_acq_time_after,
        min_acq_n_after=filters.min_acq_n_after,
        min_time=filters.min_time,
        max_gap=filters.max_gap,
        report_dir=preprocessing.output_dir,
        train_fraction=preprocessing.train_fraction,
        split_seed=preprocessing.split_seed,
        data_root=data_root,
    )

    results[dataset.dataset_name] = dataset_result

    dataset_output_paths = preprocessing_output_paths(
        dataset.dataset_name,
        preprocessing.output_dir;
        train_fraction=preprocessing.train_fraction,
        report_path=dataset_result.report_path,
    )
    log_output_paths(dataset_output_paths; header="$(dataset.dataset_name) saved output paths")
end

mimic_dataset = config.datasets[gold_standard.mimic_dataset_key].dataset_name
external_dataset = config.datasets[gold_standard.external_dataset_key].dataset_name

if haskey(results, mimic_dataset) && haskey(results, external_dataset)
    @info "Writing gold-standard cohort artifacts for $(gold_standard.run_dataset_name)."

    mimic_validation_id_set = Set(patient.id for patient in results[mimic_dataset].test)
    mimic_gold_candidates = [
        patient for patient in results[mimic_dataset].all_eligible_patients
        if patient.id in mimic_validation_id_set
    ]

    mimic_validation_ids = save_validation_patient_ids!(
        mimic_dataset,
        results[mimic_dataset].test,
        preprocessing.output_dir;
        suffix="val",
    )

    gold_outputs = save_gold_standard_artifacts!(
        (
            (
                dataset_name=mimic_dataset,
                dataset_tag=gold_standard.mimic_tag,
                patients=mimic_gold_candidates,
                raw_n=results[mimic_dataset].raw_n,
            ),
            (
                dataset_name=external_dataset,
                dataset_tag=gold_standard.external_tag,
                patients=results[external_dataset].all_eligible_patients,
                raw_n=results[external_dataset].raw_n,
            ),
        ),
        preprocessing.output_dir;
        run_dataset_name=gold_standard.run_dataset_name,
        filters=gold_standard.filters,
    )

    log_output_paths(
        (
            mimic_validation_ids=mimic_validation_ids,
            gold_ids=gold_outputs.ids_path,
            gold_report=gold_outputs.report_path,
        );
        header="Gold-standard preprocessing artifacts",
    )
else
    @warn "Gold-standard artifacts were not written because one or more required datasets were not processed." required=(mimic_dataset, external_dataset) available=collect(keys(results))
end

@info "═══ Preprocessing pipeline completed ═══"

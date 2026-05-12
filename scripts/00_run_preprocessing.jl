"""
00_run_preprocessing.jl

Run the shared dataset preprocessing/report pipeline and save reusable patient
sets for downstream training and evaluation scripts.

Pipeline:
1. Read preprocessing configuration from `config/workflow_config.jl`.
2. Resolve input/output paths.
3. Load raw Excel datasets through helpers.jl.
4. Collapse duplicate timepoints, trim, filter anomalies, and report each step.
5. Save all-eligible IDs plus JLD2 train/test artifacts.
"""

using Dates
using Logging

include(joinpath(@__DIR__, "..", "src", "MechanisticAI.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable preprocessing settings live in `config/workflow_config.jl`.
# =============================================================================

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run.
# =============================================================================

# =============================================================================
# OUTPUT PATHS
# Files and folders produced by this run.
# =============================================================================

# =============================================================================
# DERIVED SETTINGS
# Values derived from settings and paths. No heavy side effects here.
# =============================================================================

# =============================================================================
# PIPELINE
# Main readable execution flow.
# =============================================================================

config = WORKFLOW_CONFIG
preprocessing = config.preprocessing
filters = preprocessing.filters
datasets = [config.datasets[key] for key in preprocessing.dataset_keys]

train_percent_label = round(Int, preprocessing.train_fraction * 100)
test_percent_label = 100 - train_percent_label

@info "═══ Preprocessing pipeline started $(now()) ═══"
@info "MIMIC-IV split: $(train_percent_label)% train / $(test_percent_label)% test"

results = Dict{String,Any}()

for dataset in datasets
    @info "Processing $(dataset.dataset_name)..."

    results[dataset.dataset_name] = run_dataset_report(;
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
        data_root=config.paths.data_root,
    )

    @info "$(dataset.dataset_name) report: $(results[dataset.dataset_name].report_path)"
end

@info "═══ Preprocessing pipeline completed ═══"

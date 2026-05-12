"""
workflow_config.jl

Central workflow configuration for the refactored scripts.

Edit this file when paths, datasets, or run-level settings need to change.
Scripts should read values from `WORKFLOW_CONFIG` instead of redefining the
same constants locally.
"""

# =============================================================================
# PATHS
# Shared filesystem roots used by the refactored workflow.
# =============================================================================

const WORKFLOW_PATHS = (
    data_root="data",
    results_root="results",
)

# =============================================================================
# DATASETS
# Dataset registry used by preprocessing script.
# =============================================================================

const WORKFLOW_DATASETS = (
    mimic_iv=(
        dataset_name="MIMIC-IV",
        dataset_path="MIMIC-IV/NSTEMI_reorganized_skipped.xlsx",
        column_letter="B",
    ),
    umg=(
        dataset_name="UMG",
        dataset_path="UMG_NSTEMI_Dataset.xlsx",
        column_letter="A",
    ),
)

# =============================================================================
# MODEL AND TIME SETTINGS
# Shared model-domain constants used by multiple workflow steps.
#
# `t_scale` is measured in hours. It is used by preprocessing to define the
# analysis window and by cUDE models to normalize time as `t / t_scale`.
# =============================================================================

const WORKFLOW_MODEL_SETTINGS = (
    t_scale=240.0,
)

# =============================================================================
# PREPROCESSING SETTINGS
# Settings controlling dataset selection, splitting, and output location.
# =============================================================================

const PREPROCESSING_SETTINGS = (
    dataset_keys=(:mimic_iv, :umg),
    output_dir=WORKFLOW_PATHS.results_root,
    train_fraction=0.8,
    split_seed=1234,
)

# =============================================================================
# PREPROCESSING FILTERS
# Eligibility thresholds applied after duplicate collapse and time trimming.
# =============================================================================

const PREPROCESSING_FILTERS = (
    meas_min_number=5,
    min_acq_time_before=12.0,
    min_acq_n_before=1,
    min_acq_time_after=48.0,
    min_acq_n_after=1,
    min_time=72.0,
    max_gap=72.0,
)

const WORKFLOW_CONFIG = (
    paths=WORKFLOW_PATHS,
    datasets=WORKFLOW_DATASETS,
    model=WORKFLOW_MODEL_SETTINGS,
    preprocessing=merge(PREPROCESSING_SETTINGS, (filters=PREPROCESSING_FILTERS,)),
)

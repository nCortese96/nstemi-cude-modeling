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
    results_test_root="results_test",
)

# =============================================================================
# RUN MODE
# Global workflow mode switches shared by refactored scripts.
#
# Set `test_mode=true` to keep trial runs isolated under `results_test`.
# Set `test_mode=false` to write standard workflow outputs under `results`.
# =============================================================================

const WORKFLOW_RUN_MODE = (
    test_mode=true,
)

const ACTIVE_RESULTS_ROOT = WORKFLOW_RUN_MODE.test_mode ?
                            WORKFLOW_PATHS.results_test_root :
                            WORKFLOW_PATHS.results_root

# =============================================================================
# WORKFLOW OUTPUT TREE
# Canonical output directories mirrored under `results` or `results_test`.
#
# Scripts should write to these mapped paths instead of composing output roots
# locally. Additional step-level paths will be added here as each script is
# refactored.
# =============================================================================

const WORKFLOW_OUTPUT_DIRS = (
    cohorts=joinpath(ACTIVE_RESULTS_ROOT, "00_cohorts"),
    ode_evaluation=joinpath(ACTIVE_RESULTS_ROOT, "01_ode_evaluation"),
    cude_workflow=joinpath(ACTIVE_RESULTS_ROOT, "02_cude_workflow"),
    comparison_analyses=joinpath(ACTIVE_RESULTS_ROOT, "03_comparison_analyses"),
)

# =============================================================================
# DATASETS
# Dataset registry used by refactored workflow scripts.
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
    output_dir=WORKFLOW_OUTPUT_DIRS.cohorts,
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

# =============================================================================
# ODE TD-SIGMOID FIT SETTINGS
# Settings controlling workflow step 01.
#
# This step loads preprocessed cohorts from step 00 and estimates patient-level
# mechanistic ODE parameters without re-reading raw Excel files.
# =============================================================================

const ODE_TDSIGMOID_SETTINGS = (
    dataset_keys=(:mimic_iv, :umg),
    output_dir=WORKFLOW_OUTPUT_DIRS.ode_evaluation,
    pguess=log.([0.005, 0.005, 0.01, 0.01, 30.0]),
    lower=log.([0.001, 0.001, 0.001, 0.001, 0.001]),
    upper=log.([10.0, 10.0, 500.0, 500.0, 500.0]),
    lambda_back=1.0,
    n_multistart=40,
    rng_seed=1234,
    maxiters=1000,
    maxtime=80.0,
    prescreen=false,
    topk=8,
    plotting=true,
)

const WORKFLOW_CONFIG = (
    paths=merge(WORKFLOW_PATHS, (active_results_root=ACTIVE_RESULTS_ROOT,)),
    run=WORKFLOW_RUN_MODE,
    outputs=WORKFLOW_OUTPUT_DIRS,
    datasets=WORKFLOW_DATASETS,
    model=WORKFLOW_MODEL_SETTINGS,
    preprocessing=merge(PREPROCESSING_SETTINGS, (filters=PREPROCESSING_FILTERS,)),
    ode_tdsigmoid=ODE_TDSIGMOID_SETTINGS,
)

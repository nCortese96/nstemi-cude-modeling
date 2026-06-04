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
# Set `progress_bars=false` to disable progress bars in time-consuming steps.
# =============================================================================

const WORKFLOW_RUN_MODE = (
    test_mode=true,
    progress_bars=true,
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
    symbolic_surrogate=joinpath(ACTIVE_RESULTS_ROOT, "04_symbolic_surrogate"),
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
# GOLD-STANDARD COHORT SETTINGS
# Stricter high-information filters used by the truncation stress workflow.
#
# These artifacts are produced by step 00 so downstream scripts can consume a
# stable gold-standard cohort without re-reading raw Excel files.
# =============================================================================

const GOLD_STANDARD_FILTERS = (
    meas_min_number=8,
    min_acq_time_before=12.0,
    min_acq_n_before=1,
    min_acq_time_after=48.0,
    min_acq_n_after=1,
    min_time=72.0,
    max_gap=24.0,
)

const GOLD_STANDARD_SETTINGS = (
    run_dataset_name="MIMIC-UMG",
    mimic_dataset_key=:mimic_iv,
    external_dataset_key=:umg,
    mimic_tag="MIMIC",
    external_tag="UMG",
    filters=GOLD_STANDARD_FILTERS,
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
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
)

# =============================================================================
# CUDE TRAINING SETTINGS
# Settings controlling workflow step 02a.
#
# This step loads the MIMIC-IV training cohort generated by step 00 and trains
# cUDE models for each configured neural-network width.
#
# Set `reuse_existing_initials=true` to load `init_params.jld2` from each
# width output directory and train from those selected starts when available.
# =============================================================================

const CUDE_TRAINING_SETTINGS = (
    dataset_key=:mimic_iv,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.cude_workflow, "02a_cude_training"),
    widths=(4, 6, 8),
    nn_depth=2,
    input_dim=2,
    n_params=5,
    n_conditional=1,
    lower=log.([0.001, 0.001, 0.001, 0.001, 0.001]),
    upper=log.([5.0, 5.0, 500.0, 500.0, 1.0]),
    initial_guesses=25000,
    selected_initials=4,
    reuse_existing_initials=false,
    rng_seed=42,
    lambda_back=1.0,
    kappa_bounds=0.05,
    adam_maxiters=500,
    adam_eta=0.01,
    lbfgs_maxiters=300,
    lbfgs_tolerances=(g=1e-6, f=1e-6, x=1e-6),
    plotting=true,
    write_report=true,
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
)

# =============================================================================
# CUDE EVALUATION SETTINGS
# Settings controlling workflow step 02b.
#
# This step loads trained cUDE candidates from step 02a and evaluates each
# candidate on the MIMIC-IV validation/test split generated by step 00.
# =============================================================================

const CUDE_EVALUATION_SETTINGS = (
    dataset_key=:mimic_iv,
    training_input_dir=CUDE_TRAINING_SETTINGS.output_dir,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.cude_workflow, "02b_cude_evaluation"),
    widths=(4, 6, 8),
    model_indices=(1, 2, 3, 4),
    nn_depth=2,
    input_dim=2,
    n_params=5,
    lower=log.([0.001, 0.001, 0.001, 0.001, 0.001]),
    upper=log.([10.0, 10.0, 500.0, 500.0, 1.0]),
    lambda_back=1.0,
    n_multistart=40,
    rng_seed=1234,
    maxiters=1000,
    maxtime=80.0,
    prescreen=false,
    topk=12,
    bounds=true,
    plotting=true,
    display_plots=false,
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
)

# =============================================================================
# CUDE MODEL SELECTION SETTINGS
# Settings controlling workflow step 02c.
#
# This step loads model-summary CSV files from step 02b and selects the best
# cUDE candidate with a configurable ranking rule.
# =============================================================================

const CUDE_MODEL_SELECTION_SETTINGS = (
    dataset_key=:mimic_iv,
    input_dir=CUDE_EVALUATION_SETTINGS.output_dir,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.cude_workflow, "02c_cude_model_selection"),
    widths=CUDE_EVALUATION_SETTINGS.widths,
    selection_rule=:robust_loss_mean,
    plotting=true,
)

# =============================================================================
# CUDE EXTERNAL TEST SETTINGS
# Settings controlling workflow step 02d.
#
# This step loads the selected MIMIC-IV cUDE model from step 02c and evaluates
# it on the external UMG cohort generated by step 00.
# =============================================================================

const CUDE_EXTERNAL_TEST_SETTINGS = (
    training_dataset_key=:mimic_iv,
    external_dataset_key=:umg,
    model_selection_dataset_key=:mimic_iv,
    training_input_dir=CUDE_TRAINING_SETTINGS.output_dir,
    model_selection_dir=CUDE_MODEL_SELECTION_SETTINGS.output_dir,
    selected_model_path=nothing,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.cude_workflow, "02d_cude_external_test"),
    nn_depth=CUDE_EVALUATION_SETTINGS.nn_depth,
    input_dim=CUDE_EVALUATION_SETTINGS.input_dim,
    n_params=CUDE_EVALUATION_SETTINGS.n_params,
    lower=CUDE_EVALUATION_SETTINGS.lower,
    upper=CUDE_EVALUATION_SETTINGS.upper,
    lambda_back=CUDE_EVALUATION_SETTINGS.lambda_back,
    n_multistart=CUDE_EVALUATION_SETTINGS.n_multistart,
    rng_seed=CUDE_EVALUATION_SETTINGS.rng_seed,
    maxiters=CUDE_EVALUATION_SETTINGS.maxiters,
    maxtime=CUDE_EVALUATION_SETTINGS.maxtime,
    prescreen=CUDE_EVALUATION_SETTINGS.prescreen,
    topk=CUDE_EVALUATION_SETTINGS.topk,
    bounds=CUDE_EVALUATION_SETTINGS.bounds,
    plotting=true,
    display_plots=false,
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
)

# =============================================================================
# MODEL DIAGNOSTICS SETTINGS
# Settings controlling workflow step 03a.
#
# This step compares ODE and selected cUDE patient-level outputs across MIMIC-IV
# and UMG, then writes diagnostic tables and plots under the comparison tree.
# =============================================================================

const MODEL_DIAGNOSTICS_SETTINGS = (
    mimic_dataset_key=:mimic_iv,
    external_dataset_key=:umg,
    model_selection_dataset_key=:mimic_iv,
    cohort_dir=PREPROCESSING_SETTINGS.output_dir,
    ode_input_dir=ODE_TDSIGMOID_SETTINGS.output_dir,
    cude_training_input_dir=CUDE_TRAINING_SETTINGS.output_dir,
    cude_evaluation_input_dir=CUDE_EVALUATION_SETTINGS.output_dir,
    cude_external_test_input_dir=CUDE_EXTERNAL_TEST_SETTINGS.output_dir,
    model_selection_dir=CUDE_MODEL_SELECTION_SETTINGS.output_dir,
    selected_model_path=nothing,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.comparison_analyses, "03a_diagnostics"),
    input_dim=CUDE_EVALUATION_SETTINGS.input_dim,
    time_edges=(0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, WORKFLOW_MODEL_SETTINGS.t_scale),
    residual_tmax=WORKFLOW_MODEL_SETTINGS.t_scale,
    delta_smape_threshold=1.0,
    profile_selection_seed=42,
    profile_rows_per_group=10,
    plotting=true,
    metrics_paper_plots=true,
    profile_comparison=true,
    residual_plot_style=(
        figure_fontsize=18,
        bin_label_fontsize=18,
        scatter_markersize=5,
        summary_linewidth=2.5,
    ),
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
)

# =============================================================================
# PROFILE LIKELIHOOD SETTINGS
# Settings controlling workflow step 03b.
#
# This step runs profile likelihood analysis for the selected cUDE model and
# the ODE baseline on the MIMIC-IV validation/test cohort and UMG external
# cohort. The PLA numerical core intentionally keeps its own refit/profile
# optimizer path to preserve legacy behavior.
# =============================================================================

const PROFILE_LIKELIHOOD_SETTINGS = (
    mimic_dataset_key=:mimic_iv,
    external_dataset_key=:umg,
    model_selection_dataset_key=:mimic_iv,
    cohort_dir=PREPROCESSING_SETTINGS.output_dir,
    ode_input_dir=ODE_TDSIGMOID_SETTINGS.output_dir,
    cude_training_input_dir=CUDE_TRAINING_SETTINGS.output_dir,
    cude_evaluation_input_dir=CUDE_EVALUATION_SETTINGS.output_dir,
    cude_external_test_input_dir=CUDE_EXTERNAL_TEST_SETTINGS.output_dir,
    model_selection_dir=CUDE_MODEL_SELECTION_SETTINGS.output_dir,
    selected_model_path=nothing,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.comparison_analyses, "03b_pla"),
    targets=(:cude_mimic, :cude_umg, :ode_mimic, :ode_umg),
    default_target="all",
    run_compute=true,
    run_plot_patients=true,
    run_plot_aggregate=true,
    input_dim=CUDE_EVALUATION_SETTINGS.input_dim,
    n_params=5,
    cude_lower=CUDE_EVALUATION_SETTINGS.lower,
    cude_upper=CUDE_EVALUATION_SETTINGS.upper,
    ode_lower=ODE_TDSIGMOID_SETTINGS.lower,
    ode_upper=ODE_TDSIGMOID_SETTINGS.upper,
    refit_maxiters=1000,
    profile_maxiters=100000,
    step_scale=0.2,
    span=1.25,
    expand_tries=3,
    expand_factor=2.0,
    eps_bound=1e-7,
    separate=false,
    aggregate_plot_style=(
        legend_title_fontsize=20,
        legend_label_fontsize=15,
        legend_note_fontsize=13,
        subplot_legend_fontsize=15,
        subplot_tickfontsize=12,
        subplot_guidefontsize=14,
        subplot_legend_position=:topright,
        subplot_profile_linewidth=2.0,
        subplot_profile_alpha=0.85,
        subplot_threshold_linewidth=2.2,
        subplot_png_dpi=300,
        subplot_left_margin_mm=4,
        subplot_bottom_margin_mm=5,
        subplot_right_margin_mm=3,
        subplot_top_margin_mm=3,
        # Use `nothing` to hide the combined aggregate title, `:default` to
        # restore the dataset-specific legacy title, or a string for a custom title.
        combined_title=nothing,
        combined_title_fontsize=20,
        combined_left_margin_mm=4,
        combined_right_margin_mm=4,
        combined_bottom_margin_mm=6,
        combined_top_margin_mm=5,
    ),
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
)

# =============================================================================
# SYSTEMATIC TRUNCATION SETTINGS
# Settings controlling workflow step 03c.
#
# This step runs ODE and selected-cUDE stress tests on the gold-standard cohort,
# then creates summary tables and ODE-vs-cUDE overlay plots.
# =============================================================================

const SYSTEMATIC_TRUNCATION_SETTINGS = (
    mimic_dataset_key=:mimic_iv,
    external_dataset_key=:umg,
    model_selection_dataset_key=:mimic_iv,
    cohort_dir=PREPROCESSING_SETTINGS.output_dir,
    cude_training_input_dir=CUDE_TRAINING_SETTINGS.output_dir,
    model_selection_dir=CUDE_MODEL_SELECTION_SETTINGS.output_dir,
    selected_model_path=nothing,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.comparison_analyses, "03c_truncation_stress"),
    run_dataset_name=GOLD_STANDARD_SETTINGS.run_dataset_name,
    default_target="all",
    valid_targets=("all", "ode", "cude", "summary", "overlay", "plots"),
    min_keep_meas=4,
    truncation_levels=(0.35, 0.70),
    truncation_sections=(:start, :middle, :end),
    n_multistart=40,
    topk=8,
    maxiters=5000,
    maxtime=80.0,
    ode_pguess=log.([0.005, 0.005, 0.01, 0.01, 30.0]),
    ode_lower=log.([0.001, 0.001, 0.001, 0.001, 0.001]),
    ode_upper=log.([10.0, 10.0, 500.0, 500.0, 500.0]),
    cude_pguess=log.([0.005, 0.005, 0.1, 0.01, 0.5]),
    cude_lower=CUDE_EVALUATION_SETTINGS.lower,
    cude_upper=CUDE_EVALUATION_SETTINGS.upper,
    lambda_back=1.0,
    input_dim=CUDE_EVALUATION_SETTINGS.input_dim,
    plotting=true,
    display_plots=false,
    overlay_legend=false,
    # Save a mirrored overlay set without axis titles while preserving ticks.
    overlay_no_labels=true,
    plot_style=(
        size=(1000, 650),
        left_margin_mm=12,
        bottom_margin_mm=9,
        right_margin_mm=5,
        top_margin_mm=6,
        guidefontsize=13,
        tickfontsize=11,
        titlefontsize=12,
        legendfontsize=9,
    ),
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
    gold_standard=GOLD_STANDARD_SETTINGS,
)

# =============================================================================
# SYMBOLIC REGRESSION SETTINGS
# Settings controlling workflow step 04a.
#
# This step fits a symbolic surrogate to the selected cUDE neural correction
# function. The default grid and symbolic-regression options mirror the current
# legacy script.
# =============================================================================

const SYMBOLIC_REGRESSION_SETTINGS = (
    model_selection_dataset_key=:mimic_iv,
    cude_training_input_dir=CUDE_TRAINING_SETTINGS.output_dir,
    model_selection_dir=CUDE_MODEL_SELECTION_SETTINGS.output_dir,
    selected_model_path=nothing,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.symbolic_surrogate, "04a_symbolic_regression"),
    input_dim=CUDE_EVALUATION_SETTINGS.input_dim,
    t_scale=WORKFLOW_MODEL_SETTINGS.t_scale,
    tmax_sr_h=2400.0,
    # tmax_sr_h=6500.0, #2400.0,
    t_grid=unique(vcat(
        collect(0.01:2.0:248.01),
        collect(260.0:10.0:2100.0),
        collect(2102.0:2.0:2400.0),
    )),
    # t_grid=collect(0.01:10.0:6500.0),
    beta_grid=collect(range(0.1, 1.0, length=20)),
    # beta_grid=collect(range(0.2, 0.8, length=20)), #Default=18, Last=#20
    plot_beta_grid=collect(0.1:0.1:1.0),
    variable_names=("t_norm", "β"),
    binary_operators=(+, *),
    unary_operators=(inv,),
    maxsize=20,
    # maxsize=16, #20
    populations=24,
    parsimony=5e-4,
    # parsimony=1e-3, #5e-4
    complexity_of_constants=2,
    batching=true,
    batch_size=512,
    should_optimize_constants=true,
    lambda_negative=200.0,
    lambda_high=20.0,
    smooth_eps=1e-5,
    niterations_warmup=300,
    niterations_main=25000,
    seed=42,
    teacher_mse_tolerance=1.02,
    plotting=true,
    display_plots=false,
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
)

# =============================================================================
# SYMBOLIC FORMULA EVALUATION SETTINGS
# Settings controlling workflow step 04b.
#
# This step evaluates the manually promoted formula defined at the end of
# `src/models.jl`.
# =============================================================================

const SYMBOLIC_FORMULA_EVALUATION_SETTINGS = (
    dataset_keys=(:mimic_iv, :umg),
    cohort_dir=PREPROCESSING_SETTINGS.output_dir,
    output_dir=joinpath(WORKFLOW_OUTPUT_DIRS.symbolic_surrogate, "04b_surrogate_optimization"),
    n_params=5,
    pguess=log.([0.005, 0.005, 0.1, 0.01, 0.5]),
    lower=log.([0.001, 0.001, 0.001, 0.001, 0.001]),
    upper=log.([10.0, 10.0, 500.0, 500.0, 1.0]),
    lambda_back=1.0,
    n_multistart=40,
    rng_seed=1234,
    maxiters=1000,
    maxtime=80.0,
    prescreen=false,
    topk=8,
    plotting=true,
    display_plots=false,
    progress_bars=WORKFLOW_RUN_MODE.progress_bars,
    correction_t_grid=collect(0.1:0.1:2400.0),
    correction_beta_grid=collect(0.1:0.1:1.0),
)

const WORKFLOW_CONFIG = (
    paths=merge(WORKFLOW_PATHS, (active_results_root=ACTIVE_RESULTS_ROOT,)),
    run=WORKFLOW_RUN_MODE,
    outputs=WORKFLOW_OUTPUT_DIRS,
    datasets=WORKFLOW_DATASETS,
    model=WORKFLOW_MODEL_SETTINGS,
    preprocessing=merge(PREPROCESSING_SETTINGS, (filters=PREPROCESSING_FILTERS, gold_standard=GOLD_STANDARD_SETTINGS)),
    ode_tdsigmoid=ODE_TDSIGMOID_SETTINGS,
    cude_training=CUDE_TRAINING_SETTINGS,
    cude_evaluation=CUDE_EVALUATION_SETTINGS,
    cude_model_selection=CUDE_MODEL_SELECTION_SETTINGS,
    cude_external_test=CUDE_EXTERNAL_TEST_SETTINGS,
    model_diagnostics=MODEL_DIAGNOSTICS_SETTINGS,
    profile_likelihood=PROFILE_LIKELIHOOD_SETTINGS,
    systematic_truncation=SYSTEMATIC_TRUNCATION_SETTINGS,
    symbolic_regression=SYMBOLIC_REGRESSION_SETTINGS,
    symbolic_formula_evaluation=SYMBOLIC_FORMULA_EVALUATION_SETTINGS,
)

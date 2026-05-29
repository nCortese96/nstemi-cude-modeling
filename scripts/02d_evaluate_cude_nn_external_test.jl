"""
02d_evaluate_cude_nn_external_test.jl

Evaluate the selected cUDE model from step 02c on the external UMG cohort.

Pipeline:
1. Load shared helpers and workflow configuration.
2. Resolve step 00 cohorts, step 02a model artifacts, step 02c selected model,
   and step 02d output paths.
3. Load the selected cUDE candidate and derive the patient-level ODE pguess
   from its training parameters.
4. Evaluate the selected model on the external cohort with patient-level
   multi-start fitting.
5. Save optimized parameters, patient metrics, and patient profile plots.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/02d_evaluate_cude_nn_external_test.jl

Use `config/workflow_config.jl` to switch between `results/` and
`results_test/`, change optimizer settings, or disable plots/progress bars.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using Dates
using DataFrames: nrow
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
settings = config.cude_external_test
training_dataset_config = config.datasets[settings.training_dataset_key]
external_dataset_config = config.datasets[settings.external_dataset_key]
selection_dataset_config = config.datasets[settings.model_selection_dataset_key]

# =============================================================================
# INPUT PATHS
# Step 02d consumes step 00 cohorts, step 02a training artifacts, and the step
# 02c selected-model CSV.
# =============================================================================

cohort_dir = config.preprocessing.output_dir
training_input_dir = settings.training_input_dir
model_selection_dir = settings.model_selection_dir
selection_paths = cude_model_selection_output_paths(model_selection_dir, selection_dataset_config.dataset_name)
selected_model_path = settings.selected_model_path === nothing ?
                      selection_paths.selected_model :
                      settings.selected_model_path

# =============================================================================
# OUTPUT PATHS
# Step 02d follows the flat official external-test tree under
# `02d_cude_external_test`.
# =============================================================================

output_root = settings.output_dir
external_dataset_name = external_dataset_config.dataset_name
paths = cude_external_test_output_paths(output_root, external_dataset_name)

# =============================================================================
# DERIVED SETTINGS
# Values derived from config and loaded artifacts. No heavy side effects here.
# =============================================================================

training_dataset_name = training_dataset_config.dataset_name
selection_dataset_name = selection_dataset_config.dataset_name

# =============================================================================
# PIPELINE
# =============================================================================

@info "cUDE external-test workflow started at $(now())."
log_workflow_context(
    config;
    script_name="02d_evaluate_cude_nn_external_test.jl",
    output_paths=(
        cohort_dir=cohort_dir,
        training_input_dir=training_input_dir,
        selected_model=selected_model_path,
        output_root=output_root,
    ),
)

@info "Training dataset: $(training_dataset_name)"
@info "External dataset: $(external_dataset_name)"
@info "Selection dataset: $(selection_dataset_name)"
@info "Julia threads: $(nthreads())"

ensure_output_dirs!(
    (output_root=paths.output_dir, profiles=paths.profiles_dir);
    header="Ensured cUDE external-test output directories",
)
log_output_paths(
    (
        best_params=paths.best_params,
        patients_params_train=paths.patients_params_train,
        patients_params_val=paths.patients_params_val,
        patients_metrics_val=paths.patients_metrics_val,
        profiles=paths.profiles_dir,
    );
    header="cUDE external-test output files",
)

@info "Loading step 00 cohorts."
training_cohort = load_preprocessed_cohort(training_dataset_name, cohort_dir)
external_cohort = load_preprocessed_cohort(external_dataset_name, cohort_dir)

training_dataset = hasproperty(training_cohort, :training) ? training_cohort.training : training_cohort.test
external_dataset = external_cohort.test
training_ids = [patient.id for patient in training_dataset]

@info "Loaded $(length(training_dataset)) training patients from $(training_dataset_name)."
@info "Loaded $(length(external_dataset)) external-test patients from $(external_dataset_name)."

@info "Loading selected cUDE model from step 02c."
selected_model = load_selected_cude_model(selected_model_path)
width = selected_model.nn_width
model_idx = selected_model.model_idx

settings.nn_depth == selected_model.nn_depth ||
    error("Configured nn_depth=$(settings.nn_depth) does not match selected model nn_depth=$(selected_model.nn_depth).")

@info "Selected cUDE model: $(selected_model.model_id) | width=$(width) | model_idx=$(model_idx)"

@info "Loading step 02a cUDE training artifacts for width=$(width)."
artifacts = load_cude_training_artifacts(training_input_dir, width)
n_candidates = length(artifacts.neural_network_parameters)
1 <= model_idx <= n_candidates ||
    error("Selected model_idx=$(model_idx) is outside available candidates 1:$(n_candidates) for width $(width).")

chain = neural_network_model(settings.nn_depth, width; input_dims=settings.input_dim)
neural_params = artifacts.neural_network_parameters[model_idx]
training_log_params = Vector(artifacts.ode_params[model_idx])
pguess_stats = median_log_parameter_guess(training_log_params; n_params=settings.n_params)
pguess = pguess_stats.pguess

@info "Selected external-test pguess for width=$(width), model=$(model_idx): $(exp.(pguess))"
@info "Training parameter median [Q1-Q3] for width=$(width), model=$(model_idx): $(pguess_stats.median_natural) [$(pguess_stats.q1_natural), $(pguess_stats.q3_natural)]"

evaluation = evaluate_cude_model(
    external_dataset,
    chain,
    neural_params,
    pguess,
    settings;
    dataset_name=external_dataset_name,
    width=width,
    model_idx=model_idx,
)

isempty(evaluation.successful_patients) &&
    error("No successful external-test patients for $(external_dataset_name), width=$(width), model=$(model_idx).")

for (patient, result) in zip(evaluation.successful_patients, evaluation.results)
    save_cude_patient_plots(
        result.sol,
        patient,
        external_dataset_name,
        paths.profiles_dir;
        plotting=settings.plotting,
        display_plots=settings.display_plots,
    )
end

saved = save_cude_evaluation_artifacts(
    paths;
    training_ids=training_ids,
    training_log_params=training_log_params,
    validation_ids=evaluation.patient_ids,
    validation_log_params=evaluation.ode_params_val,
    smapes=evaluation.smape_values,
    rmsles=evaluation.rmsle_values,
    losses=evaluation.loss_values,
    n_params=settings.n_params,
)

@info "Saved external-test metrics for $(external_dataset_name): $(nrow(saved.metrics)) patient rows."
@info "cUDE external-test workflow completed at $(now())."

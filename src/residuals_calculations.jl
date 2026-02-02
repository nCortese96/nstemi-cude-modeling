using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using JLD2, CSV
using ProgressMeter
using Statistics
using Dates
using Logging

using Revise

includet("ctnt-ude-model.jl")

@info "Starting residual calculation script"

UMG_data = true;

UDE = false; # false for cUDE

best_idx = 4; # index of the best model to test 

if UDE
    @info "Using UDE model"
    input_dim = 1;
    nn_depth = 2;
    nn_width = 8;
    N_params = 4;
    inputs_str = "τ";
else
    @info "Using cUDE model"
    input_dim = 2;
    nn_depth = 2;
    nn_width = 8;
    N_params = 5;
    inputs_str = "τ, β";
end 


T_SCALE = 350.0;

const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, 350.0];

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

# experiment = "NSTEMI_UDE_UMG_MSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus";
# experiment = "NSTEMI_partrval_UMG_MSE_ts350.0_28_inp2_multipl_softplus";

fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
modelssave_path = "res/$(experiment)/models/test_NN";

@info "Loading dataset"
test_dataset = if UMG_data
    dataset = "UMG";
    
    figsave_path = "$(fig_path)/umg_test_nn_$(best_idx)";
    modelssave_path = "$(models_path)/umg_test_nn_$(best_idx)";   
        
    mkpath(figsave_path)
    mkpath(modelssave_path)

    @load "$(modelssave_path)/UMG_testset.jld2" test_dataset;
    @info "$dataset test dataset loaded from previous save"
    test_dataset;
else
    dataset = "MIMIC-IV";

    figsave_path = "res/$(experiment)/figs/test_NN_$(best_idx)";
    modelssave_path = "res/$(experiment)/models/test_NN_$(best_idx)";

    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
    @info "Using $dataset dataset for residuals"

    test_dataset;
end
@info "Dataset $dataset loaded with $(length(test_dataset)) patients"

@info "Loading best model and parameters"
best_nn, ode_params_val = try
    @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
    @load "$(modelssave_path)/best_params_val_$(dataset).jld2" ode_params_val;
    best_nn, ode_params_val;
catch e

    @warn "Error loading model: $e"

    @info "Using best idx [$(best_idx)] for load model"
    
    @load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
    @load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

    best_nn = neural_network_parameters[best_idx];
    ode_params_val = ode_params[best_idx]; # log version where 1 is the index of the best model in info_output
    best_nn, ode_params_val;
end
@info "Best model and parameters loaded successfully"
@info "Loaded $(length(ode_params_val)) parameters"

hi_ids = CSV.read("res/ids_high_information_$(dataset)_minafter.csv", DataFrame);
# high_information = filter(p -> p.id in hi_ids.patient, test_dataset);
high_information_idxs = findall(p -> p.id in hi_ids.patient, test_dataset);
high_information = test_dataset[high_information_idxs];
patient_dims(high_information)

ode_params_val_hi = [ode_params_val[N_params * (i-1) + 1:N_params * i] for i in high_information_idxs];
ode_params_val_hi = vcat(ode_params_val_hi...);

@info "Total sample number for high information: $(length(high_information))"

@info "Processing $(experiment) residuals data"

residuals_ae, smape_ae = compute_plot_residuals(test_dataset, ode_params_val, best_nn, chain; 
    EDGES = EDGES, N_params = N_params, UDE = UDE, hi = false, show_plots = true,
    # figsave_path=figsave_path, modelssave_path=modelssave_path
    );

@info "Computed residuals for AE test dataset. Median sMAPE: $(median(smape_ae.smape))"

residuals_hi, smape_hi = compute_plot_residuals(high_information, ode_params_val_hi, best_nn, chain; 
    EDGES = EDGES, N_params = N_params, UDE = UDE, hi = true, show_plots = true,
    # figsave_path=figsave_path, modelssave_path=modelssave_path
    );

@info "Computed residuals for HI test dataset. Median sMAPE: $(median(smape_hi.smape))"

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "--> Median sMAPE with NN on HI dataset: $(median(smape_hi.smape))")
# end

@info "Residual calculation script completed"
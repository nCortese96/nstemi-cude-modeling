using StableRNGs, DataFrames, StatsBase, Random
using Optimization, OptimizationOptimisers, LineSearches
using JLD2, CSV, XLSX
using ProgressMeter
using Statistics
using Dates
using Logging

using Revise

includet("ctnt-ude-model.jl")

@info "Starting residual calculation script"

dataset_id = 0; # 0 for MIMIC-IV, 1 for UMG
if dataset_id == 0
    dataset_name = "MIMIC-IV"
    UMG_data = false
elseif dataset_id == 1
    dataset_name = "UMG"
    UMG_data = true
else
    error("dataset_id must be 0 (MIMIC-IV) or 1 (UMG)")
end

UDE = false; # false for cUDE
if UDE
    @info "Using UDE model"
    input_dim = 1
    nn_depth = 2
    nn_width = 8
    N_params = 4
    inputs_str = "τ"
else
    @info "Using cUDE model"
    input_dim = 2
    nn_depth = 2
    nn_width = 4
    N_params = 5
    inputs_str = "τ, β"
end

# UMG_data = false; # true for UDE with UMG data, false for cUDE with MIMIC-IV data

best_idx = 4; # index of the best model to test 
use_multistart = true; # whether to use the multistart version of the best model for testing (only if multistart was used for training)
T_SCALE = 240.0;# 350.0;

const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, T_SCALE];

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

# experiment = "NSTEMI_UDE_UMG_MSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
# experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus";
# experiment = "NSTEMI_partrval_UMG_MSE_ts350.0_28_inp2_multipl_softplus";

experiment = "NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback";

fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
# modelssave_path = "res/$(experiment)/models/test_NN";

figsave_path = "$(fig_path)/$(dataset_name)_test_NN_$(best_idx)$(use_multistart ? "_ms_test" : "")"
modelssave_path = "$(models_path)/$(dataset_name)_test_NN_$(best_idx)$(use_multistart ? "_ms_test" : "")"

@info "Loading dataset"
test_dataset = if UMG_data
    @info "Using UMG dataset for testing"

    @load "$(models_path)/UMG_testset.jld2" test_dataset
    @info "$dataset_name test dataset loaded from previous save"
    test_dataset   
else
    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset
    @info "Using $dataset_name dataset for testing"
    test_dataset
end

ids = DataFrame(patient=[p.id for p in test_dataset])
CSV.write("res/ids_all_eligible_$(dataset_name)_val.csv", ids)

@info "Dataset $dataset_name loaded with $(length(test_dataset)) patients"
# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Residual Calculation Results for $best_idx-th model on $dataset dataset")
#     println(io, "********************************")
#     println(io, "Total sample number for All eligible: $(length(test_dataset))")
# end

@info "Loading best model and parameters"
@load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
best_nn = neural_network_parameters[best_idx]

ode_params_val = try
    @load "$(modelssave_path)/best_params_val_$(dataset_name).jld2" ode_params_val
    ode_params_val
catch e
    @error "Error loading best parameters: $e. 
            \nPlease ensure that the best parameters for $dataset_name dataset are saved in the expected location.
            \nPlease run test_NN.jl script to compute best parameters for $dataset_name dataset, before running this script"
end

@info "Best model and parameters loaded successfully"
@info "Loaded $(length(ode_params_val)) parameters"

@info "Processing $(experiment) residuals data"

residuals_ae, smape_ae = compute_plot_residuals(test_dataset, ode_params_val, best_nn, chain;
    EDGES=EDGES, N_params=N_params, UDE=UDE, hi=false, show_plots=true,
    figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label=dataset_name
);

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Average $(dataset_name) sMAPE AE : $(mean(smape_ae.smape)) - $(std(smape_ae.smape))")
    println(io, "--> Median $(dataset_name) sMAPE AE : $(median(smape_ae.smape)) [$(quantile(smape_ae.smape, 0.25)) - $(quantile(smape_ae.smape, 0.75))]")
end

@info "Computed residuals for AE test dataset. Average sMAPE: $(mean(smape_ae.smape)), Median sMAPE: $(median(smape_ae.smape))"

# HIGH INFORMATION

# hi_ids = if !UMG_data

#     hi_ids = try
#         CSV.read("res/ids_high_information_$(dataset_name)_val.csv", DataFrame)
#     catch e
#         @warn "Error loading high information ids: $e"
#         @info "Calculating high information ids from dataset"

#         meas_min_number = 8
#         min_acq_time_before = 24.0
#         min_acq_n_before = 2
#         min_acq_time_after = 36.0
#         min_acq_n_after = 1
#         min_time = 12.0
#         max_gap = 36.0

#         anoms = find_anomalies(
#             test_dataset,
#             meas_min_number,
#             min_acq_time_before, min_acq_n_before,
#             min_acq_time_after, min_acq_n_after,
#             min_time;
#             max_gap_h=max_gap
#         )

#         @info "Removed sample number for high information: $(length(anoms))"
#         high_information_val = filter(p -> !haskey(anoms, p.id), test_dataset)
#         patient_dims(high_information_val)
#         @info "Total sample number for high information: $(length(high_information_val))"

#         hi_ids = DataFrame(patient=[p.id for p in high_information_val])
#         CSV.write(joinpath("./res", "ids_high_information_$(dataset_name)_val.csv"), hi_ids)
#         hi_ids
#     end

#     hi_ids

# else

#     hi_ids = CSV.read("res/ids_high_information_$(dataset_name)_minafter.csv", DataFrame) # calcuated in TroponinReleaseDiffEqs.jl
#     # high_information = filter(p -> p.id in hi_ids.patient, test_dataset);
#     hi_ids

# end

# high_information_idxs = findall(p -> p.id in hi_ids.patient, test_dataset);
# high_information = test_dataset[high_information_idxs];
# patient_dims(high_information)

# ode_params_val_hi = [ode_params_val[N_params*(i-1)+1:N_params*i] for i in high_information_idxs];
# ode_params_val_hi = vcat(ode_params_val_hi...);

# @info "Total sample number for high information: $(length(high_information))"

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "Total sample number for high information: $(length(high_information))")
# end

# residuals_hi, smape_hi = compute_plot_residuals(high_information, ode_params_val_hi, best_nn, chain;
#     EDGES=EDGES, N_params=N_params, UDE=UDE, hi=true, show_plots=true,
#     figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label="$(dataset_name)_HI"
# );

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "--> Average $(dataset_name) sMAPE HI : $(mean(smape_hi.smape)) - $(std(smape_hi.smape))")
#     println(io, "--> Median $(dataset_name) sMAPE HI : $(median(smape_hi.smape)) [$(quantile(smape_hi.smape, 0.25)) - $(quantile(smape_hi.smape, 0.75))]")
# end

# @info "Computed residuals for HI test dataset. Average sMAPE: $(mean(smape_hi.smape)), Median sMAPE: $(median(smape_hi.smape))"

# # if UMG_data

# #     residuals_hi, smape_hi = compute_plot_residuals(high_information, ode_params_val_hi, best_nn, chain; 
# #     EDGES = EDGES, N_params = N_params, UDE = UDE, hi = true, show_plots = true,
# #     figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label="$(dataset_name)_HI"
# #     );
# #     @info "Computed residuals for HI test dataset. Median sMAPE: $(median(smape_hi.smape))"

# #     open("res/$(experiment)/info_output.txt", "a") do io
# #         println(io, "--> Median $(dataset_name) sMAPE HI : $(median(smape_hi.smape)) [$(quantile(smape_hi.smape, 0.25)) - $(quantile(smape_hi.smape, 0.75))]")
# #     end

# # end

@info "Residual calculation script completed"
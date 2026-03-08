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

UMG_data = false; # true for UDE with UMG data, false for cUDE with MIMIC-IV data

UDE = false; # false for cUDE

best_idx = 1; # index of the best model to test 

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
    nn_width = 6
    N_params = 5
    inputs_str = "τ, β"
end

T_SCALE = 240.0;# 350.0;

const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, T_SCALE];

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

# experiment = "NSTEMI_UDE_UMG_MSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
# experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus";
# experiment = "NSTEMI_partrval_UMG_MSE_ts350.0_28_inp2_multipl_softplus";

experiment = "NSTEMI_cUDE_MIMIC-IV_MSE_26_sigmoid_regback";

fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
# modelssave_path = "res/$(experiment)/models/test_NN";

@info "Loading dataset"
test_dataset = if UMG_data
    dataset = "UMG"

    figsave_path = "$(fig_path)/umg_test_nn_$(best_idx)"
    modelssave_path = "$(models_path)/umg_test_nn_$(best_idx)"

    mkpath(figsave_path)
    mkpath(modelssave_path)

    @load "$(modelssave_path)/UMG_testset.jld2" test_dataset
    @info "$dataset test dataset loaded from previous save"
    test_dataset
else
    dataset = "MIMIC-IV"

    figsave_path = "res/$(experiment)/figs/test_NN_$(best_idx)"
    modelssave_path = "res/$(experiment)/models/test_NN_$(best_idx)"

    mkpath(figsave_path)
    mkpath(modelssave_path)

    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset
    @info "Using $dataset dataset for residuals"

    ae_ids = try
        ids = CSV.read("$(models_path)/ids_all_eligible_$(dataset)_val.csv", DataFrame)
        ids
    catch e
        @warn "Could not load existing ids CSV: $e. Recreating it from test dataset."
        ids = DataFrame(patient=[p.id for p in test_dataset])
        CSV.write("$(models_path)/ids_all_eligible_$(dataset)_val.csv", ids)
        ids
    end
    test_dataset

    # ae_ids = DataFrame(patient = [p.id for p in test_dataset]);
    # CSV.write(joinpath("./res", "ids_all_eligible_$(dataset)_val.csv"), ae_ids)
    # test_dataset;
end
@info "Dataset $dataset loaded with $(length(test_dataset)) patients"
# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Residual Calculation Results for $best_idx-th model on $dataset dataset")
#     println(io, "********************************")
#     println(io, "Total sample number for All eligible: $(length(test_dataset))")
# end

@info "Loading best model and parameters"
best_nn, ode_params_val = try
    @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn
    @load "$(modelssave_path)/best_params_val_$(dataset).jld2" ode_params_val
    best_nn, ode_params_val
catch e

    if UMG_data
        @error "Error loading model: $e. Please ensure that the best model and parameters for UMG dataset are saved in the expected location."
    end
    @warn "Error loading model: $e"

    @info "Using best idx [$(best_idx)] for load model"

    @load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters
    @load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params

    best_nn = neural_network_parameters[best_idx]
    ode_params_val = ode_params[best_idx] # log version where 1 is the index of the best model in info_output
    best_nn, ode_params_val
end
@info "Best model and parameters loaded successfully"
@info "Loaded $(length(ode_params_val)) parameters"

@info "Processing $(experiment) residuals data"

residuals_ae, smape_ae = compute_plot_residuals(test_dataset, ode_params_val, best_nn, chain;
    EDGES=EDGES, N_params=N_params, UDE=UDE, hi=false, show_plots=true,
    figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label=dataset
);

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Average $(dataset) sMAPE AE : $(mean(smape_ae.smape)) - $(std(smape_ae.smape))")
    println(io, "--> Median $(dataset) sMAPE AE : $(median(smape_ae.smape)) [$(quantile(smape_ae.smape, 0.25)) - $(quantile(smape_ae.smape, 0.75))]")
end

@info "Computed residuals for AE test dataset. Average sMAPE: $(mean(smape_ae.smape)), Median sMAPE: $(median(smape_ae.smape))"

# HIGH INFORMATION

hi_ids = if !UMG_data

    hi_ids = try
        CSV.read("res/ids_high_information_$(dataset)_val.csv", DataFrame)
    catch e
        @warn "Error loading high information ids: $e"
        @info "Calculating high information ids from dataset"

        meas_min_number = 8
        min_acq_time_before = 24.0
        min_acq_n_before = 2
        min_acq_time_after = 36.0
        min_acq_n_after = 1
        min_time = 12.0
        max_gap = 36.0

        anoms = find_anomalies(
            test_dataset,
            meas_min_number,
            min_acq_time_before, min_acq_n_before,
            min_acq_time_after, min_acq_n_after,
            min_time;
            max_gap_h=max_gap
        )

        @info "Removed sample number for high information: $(length(anoms))"
        high_information_val = filter(p -> !haskey(anoms, p.id), test_dataset)
        patient_dims(high_information_val)
        @info "Total sample number for high information: $(length(high_information_val))"

        hi_ids = DataFrame(patient=[p.id for p in high_information_val])
        CSV.write(joinpath("./res", "ids_high_information_$(dataset)_val.csv"), hi_ids)
        hi_ids
    end

    hi_ids

else

    hi_ids = CSV.read("res/ids_high_information_$(dataset)_minafter.csv", DataFrame) # calcuated in TroponinReleaseDiffEqs.jl
    # high_information = filter(p -> p.id in hi_ids.patient, test_dataset);
    hi_ids

end

high_information_idxs = findall(p -> p.id in hi_ids.patient, test_dataset);
high_information = test_dataset[high_information_idxs];
patient_dims(high_information)

ode_params_val_hi = [ode_params_val[N_params*(i-1)+1:N_params*i] for i in high_information_idxs];
ode_params_val_hi = vcat(ode_params_val_hi...);

@info "Total sample number for high information: $(length(high_information))"

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "Total sample number for high information: $(length(high_information))")
end

residuals_hi, smape_hi = compute_plot_residuals(high_information, ode_params_val_hi, best_nn, chain;
    EDGES=EDGES, N_params=N_params, UDE=UDE, hi=true, show_plots=true,
    figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label="$(dataset)_HI"
);

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Average $(dataset) sMAPE HI : $(mean(smape_hi.smape)) - $(std(smape_hi.smape))")
    println(io, "--> Median $(dataset) sMAPE HI : $(median(smape_hi.smape)) [$(quantile(smape_hi.smape, 0.25)) - $(quantile(smape_hi.smape, 0.75))]")
end

@info "Computed residuals for HI test dataset. Average sMAPE: $(mean(smape_hi.smape)), Median sMAPE: $(median(smape_hi.smape))"

# if UMG_data

#     residuals_hi, smape_hi = compute_plot_residuals(high_information, ode_params_val_hi, best_nn, chain; 
#     EDGES = EDGES, N_params = N_params, UDE = UDE, hi = true, show_plots = true,
#     figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label="$(dataset)_HI"
#     );
#     @info "Computed residuals for HI test dataset. Median sMAPE: $(median(smape_hi.smape))"

#     open("res/$(experiment)/info_output.txt", "a") do io
#         println(io, "--> Median $(dataset) sMAPE HI : $(median(smape_hi.smape)) [$(quantile(smape_hi.smape, 0.25)) - $(quantile(smape_hi.smape, 0.75))]")
#     end

# end

@info "Residual calculation script completed"
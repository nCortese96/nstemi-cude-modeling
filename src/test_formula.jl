using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2, CSV
using ProgressMeter
using Statistics
using Dates
using Logging

using Revise

println("⚠️ Test formula algorithm started $(now())")

includet("ctnt-ude-model.jl")

UMG_data = true;

best_idx = 4; # index of the best model to test

UDE = false; # false for cUDE

complexity = 14; # complexity of the formula to test

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

# experiment = "NSTEMI_partrvalMIMIC_SSEf_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus"
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

# figsave_path = "$(fig_path)/test_formula";
# modelssave_path = "$(models_path)/test_formula";

# mkpath(figsave_path)
# mkpath(modelssave_path)

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Test formula algorithm Started")
#     println(io, "********************************")
# end

# @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn
@info "Loading dataset"
test_dataset = if UMG_data
    dataset = "UMG";
    
    figsave_path = "$(fig_path)/umg_test_formula_$(best_idx)_compl$(complexity)";
    modelssave_path = "$(models_path)/umg_test_formula_$(best_idx)_compl$(complexity)"; 

    mkpath(figsave_path)
    mkpath(modelssave_path)

    modelsload_path = "$(models_path)/umg_test_nn_$(best_idx)";   

    @load "$(modelsload_path)/UMG_testset.jld2" test_dataset;
    @info "$dataset test dataset loaded from previous save"
    test_dataset;
else
    dataset = "MIMIC-IV";

    figsave_path = "res/$(experiment)/figs/test_formula_$(best_idx)";
    modelssave_path = "res/$(experiment)/models/test_formula_$(best_idx)";

    mkpath(figsave_path)
    mkpath(modelssave_path)

    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
    @info "Using $dataset dataset for residuals"

    test_dataset;
end;

@info "Dataset $dataset loaded with $(length(test_dataset)) patients"
patient_dims(test_dataset)
all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(test_dataset);
display(dist)

lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1]); # 5.0, 5.0, 300.0, 400.0, 3

# ode_p = best_solution;

# const c1 = 0.13256218391741428;
# const c2 = 13.260793182859695;
# const c3 = 67.46423088085042;
# corr(t, β) = c1 / (c2*t + c3*β + 1/(t + eps(Float64)))   # guardrail su t≈0

# corr(t_norm, β) = ((((β * t_norm) * -111.57808469280414) * (inv(β + -2.282974072276526) + t_norm)) * (β + -0.40181845386338866)) * β

#complexity 18

corr(t_norm, β) = complexity == 18 ?
 inv((t_norm * ((t_norm * (t_norm + 0.1944756325708591)) + ((β * β) * 96.22897420906513))) + 0.002123782221456696) * 0.010301680893026832 :
 inv(((((β * 281.86) * β) + t_norm) * t_norm) + 0.0057263) * 0.030174
#complexity 14
# corr(t_norm, β) = inv(((((β * 281.86) * β) + t_norm) * t_norm) + 0.0057263) * 0.030174

function ctnt_ode!(du, u, p, t)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]
    # println("ode")
    # println(p)
    β = exp(p[5]) # Positive conditional parameter

    a = exp(p[1])
    b = exp(p[2])
    # Cs0 = exp(p.ode[3])
    # Cc0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    # correction = chain([u[1], t, a, b, Cs0, Cc0, β], p.neural)[1]

    # correction = chain([u[1], t, β], p.neural)[1]

    t_norm   = t / T_SCALE

    correction = corr(t_norm, β)

    du[1] = - (Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a*(Cc - Cp)
    du[3] = a*(Cc - Cp) - b*Cp
end

@load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

# pguess = mean([optsol.u for optsol in best_solution]);
best_ode_p = ode_params[best_idx]; # log version where 1 is the index of the best model in info_output
pguess = vec(mean(reshape(best_ode_p, :, N_params), dims=1));
println("Initial: ", exp.(pguess))

optfunc = OptimizationFunction(patient_loss_formula, AutoForwardDiff());

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "*********************************")
    println(io, "Evaluating $(dataset) Formula from SymReg with sMAPE - complexity $(complexity)")
end

smape_values = [];
loss_values = [];
params_list = [];

ev_bar = Progress(length(test_dataset); desc = "Validating", color = :cyan, showspeed = true);
for (i, patient) in enumerate(test_dataset)

    u0_init = [exp(pguess[3]), exp(pguess[4]), 0.0];

    tspan = (0.0, patient.timepoints[end] + 10.0);

    problem = ODEProblem(ctnt_ode!, u0_init, tspan);

    optprob = OptimizationProblem(optfunc, pguess, (problem, patient.timepoints, patient.ctnt_data), lb = lhs_lb, ub = lhs_ub);

    optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000);

    p_opt = optsol_lbfgs.u;

    println("For $(patient.id), params: ", p_opt)
    println("Params: ", exp.(p_opt))
    # push!(optsols_valid, optsol_lbfgs);
    push!(params_list, p_opt);

    u0_new = [exp(p_opt[3]), exp(p_opt[4]), 0.0]
    prob   = remake(problem; u0 = u0_new, p = p_opt)

    # sol = Array(solve(prob, Tsit5(); p=p_opt, saveat=1));
    sol = solve(prob, Tsit5(); p=p_opt, saveat=1);
    println("Patient loss: ", patient_loss_formula(p_opt, (prob, patient.timepoints, patient.ctnt_data)))
    # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
    push!(loss_values, optsol_lbfgs.objective)
    println("Objective:    ", optsol_lbfgs.objective)
    validation_metric = smape_loss_formula(p_opt, (prob, patient.timepoints, patient.ctnt_data));
    println("sMAPE: ", validation_metric)
    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Patient $(patient.id) sMAPE Formula validation: $validation_metric")
    end
    push!(smape_values, validation_metric)
    pred = sol[3,:];

    pl = Plots.plot(pred; lw=2, label="Model with Formula Prediction", xlabel="Time", ylabel="Troponin", title="Patient $(patient.id)")
    Plots.scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

    display(pl)

    save("$(figsave_path)/patient_$(patient.id)_compl$(complexity).svg", pl)
    next!(ev_bar)
end

println(median(smape_values))
println(median(loss_values))

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Median sMAPE with formula: $(median(smape_values))")
    println(io, "--> Median SSE with formula: $(median(loss_values))")
end

if UMG_data
    @info "Best model and parameters loaded successfully"
    @info "Loaded $(length(params_list)) parameters"

    hi_ids = CSV.read("res/ids_high_information_$(dataset)_minafter.csv", DataFrame);
    # high_information = filter(p -> p.id in hi_ids.patient, test_dataset);
    high_information_idxs = findall(p -> p.id in hi_ids.patient, test_dataset);
    high_information = test_dataset[high_information_idxs];
    patient_dims(high_information)

    params_list_hi = [params_list[i] for i in high_information_idxs];
    # ode_params_val_hi = vcat(ode_params_val_hi...);

    @info "Total sample number for high information: $(length(high_information))"

    @info "Processing $(experiment) residuals data"

    residuals_ae, smape_ae = compute_plot_residuals(test_dataset, params_list, ctnt_ode!;
        N_params = N_params, UDE = UDE, hi = false, show_plots = true,
        figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label=dataset
        );

    @info "Computed residuals for AE test dataset. Median sMAPE: $(median(smape_ae.smape))"

    residuals_hi, smape_hi = compute_plot_residuals(high_information, params_list_hi, ctnt_ode!;
        N_params = N_params, UDE = UDE, hi = true, show_plots = true,
        figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label="$(dataset)_HI"
        );

    @info "Computed residuals for HI test dataset. Median sMAPE: $(median(smape_hi.smape))"

    open("res/$(experiment)/info_output.txt", "a") do io
        println(io, "--> Median sMAPE with formula on HI dataset: $(median(smape_hi.smape))")
    end

    @info "Residual calculation script completed"
end
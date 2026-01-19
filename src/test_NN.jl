using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics
using Dates
using Logging
using Base.Threads: @threads, nthreads

println("⚠️ Test NN algorithm started $(now())")

include("ctnt-ude-model.jl")

input_dim = 2;
nn_depth = 2;
nn_width = 8;
inputs_str = "t, β";
if input_dim == 3
    inputs_str = "u[1], t, β";
elseif input_dim == 7
    inputs_str = "u[1], t, a, b, Cs0, Cc0, β";
end

T_SCALE = 350.0;

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

experiment = "NSTEMI_partrvalMIMIC_balancedf18_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

figsave_path = "$(fig_path)/test_NN";
modelssave_path = "$(models_path)/test_NN";

mkpath(figsave_path)
mkpath(modelssave_path)

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Test NN algorithm Started")
#     println(io, "********************************")
# end

@load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
@load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;
# @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
best_ode_p = ode_params[1]; # log version where 1 is the index of the best model in info_output
pguess = vec(mean(reshape(best_ode_p, :, length(best_ode_p)), dims=1));
println("Initial: ", exp.(pguess))
########### SET THIS PARAMETER FOR VALIDATION/TEST as FALSE/TRUE ###############################################
UMG_data = false;
########### SET THIS PARAMETER FOR VALIDATION/TEST as FALSE/TRUE ###############################################

Dataset = "MIMIC-IV";

if UMG_data
    Dataset = "UMG";
    figsave_path = "$(fig_path)/umg_test_nn";
    modelssave_path = "$(models_path)/umg_test_nn";   
    
    mkpath(figsave_path)
    mkpath(modelssave_path)

    file_path = "data/UMG_NSTEMI_Dataset.xlsx"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
    sheet_ids = "IDs";
    sheet_times = "times";
    sheet_values = "values";
    xf = XLSX.readxlsx(file_path);
    # Caricamento dei fogli in DataFrame
    # ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
    ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true));
    timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
    troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

    println("Patient loaded: ", nrow(ids))
    patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

    # Trimming to T_SCALE
    trimmed_p = trim_time(patients, T_SCALE);
    patient_dims(trimmed_p)

    # 0. Pre-processing
    meas_min_number = 6;
    anoms = find_anomalies(trimmed_p, meas_min_number);
    println("Removed: $(length(anoms))")

    cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p);
    patient_dims(cleaned_patients)
    println("Total sample: $(length(cleaned_patients))")

    all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(cleaned_patients);
    display(dist)
    savefig("$(figsave_path)/umg_distributions.svg")

    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "$Dataset - Test NN started $(now())")
        println(io, "   Patient loaded: ", nrow(ids))
        println(io, "   Time: min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h")
        println(io, "   cTnT: min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL")
    end

    plt = scutter_patients(cleaned_patients)
    # display(plt)
    savefig("$(figsave_path)/scatter_post.svg")

    test_dataset = cleaned_patients;

else
    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
end

lhs_lb = log.([0.001, 0.001, 0.01, 0.01, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1]); # 5.0, 5.0, 300.0, 400.0, 3

# ode_p = best_solution;
nn_p = best_nn;

# pguess = mean([optsol.u for optsol in ode_p]);

optfunc = OptimizationFunction(patient_loss, AutoForwardDiff());

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "*********************************")
    println(io, "$Dataset - Evaluating NN with sMAPE")
end

smape_values = [];
loss_values = [];

u0_init = [exp(pguess[end-2]), exp(pguess[end-1]), 0.0];

ev_bar = Progress(length(test_dataset); desc = "Validating", color = :cyan, showspeed = true)
for (i, patient) in enumerate(test_dataset)

    tspan = (0.0, patient.timepoints[end] + 10.0);

    model = ctntCUDEModel(pguess, chain, tspan);

    optprob = OptimizationProblem(optfunc, pguess,
                (model, patient.timepoints, patient.ctnt_data, nn_p),
                lb = lhs_lb, ub = lhs_ub);

    optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()),
                maxiters=1000);

    p_opt = ComponentArray(ode = optsol_lbfgs.u, neural = nn_p);

    println("For $(patient.id), params: ", p_opt.ode)
    println("Params: ", exp.(p_opt.ode))
    # push!(optsols_valid, optsol_lbfgs);

    u0_new = [exp(p_opt.ode[end-2]), exp(p_opt.ode[end-1]), 0.0]
    prob   = remake(model.problem; u0 = u0_new, p = p_opt)

    opt_model = ctntCUDEModel(prob, chain);

    sol = Array(solve(prob, AutoTsit5(Rosenbrock23()); p=p_opt, saveat=1));
    println("Patient loss: ", patient_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
    # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
    push!(loss_values, optsol_lbfgs.objective)
    println("Objective:    ", optsol_lbfgs.objective)
    validation_metric = smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural));
    println("sMAPE: ", validation_metric)
    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Patient $(patient.id) sMAPE NN validation: $validation_metric")
    end
    push!(smape_values, validation_metric)
    pred = sol[3,:];

    pl = plot(pred; lw=2, label="Model with NN Prediction", xlabel="Time", ylabel="Troponin", title="Patient $(patient.id)")
    scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

    display(pl)

    save("$(figsave_path)/patient_$(patient.id).svg", pl)
    next!(ev_bar)
end

println(median(smape_values))
println(median(loss_values))

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Median sMAPE: $(median(smape_values))")
    println(io, "--> Median loss: $(median(loss_values))")
end


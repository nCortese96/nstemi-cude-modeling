using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics
using Dates
using StaticArrays
using Logging
using Base.Threads: @threads, nthreads

using Revise
includet("ctnt-ude-model.jl")

println("⚠️ Algorithm started $(now())")

############################
# 1. Caricamento del dataset
############################
println("Dataset loading...")
# Percorso del file Excel
# file_path = "data/STEMI_merged.xlsx";
# sheet_times = "Tempi cleaned";
# sheet_values = "Misurazioni cleaned";

# 0 - MIMIC-IV NSTEMI
# 1 - UMG NSTEMI
# 2 - UMG STEMI

dataset_id = 1; # change here for different datasets
# plotting = true; # set true to enable plotting of each patient during optimization and residual calculation

if dataset_id == 0
    dataset_name = "MIMIC-IV";
    dataset_path = "MIMIC-IV/NSTEMI_reorganized_skipped.xlsx";
    column_letter = "B";
elseif dataset_id == 1
    dataset_name = "UMG";
    dataset_path = "UMG_NSTEMI_Dataset.xlsx";
    column_letter = "A";
# elseif dataset_id == 2
#     dataset_name = "UMG_STEMI";
#     dataset_path = "UMG_STEMI_Dataset.xlsx";
#     column_letter = "A";
end

file_path = "data/$(dataset_path)" # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

input_dim = 2;
nn_depth = 2;
nn_width = 8;
inputs_str = "τ, β";
if input_dim == 4
    inputs_str = "Cs0, Cc0, τ, β";
elseif input_dim == 6
    inputs_str = "τ, a, b, Cs0, Cc0, β";
end

# USE_GPU = true;
T_SCALE = 350.0;
N_params = 5;
# dt = 0.1;

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

experiment = "NSTEMI_cUDEabs1_$(dataset_name)_MSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus"
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
mkpath(fig_path)
mkpath(models_path)
open("res/$(experiment)/info_output.txt", "w") do io
    println(io, "Experiment $(experiment) log file")
    # println(io, "be = bounds edited")
    println(io, "Neural network settings:")
    println(io, "dept: $(nn_depth); width: $(nn_width); inputs($(input_dim)): $(inputs_str)")
    println(io, "dataset: $(file_path)")
end

xf = XLSX.readxlsx(file_path);
# Caricamento dei fogli in DataFrame
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

println("Patient loaded: ", nrow(ids))
println("Initialize...")

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

# Trimming to T_SCALE
trimmed_p = trim_time(patients, T_SCALE);
patient_dims(trimmed_p)

# 0. Pre-processing
meas_min_number = 6;
anoms = find_anomalies(trimmed_p, meas_min_number);
println("Campioni rimossi in totale: $(length(anoms))")

cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p);
patient_dims(cleaned_patients)
println("Totale campioni: $(length(cleaned_patients))")

all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(cleaned_patients);
display(dist)
savefig("$(fig_path)/dataset_distributions.svg")

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Patient loaded: ", nrow(ids))
    println(io, "Time: min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h")
    println(io, "cTnT: min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL")
end

plt = scutter_patients(cleaned_patients)
# display(plt)
savefig("$(fig_path)/scatter_post.svg")

Random.seed!(1234);
rng = StableRNG(42);
shuffle!(cleaned_patients);
n_train = Int(round(length(cleaned_patients) * 0.8));
training_dataset = cleaned_patients[1:n_train];
test_dataset = cleaned_patients[n_train+1:end];
# n_patients = length(training_dataset);
println("Training split: ", length(training_dataset))
println("Validation split: ", length(test_dataset))

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Training split: ", length(training_dataset))
    println(io, "Validation split: ", length(test_dataset))
end

training_id = [patient.id for patient in training_dataset];
test_id = [patient.id for patient in test_dataset];

# check = [];
# check = load("$(models_path)/testsetNSTEMI_MIMIC_0706log.jld2", "test_dataset")
# @load "$(models_path)/testsetNSTEMI_MIMIC_0706log.jld2" check;

@save "$(models_path)/trainingsetNSTEMI_$(experiment).jld2" training_dataset;
@save "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;

# nn_params_init = init_params(chain);
# println("NN parameter len: $(length(nn_params_init))")

n_conditional = 1;
lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1]); # 5.0, 5.0, 200.0, 400.0, 3
# [a, b, Cs0, Cc0 ... last one is conditional parameter β]
initial_guesses = dataset_id == 0 ? 25000 : 1000; # number of initial guesses

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "LB exp: ", exp.(lhs_lb))
    println(io, "UB exp: ", exp.(lhs_ub))
end

initial_nn = sample_initial_neural_parameters(initial_guesses, chain, rng);
initial_ode = sample_initial_parameters(length(training_dataset), initial_guesses, lhs_lb, lhs_ub, rng);

initial_parameters = [ComponentArray(
        neural = initial_nn[i],
        ode = repeat(initial_ode[:,i], 1, n_conditional)
        ) for i in eachindex(initial_nn)];

init_bar = Progress(initial_guesses; dt=1, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick);

# losses_initial = Float64[];
# models = [];
# # models = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (training_dataset[j].timepoints[1], training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];
# for p in initial_parameters # p = initial_parameters[k]
#     models_array = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (0.0, training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];
#     push!(models, models_array);
#     loss_value = training_loss(p, (models_array, training_dataset));
#     # loss_value = training_loss(p, training_dataset);
#     # println(loss_value)
#     push!(losses_initial, loss_value);
#     next!(init_bar; showvalues = [(:loss, loss_value)]);
#     # init_bar.desc = "LOSS: $loss_value"
# end

losses_initial = Vector{Float64}(undef, initial_guesses);
models        = Vector{Vector{ctntUDEModel}}(undef, initial_guesses);

θ_dummy = initial_parameters[1];
local_models = [
    ctntCUDEModel(θ_dummy.ode[N_params*(j-1)+1:N_params*j], chain,
                    (0.0, training_dataset[j].timepoints[end]))
    for j in eachindex(training_dataset)
];

@threads for k in eachindex(initial_parameters)
# for k in eachindex(initial_parameters)
    p = initial_parameters[k]
    # local_models = [
    #     ctntCUDEModel(p.ode[5*(j-1)+1:5*j], chain,
    #                   (0.0, training_dataset[j].timepoints[end]))
    #     for j in eachindex(training_dataset)
    # ]
    # models[k] = local_models
    losses_initial[k] =
        serial_training_loss(p, (local_models, training_dataset); n_params=N_params)
        # par_training_loss(p, (local_models, training_dataset); n_params=N_params)

    next!(init_bar)      # thread-safe
end

selected_initials = Threads.nthreads(); # 2 number of best initializations to select
@info "Selecting $selected_initials best initializations out of $initial_guesses"

param_indxs = partialsortperm(losses_initial, 1:selected_initials);
out_params = initial_parameters[param_indxs];

# println("Calculated losses")
# println(losses_initial)
println("Starting points: ", selected_initials)
println("Best starting points loss indexes: ", param_indxs)
println("Best starting points losses: ", losses_initial[param_indxs])

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Starting points: ", selected_initials)
    println(io, "Best starting points loss indexes: ", param_indxs)
    println(io, "Best starting points losses: ", losses_initial[param_indxs])
end

# models = local_models[param_indxs];

@save "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;
# @load "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;

#### TRAINING ####

adam_maxiters = 500;
lbfgs_maxiters = 400;

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "adam_maxiters: ", adam_maxiters)
    println(io, "lbfgs_maxiters: ", lbfgs_maxiters)
end

# optsols = OptimizationSolution[];
# losses_per_model = Vector{Vector{Float64}}()

n_start = length(out_params);            # = selected_initials

optsols          = Vector{OptimizationSolution}(undef, n_start);
losses_per_model = Vector{Vector{Float64}}(undef, n_start);

const PROG_EVERY_ITERS = 10
const PROG_EVERY_SECS  = 10.0

function make_progress_cb!(pbar, losses; offset=0, every_iters=10, every_secs=1.0)
    last_t    = Ref(time())
    last_k    = Ref(0)   # ultimo "global iter" scritto sulla barra

    return (state, l) -> begin
        push!(losses, l)

        k = offset + state.iter  # <-- contatore globale monotono

        if (k - last_k[] >= every_iters) || (time() - last_t[] > every_secs)
            ProgressMeter.update!(pbar, k;
                showvalues = () -> [(:iter, k), (:loss, l)]
            )
            last_k[] = k
            last_t[] = time()
        end
        return false
    end
end

optfunc = OptimizationFunction(
    # (p, data) -> par_training_loss(p, data; n_params=N_params), 
    (p, data) -> serial_training_loss(p, data; n_params=N_params),
    AutoForwardDiff()
    ); # training_loss AutoForwardDiff() AutoZygote()

# for (i, θ_init) in enumerate(out_params)
@threads for i in eachindex(out_params)
    θ_init = out_params[i]
    train_bar = Progress(adam_maxiters; dt=1, desc="ADAM phase param set θ$(i)", showspeed=true, color=:firebrick);
    # train_bar.desc = "ADAM phase param set θ$(i)"
    
    losses_this = Float64[]
    # try

    # cb = make_progress_cb!(train_bar, losses_this)

    cb_adam  = make_progress_cb!(train_bar, losses_this; offset=0, every_iters=10, every_secs=10.0)
    
    println("ADAM for parameter set: θ$(i)")

    # local_models = [
    #     ctntCUDEModel(
    #         θ_init.ode[5*(j-1)+1:5*j], chain,
    #         (0.0, training_dataset[j].timepoints[end])
    #     ) for j in eachindex(training_dataset)
    # ]

    optprob = Optimization.OptimizationProblem(
        optfunc, θ_init, 
        (local_models, training_dataset)
        ); # models[i]
    # optprob = Optimization.OptimizationProblem(optfunc, θ_init, (training_dataset, USE_GPU));

    # cb = make_callback(losses_this, state0)

    # opt_adamw = AdamW(η, betas, λdecay)

    opt_result1 = Optimization.solve(
        optprob, 
        Optimisers.Adam(0.01), 
        maxiters=adam_maxiters,
        callback = cb_adam
        # callback = (state, l) -> begin
        #                 push!(losses_this, l)
        #                 next!(train_bar; showvalues = [(:iter, state.iter), (:loss, l)]);
        #                 # if length(losses_this) % 10 == 0
        #                 #     println("Current loss after $(length(losses_this)) iterations: $(losses_this[end])")
        #                 # end
        #                 return false
        #             end
        ); # Optimisers.Adam(0.01)
    println("LBFGS for parameter set: θ$(i)")
    println(opt_result1.retcode)

    println("Adam iterations: $(length(losses_this))")
    # push!(adam_iters_per_model, length(losses_this))

    finish!(train_bar) # Reset Adam per LBFGS

    # train_bar.desc = "LBFGS phase param set θ$(i)"
    train_bar = Progress(lbfgs_maxiters; dt=1, desc="LBFGS phase param set θ$(i)", showspeed=true, color=:firebrick);
    cb_lbfgs = make_progress_cb!(train_bar, losses_this; offset=0, every_iters=10, every_secs=10.0)
    
    optprob2 = Optimization.OptimizationProblem(
        optfunc, opt_result1.u, 
        (local_models, training_dataset)
        ); # models[i]
    # optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u, (training_dataset, USE_GPU));
    opt_result2 = Optimization.solve(
        optprob2,
        LBFGS(linesearch=LineSearches.BackTracking()),
        maxiters=lbfgs_maxiters,
        g_abstol  = 1e-6,
        f_abstol = 1e-6,
        x_abstol = 1e-6,
        callback = cb_lbfgs
        # callback = (state, l) -> begin
        #                 push!(losses_this, l)
        #                 next!(train_bar; showvalues = [(:iter, state.iter), (:loss, l)]);
        #                 # if length(losses_this) % 10 == 0
        #                 #     println("Current loss after $(length(losses_this)) iterations: $(losses_this[end])")
        #                 # end
        #                 return false
        #             end
    );

    # push!(optsols, opt_result2)
    optsols[i] = opt_result2;
    
    println("Solutions: $(length(optsols))/$selected_initials")
    println(opt_result2.retcode)

    println("LBFGS iterations: $(length(losses_this))")

    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Returncode model θ$(i): ", opt_result2.retcode)
        println(io, "Final loss model θ$(i): ", opt_result2.objective)
    end

    finish!(train_bar)
    
    # push!(losses_per_model, losses_this)
    losses_per_model[i] = losses_this;

    # catch
        # println("Optimization failed... Skipping")
    # end
    # next!(global_prog)
end

n_adam = adam_maxiters;
@showprogress desc="Plottng loss" for (k, loss_vec) in enumerate(losses_per_model)
    # n_adam = adam_iters_per_model[k]      # confine reale

    # Adam
    Plots.plot(1:n_adam, loss_vec[1:n_adam];
         yaxis = :log10, xaxis = :log10,
         label = "Adam", color = :blue)

    # LBFGS (solo se c’è qualcosa dopo)
    if n_adam < length(loss_vec)
        Plots.plot!(n_adam+1:length(loss_vec),
              loss_vec[n_adam+1:end];
              label = "LBFGS", color = :red)
    end

    savefig("$(fig_path)/loss_$(experiment)_$(k).svg")
end

@save "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses_per_model;
# @load "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses_per_model;
# @save "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;
# @load "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;

neural_network_parameters = [optsol.u.neural[:] for optsol in optsols];
ode_params = [optsol.u.ode[:] for optsol in optsols];

@save "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@save "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;
@load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

# lb = log.([0.001, 0.001, 0.001, 0.01, 0.001]);
# ub = log.([5.0, 5.0, 300.0, 400.0, 3]);

#### Evaluation ####

# n_models = length(optsols)                      # = selected_initials
n_optsol = length(neural_network_parameters)

opt_solutions      = Vector{Vector{OptimizationSolution}}(undef, n_optsol)
optimized_models   = Vector{Vector{ctntUDEModel}}(undef, n_optsol)
model_objectives   = Vector{Vector{Float64}}(undef, n_optsol)
opt_smapes      = Vector{Vector{Float64}}(undef, n_optsol)

# opt_solutions = []
# model_objectives = []
# optimized_models = []
ev_bar = Progress(n_optsol * length(test_dataset); desc = "Validating", color = :cyan, showspeed = true);
optfunc_val = OptimizationFunction(patient_loss, AutoForwardDiff());
# @showprogress desc="Evaluating" for (k, opt_sol) in enumerate(optsols)
for k in 1:n_optsol
    # opt_sol = optsols[k]
    # nn_p = opt_sol.u.neural;
    # ode_p = opt_sol.u.ode;
    nn_p = neural_network_parameters[k];
    ode_p = ode_params[k];
    # try
        println("Optsolution n: $k")
        models_valid = [
            ctntCUDEModel(
                ode_p[N_params*(j-1) + 1 : N_params*j], chain,
                (0.0, test_dataset[j].timepoints[end])
            )
            for j in eachindex(test_dataset)];

        # println(models_valid)

        initial = vec(mean(reshape(ode_p, :, N_params), dims=1));
        println("Initial: ", initial)
        open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
            println(io, "Initial: ", initial)
        end
        opt_models = [];
        optsols_valid = OptimizationSolution[];
        model_smapes = Float64[];

        for (i, model) in enumerate(models_valid)
            patient = test_dataset[i]
            # mean_params = mean ode params and β
            optprob = OptimizationProblem(optfunc_val, initial,
                (model, patient.timepoints, patient.ctnt_data, nn_p),
                lb = lhs_lb, ub = lhs_ub);

            optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()),
                maxiters=1000);

            println("For $(patient.id), params: ", optsol_lbfgs.u)
            println("Params: ", exp.(optsol_lbfgs.u))
            push!(optsols_valid, optsol_lbfgs);
            
            # Print results
            tspan = (0.0, patient.timepoints[end]+10);

            # println(solutions[:, best_model_index][i].u == opt_solutions[best_model_index][i].u)
            open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
                println(io, "For $(patient.id), params: ", optsol_lbfgs.u)
                println(io, "Params: ", exp.(optsol_lbfgs.u))
            end
            # initial = vec(mean(reshape(best_ode_beta, :, 5), dims=1))
            # println(exp.(initial))

            # Costruisci il modello per questo paziente:
            # opt_model = ctntCUDEModel(optsol_lbfgs.u, chain, tspan);
            # push!(opt_models, opt_model);
            p_opt = ComponentArray(ode = optsol_lbfgs.u, neural = nn_p);

            u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]
            
            prob   = remake(model.problem; u0 = u0_new, p = p_opt)
            opt_model = ctntUDEModel(prob, chain);

            push!(opt_models, opt_model);
            sol = Array(solve(opt_model.problem, Tsit5(); p=p_opt, saveat=1));
            # sol = Array(solve_model(p_opt, (model, patient.timepoints, patient.ctnt_data)))
            println("Patient loss: ", patient_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
            # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
            println("Objective:    ", optsol_lbfgs.objective)
            smape_val = smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural))
            println("sMAPE: ", smape_val)
            push!(model_smapes, smape_val);

            open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
                println(io, "Patient loss model $(k): ", patient_loss(p_opt.ode, (model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
                # println(io, "Compute loss model $(k): ", compute_loss(p_opt, (model, patient.timepoints, patient.ctnt_data)))
                println(io, "Objective model $(k):    ", optsol_lbfgs.objective)
                println(io, "sMAPE: ", smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
            end
            
            pred = sol[3,:];
            # pred = [u[3] for u in sol.u]

            pl = Plots.plot(pred; lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
            Plots.scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

            save("$(fig_path)/$(experiment)_model_$(k)_$(patient.id).svg", pl)
            next!(ev_bar)
        end

        opt_solutions[k] = optsols_valid
        optimized_models[k] = opt_models
        opt_smapes[k] = model_smapes
        # push!(opt_solutions, optsols_valid);
        # push!(optimized_models, opt_models);

        objectives = [sol.objective for sol in optsols_valid];
        println("Median: ", median(objectives))
        println("Mean: ", mean(objectives))
        # push!(model_objectives, objectives)
        model_objectives[k] = objectives
    # catch
    #     push!(model_objectives, repeat([Inf], length(models)))
    # end
end
finish!(ev_bar)
@info "Validation completed."

# model_objectives = model_objectives[2]
# find the model that performs best on each individual
objectives = hcat(model_objectives...);
solutions = hcat(opt_solutions...);
new_models = hcat(optimized_models...);
smapes = hcat(opt_smapes...);
@save "$(models_path)/objectivesNSTEMI_$(experiment).jld2" objectives;


best_model_index = argmin(median(objectives, dims=1)[:]);
println("Average in validation: ", mean(objectives, dims=1)[:])
println("Median in validation: ", median(objectives, dims=1)[:])
println("Median sMAPE in validation: ", median(smapes... , dims=1)[:])
println("Best model id: $best_model_index")

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Average in validation: ", mean(objectives, dims=1)[:])
    println(io, "Median in validation: ", median(objectives, dims=1)[:])
    println(io, "Median sMAPE in validation: ", median(hcat(smapes...) , dims=1)[:])
    println(io, "Best model id: $best_model_index")
end

# best_model_index = argmin(sum(objectives, dims=2)[:])
# best_model = optsols[best_model_index];

# best_nn = best_model.u.neural;
# best_ode_beta = best_model.u.ode
best_nn = neural_network_parameters[best_model_index];

# ode_betas_test = [optsol.u for optsol in opt_solutions]
best_solution = opt_solutions[best_model_index];
@save "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;
best_models = new_models[best_model_index];
@save "$(models_path)/optimized_models_NSTEMI_$(experiment).jld2" new_models;
# losses_test = [optsol.objective for optsol in opt_solutions]
# @save "$(models_path)/lossestestNSTEMI_$(experiment).jld2" losses_test;

@save "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
# @save "$(models_path)/best_ode_beta_NSTEMI_$(experiment).jld2" best_ode_beta

# @load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;

a_dist = [exp(sol.u[1]) for sol in best_solution]
b_dist = [exp(sol.u[2]) for sol in best_solution]
Cs0_dist = [exp(sol.u[3]) for sol in best_solution]
Cc0_dist = [exp(sol.u[4]) for sol in best_solution]
β_dist = [exp(sol.u[end]) for sol in best_solution]


plt_a = histogram(a_dist;
                 bins = 5,
                 xlabel = "Value",
                 ylabel = "#",
                 title = "Params a",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_b = histogram(b_dist;
                 bins = 5,
                 xlabel = "Value",
                 ylabel = "#",
                 title = "Params b",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_Cs0 = histogram(Cs0_dist;
                 bins = 5,
                 xlabel = "Value",
                 ylabel = "#",
                 title = "Params Cs0",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_Cc0 = histogram(Cc0_dist;
                 bins = 5,
                 xlabel = "Value",
                 ylabel = "#",
                 title = "Params Cc0",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_β = histogram(β_dist;
                 bins = 5,
                 xlabel = "Value",
                 ylabel = "#",
                 title = "Params β",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

@info "⚠️ Algorithm ended $(now())"
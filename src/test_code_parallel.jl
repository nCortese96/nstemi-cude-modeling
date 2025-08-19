using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics
using Logging
using Base.Threads: @threads, nthreads

println("⚠️ Algorithm started")

include("ctnt-ude-model.jl")
include("ensemble_training.jl")
using .EnsembleTraining

############################
# 1. Caricamento del dataset
############################
println("Dataset loading...")
# Percorso del file Excel
# file_path = "data/STEMI_merged.xlsx";
# sheet_times = "Tempi cleaned";
# sheet_values = "Misurazioni cleaned";

file_path = "data/MIMIC-IV/NSTEMI_reorganized_skipped.xlsx"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

input_dim = 2;
nn_depth = 2;
nn_width = 8;
inputs_str = "t, β";
if input_dim == 3
    inputs_str = "u[1], t, β";
elseif input_dim == 7
    inputs_str = "u[1], t, a, b, Cs0, Cc0, β";
end

USE_GPU = true;
T_SCALE = 350.0;
# dt = 0.1;

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

experiment = "NSTEMI_ensMIMIC_logSSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
mkpath(fig_path)
mkpath(models_path)
open("res/$(experiment)/info_output.txt", "w") do io
    println(io, "Experiment $(experiment) log file")
    println(io, "be = bounds edited")
    println(io, "Neural network settings:")
    println(io, "dept: $(nn_depth); width: $(nn_width); inputs($(input_dim)): $(inputs_str)")
    println(io, "dataset: $(file_path)")
end

xf = XLSX.readxlsx(file_path);
# Caricamento dei fogli in DataFrame
# ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "B:B", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

println("Patient loaded: ", nrow(ids))
println("Initialize...")

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

# Trimming to T_SCALE
trimmed_p = trim_time(patients, T_SCALE);
patient_dims(trimmed_p)

# 0. Pre-processing
meas_min_number = 5;
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
lhs_lb = log.([0.001, 0.001, 0.01, 0.01, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1]); # 5.0, 5.0, 300.0, 400.0, 3
# [a, b, Cs0, Cc0 ... last one is conditional parameter β]
initial_guesses = 1000;

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

losses_initial = Float64[];
# models = [];
# models = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (training_dataset[j].timepoints[1], training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];
for p in initial_parameters # p = initial_parameters[k]
    # models_array = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (0.0, training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];
    # push!(models, models_array);
    # loss_value = training_loss(p, (models_array, training_dataset));
    # loss_value = training_loss(p, training_dataset);
    loss_value = EnsembleTraining.ensemble_training_loss(p, (training_dataset, chain); backend = :threads)
    # println(loss_value)
    push!(losses_initial, loss_value);
    next!(init_bar; showvalues = [(:loss, loss_value)]);
    # init_bar.desc = "LOSS: $loss_value"
end

selected_initials = 2;
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

# models = models[param_indxs];

# @save "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;
# @load "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;

# for param_indx in partialsortperm(losses_initial, 1:25)
#     println(initial_parameters[param_indx])
# end

# patience   = 20;          # numero max di iterazioni senza miglioramenti
# min_delta  = 1e-6;        # miglioramento minimo per essere accettato
# best_loss  = Inf;         # parte a +∞
# stagnation = 0;           # contatore interno
# best_θ     = copy(out_params[1]);
# losses     = Float64[];   # vettore per il log, come prima

# callback_func = (state, l) -> begin
#     push!(losses, l)
#     if length(losses) % 10 == 0
#         println("Current loss after $(length(losses)) iterations: $(losses[end])")
#     end
#     return false
# end

# const STEP_LR   = 200          # ogni 200 iter dimezzo lr
# const PATIENCE  = 20           # early-stop se nessun miglioramento
# const MIN_DELTA = 1e-6         # quanto deve scendere la loss per "migliorare"

# """
#     make_callback(loss_vec, opt_state)

# Crea un callback chiuso sul vettore `loss_vec` e sullo stato `opt_state`
# (AdamW) in modo da:
#   • salvare tutte le loss            → push!
#   • stampare ogni 10 iterazioni
#   • dimezzare il learning-rate ogni STEP_LR iter
#   • fermare Adam se la valid-loss (train, nel tuo caso) non migliora più
# """
# function make_callback(loss_vec, opt_state)
#     best_loss      = Inf
#     patience_left  = PATIENCE

#     function cb(state, l)
#         push!(loss_vec, l)

#         # log ogni 10 step
#         if length(loss_vec) % 10 == 0
#             @info "iter $(state.iteration)  loss = $(round(l; digits=5))"
#         end

#         # step-decay LR
#         if state.iteration % STEP_LR == 0
#             opt_state.opt.inner.eta *= 0.5
#             @info "lr  -> $(opt_state.opt.inner.eta)"
#         end

#         # early-stopping semplice su train-loss
#         if l < best_loss - MIN_DELTA
#             best_loss     = l
#             patience_left = PATIENCE
#         else
#             patience_left -= 1
#             if patience_left == 0
#                 @info "Early stop: best = $(round(best_loss; digits=5))"
#                 return true        # ← dice a Optimization di fermarsi
#             end
#         end
#         return false
#     end

#     return cb
# end

adam_maxiters = 400;
lbfgs_maxiters = 300;

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "adam_maxiters: ", adam_maxiters)
    println(io, "lbfgs_maxiters: ", lbfgs_maxiters)
end

optsols = OptimizationSolution[];
# optfunc = OptimizationFunction(training_loss, AutoForwardDiff()); # ensamble_training_loss
optfunc = OptimizationFunction((θ, _) -> EnsembleTraining.ensemble_training_loss(
    θ, (training_dataset, chain);
    backend = :threads), AutoZygote())
losses_per_model = Vector{Vector{Float64}}()
adam_iters_per_model = Int[] 
# lower_bound = log.([0.001, 0.001, 0.001, 0.001, 0.001])
# upper_bound = log.([5, 5, 400, 400, 1])
# train_bar = Progress(100; dt=0.5, desc="Single solution optimizing...", showspeed=true, color=:firebrick)
# global_prog = Progress(selected_initials; dt=1, desc="Global process optimizing...", showspeed=true, color=:blue);
# η      = 0.01             # learning-rate che usavi già
# betas  = (0.9, 0.999)     # default di Adam
# λdecay = 1f-4             # weight-decay (prova 10-4)

# opt_adamw = AdamW(η, betas; decay = λdecay)
#   oppure forma posizionale equivalente:
# opt_adamw = AdamW(η, betas, λdecay)

for (i, θ_init) in enumerate(out_params)
    train_bar = Progress(adam_maxiters+lbfgs_maxiters; dt=0.5, desc="Training start ", showspeed=true, color=:firebrick);
    train_bar.desc = "ADAM phase param set $(i)"
    losses_this = Float64[]
    # try
    println("ADAM for parameter set: $(i)")

    # optprob = Optimization.OptimizationProblem(optfunc, θ_init, (models[i], training_dataset));
    # optprob = Optimization.OptimizationProblem(optfunc, θ_init, (training_dataset, USE_GPU));
    optprob = Optimization.OptimizationProblem(optfunc, θ_init, nothing)

    # cb = make_callback(losses_this, state0)

    # opt_adamw = AdamW(η, betas, λdecay)

    opt_result1 = Optimization.solve(optprob, Optimisers.Adam(0.01), maxiters=adam_maxiters,
            callback = (state, l) -> begin
                            push!(losses_this, l)
                            if state.iter % 10 == 0
                                next!(train_bar; showvalues = [(:iter, state.iter), (:loss, l)]);
                            end
                            # if length(losses_this) % 10 == 0
                            #     println("Current loss after $(length(losses_this)) iterations: $(losses_this[end])")
                            # end
                            return false
                        end); # Optimisers.Adam(0.01)
    println("LBFGS for parameter set: $(i)")
    println(opt_result1.retcode)

    println("Adam iterations: $(length(losses_this))")
    push!(adam_iters_per_model, length(losses_this))
    
    train_bar.desc = "LBFGS phase param set $(i)"
    # optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u, (models[i], training_dataset));
    # optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u, (training_dataset, USE_GPU));
    optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u, nothing)
    opt_result2 = Optimization.solve(
        optprob2,
        LBFGS(linesearch=LineSearches.BackTracking()),
        maxiters=lbfgs_maxiters,
        callback = (state, l) -> begin
                        push!(losses_this, l)
                        if state.iter % 10 == 0
                            next!(train_bar; showvalues = [(:iter, state.iter), (:loss, l)]);
                        end
                        # if length(losses_this) % 10 == 0
                        #     println("Current loss after $(length(losses_this)) iterations: $(losses_this[end])")
                        # end
                        return false
                    end
    );

    push!(optsols, opt_result2)
    
    println("Solutions: $(length(optsols))/$selected_initials")
    println(opt_result2.retcode)

    println("LBFGS iterations: $(length(losses_this))")

    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Returncode model $(i): ", opt_result2.retcode)
        println(io, "Final loss model $(i): ", opt_result2.objective)
    end

    push!(losses_per_model, losses_this)

    # catch
        # println("Optimization failed... Skipping")
    # end
    # next!(global_prog)
end

@showprogress desc="Plottng loss" for (k, loss_vec) in enumerate(losses_per_model)
    n_adam = adam_iters_per_model[k]      # confine reale

    # Adam
    plot(1:n_adam, loss_vec[1:n_adam];
         yaxis = :log10, xaxis = :log10,
         label = "Adam", color = :blue)

    # LBFGS (solo se c’è qualcosa dopo)
    if n_adam < length(loss_vec)
        plot!(n_adam+1:length(loss_vec),
              loss_vec[n_adam+1:end];
              label = "LBFGS", color = :red)
    end

    savefig("$(fig_path)/loss_$(experiment)_$(k).svg")
end

@save "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses_per_model;
# @load "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses_per_model;
# @save "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;
# @load "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;

neural_network_parameters = [optsol.u.neural[:] for optsol in optsols]
ode_params = [optsol.u.ode[:] for optsol in optsols]

@save "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@save "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

# lb = log.([0.001, 0.001, 0.001, 0.01, 0.001]);
# ub = log.([5.0, 5.0, 300.0, 400.0, 3]);

opt_solutions = []
model_objectives = []
optimized_models = []
@showprogress desc="Evaluating" for (k, opt_sol) in enumerate(optsols)
    # try
        println("Optsolution n: $k")
        models_valid = [
            ctntCUDEModel(
                opt_sol.u.ode[5*(j-1) + 1 : 5*j], chain,
                (0.0, test_dataset[j].timepoints[end])
            )
            for j in eachindex(test_dataset)];

        # println(models_valid)

        initial = vec(mean(reshape(opt_sol.u.ode, :, 5), dims=1));
        println("Initial: ", initial)
        opt_models = [];
        optsols_valid = OptimizationSolution[];
        optfunc = OptimizationFunction(patient_loss, AutoForwardDiff());
        for (i, model) in enumerate(models_valid)
            patient = test_dataset[i]
            # mean_params = mean ode params and β
            optprob = OptimizationProblem(optfunc, initial,
                (model, patient.timepoints, patient.ctnt_data, opt_sol.u.neural),
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
            p_opt = ComponentArray(ode = optsol_lbfgs.u, neural = opt_sol.u.neural);

            u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]
            
            prob   = remake(model.problem; u0 = u0_new, p = p_opt)
            opt_model = ctntCUDEModel(prob, chain);

            push!(opt_models, opt_model);
            sol = Array(solve(opt_model.problem, AutoTsit5(Rosenbrock23()); p=p_opt, saveat=1));
            # sol = Array(solve_model(p_opt, (model, patient.timepoints, patient.ctnt_data)))
            println("Patient loss: ", patient_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
            # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
            println("Objective:    ", optsol_lbfgs.objective)
            println("sMAPE: ", smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))

            open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
                println(io, "Patient loss model $(k): ", patient_loss(p_opt.ode, (model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
                # println(io, "Compute loss model $(k): ", compute_loss(p_opt, (model, patient.timepoints, patient.ctnt_data)))
                println(io, "Objective model $(k):    ", optsol_lbfgs.objective)
            end

            pred = sol[3,:];
            # pred = [u[3] for u in sol.u]

            pl = plot(pred; lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
            scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

            save("$(fig_path)/$(experiment)_model_$(k)_$(patient.id).svg", pl)

        end
        push!(opt_solutions, optsols_valid);
        push!(optimized_models, opt_models);

        objectives = [sol.objective for sol in optsols_valid];
        println("Median: ", median(objectives))
        println("Mean: ", mean(objectives))
        push!(model_objectives, objectives)
    # catch
    #     push!(model_objectives, repeat([Inf], length(models)))
    # end
end

# model_objectives = model_objectives[2]
# find the model that performs best on each individual
objectives = hcat(model_objectives...);
solutions = hcat(opt_solutions...);
new_models = hcat(optimized_models...);
@save "$(models_path)/objectivesNSTEMI_$(experiment).jld2" objectives;

best_model_index = argmin(median(objectives, dims=1)[:]);
println("Best model id: $best_model_index")

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Average in validation: ", mean(objectives, dims=1)[:])
    println(io, "Median in validation: ", median(objectives, dims=1)[:])
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
β_dist = [exp(sol.u[5]) for sol in best_solution]


plt_a = histogram(a_dist;
                 bins = 5,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Time-points distribution",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_b = histogram(b_dist;
                 bins = 5,
                 xlabel = "Time (h)",
                 ylabel = "#",
                 title = "Time-points distribution",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_Cs0 = histogram(Cs0_dist;
                 bins = 5,
                 xlabel = "Time (h)",
                 ylabel = "#",
                 title = "Time-points distribution",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_Cc0 = histogram(Cc0_dist;
                 bins = 5,
                 xlabel = "Time (h)",
                 ylabel = "#",
                 title = "Time-points distribution",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")

plt_β = histogram(β_dist;
                 bins = 5,
                 xlabel = "Time (h)",
                 ylabel = "#",
                 title = "Time-points distribution",
                 legend = false)
savefig("$(fig_path)/a_dist.svg")
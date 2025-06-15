using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics

println("⚠️ Algorithm started")

include("ctnt-ude-model.jl")

############################
# 1. Caricamento del dataset
############################
println("Dataset loading...")
# Percorso del file Excel
# file_path = "data/STEMI_merged.xlsx";
# sheet_times = "Tempi cleaned";
# sheet_values = "Misurazioni cleaned";

file_path = "data/UMG_NSTEMI_Dataset.xlsx"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

input_dim = 2;
depth = 2;
width = 6;
inputs_str = "t, β"
if input_dim == 3
    inputs_str = "u[1], t, β"
elseif input_dim == 7
    inputs_str == "u[1], t, a, b, Cs0, Cc0, β"
end

experiment = "NSTEMI_SSE_26_inp2_multip_sigmoid";
fig_path = "res/$(experiment)/figs"
models_path = "res/$(experiment)/models"
mkpath(fig_path)
mkpath(models_path)
open("res/$(experiment)/info_output.txt", "w") do io          # "w" = write (sovrascrive)
    println(io, "Experiment $(experiment) log file")
    println(io, "Neural network settings:")
    println(io, "dept: $(depth); width: $(width); inputs($(input_dim)): $(inputs_str)")
    println(io, "dataset: $(file_path)")
end

chain = neural_network_model(2, 6; input_dims=input_dim);

xf = XLSX.readxlsx(file_path)
# Caricamento dei fogli in DataFrame
# ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

println("Patient loaded: ", nrow(ids))
println("Initialize...")

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Patient loaded: ", nrow(ids))
end

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

bad = String[]
for p in patients
    if length(p.timepoints) != length(p.ctnt_data)
        println(p.id)
        println(length(p.timepoints))
        println(length(p.ctnt_data))
        push!(bad, p.id)
    end
end
return bad

if !isempty(bad)
    @warn "⚠️  Pazienti con lunghezze discordanti" bad_ids
    # opzionale: rimuovili o gestiscili
    # patients_raw = filter(p -> !(p.id in bad_ids), patients_raw)
end

function deduplicate_times!(p::PatientData)
    tp, ct = p.timepoints, p.ctnt_data
    @assert length(tp) == length(ct) "Lunghezze diverse per ID=$(p.id)"

    seen = Set{Float64}()       # tempi già incontrati
    removed = 0
    # scorri al contrario così gli indici restanti non cambiano
    for i in reverse(eachindex(tp))
        t = tp[i]
        if t in seen            # duplicato → rimuovi
            println("ID=$(p.id)  t=$(t)  ctnt=$(ct[i])  [rimosso]")
            splice!(tp, i)
            splice!(ct, i)
            removed += 1
        else
            push!(seen, t)
        end
    end
    return removed
end

total_removed = sum(deduplicate_times!(p) for p in patients);
println("Campioni duplicati rimossi in totale: $total_removed")

# dernormalization_time = Dict{String, Tuple{Float64,Float64}}()

# for p in patients
#     t0, tf = first(p.timepoints), last(p.timepoints)   # estremi del paziente
#     Δt = tf - t0 + eps()
#     p.timepoints .= (p.timepoints .- t0) ./ Δt
#     dernormalization_time[p.id] = (t0, tf)         # ora 0–1
# end

# patients[2].timepoints

Random.seed!(1234);
rng = StableRNG(42);

# initial_params = [0.005, 0.005, 0.1, 0.001, 0.1];

# Costruisci l'array di PatientData iterando su tutte le righe.

shuffle!(patients);
n_train = Int(round(length(patients) * 0.8));
training_dataset = patients[1:n_train];
test_dataset = patients[n_train+1:end];
println("Training split: ", length(training_dataset))
println("Validation split: ", length(test_dataset))

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Training split: ", length(training_dataset))
    println(io, "Validation split: ", length(test_dataset))
end

training_id = [patient.id for patient in training_dataset]
test_id = [patient.id for patient in test_dataset]

# check = [];
# check = load("$(models_path)/testsetNSTEMI_MIMIC_0706log.jld2", "test_dataset")
# @load "$(models_path)/testsetNSTEMI_MIMIC_0706log.jld2" check;

@save "$(models_path)/trainingsetNSTEMI_$(experiment).jld2" training_dataset;
@save "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;

# nn_params_init = init_params(chain);
# println("NN parameter len: $(length(nn_params_init))")

n_conditional = 1;
lhs_lb = log.([0.001, 0.001, 0.01, 0.01, 0.001]);
lhs_ub = log.([5.0, 5.0, 300.0, 400.0, 3]);
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

losses_initial = Float64[];
models = [];
prog = Progress(initial_guesses; dt=1, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick);
# models = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (training_dataset[j].timepoints[1], training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];

for p in initial_parameters # p = initial_parameters[k]
    models_array = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (0.0, training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];
    push!(models, models_array);
    loss_value = training_loss(p, (models_array, training_dataset));
    push!(losses_initial, loss_value);
    next!(prog);
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

models = models[param_indxs];

@save "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;
@load "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;

# for param_indx in partialsortperm(losses_initial, 1:25)
#     println(initial_parameters[param_indx])
# end

# patience   = 20;          # numero max di iterazioni senza miglioramenti
# min_delta  = 1e-6;        # miglioramento minimo per essere accettato
# best_loss  = Inf;         # parte a +∞
# stagnation = 0;           # contatore interno
# best_θ     = copy(out_params[1]);
losses     = Float64[];   # vettore per il log, come prima

callback_func = (state, l) -> begin
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adam_maxiters = 800;
lbfgs_maxiters = 500;

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "adam_maxiters: ", adam_maxiters)
    println(io, "lbfgs_maxiters: ", lbfgs_maxiters)
end

optsols = OptimizationSolution[];
optfunc = OptimizationFunction(training_loss, AutoForwardDiff());
# lower_bound = log.([0.001, 0.001, 0.001, 0.001, 0.001])
# upper_bound = log.([5, 5, 400, 400, 1])
# prog = Progress(100; dt=0.5, desc="Single solution optimizing...", showspeed=true, color=:firebrick)
# global_prog = Progress(selected_initials; dt=1, desc="Global process optimizing...", showspeed=true, color=:blue);
for (i, θ_init) in enumerate(out_params)
    # try
    println("ADAM for parameter set: $(i)")
    # global_progress = Progress(100, desc="Ottimizzazione globale ADAM", dt=0.5);
    optprob = Optimization.OptimizationProblem(optfunc, θ_init, (models[i], training_dataset));
    opt_result1 = Optimization.solve(optprob, Optimisers.Adam(0.01), maxiters=adam_maxiters, callback=callback_func);
    println("LBFGS for parameter set: $(i)")
    # global_progress = Progress(100, desc="Affinament LBFGS", dt=0.5);
    println(opt_result1.retcode)
    optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u, (models[i], training_dataset));
    opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=lbfgs_maxiters, callback=callback_func);
    push!(optsols, opt_result2)
    println("Solutions: $(length(optsols))/$selected_initials")
    println(opt_result2.retcode)

    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Returncode model $(i): ", opt_result2.retcode)
        println(io, "Final loss model $(i): ", opt_result2.objective)
    end

    # catch
        # println("Optimization failed... Skipping")
    # end
    # next!(global_prog)
end

segments = []         # raccoglierà i range di indici
sp = 1             # inizio del primo segmento
n = length(losses)
@assert n > 0 "Array vuoto!"

for i in 1:(n-1)
    global sp
    if i+1 < length(losses) && abs(losses[i+1] / losses[i]) ≥ 10
        push!(segments, sp:i)
        sp = i + 1
    end
end
# chiudi l’ultimo segmento
push!(segments, sp:n)
println(segments)

n = length(segments)

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Loss segments: ", n)
    println(io, segments)
end

for i in 1:n
    loss = losses[segments[i]]
    pl_losses = plot(1:800, loss[1:800], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
    plot!(801:length(loss), loss[801:end], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)
    display(pl_losses)
    save("$(fig_path)/loss_$(experiment)_$(i).svg", pl_losses) 
end

@save "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses;
@load "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses;
@save "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;
@load "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;

neural_network_parameters = [optsol.u.neural[:] for optsol in optsols]
ode_params = [optsol.u.ode[:] for optsol in optsols]

@save "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@save "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

# lb = log.([0.001, 0.001, 0.001, 0.01, 0.001]);
# ub = log.([5.0, 5.0, 300.0, 400.0, 3]);

opt_solutions = []
model_objectives = []
optimized_models = []
for (k, opt_sol) in enumerate(optsols)
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
            opt_model = model;
            p_opt = ComponentArray(ode = optsol_lbfgs.u, neural = opt_sol.u.neural);

            u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]
            
            prob   = remake(opt_model.problem; u0 = u0_new, p = p_opt)
            push!(opt_models, ctntCUDEModel(prob, chain));
            sol = Array(solve(opt_model.problem, AutoTsit5(Rosenbrock23()); p=p_opt, saveat=1));
            # sol = Array(solve_model(p_opt, (model, patient.timepoints, patient.ctnt_data)))
            println("Patient loss: ", patient_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
            println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
            println("Objective:    ", optsol_lbfgs.objective)

            open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
                println(io, "Patient loss model $(k): ", patient_loss(p_opt.ode, (model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
                println(io, "Compute loss model $(k): ", compute_loss(p_opt, (model, patient.timepoints, patient.ctnt_data)))
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
best_model = optsols[best_model_index];

best_nn = best_model.u.neural;
# best_ode_beta = best_model.u.ode 

# ode_betas_test = [optsol.u for optsol in opt_solutions]
best_solution = opt_solutions[best_model_index];
@save "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;
best_models = new_models[best_model_index];
@save "$(models_path)/optimized_models_NSTEMI_$(experiment).jld2" new_models;
# losses_test = [optsol.objective for optsol in opt_solutions]
# @save "$(models_path)/lossestestNSTEMI_$(experiment).jld2" losses_test;

@save "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
# @save "$(models_path)/best_ode_beta_NSTEMI_$(experiment).jld2" best_ode_beta

# for i in eachindex(test_dataset)
#     patient = test_dataset[i]
#     # idx_start = 5*(i-1) + 1
#     # idx_end = 5*i  # usa il guess iniziale, per esempio
#     tspan = (0.0, patient.timepoints[end]+10)

#     println("Patient: ", patient.id)
    
#     best_ode = best_solution[i].u
#     println("Params in log: ", best_ode)
#     println("Params: ", exp.(best_ode))
#     # println(solutions[:, best_model_index][i].u == opt_solutions[best_model_index][i].u)
#     open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
#         println(io, "Patient: ", patient.id)
#         println(io, "Params in log: ", best_ode)
#         println(io, "Params: ", exp.(best_ode))
#     end
#     # initial = vec(mean(reshape(best_ode_beta, :, 5), dims=1))
#     # println(exp.(initial))

#     # Costruisci il modello per questo paziente:
#     opt_model = ctntCUDEModel(best_ode, chain, tspan)

#     p_opt = ComponentArray(ode = best_ode, neural = best_nn)

#     sol = Array(solve(opt_model.problem, AutoTsit5(Rosenbrock23()); p=p_opt, saveat=1))
#     # sol = Array(solve_model(p_opt, (model, patient.timepoints, patient.ctnt_data)))
#     println("Patient loss: ", patient_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
#     println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
#     println("Objective:    ", objectives[:, best_model_index][i])

#     open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
#         println(io, "Patient loss: ", patient_loss(p_opt, (model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
#         println(io, "Compute loss: ", compute_loss(p_opt, (model, patient.timepoints, patient.ctnt_data)))
#         println(io, "Objective:    ", objectives[:, best_model_index][i])
#     end

#     pred = sol[3,:];
#     # pred = [u[3] for u in sol.u]

#     pl = plot(pred; lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
#     scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

#     save("$(fig_path)/$(experiment)_$(patient.id).svg", pl)
# end

# i = 20
# patient = test_dataset[i]
# patient.id
# # println(solutions[:, best_model_index][i].u == best_solution[i].u)
# best_ode = best_solution[i].u  # usa il guess iniziale, per esempio
# println(exp.(best_ode))
# tspan = (0.0, patient.timepoints[end]+10)

# initial = mean(best_solution)
# println(exp.(initial))

# chain = neural_network_model(2, 6; input_dims=input_dim);

# # Costruisci il modello per questo paziente:
# model = ctntCUDEModel(initial, chain, tspan)

# p = ComponentArray(ode = initial, neural = best_nn)

# lhs_lb = log.([0.001, 0.0001, 0.01, 0.01, 0.001]);
# lhs_ub = log.([5.0, 5.0, 300.0, 400.0, 3]);

# optfunc = OptimizationFunction(patient_loss, AutoForwardDiff())

# optprob = OptimizationProblem(optfunc, initial,
#                 (model, patient.timepoints, patient.ctnt_data, p.neural),
#                 lb = lhs_lb, ub = lhs_ub)

# optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()),
#     maxiters=1000)

# println(optsol.u == best_ode)

# println(exp.(optsol.u))
# println(optsol.objective)
# println(objectives[:, 2][20])

# p_opt = ComponentArray(ode = optsol.u, neural = best_nn)

# opt_model = ctntCUDEModel(optsol.u, chain, tspan)

# sol = Array(solve(model.problem, AutoTsit5(Rosenbrock23()); p=p_opt, saveat=1))
# # sol = Array(solve_model(p_opt, (model, patient.timepoints, patient.ctnt_data)))
# println(patient_loss(p_opt, (model, patient.timepoints, patient.ctnt_data, p.neural)))
# println(compute_loss(p_opt, (model, patient.timepoints, patient.ctnt_data)))

# pred = sol[3,:];
# # pred = [u[3] for u in sol.u]

# plot(pred; lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
# scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

# Plot the losses
# pl_losses = plot(1:adam_maxiters, losses[1:adam_maxiters], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
# plot!(adam_maxiters+1:adam_maxiters+lbfgs_maxiters, losses[adam_maxiters+1:adam_maxiters+lbfgs_maxiters], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)

# # Plot the losses
# pl_losses = plot(753:753+adam_maxiters, losses[753:753+adam_maxiters], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
# plot!(753+adam_maxiters:length(losses), losses[753+adam_maxiters:end], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)

# #example
# param_indx = 1
# θ_init = initial_parameters[param_indx]

# #############################################################
# # 4. Preparazione del vettore iniziale per l'ottimizzazione
# #############################################################
# # Ogni paziente ha 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
# # θ è il vettore completo concatenato: [parametri globali della rete; guess per paziente1; guess per paziente2; ...]
# # θ_init = vcat(nn_params_init, [patient.init_params for patient in training_dataset]...);
# println("Vettore iniziale θ_init creato (dimensione: ", length(θ_init), ")")

# ################################################################################
# # 7. FASE DI TRAINING: OTTIMIZZAZIONE IN DUE STEP
# ################################################################################
# println("Inizio fase di training...")

# global_progress = Progress(100, desc="Ottimizzazione globale ADAM", dt=0.5);

# # Definisci una callback che verrà chiamata ad ogni iterazione.
# # La callback riceve lo "stato" corrente dell'ottimizzazione (state).

# losses = Float64[];

# callback_func = (state, l) -> begin
#     push!(losses, l)
#     if length(losses) % 10 == 0
#         println("Current loss after $(length(losses)) iterations: $(losses[end])")
#     end
#     next!(global_progress)  # Avanza la progress bar di uno step.
#     return false            # Restituisce false per non terminare prematuramente.
# end

# # Definisci la funzione di loss come una funzione che accetta due argomenti:
# # - θ: il vettore dei parametri
# # - data: una tupla contenente il training_dataset (in questo caso)
# optfunc = OptimizationFunction((θ, x) -> training_loss(θ, training_dataset, nn_params_init), AutoForwardDiff());
# println("- Optimization function defined")

# println("- Adam Optimization started")
# # Primo step: utilizziamo Gradient Descent per una convergenza rapida
# optprob = Optimization.OptimizationProblem(optfunc, θ_init);
# opt_result1 = Optimization.solve(optprob, Optimisers.Adam(0.01), maxiters=100, callback=callback_func);
# θ_intermediate = opt_result1.u;
# println("Training loss after $(length(losses)) iterations: $(losses[end])")

# global_progress = Progress(100, desc="Affinament LBFGS", dt=0.5); # reset progressbar

# println("- LBFGS Optimization started")
# # Per il secondo step, creiamo una nuova progress bar con, ad esempio, 100 iterazioni
# # Secondo step: affinamento con LBFGS usando BackTracking per la line search
# optprob2 = Optimization.OptimizationProblem(optfunc, θ_intermediate);
# opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100, callback=callback_func);

# θ_opt = opt_result2.u; # risultato per un set di parametri, va messo in un array

# println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# final_loss = training_loss(θ_opt, training_dataset, nn_params_init);
# println("Training completato. Loss finale: ", final_loss)

# # Plot the losses
# pl_losses = plot(1:adam_maxiters, losses[1:adam_maxiters], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
# plot!(101:length(losses), losses[101:end], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)

# # Ho bisogno ora di ottimizzare il processo, magari creando un multistart

# println("Flusso di esecuzione completato.")

# # @save "$(models_path)/theta_opt_ann.jld2" θ_opt

# nn_params = θ_opt[1:97]

# i = 1
# patient = training_dataset[i]
# idx_start = length(nn_params) + 5*(i-1) + 1  # per il primo paziente
# idx_end   = length(nn_params) + 5*i
# patient_params = θ_opt[idx_start:idx_end]  # usa il guess iniziale, per esempio
# tspan = (0.0, patient.timepoints[end]+10)

# chain = neural_network_model(2, 6; input_dims=7);

# # Costruisci il modello per questo paziente:
# model = ctntCUDEModel(patient_params, chain, tspan)

# p = ComponentArray(ode = patient_params, neural = nn_params)
# sol = solve(model.problem, p=p, saveat=1)
# pred = [u[3] for u in sol.u]

# plot(pred, lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
# scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")
using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter: Progress, next!

include("ctnt-ude-model.jl")

############################
# 1. Caricamento del dataset
############################
println("Dataset loading...")
# Percorso del file Excel
# file_path = "data/STEMI_merged.xlsx";
# sheet_times = "Tempi cleaned";
# sheet_values = "Misurazioni cleaned";

file_path = "data/UMG_NSTEMI_Dataset.xlsx";
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

xf = XLSX.readxlsx(file_path)
# Caricamento dei fogli in DataFrame
# ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:O", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:O", header=false, infer_eltypes=true));

foreach(names(troponin_df)) do col
    println(col, " ⇒ ", unique(typeof.(troponin_df[!, col])))
end

println("Patient loaded: ", nrow(ids))
println("Initialize...")

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

Random.seed!(1234);
rng = StableRNG(42)

# initial_params = [0.005, 0.005, 0.1, 0.001, 0.1];
# β_init = 0.1

# initial_ode = sample_initial_parameters(nrow(ids), 1, lhs_lb, lhs_ub, rng)

chain = neural_network_model(2, 6; input_dims=7);
nn_params_init = init_params(chain);
println("NN parameter len: $(length(nn_params_init))")
# init_model = ComponentArray(neural = nn_params_init, ode = initial_params, β=β_init)
# Costruisci l'array di PatientData iterando su tutte le righe.

shuffle!(patients);
n_train = Int(round(length(patients) * 0.8));
training_dataset = patients[1:n_train];
test_dataset = patients[n_train+1:end];
println("Training split: ", length(training_dataset))
println("Validation split: ", length(test_dataset))

n_conditional = 1
lhs_lb = log.[0.001, 0.001, 0.01, 0.001, 0.001]
lhs_ub = log.[5.0, 5.0, 300.0, 400.0, 1]
# [a, b, Cs0, Cc0 ... last one is conditional parameter β]
initial_guesses = 10

initial_nn = sample_initial_neural_parameters(initial_guesses, chain, rng)
initial_ode = sample_initial_parameters(length(training_dataset), initial_guesses, lhs_lb, lhs_ub, rng)

initial_parameters = [ComponentArray(
        neural = initial_nn[i],
        ode = repeat(initial_ode[:,i], 1, n_conditional)
        ) for i in eachindex(initial_nn)]

# train_patients = []
# for j in 1:initial_guesses
#     patient = [row_to_patient(
#         training_dataset[i].id,
#         training_dataset[i].timepoints,
#         training_dataset[i].ctnt_data,
#         ComponentArray(neural = initial_nn[j],
#                         ode = initial_ode[(length(lhs_lb)*(j-1) + 1):(length(lhs_lb)*j-1)],
#                         β=initial_ode[length(lhs_lb)*j]))
#         for i in eachindex(training_dataset)]
#     push!(train_patients, patient...);
# end

# i = 1; k = 250;
# #iterazione del for che somma le loss e calcola la media
# p_model = initial_parameters[k] # k = indice random dei component arrays non collegato a i
# θ_set = p_model.ode
# neural = p_model.neural
# idx_start = 5*(i-1) + 1  # per il primo paziente
# idx_end   = 5*i
# patient_params = log.(θ_set[idx_start:idx_end])  # usa il guess iniziale, per esempio
# patient = training_dataset[i]
# tspan = (patient.timepoints[1], patient.timepoints[end])
# model = ctntCUDEModel(patient_params, chain, tspan)
# test_loss = patient_loss(patient_params, model, patient.timepoints, patient.ctnt_data, neural) # patient_params, model::ctntCUDEModel, timepoints, ctnt_data, fixed_nn_params

losses_initial = Float64[]
models = []
prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
# models = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (training_dataset[j].timepoints[1], training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];

for p in initial_parameters # p = initial_parameters[k]
    models_array = [ctntCUDEModel(p.ode[5*(j-1) + 1:5*j], chain, (training_dataset[j].timepoints[1], training_dataset[j].timepoints[end])) for j in eachindex(training_dataset)];
    push!(models, models_array)
    loss_value = training_loss(p, (models_array, training_dataset))
    push!(losses_initial, loss_value)
    next!(prog)
end
    # θ_set = p.ode
    # neural = p.neural
    # N_nn = length(nn_params_init)
    # loss_tot = 0.0
    # for (i, patient) in enumerate(training_dataset) # patient = training_dataset[i]
    #     idx_start = 5*(i-1) + 1
    #     idx_end   = 5*i
    #     patient_params = log.(θ_set[idx_start:idx_end])
    #     tspan = (patient.timepoints[1], patient.timepoints[end])

    #     model = ctntCUDEModel(patient_params, chain, tspan)
    #     ### Calcolo cost function ###
    #     loss_tot += patient_loss(patient_params, model, patient.timepoints, patient.ctnt_data, neural)
    #     # loss_tot += sum(abs2, sol[3,:] - patient.ctnt_data)
    # end
    # loss_value = loss_tot / length(training_dataset)
    # loss_value = training_loss(p, training_dataset)
    # push!(losses_initial, loss_value)
    # next!(prog)

selected_initials = 2
param_indxs = partialsortperm(losses_initial, 1:selected_initials)
out_params = initial_parameters[param_indxs]

models = models[param_indxs];

@save "res/models/out_paramsODE0506log.jld2" out_params
@load "res/models/out_paramsODE0506log.jld2" out_params

# for param_indx in partialsortperm(losses_initial, 1:25)
#     println(initial_parameters[param_indx])
# end

patience   = 20          # numero max di iterazioni senza miglioramenti
min_delta  = 1e-6        # miglioramento minimo per essere accettato
best_loss  = Inf         # parte a +∞
stagnation = 0           # contatore interno
best_θ     = copy(out_params[1])
losses     = Float64[]   # vettore per il log, come prima

callback_func = (state, l) -> begin
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    # next!(prog)  # Avanza la progress bar di uno step.
        # 2. early-stop: aggiorna record o conta stagnazione
    global best_loss, stagnation
    if l < best_loss - min_delta          # miglioramento significativo?
        best_loss  = l
        global best_θ = copy(state.u)
        stagnation = 0                    # reset pazienza
    else
        stagnation += 1                   # passo “a vuoto”
    end

    # 3. se ho superato la patience → fermo l’ottimizzatore
    return stagnation ≥ patience             # Restituisce false per non terminare prematuramente.
end

adam_maxiters = 500
lbfgs_maxiters = 250

optsols = OptimizationSolution[]
optfunc = OptimizationFunction(training_loss, AutoForwardDiff())
# lower_bound = log.([0.001, 0.001, 0.001, 0.001, 0.001])
# upper_bound = log.([5, 5, 400, 400, 1])
# prog = Progress(100; dt=0.5, desc="Single solution optimizing...", showspeed=true, color=:firebrick)
global_prog = Progress(selected_initials; dt=1, desc="Global process optimizing...", showspeed=true, color=:blue)
for (i, θ_init) in enumerate(out_params)
    # try
    println("ADAM for parameter set: $(i)")
    # global_progress = Progress(100, desc="Ottimizzazione globale ADAM", dt=0.5);
    optprob = Optimization.OptimizationProblem(optfunc, θ_init, (models[i], training_dataset));
    opt_result1 = Optimization.solve(optprob, Optimisers.Adam(0.01), maxiters=adam_maxiters, callback=callback_func);
    println("LBFGS for parameter set: $(i)")
    # global_progress = Progress(100, desc="Affinament LBFGS", dt=0.5);
    # println(opt_result1.u)
    optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u, (models[i], training_dataset));
    opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=lbfgs_maxiters, callback=callback_func);
    push!(optsols, opt_result2)
    println("Solutions: $(length(optsols))/$selected_initials")
    # catch
        # println("Optimization failed... Skipping")
    # end
    next!(global_prog)
end

@save "res/models/lossesUDE0506log.jld2" losses
@save "res/models/optsolsUDE0506log.jld2" optsols



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

# # @save "res/models/theta_opt_ann.jld2" θ_opt

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
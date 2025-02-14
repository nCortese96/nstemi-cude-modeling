using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches

include("ctnt-ude-model.jl")

############################
# 1. Caricamento del dataset
############################
println("Caricamento del dataset...")
# Percorso del file Excel
file_path = "data/STEMI_merged.xlsx";
sheet_times = "Tempi cleaned";
sheet_values = "Misurazioni cleaned";

# Caricamento dei fogli in DataFrame
ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "B:X", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "B:X", header=false, infer_eltypes=true));

initial_params = [0.005, 0.005, 0.1, 0.001, log(0.1)]

# Costruisci l'array di PatientData iterando su tutte le righe.
n_rows = nrow(ids)
patients = [row_to_patient(i, ids, timepoints_df, troponin_df, initial_params) for i in 1:n_rows]

println("Numero di pazienti caricati: ", length(patients))

##############################################
# 2. Suddivisione in training e validation
##############################################
Random.seed!(1234)
shuffle!(patients)
n_train = Int(round(length(patients) * 0.8))
training_dataset = patients[1:n_train]
test_dataset = patients[n_train+1:end]

println("Pazienti per il training: ", length(training_dataset))
println("Pazienti per la validation: ", length(test_dataset))

##############################################
# 3. Creazione della rete neurale e parametri
##############################################
println("Creazione della rete neurale...")
# La rete riceve un vettore di 7 elementi: [u[1], t, a, b, Cs0, Cc0, β]
chain = neural_network_model(2, 6; input_dims=7)
nn_params_init = init_params(chain)

#############################################################
# 4. Preparazione del vettore iniziale per l'ottimizzazione
#############################################################
# Ogni paziente ha 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
# θ è il vettore completo concatenato: [parametri globali della rete; guess per paziente1; guess per paziente2; ...]
θ_init = vcat(nn_params_init, [patient.init_params for patient in training_dataset]...)
println("Vettore iniziale θ_init creato (dimensione: ", length(θ_init), ")")

################################################################################
# 7. FASE DI TRAINING: OTTIMIZZAZIONE IN DUE STEP
################################################################################
println("Inizio fase di training...")

# Definisci la funzione di loss come una funzione che accetta due argomenti:
# - θ: il vettore dei parametri
# - data: una tupla contenente il training_dataset (in questo caso)
optfunc = OptimizationFunction((θ, x) -> training_loss(θ, training_dataset, nn_params_init), AutoForwardDiff())

# Primo step: utilizziamo Gradient Descent per una convergenza rapida
optprob = Optimization.OptimizationProblem(optfunc, θ_init)
opt_result1 = Optimization.solve(optprob, Optimisers.Adam(0.01), maxiters=1000)
θ_intermediate = opt_result1.u

# Secondo step: affinamento con LBFGS usando BackTracking per la line search
optprob2 = Optimization.OptimizationProblem(optfunc, θ_intermediate)
opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
θ_opt = opt_result2.u

final_loss = training_loss(θ_opt, training_dataset, nn_params_init)
println("Training completato. Loss finale: ", final_loss)

#############################################################
# 6. Selezione del modello migliore sul validation set
#############################################################
# In un flusso reale, candidate_nn_params sarebbero ottenuti dal training.
# Qui usiamo due candidate dummy uguali (nn_params_init) per esempio.
candidate_nn_params = [nn_params_init, nn_params_init]
best_model_index, votes = select_model(validation_dataset, candidate_nn_params)
println("Il miglior candidato (modello globale) è il numero: ", best_model_index)
println("Voti ricevuti: ", votes)

println("Flusso di esecuzione completato.")
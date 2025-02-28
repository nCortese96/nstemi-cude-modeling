using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2

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

initial_params = [log(0.005), log(0.005), log(0.1), log(0.001), log(0.1)];

# Costruisci l'array di PatientData iterando su tutte le righe.
n_rows = nrow(ids);
patients = [row_to_patient(i, ids, timepoints_df, troponin_df, initial_params) for i in 1:n_rows];

println("Numero di pazienti caricati: ", length(patients))

##############################################
# 2. Suddivisione in training e validation
##############################################
Random.seed!(1234);
shuffle!(patients);
n_train = Int(round(length(patients) * 0.8));
training_dataset = patients[1:n_train];
test_dataset = patients[n_train+1:end];

println("Pazienti per il training: ", length(training_dataset))
println("Pazienti per la validation: ", length(test_dataset))

##############################################
# 3. Creazione della rete neurale e parametri
##############################################
println("Creazione della rete neurale...")
# La rete riceve un vettore di 7 elementi: [u[1], t, a, b, Cs0, Cc0, β]
chain = neural_network_model(2, 6; input_dims=7);
nn_params_init = init_params(chain);

#############################################################
# 4. Preparazione del vettore iniziale per l'ottimizzazione
#############################################################
# Ogni paziente ha 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
# θ è il vettore completo concatenato: [parametri globali della rete; guess per paziente1; guess per paziente2; ...]
θ_init = vcat(nn_params_init, [patient.init_params for patient in training_dataset]...);
println("Vettore iniziale θ_init creato (dimensione: ", length(θ_init), ")")

################################################################################
# 7. FASE DI TRAINING: OTTIMIZZAZIONE IN DUE STEP
################################################################################
println("Inizio fase di training...")

global_progress = Progress(100, desc="Ottimizzazione globale ADAM", dt=0.5);

# Definisci una callback che verrà chiamata ad ogni iterazione.
# La callback riceve lo "stato" corrente dell'ottimizzazione (state).

losses = Float64[];

callback_func = (state, l) -> begin
    push!(losses, l)
    if length(losses) % 10 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    next!(global_progress)  # Avanza la progress bar di uno step.
    return false            # Restituisce false per non terminare prematuramente.
end

# Definisci la funzione di loss come una funzione che accetta due argomenti:
# - θ: il vettore dei parametri
# - data: una tupla contenente il training_dataset (in questo caso)
optfunc = OptimizationFunction((θ, x) -> training_loss(θ, training_dataset, nn_params_init), AutoForwardDiff());
println("- Optimization function defined")

println("- Adam Optimization started")
# Primo step: utilizziamo Gradient Descent per una convergenza rapida
optprob = Optimization.OptimizationProblem(optfunc, θ_init);
opt_result1 = Optimization.solve(optprob, Optimisers.Adam(0.01), maxiters=100, callback=callback_func);
θ_intermediate = opt_result1.u;
println("Training loss after $(length(losses)) iterations: $(losses[end])")

global_progress = Progress(100, desc="Affinament LBFGS", dt=0.5); # reset progressbar

println("- LBFGS Optimization started")
# Per il secondo step, creiamo una nuova progress bar con, ad esempio, 100 iterazioni
# Secondo step: affinamento con LBFGS usando BackTracking per la line search
optprob2 = Optimization.OptimizationProblem(optfunc, θ_intermediate);
opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=100, callback=callback_func);
θ_opt = opt_result2.u;
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

final_loss = training_loss(θ_opt, training_dataset, nn_params_init);
println("Training completato. Loss finale: ", final_loss)

# Plot the losses
pl_losses = plot(1:100, losses[1:100], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(101:length(losses), losses[101:end], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)

# Ho bisogno ora di ottimizzare il processo, magari creando un multistart

println("Flusso di esecuzione completato.")

using JLD2
@save "theta_opt.jld2" θ_opt
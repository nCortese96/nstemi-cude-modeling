
################################################################################
# Import delle librerie necessarie
using OrdinaryDiffEq
using SciMLBase: ODEProblem
using SimpleChains: SimpleChain, TurboDense, static, init_params
using ComponentArrays: ComponentArray
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using JLD2, DataFrames, CSV

################################################################################
# 1. Definizione della funzione cUDE per il modello troponin
#
# Stato:
#   u[1] = ctnt nel sarcomero,
#   u[2] = ctnt nel citosol,
#   u[3] = ctnt nel plasma.
#
# p.ode = [a, b, Cs0, Cc0, log(β)]
#   - a: tasso base di diffusione
#   - b: tasso base di clearance
#   - Cs0: condizione iniziale nel sarcomero
#   - Cc0: condizione iniziale nel citosol
#   - log(β): parametro condizionale in log‑scala
#
# L'input alla rete neurale è:
#   [u[1], t, p.ode[1:4]..., β]
# dove β = exp(p.ode[5])
#
function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    β = exp(p.ode[5])
    a = p.ode[1]
    b = p.ode[2]
    # Input: u[1] (stato attuale nel sarcomero), t, [a, b, Cs0, Cc0], β
    correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]
    du[1] = - (u[1] - u[2]) + correction
    du[2] = (u[1] - u[2]) - correction - a*(u[2] - u[3])
    du[3] = a*(u[2] - u[3]) - b*u[3]
end

################################################################################
# 2. Costruzione del modello troponin cUDE
#
# La struttura contiene l'ODEProblem e la rete neurale (chain).
#
struct TroponinCUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
TroponinCUDEModel(a, b, Cs0, Cc0, chain, tspan)

Costruisce il modello per la dinamica della ctnt.
- a, b: parametri di base del modello.
- Cs0, Cc0: condizioni iniziali per il sarcomero e il citosol.
- chain: la rete neurale (SimpleChain).
- tspan: intervallo temporale della simulazione.
"""
function TroponinCUDEModel(a::Float64, b::Float64, Cs0::Float64, Cc0::Float64,
                           chain::SimpleChain, tspan::Tuple{Float64,Float64})
    u0 = [Cs0, Cc0, 0.0]
    f(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
    prob = ODEProblem(f, u0, tspan)
    return TroponinCUDEModel(prob, chain)
end

################################################################################
# 3. Funzione di loss per un singolo modello
#
# Confronta le predizioni del compartimento plasma (u[3]) con i dati osservati.
function loss(θ, args::Tuple{TroponinCUDEModel, AbstractVector{Float64}, AbstractVector{Float64}})
    model, timepoints, ctnt_data = args
    sol = solve(model.problem, Tsit5(); p=θ, saveat=timepoints)
    pred = [u[3] for u in sol.u]
    return sum((pred .- ctnt_data).^2)
end

################################################################################
# 4. Funzione training_loss: somma la loss su tutti i pazienti del training
#
# x è un vettore contenente:
#   - i parametri della rete neurale (globali) (primi N_nn elementi)
#   - per ciascun paziente, 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
function training_loss(x, training_dataset)
    N_nn = length(nn_params_init)
    loss_tot = 0.0
    nn_param_vec = x[1:N_nn]
    for (i, patient) in enumerate(training_dataset)
        idx_start = N_nn + 5*(i-1) + 1
        idx_end   = N_nn + 5*i
        patient_params = x[idx_start:idx_end]
        tspan = (patient.timepoints[1], patient.timepoints[end])
        model = TroponinCUDEModel(patient_params[1], patient_params[2],
                                  patient_params[3], patient_params[4],
                                  chain, tspan)
        p = ComponentArray(ode = patient_params, neural = nn_param_vec)
        sol = solve(model.problem, Tsit5(); p=p, saveat=patient.timepoints)
        pred = [u[3] for u in sol.u]
        loss_tot += sum((pred .- patient.ctnt_data).^2)
    end
    return loss_tot
end

################################################################################
# 5. Funzione patient_loss: loss per un singolo paziente
#
# Data una configurazione specifica del paziente (patient_params) e i parametri fissi della rete,
# calcola la loss sul compartimento plasma.
function patient_loss(patient_params, model::TroponinCUDEModel, fixed_nn_params, timepoints, ctnt_data)
    p = ComponentArray(ode = patient_params, neural = fixed_nn_params)
    sol = solve(model.problem, Tsit5(); p=p, saveat=timepoints)
    pred = [u[3] for u in sol.u]
    return sum((pred .- ctnt_data).^2)
end

################################################################################
# 6. Funzione select_model: seleziona il candidato migliore "a voto"
#
# validation_dataset: vettore di dati di validazione, dove ogni elemento (di tipo PatientData)
# contiene: timepoints, ctnt_data, init_params (guess iniziale [a, b, Cs0, Cc0])
#
# candidate_nn_params: vettore di candidate globali (set di parametri della rete neurale ottenuti dal training)
#
# Per ogni paziente, valuta la loss per ogni candidato (usando un guess iniziale per i parametri specifici)
# e incrementa il voto del candidato che minimizza la loss. Alla fine restituisce l'indice del candidato
# con il maggior numero di voti.
function select_model(validation_dataset, candidate_nn_params)
    counts = Dict{Int,Int}()
    for i in 1:length(candidate_nn_params)
        counts[i] = 0
    end

    for patient in validation_dataset
        candidate_losses = Float64[]
        tspan = (patient.timepoints[1], patient.timepoints[end])
        for (i, candidate_nn) in enumerate(candidate_nn_params)
            model_candidate = TroponinCUDEModel(patient.init_params[1],
                                                patient.init_params[2],
                                                patient.init_params[3],
                                                patient.init_params[4],
                                                candidate_nn, tspan)
            # Usa un guess iniziale per i parametri specifici; ad esempio, log(β) viene fissato a log(0.1)
            initial_guess = [patient.init_params[1],
                             patient.init_params[2],
                             patient.init_params[3],
                             patient.init_params[4],
                             log(0.1)]
            l = patient_loss(initial_guess, model_candidate, candidate_nn, patient.timepoints, patient.ctnt_data)
            push!(candidate_losses, l)
        end
        best_candidate_idx = argmin(candidate_losses)
        counts[best_candidate_idx] += 1
    end

    best_model_index = argmax(values(counts))
    return best_model_index, counts
end

################################################################################
# 7. Definizione della struttura per i dati paziente
#
struct PatientData
    timepoints::Vector{Float64}
    ctnt_data::Vector{Float64}
    init_params::Vector{Float64}  # [a, b, Cs0, Cc0]
end

################################################################################
# 8. Creazione della rete neurale con SimpleChains
#
# La rete riceve in input un vettore di 7 elementi: [u[1], t, a, b, Cs0, Cc0, β]
function neural_network_model(depth::Int, width::Int; input_dims::Int)
    layers = []
    for i in 1:depth
        push!(layers, TurboDense(static(input_dims), width, tanh))
    end
    # Ultimo layer con attivazione softplus (implementata qui come log(1+exp(x)))
    push!(layers, TurboDense(width, 1, x -> log(1+exp(x))))
    return SimpleChain(static(input_dims), layers...)
end

# Creazione della rete neurale; input_dims = 7
chain = neural_network_model(2, 6; input_dims=7)

# Inizializzazione dei parametri globali della rete
nn_params_init = init_params(chain)

################################################################################
# 9. Creazione di un dataset di esempio per training e validazione
#
# In una vera applicazione, questi dati verranno caricati da file o raccolti sperimentalmente.
patient1 = PatientData(collect(0.0:0.1:10.0), rand(101), [0.005, 0.005, 0.1, 0.001])
patient2 = PatientData(collect(0.0:0.1:10.0), rand(101), [0.005, 0.005, 0.1, 0.001])
training_dataset = [patient1, patient2]
validation_dataset = [patient1]  # Per esempio, useremo patient1 come validazione

################################################################################
# 10. Definizione di un guess iniziale per i parametri specifici del paziente
#
# Ogni paziente avrà 5 parametri: [a, b, Cs0, Cc0, log(β)]
function initial_patient_guess()
    return [0.005, 0.005, 0.1, 0.001, log(0.1)]
end

# Creazione del vettore iniziale x0 per l'ottimizzazione congiunta:
# x0 = [nn_params_init; guess per paziente1; guess per paziente2; ...]
x0 = vcat(nn_params_init, initial_patient_guess(), initial_patient_guess())

################################################################################
# 11. Fase di Training (esempio)
#
# Si assume di ottimizzare la funzione training_loss per ottenere i parametri ottimizzati.
# Ad esempio, usando una funzione di ottimizzazione (qui si lascia come commento un placeholder).
#
# x_opt = Optimization.optimize(x -> training_loss(x, training_dataset), x0, <algoritmo>, maxiters=<num_iter>)
#
# Dopo il training, si ottengono candidate_nn_params, cioè diversi set di parametri della rete neurale
# ottenuti da diverse esecuzioni (qui per esempio usiamo due candidate dummy).
candidate_nn_params = [nn_params_init, nn_params_init]  # In pratica, saranno differenti

################################################################################
# 12. Selezione del modello migliore sul validation set
best_model_index, votes = select_model(validation_dataset, candidate_nn_params)
println("Il miglior candidato è il numero: ", best_model_index)
println("Voti: ", votes)

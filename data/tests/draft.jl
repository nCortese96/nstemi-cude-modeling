################################################################################
# IMPORT LIBRERIE
using OrdinaryDiffEq
using SciMLBase: ODEProblem
using SimpleChains: SimpleChain, TurboDense, static, init_params
using ComponentArrays: ComponentArray
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using DataFrames, CSV
using XLSX           # Per leggere file Excel
using Random

################################################################################
# 1. CARICAMENTO DEL DATASET DA EXCEL E CREAZIONE DEL VETTORE DI PATIENTDATA
#
# Il file Excel ("data/dataset.xlsx") contiene due fogli:
# - "Timepoints": la prima colonna è "id" (es. "s1", "s2", …) e le colonne successive
#   contengono i timepoints (prefisso "tp").
# - "Troponin": la prima colonna è "id" e le colonne successive contengono i valori
#   di troponina (prefisso "troponin").
# I dati vengono uniti tramite innerjoin sul campo "id".

excel_file = "data/dataset.xlsx"

# Caricamento dei fogli in DataFrame
timepoints_df = DataFrame(XLSX.readtable(excel_file, "Timepoints")...)
troponin_df  = DataFrame(XLSX.readtable(excel_file, "Troponin")...)

# Unione dei due DataFrame in base a "id"
data_df = innerjoin(timepoints_df, troponin_df, on="id")

# Definizione della struttura per i dati del paziente
struct PatientData
    timepoints::Vector{Float64}   # vettore dei timepoints per il paziente
    ctnt_data::Vector{Float64}      # vettore dei valori di troponina (ctnt)
    init_params::Vector{Float64}    # guess iniziale: [a, b, Cs0, Cc0]
end

# Funzione per trasformare una riga del DataFrame in un PatientData.
function row_to_patient(row)
    id = row.id  # es. "s1", "s2", ...
    # Estrae le colonne dei timepoints (prefisso "tp") e le ordina
    tp_cols = filter(name -> startswith(name, "tp"), names(row))
    timepoints = [parse(Float64, row[col]) for col in sort(tp_cols)]
    # Estrae le colonne dei valori di troponina (prefisso "troponin") e le ordina
    ctnt_cols = filter(name -> startswith(name, "troponin"), names(row))
    ctnt_data = [parse(Float64, row[col]) for col in sort(ctnt_cols)]
    # Se non ci sono guess individuali, usiamo uno standard: [a, b, Cs0, Cc0]
    init_params = [0.005, 0.005, 0.1, 0.001]
    return PatientData(timepoints, ctnt_data, init_params)
end

# Costruzione dell'array di PatientData
patients = [row_to_patient(row) for row in eachrow(data_df)]

# Suddivisione casuale in training (70%) e validation (30%)
Random.seed!(1234)
shuffle!(patients)
n_train = Int(round(length(patients) * 0.7))
training_dataset = patients[1:n_train]
validation_dataset = patients[n_train+1:end]

################################################################################
# 2. DEFINIZIONE DEL MODELLO cUDE PER LA TROPONINA
#
# Il modello considera tre compartimenti:
#   u[1] = ctnt nel sarcomero,
#   u[2] = ctnt nel citosol,
#   u[3] = ctnt nel plasma.
#
# I parametri sono passati in un ComponentArray p.ode = [a, b, Cs0, Cc0, log(β)],
# dove log(β) verrà trasformato in β positivo con exp().
#
# L'input alla rete neurale è: [u[1], t, a, b, Cs0, Cc0, β]

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    β = exp(p.ode[5])
    a = p.ode[1]
    b = p.ode[2]
    # Calcola il termine di correzione usando l'input di dimensione 7:
    # [u[1], t, p.ode[1:4]..., β] equivale a [u[1], t, a, b, Cs0, Cc0, β]
    correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]
    du[1] = - (u[1] - u[2]) + correction
    du[2] = (u[1] - u[2]) - correction - a*(u[2] - u[3])
    du[3] = a*(u[2] - u[3]) - b*u[3]
end

# Struttura per il modello cUDE
struct TroponinCUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
TroponinCUDEModel(a, b, Cs0, Cc0, chain, tspan)
Costruisce il modello per la dinamica della ctnt.
- a, b: parametri base
- Cs0, Cc0: condizioni iniziali per il sarcomero e il citosol
- chain: rete neurale (SimpleChain)
- tspan: intervallo temporale della simulazione
"""
function TroponinCUDEModel(a::Float64, b::Float64, Cs0::Float64, Cc0::Float64,
                           chain::SimpleChain, tspan::Tuple{Float64,Float64})
    u0 = [Cs0, Cc0, 0.0]   # condizioni iniziali: plasma inizia a 0
    f(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
    prob = ODEProblem(f, u0, tspan)
    return TroponinCUDEModel(prob, chain)
end

################################################################################
# 3. DEFINIZIONE DELLE FUNZIONI DI LOSS
#
# loss: calcola la loss per un modello (sul compartimento plasma) su un set di timepoints
function loss(θ, args::Tuple{TroponinCUDEModel, AbstractVector{Float64}, AbstractVector{Float64}})
    model, timepoints, ctnt_data = args
    sol = solve(model.problem, Tsit5(); p=θ, saveat=timepoints)
    pred = [u[3] for u in sol.u]
    return sum((pred .- ctnt_data).^2)
end

# patient_loss: calcola la loss per un singolo paziente dato un vettore di parametri specifici
function patient_loss(patient_params, model::TroponinCUDEModel, fixed_nn_params, timepoints, ctnt_data)
    p = ComponentArray(ode = patient_params, neural = fixed_nn_params)
    sol = solve(model.problem, Tsit5(); p=p, saveat=timepoints)
    pred = [u[3] for u in sol.u]
    return sum((pred .- ctnt_data).^2)
end

# training_loss: somma la loss su tutto il training dataset.
# x è un vettore concatenato composto da:
#   - Parametri globali della rete (primi N_nn elementi)
#   - Per ogni paziente, 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
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
# 4. FUNZIONE select_model: seleziona il candidato migliore "a voto" sul validation set
#
# validation_dataset: vettore di PatientData
# candidate_nn_params: vettore di set di parametri globali della rete (candidati)
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
            # Guess iniziale per i parametri specifici: fissiamo log(β) = log(0.1)
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
# 5. CREAZIONE DELLA RETE NEURALE CON SIMPLECHAINS
#
# La rete riceve in input un vettore di 7 elementi: [u[1], t, a, b, Cs0, Cc0, β]
function neural_network_model(depth::Int, width::Int; input_dims::Int)
    layers = []
    for i in 1:depth
        push!(layers, TurboDense(static(input_dims), width, tanh))
    end
    # Ultimo layer con attivazione softplus: log(1+exp(x))
    push!(layers, TurboDense(width, 1, x -> log(1+exp(x))))
    return SimpleChain(static(input_dims), layers...)
end

# Creazione della rete neurale; qui input_dims = 7
chain = neural_network_model(2, 6; input_dims=7)
nn_params_init = init_params(chain)

################################################################################
# 6. PREPARAZIONE DEI GUESS INIZIALI PER I PARAMETRI SPECIFICI DEI PAZIENTI
#
# Ogni paziente ha 5 parametri: [a, b, Cs0, Cc0, log(β)]
function initial_patient_guess()
    return [0.005, 0.005, 0.1, 0.001, log(0.1)]
end

# Creazione del vettore iniziale x0 per l'ottimizzazione congiunta:
# x0 = [parametri globali della rete; guess per paziente1; guess per paziente2; ...]
x0 = vcat(nn_params_init, initial_patient_guess(), initial_patient_guess())
# In questo esempio abbiamo due pazienti nel training_dataset.

################################################################################
# 7. FASE DI TRAINING (placeholder)
#
# Qui si eseguirebbe l'ottimizzazione della funzione training_loss sul training_dataset,
# aggiornando x0 per ottenere il vettore ottimizzato x_opt.
#
# Ad esempio, potresti usare un algoritmo come GradientDescent seguito da LBFGS.
# Esempio (commentato):
# using OptimizationOptimisers
# optsol = Optimization.optimize(x -> training_loss(x, training_dataset), x0, GradientDescent(0.01), maxiters=100)
# x_opt = optsol.u

################################################################################
# 8. SELEZIONE DEL MODELLO MIGLIORE SUL VALIDATION SET
#
# Supponiamo che, dal training, siano stati ottenuti diversi candidate globali (candidate_nn_params).
# Qui, per semplicità, usiamo due candidate dummy (identiche).
candidate_nn_params = [nn_params_init, nn_params_init]  # In pratica saranno differenti
best_model_index, votes = select_model(validation_dataset, candidate_nn_params)
println("Il miglior candidato è il numero: ", best_model_index)
println("Voti: ", votes)

################################################################################
# Fine dello script
#
# Il flusso completo è:
#  - Caricamento e preparazione dei dati da Excel.
#  - Suddivisione in training e validation.
#  - Definizione del modello cUDE con rete neurale.
#  - Definizione delle funzioni di loss e delle procedure di ottimizzazione.
#  - (Placeholder) Ottimizzazione sul training set.
#  - Selezione del modello migliore sul validation set.

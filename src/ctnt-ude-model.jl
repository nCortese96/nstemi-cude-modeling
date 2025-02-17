using SimpleChains: SimpleChain, TurboDense, static, init_params
using SciMLBase: ODEProblem, OptimizationSolution
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray

using ProgressMeter: Progress, next!

using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches

softplus(x) = log(1 + exp(x))

# Definizione della struttura per i dati del paziente
struct PatientData
    id::String
    timepoints::Vector{Float64}   # vettore dei timepoints per il paziente
    ctnt_data::Vector{Float64}      # vettore dei valori di troponina
    init_params::Vector{Float64}    # guess iniziale, ad esempio: [a, b, Cs0, Cc0]
end

# Funzione per convertire la riga i-esima dei tre DataFrame in una struttura PatientData.
# Per ogni riga vengono rimossi i valori mancanti (missing) e si mantengono solo i valori validi.
function row_to_patient(i, ids::DataFrame, timepoints_df::DataFrame, troponin_df::DataFrame, initial_params::AbstractVector{Float64})
    # Estrai l'ID (se serve)
    id_val = ids[i, :][1]
    
    # Estrai i valori della riga come vettori.
    # timepoints_df[i, :] restituisce una NamedTuple; convertiamola in vettore.
    tp_row = [x for x in collect(values(timepoints_df[i, :])) if !ismissing(x)]
    ctnt_row = [x for x in collect(values(troponin_df[i, :])) if !ismissing(x)]
    
    # Opzionalmente, se i valori sono stringhe, li puoi convertire in Float64.
    # In questo esempio assumiamo che infer_eltypes=true li abbia già convertiti.
    
    # Definisci dei guess iniziali standard, ad esempio:
    # init_params = [0.005, 0.005, 0.1, 0.001, log(0.1)]
    
    return PatientData(id_val, tp_row, ctnt_row, initial_params)
end

function neural_network_model(depth::Int, width::Int; input_dims::Int = 7)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    # Esempio di termini dinamici (da adattare al modello specifico)
    # Termini base (senza correzione)
    # p.ode = [a, b, Cs0, Cc0]
    # Cs_ctnt = u[1]
    # Cc_ctnt = u[2]
    # Cp_ctnt = u[3]

    β = exp(p.ode[5]) # sempre positivo

    # a = 10 ^ θ[1]
    # b = 10 ^ θ[2]

    a = exp(p.ode[1])
    b = exp(p.ode[2])
    Cc0 = exp(p.ode[3])
    Cs0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    correction = chain([u[1], t, a, b, Cc0, Cs0, β], p.neural)[1]

    du[1] = - (u[1] - u[2] + correction)
    du[2] = (u[1] - u[2] + correction) - a*(u[2] - u[3])
    du[3] = a*(u[2] - u[3]) - b*u[3]

end

struct ctntCUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

function ctntCUDEModel(
    # ctnt_timepoints::AbstractVector{T},
    θ,
    chain::SimpleChain,
    tspan::Tuple{T,T}
    ) where T <: Real

    # construct the ude function
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)

    # tspan = (ctnt_timepoints[1], ctnt_timepoints[end])

    Cc0 = exp(θ[3])
    Cs0 = exp(θ[4])

    u0 = [Cc0, Cs0, 0];

    # ode = ODEProblem(cude!, u0, tspan, θ)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntCUDEModel(ode, chain)
end


########################## LOSS FUNCTIONS ##########################################


function loss(θ, (model, timepoints, ctnt_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: ctntCUDEModel

    # solve the ODE problem
    sol = solve(model.problem, p=θ, saveat=patient.timepoints)
    pred = [u[3] for u in sol.u]
    # Calculate the mean squared error
    return sum((pred .- patient.ctnt_data).^2)
end

# 4. Funzione training_loss: somma la loss su tutti i pazienti del training
#
# x è un vettore contenente:
#   - i parametri della rete neurale (globali) (primi N_nn elementi)
#   - per ciascun paziente, 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
function training_loss(θ, training_dataset, nn_params_init)
    N_nn = length(nn_params_init)
    loss_tot = 0.0
    nn_param_vec = θ[1:N_nn]
    # pbar = Progress(length(training_dataset), desc="Calcolo training_loss")
    # println("Calcolo training loss...")
    for (i, patient) in enumerate(training_dataset)
        # println(patient.id)
        idx_start = N_nn + 5*(i-1) + 1
        idx_end   = N_nn + 5*i
        patient_params = θ[idx_start:idx_end]
        tspan = (patient.timepoints[1], patient.timepoints[end])

        model = ctntCUDEModel(patient_params, chain, tspan)
        p = ComponentArray(ode = patient_params, neural = nn_param_vec)
        # println(patient.timepoints)
        ### Calcolo cost function ###
        sol = solve(model.problem, p=p, saveat=patient.timepoints)
        # println(sol)
        pred = [u[3] for u in sol.u]
        loss_tot += sum((pred .- patient.ctnt_data).^2)
        # println(sol)
        # loss_tot += sum(abs2, sol[3,:] - patient.ctnt_data)
        # next!(pbar)
    end
    return loss_tot
end

# patient_loss: calcola la loss per un singolo paziente dato un vettore di parametri specifici
function patient_loss(patient_params, model::ctntCUDEModel, fixed_nn_params, timepoints, ctnt_data)
    p = ComponentArray(ode = patient_params, neural = fixed_nn_params)
    sol = solve(model.problem, Tsit5(); p=p, saveat=timepoints)
    pred = [u[3] for u in sol.u]
    return sum((pred .- ctnt_data).^2)
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
            model_candidate = ctntCUDEModel(θ, candidate_nn, tspan)
            # Guess iniziale per i parametri specifici: fissiamo log(β) = log(0.1)
            initial_guess = [log(patient.init_params[1]),
                             log(patient.init_params[2]),
                             log(patient.init_params[3]),
                             log(patient.init_params[4]),
                             log(patient.init_params[5])]
            l = patient_loss(initial_guess, model_candidate, candidate_nn, patient.timepoints, patient.ctnt_data)
            push!(candidate_losses, l)
        end
        best_candidate_idx = argmin(candidate_losses)
        counts[best_candidate_idx] += 1
    end

    best_model_index = argmax(values(counts))
    return best_model_index, counts
end

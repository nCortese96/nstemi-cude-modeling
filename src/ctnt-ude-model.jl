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


"""
neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

Constructs a neural network model with a given depth and width. The input dimensions are set to 2 by default.

# Arguments
- `depth::Int`: The depth of the neural network.
- `width::Int`: The width of the neural network.
- `input_dims::Int`: The number of input dimensions. Default is 2.

# Returns
- `SimpleChain`: A neural network model.
"""
function neural_network_model(depth::Int, width::Int; input_dims::Int = 7)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

"""
    ctnt_cude!(du, u, p, t)

Compartments. 
- u[1]: sarcomere
- u[2]: citosol
- u[3]: plasma
- p: model parameters
"""
function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    # Esempio di termini dinamici (da adattare al modello specifico)
    # Termini base (senza correzione)
    # p.ode = [a, b, Cs0, Cc0]
    # Cs_ctnt = u[1]
    # Cc_ctnt = u[2]
    # Cp_ctnt = u[3]

    β = exp(p.ode[5])

    # a = 10 ^ θ[1]
    # b = 10 ^ θ[2]

    a = p.ode[1]
    b = p.ode[2]

    correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    du[1] = - (u[1] - u[2] + correction)
    du[2] = (u[1] - u[2] + correction) - a*(u[2] - u[3])
    du[3] = a*(u[2] - u[3]) - b*u[3]

end

struct ctntCUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
ctntCUDEModel

# Arguments

# Returns
- `ctntCUDEModel`: A ctnt model with conditional neural network for module flux between sarcomere and cytosol

TODO: Add docs
"""
function ctntCUDEModel(
    # ctnt_timepoints::AbstractVector{T},
    θ,
    chain::SimpleChain,
    tspan::Tuple{T,T}
    ) where T <: Real

    # construct the ude function
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)

    # tspan = (ctnt_timepoints[1], ctnt_timepoints[end])

    u0 = [θ[3], θ[4], 0];

    # ode = ODEProblem(cude!, u0, tspan, θ)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntCUDEModel(ode, chain)
end


########################## LOSS FUNCTIONS ##########################################


"""
loss(θ, (model, timepoints, ctnt_data))

Sum of squared errors loss function for the ctnt release model.

# Arguments
- `θ`: The parameter vector.
- `model::ctntCUDEModel`: The ctnt release model.
- `timepoints::AbstractVector{T}`: The timepoints.
- `ctnt_data::AbstractVector{T}`: The ctnt release data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, ctnt_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: ctntCUDEModel

    # solve the ODE problem
    sol = Array(solve(model.problem, p=θ, saveat=timepoints))
    # Calculate the mean squared error
    return sum(abs2, sol[3,:] - ctnt_data)
end


"""
# 4. Funzione training_loss: somma la loss su tutti i pazienti del training
#
# x è un vettore contenente:
#   - i parametri della rete neurale (globali) (primi N_nn elementi)
#   - per ciascun paziente, 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
"""
function training_loss(θ, training_dataset, nn_params_init)
    N_nn = length(nn_params_init)
    loss_tot = 0.0
    nn_param_vec = θ[1:N_nn]
    for (i, patient) in enumerate(training_dataset)
        idx_start = N_nn + 5*(i-1) + 1
        idx_end   = N_nn + 5*i
        patient_params = θ[idx_start:idx_end]
        tspan = (patient.timepoints[1], patient.timepoints[end])

        model = ctntCUDEModel(θ[idx_start:idx_end], chain, tspan)
        p = ComponentArray(ode = patient_params, neural = nn_param_vec)
        # println(patient.timepoints)
        sol = Array(solve(model.problem, Tsit5(); p=p, saveat=patient.timepoints))
        # println(sol)
        # pred = [u[3] for u in sol.u]
        # loss_tot += sum((pred .- patient.ctnt_data).^2)
        loss_tot += sum(abs2, sol[3,:] - patient.ctnt_data)
    end
    return loss_tot
end


function create_progressbar_callback(its, run)
    prog = Progress(its; dt=1, desc="Optimizing run $(run) ", showspeed=true, color=:blue)
    function callback(_, _)
        next!(prog)
        false
    end

    return callback
end


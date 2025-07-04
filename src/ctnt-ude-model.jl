using SimpleChains: SimpleChain, TurboDense, static, init_params
using SciMLBase: successful_retcode, ODEProblem, OptimizationSolution, OptimizationFunction, OptimizationProblem
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using DataFrames: DataFrame
using StableRNGs, StatsBase
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches
using OrdinaryDiffEq: AutoTsit5, Rosenbrock23

using ProgressMeter: Progress, next!

softplus(x) = log(1 + exp(x))

sigmoid(x) = 1 / (1 + exp(-x))

const DELTA = 1e-6 # 0.007 # con cutoff 0.014 ng/mL # 1e-3
const EPS   = 0.014
T_SCALE = 350
μ_u1 = -1.5
σ_u1 = 1.5

smape(pred, obs) = 200 * mean(abs.(pred .- obs) ./ (abs.(pred) .+ abs.(obs) .+ EPS))

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    β = exp(p.ode[5]) # Positive conditional parameter

    a = exp(p.ode[1])
    b = exp(p.ode[2])
    # Cs0 = exp(p.ode[3])
    # Cc0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    # correction = chain([u[1], t, a, b, Cs0, Cc0, β], p.neural)[1]

    # correction = chain([u[1], t, β], p.neural)[1]

    t_norm   = t / T_SCALE

    correction = chain([t_norm, β], p.neural)[1]

    du[1] = - (Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a*(Cc - u[3])
    du[3] = a*(Cc - Cp) - b*Cp

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
    
    # println("In model: ", θ)
    # construct the ude function
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)

    # tspan = (ctnt_timepoints[1], ctnt_timepoints[end])
    
    Cs0 = exp(θ[3]) # exp both if params in log
    Cc0 = exp(θ[4])

    u0 = [Cc0, Cs0, 0];

    # ode = ODEProblem(cude!, u0, tspan, θ)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntCUDEModel(ode, chain)
end

# Definizione della struttura per i dati del paziente
struct PatientData
    id::String
    timepoints::Vector{Float64}   # vettore dei timepoints per il paziente
    ctnt_data::Vector{Float64}    # vettore dei valori di troponina
end

function row2Patient(id::String, timepoints_df::AbstractVector, troponin_df::AbstractVector)
    tp_row = [x for x in collect(values(timepoints_df)) if !ismissing(x)]
    ctnt_row = [x for x in collect(values(troponin_df)) if !ismissing(x)]
    return PatientData(id, tp_row, ctnt_row)
end

function row2Patient(ids::DataFrameRow, timepoints_df::DataFrameRow, troponin_df::DataFrameRow)
    id_val = ids[1]
    tp_row = [x for x in collect(values(timepoints_df)) if !ismissing(x)]
    ctnt_row = [x for x in collect(values(troponin_df)) if !ismissing(x)]
    return PatientData(id_val, tp_row, ctnt_row)
end

function neural_network_model(depth::Int, width::Int; input_dims::Int = 7)

    layers = []

    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))
    # push!(layers, TurboDense{true}(sigmoid, 1))

    SimpleChain(static(input_dims), layers...)
end

function sample_initial_neural_parameters(n_initials::Int, chain::SimpleChain, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_parameters(n_patients::Int, n_initials::Int, lhs_lb::AbstractVector{T}, lhs_ub::AbstractVector{T}, rng::AbstractRNG) where T <: Real
    # return sample(n_initials, lhs_lb, lhs_ub, LatinHypercubeSample(rng))
    return sample(n_initials, repeat(lhs_lb, n_patients), repeat(lhs_ub, n_patients), LatinHypercubeSample(rng))
end

########################## LOSS FUNCTIONS ##########################################

function compute_loss(θ, (model, timepoints, ctnt_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: ctntCUDEModel
    # solve the ODE problem
        sol = solve(model.problem, AutoTsit5(Rosenbrock23()); p=θ, saveat=timepoints)

        if !successful_retcode(sol)
            # If the solver fails, return infinity
            return Inf
        end
        solution = Array(sol);
        pred = solution[3,:];
        return sum(abs2, pred - ctnt_data)
        # return sum(((solution[3,:] - ctnt_data).^2).*ctnt_data)
        # return sum(abs2, log.(pred .+ DELTA) .- log.(ctnt_data .+ DELTA))
        # return smape(pred, ctnt_data)   # % su base 0–100
        # return 100 * mean(abs, (pred .- ctnt_data) ./ (ctnt_data .+ EPS))
        # return sqrt(mean((log.(pred .+ DELTA) .- log.(ctnt_data .+ DELTA)).^2))
end

## Finito il train si estraggono i parametri della rete 
# patient_loss: Quando sono noti i parametri della rete
function patient_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode = θ, neural = fixed_nn_params)

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]

    # ODEProblem aggiornato
    prob = remake(model.problem; u0 = u0, p = p)

    sol = solve(prob, AutoTsit5(Rosenbrock23()); p=p, saveat=timepoints) 

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end
    solution = Array(sol)
    pred = solution[3,:];
    return sum(abs2, pred - ctnt_data)
    # return sum(abs2, log.(pred .+ DELTA) .- log.(ctnt_data .+ DELTA))
    # return sum(((solution[3,:] - ctnt_data).^2).*ctnt_data)
    # return smape(pred, ctnt_data)
end
# La differenza sta nel dove si crea il component array:
# Se lo dai in pasto alla loss lo ottimizza tutto,
# se lo costruisci dentro ottimizza solo i parametri del modello

function smape_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode = θ, neural = fixed_nn_params)

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]

    # ODEProblem aggiornato
    prob = remake(model.problem; u0 = u0, p = p)

    sol = solve(prob, AutoTsit5(Rosenbrock23()); p=p, saveat=timepoints) 

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end
    solution = Array(sol)
    pred = solution[3,:];
    # return sum(abs2, pred - ctnt_data)
    # return sum(abs2, log.(pred .+ DELTA) .- log.(ctnt_data .+ DELTA))
    # return sum(((solution[3,:] - ctnt_data).^2).*ctnt_data)
    return smape(pred, ctnt_data)
end

function training_loss(p, (models, training_dataset))
    loss_tot = 0.0
    for (i, model) in enumerate(models)
        patient = training_dataset[i];
        idx_start = 5*(i-1) + 1
        idx_end   = 5*i
        θ = ComponentArray(ode = p.ode[idx_start:idx_end], neural = p.neural)
        u0_new = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]
        prob = remake(model.problem; u0 = u0_new, p = θ)
        new_model = ctntCUDEModel(prob, model.chain) 
        loss_tot += compute_loss(θ, (new_model, patient.timepoints, patient.ctnt_data))
        # loss_tot += sum(abs2, sol[3,:] - patient.ctnt_data)
    end
    return loss_tot / length(training_dataset)
end
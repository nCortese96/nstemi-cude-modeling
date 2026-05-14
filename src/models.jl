"""
models.jl

ODE and cUDE model definitions shared by the refactored workflow.

Sections:
- Math Utilities: stable transforms, shared constants, and metrics.
- ODE Models: mechanistic Td-sigmoid model and cUDE model containers.
- Loss Functions: patient-level and cohort-level objectives.
"""

using Base.Threads
using ComponentArrays: ComponentArray
using OrdinaryDiffEq: Tsit5
using SciMLBase: ODEProblem, successful_retcode, solve, remake
using SimpleChains: SimpleChain, TurboDense, static
using Statistics: mean

# =============================================================================
# Math Utilities
# =============================================================================

softplus(x) = log(1 + exp(x))
softplus_stable(x) = log1p(exp(-abs(x))) + max(x, 0)
relu_smooth(x; κ=0.05) = κ * softplus_stable(x / κ)

sigmoid(x) = 1 / (1 + exp(-x))
relu(x) = ifelse(x > 0, x, zero(x))

to_bounds(x, lb, ub; κ=0.05) = x +
                               κ * softplus_stable((lb - x) / κ) -
                               κ * softplus_stable((x - ub) / κ)

const DELTA = 1e-6
const T_SCALE = 240.0
const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, T_SCALE]

smape(pred, obs) = 200 * mean(abs.(pred .- obs) ./ (abs.(pred) .+ abs.(obs)))
rmsle(y_true, y_pred) = sqrt(mean((log.(y_pred .+ 1) .- log.(y_true .+ 1)) .^ 2))

"""
    initial_conditions_from_log_params(θ)

Return the ODE initial state `[Cs0, Cc0, 0.0]` from log-scale patient
parameters where `θ[3] = log(Cs0)` and `θ[4] = log(Cc0)`.
"""
initial_conditions_from_log_params(θ) = [exp(θ[3]), exp(θ[4]), 0.0]

"""
    log_mse_loss(pred, obs; delta=DELTA)

Return the mean squared error on `log(x + delta)` for predicted and observed
troponin concentrations.
"""
log_mse_loss(pred, obs; delta=DELTA) = mean(abs2, log.(pred .+ delta) .- log.(obs .+ delta))

"""
    initial_condition_penalty(θ)

Return the smooth penalty used when `Cc0 > Cs0`.
"""
initial_condition_penalty(θ) = relu_smooth(θ[4] - θ[3])^2

"""
    bounded_patient_parameters(ode_params, patient_index, n_params, lb_param, ub_param, κ_bounds)

Return the log-scale ODE parameter slice for one patient, optionally mapped
inside lower/upper bounds with a smooth transform.
"""
function bounded_patient_parameters(ode_params, patient_index::Integer, n_params::Integer, lb_param, ub_param, κ_bounds::Real)
    idx1 = n_params * (patient_index - 1) + 1
    idx2 = n_params * patient_index

    if κ_bounds != 0.0
        @views ode_raw = ode_params[idx1:idx2]
        lb_param === nothing && return ode_raw

        ode_i = similar(ode_raw)
        @inbounds for k in 1:n_params
            ode_i[k] = to_bounds(ode_raw[k], lb_param[k], ub_param[k]; κ=κ_bounds)
        end
        return ode_i
    end

    return ode_params[idx1:idx2]
end

# =============================================================================
# ODE Models
# =============================================================================

"""
    ctntUDEModel

Container for a patient ODE problem and the neural correction chain used by cUDE
training and evaluation scripts.
"""
struct ctntUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
    troponin_ode!(du, u, p, τ)

Mechanistic Td-sigmoid troponin ODE used by step 01.
"""
function troponin_ode!(du, u, p, τ)
    Cs, Cc, Cp = u

    a = exp(p[1])
    b = exp(p[2])
    Td = exp(p[5])

    n = 3.0
    τn = τ^n
    Td_n = Td^n
    fτ = τn / (τn + Td_n)

    du[1] = -(Cs - Cc) * fτ
    du[2] = (Cs - Cc) * fτ - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp
end

function ctnt_ude!(du, u, p, t, chain::SimpleChain)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    a = exp(p.ode[1])
    b = exp(p.ode[2])

    t_norm = t / T_SCALE
    correction = chain([t_norm], p.neural)[1]

    du[1] = -(Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp
end

function ctntUDEModel(θ, chain::SimpleChain, pat_times::Vector{Float64}=Float64[])
    cude!(du, u, p, t) = ctnt_ude!(du, u, p, t, chain)
    u0 = initial_conditions_from_log_params(θ)
    tspan = (0.0, pat_times[end] + 10.0)
    ode = ODEProblem(cude!, u0, tspan)
    return ctntUDEModel(ode, chain)
end

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    β = exp(p.ode[end])
    a = exp(p.ode[1])
    b = exp(p.ode[2])

    t_norm = t / T_SCALE
    correction = chain([t_norm, β], p.neural)[1]

    du[1] = -(Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp
end

function ctntCUDEModel(θ, chain::SimpleChain, pat_times::Vector{Float64}=Float64[])
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
    u0 = initial_conditions_from_log_params(θ)
    tspan = (0.0, pat_times[end] + 10.0)
    ode = ODEProblem(cude!, u0, tspan)
    return ctntUDEModel(ode, chain)
end

function neural_network_model(depth::Int, width::Int; input_dims::Int=7)
    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(sigmoid, 1))
    return SimpleChain(static(input_dims), layers...)
end

# =============================================================================
# Loss Functions
# =============================================================================

function patient_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params); λ_back=0.0)
    p = ComponentArray(ode=θ, neural=fixed_nn_params)
    u0 = initial_conditions_from_log_params(θ)
    prob = remake(model.problem; u0=u0, p=p)

    sol = solve(
        prob, Tsit5(); p=p, saveat=timepoints,
        abstol=1e-8, reltol=1e-6
    )
    if !successful_retcode(sol)
        return Inf
    end

    plasm = sol[3, :]
    return log_mse_loss(plasm, ctnt_data) + λ_back * initial_condition_penalty(θ)
end

function patient_loss_formula(θ, (problem, timepoints, ctnt_data); λ_back=0.0)
    u0 = initial_conditions_from_log_params(θ)
    prob = remake(problem; u0=u0, p=θ)

    sol = solve(
        prob,
        Tsit5();
        p=θ,
        saveat=timepoints,
        abstol=1e-8, reltol=1e-6
    )

    if !successful_retcode(sol)
        return Inf
    end

    plasm = sol[3, :]
    return log_mse_loss(plasm, ctnt_data) + λ_back * initial_condition_penalty(θ)
end

function par_training_loss(p, (models, training_dataset);
    n_params::Int=5,
    lb_param=nothing, ub_param=nothing,
    κ_bounds::Real=0.05,
    λ_back::Real=0.0
)
    n = length(training_dataset)
    T = eltype(p.ode)
    partial = fill(zero(T), Threads.maxthreadid())

    if Threads.nthreads() == 1
        loss_tot = zero(eltype(p.ode))

        @inbounds @views for i in 1:n
            patient = training_dataset[i]
            model = models[i]

            ode_i = bounded_patient_parameters(p.ode, i, n_params, lb_param, ub_param, κ_bounds)
            θ = ComponentArray(ode=ode_i, neural=p.neural)

            u0_new = initial_conditions_from_log_params(θ.ode)
            prob = remake(model.problem; u0=u0_new, p=θ)

            sol = solve(
                prob, Tsit5();
                saveat=patient.timepoints,
                abstol=1e-8, reltol=1e-6
            )

            if !successful_retcode(sol)
                return oftype(loss_tot, Inf)
            end
            plasm = sol[3, :]
            loss_tot += log_mse_loss(plasm, patient.ctnt_data) + λ_back * initial_condition_penalty(θ.ode)
        end

        return loss_tot / n
    end

    Threads.@threads for i in 1:n
        patient = training_dataset[i]
        model = models[i]

        ode_i = bounded_patient_parameters(p.ode, i, n_params, lb_param, ub_param, κ_bounds)
        θ = ComponentArray(ode=ode_i, neural=p.neural)

        u0_new = initial_conditions_from_log_params(θ.ode)
        prob = remake(model.problem; u0=u0_new, p=θ)

        sol = solve(
            prob, Tsit5();
            p=θ, saveat=patient.timepoints,
            abstol=1e-8, reltol=1e-6
        )
        loss_i = if !successful_retcode(sol)
            oftype(zero(T), Inf)
        else
            plasm = sol[3, :]
            log_mse_loss(plasm, patient.ctnt_data) + λ_back * initial_condition_penalty(θ.ode)
        end

        partial[Threads.threadid()] += loss_i
    end

    return sum(partial) / n
end

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
using DataFrames: DataFrame
using OrdinaryDiffEq: Tsit5
using SciMLBase: ODEProblem, successful_retcode, solve, remake
using SimpleChains: SimpleChain, TurboDense, static
using Statistics: mean, median, quantile, std

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
    summarize_metric_vector(values)

Return mean, standard deviation, median, quartiles, and IQR after dropping
non-finite values.
"""
# Planned use: downstream model-comparison and evaluation scripts.
function summarize_metric_vector(values::AbstractVector{<:Real})
    finite_values = collect(filter(isfinite, values))
    isempty(finite_values) && return (mean=NaN, std=NaN, median=NaN, q1=NaN, q3=NaN, iqr=NaN)
    q1 = quantile(finite_values, 0.25)
    q3 = quantile(finite_values, 0.75)
    return (
        mean=mean(finite_values),
        std=length(finite_values) > 1 ? std(finite_values) : 0.0,
        median=median(finite_values),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
    )
end

"""
    summarize_metrics(losses, smapes, rmsles)

Return grouped summary statistics for loss, sMAPE, and RMSLE vectors.
"""
# Planned use: downstream model-comparison and evaluation scripts.
function summarize_metrics(losses, smapes, rmsles)
    return (
        loss=summarize_metric_vector(losses),
        smape=summarize_metric_vector(smapes),
        rmsle=summarize_metric_vector(rmsles),
    )
end

"""
    empty_natural_parameter_store()

Return mutable vectors for natural-scale ODE/cUDE parameters.
"""
# Used by: src/models.jl, src/plotting.jl. Planned use: downstream diagnostics.
function empty_natural_parameter_store()
    return (a=Float64[], b=Float64[], Cs0=Float64[], Cc0=Float64[], β=Float64[])
end

"""
    append_natural_parameters!(store, log_params; n_params=5)

Append one patient's natural-scale parameters from a log-scale parameter vector.
"""
# Used by: src/models.jl (extract_natural_parameters).
function append_natural_parameters!(store, log_params; n_params::Int=5)
    push!(store.a, exp(log_params[1]))
    push!(store.b, exp(log_params[2]))
    push!(store.Cs0, exp(log_params[3]))
    push!(store.Cc0, exp(log_params[4]))
    if n_params == 5
        push!(store.β, exp(log_params[5]))
    end
    return store
end

parameter_names(UDE::Bool) = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"]
parameter_vectors(store, UDE::Bool) = UDE ? [store.a, store.b, store.Cs0, store.Cc0] :
                                      [store.a, store.b, store.Cs0, store.Cc0, store.β]

"""
    extract_natural_parameters(flat_log_params; n_params=5)

Convert a flat patient-major log-parameter vector into natural-scale parameter
vectors.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl through src/data_io.jl and src/plotting.jl.
function extract_natural_parameters(flat_log_params::AbstractVector; n_params::Int=5)
    length(flat_log_params) % n_params == 0 ||
        error("Parameter vector length $(length(flat_log_params)) is not divisible by n_params=$(n_params).")

    store = empty_natural_parameter_store()
    n_patients = length(flat_log_params) ÷ n_params

    for i in 1:n_patients
        idx1 = n_params * (i - 1) + 1
        idx2 = n_params * i
        append_natural_parameters!(store, flat_log_params[idx1:idx2]; n_params=n_params)
    end

    return store
end

"""
    natural_parameters_dataframe(patient_ids, flat_log_params; n_params=5)

Build the canonical patient-level natural-parameter table used by cUDE
evaluation outputs.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl through src/data_io.jl.
function natural_parameters_dataframe(patient_ids, flat_log_params::AbstractVector; n_params::Int=5)
    n_patients = length(flat_log_params) ÷ n_params
    length(patient_ids) == n_patients ||
        error("Patient ID count $(length(patient_ids)) does not match parameter count $(n_patients).")

    params = extract_natural_parameters(flat_log_params; n_params=n_params)
    df = DataFrame(
        patient_id=patient_ids,
        a=params.a,
        b=params.b,
        Cs0=params.Cs0,
        Cc0=params.Cc0,
    )

    if n_params == 5
        df[!, :beta] = params.β
    end

    return df
end

"""
    median_log_parameter_guess(flat_log_params; n_params=5)

Return the legacy cUDE evaluation parameter guess: the log of the natural-scale
patient-wise median of training ODE parameters.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl.
function median_log_parameter_guess(flat_log_params::AbstractVector; n_params::Int=5)
    length(flat_log_params) % n_params == 0 ||
        error("Parameter vector length $(length(flat_log_params)) is not divisible by n_params=$(n_params).")

    reshaped = permutedims(reshape(flat_log_params, n_params, :))
    natural = exp.(reshaped)
    median_natural = vec(median(natural, dims=1))

    return (
        pguess=log.(median_natural),
        mean_log=vec(mean(reshaped, dims=1)),
        std_natural=vec(std(natural, dims=1)),
        median_natural=median_natural,
        q1_natural=vec([quantile(natural[:, i], 0.25) for i in 1:n_params]),
        q3_natural=vec([quantile(natural[:, i], 0.75) for i in 1:n_params]),
    )
end

"""
    cude_model_summary_row(; model_id, model_idx, nn_depth, nn_width, losses, smapes, rmsles)

Build one row of the canonical cUDE model-summary table.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl.
function cude_model_summary_row(;
    model_id::AbstractString,
    model_idx::Integer,
    nn_depth::Integer,
    nn_width::Integer,
    losses,
    smapes,
    rmsles,
)
    loss = summarize_metric_vector(losses)
    smape_stats = summarize_metric_vector(smapes)
    rmsle_stats = summarize_metric_vector(rmsles)

    return (
        model_id=model_id,
        model_idx=model_idx,
        nn_depth=nn_depth,
        nn_width=nn_width,
        n_patients=length(losses),
        loss_mean=loss.mean,
        loss_std=loss.std,
        loss_median=loss.median,
        loss_q1=loss.q1,
        loss_q3=loss.q3,
        loss_iqr=loss.iqr,
        smape_mean=smape_stats.mean,
        smape_std=smape_stats.std,
        smape_median=smape_stats.median,
        smape_q1=smape_stats.q1,
        smape_q3=smape_stats.q3,
        smape_iqr=smape_stats.iqr,
        rmsle_mean=rmsle_stats.mean,
        rmsle_std=rmsle_stats.std,
        rmsle_median=rmsle_stats.median,
        rmsle_q1=rmsle_stats.q1,
        rmsle_q3=rmsle_stats.q3,
        rmsle_iqr=rmsle_stats.iqr,
    )
end

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

# =============================================================================
# Symbolic Surrogate Formula Model
# =============================================================================

const OFFICIAL_SYMBOLIC_SURROGATE_C1 = 0.0007780399162888297
const OFFICIAL_SYMBOLIC_SURROGATE_C2 = 1.0553531103104006
const OFFICIAL_SYMBOLIC_SURROGATE_C = 1 / OFFICIAL_SYMBOLIC_SURROGATE_C2
const OFFICIAL_SYMBOLIC_SURROGATE_C_BETA = OFFICIAL_SYMBOLIC_SURROGATE_C2 * OFFICIAL_SYMBOLIC_SURROGATE_C1

"""
    symbolic_surrogate_effective_time(beta)

Return the official symbolic surrogate effective-time term for a natural-scale
`beta` parameter.
"""
# Used by: src/models.jl, src/plotting.jl, scripts/04b_evaluate_symbolic_formula.jl.
symbolic_surrogate_effective_time(beta) = beta^2 / OFFICIAL_SYMBOLIC_SURROGATE_C_BETA

"""
    symbolic_surrogate_correction(t_norm, beta)

Evaluate the fixed official symbolic surrogate correction function at
normalized time `t_norm` and natural-scale `beta`.
"""
# Used by: src/models.jl, src/plotting.jl, scripts/04b_evaluate_symbolic_formula.jl.
function symbolic_surrogate_correction(t_norm, beta)
    t_eff = symbolic_surrogate_effective_time(beta)
    t4 = t_norm^4
    return (OFFICIAL_SYMBOLIC_SURROGATE_C * t4) / (t4 + t_eff)
end

"""
    symbolic_formula_ode!(du, u, p, t)

Official symbolic-surrogate ODE used by step 04b. Parameters are log-scale and
ordered as `[a, b, Cs0, Cc0, beta]`.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl through src/fitting.jl and src/diagnostics.jl.
function symbolic_formula_ode!(du, u, p, t)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    a = exp(p[1])
    b = exp(p[2])
    beta = exp(p[5])
    t_norm = t / T_SCALE
    correction = symbolic_surrogate_correction(t_norm, beta)

    du[1] = -(Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp
end

"""
    symbolic_formula_problem(pguess, patient)

Build the patient-specific ODEProblem for the fixed symbolic surrogate formula.
"""
# Used by: src/fitting.jl and src/diagnostics.jl for step 04b.
function symbolic_formula_problem(pguess::AbstractVector, patient::PatientData)
    u0 = initial_conditions_from_log_params(pguess)
    tspan = (0.0, patient.timepoints[end] + 10.0)
    return ODEProblem(symbolic_formula_ode!, u0, tspan)
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

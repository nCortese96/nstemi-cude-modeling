"""
helpers.jl

Reorganized compatibility copy of `ctnt-ude-model.jl`.

This file keeps the current scientific behavior available while making the
future package split visible. Each section states the future `.jl` file where
the contained helpers should eventually move. Public function names are kept
compatible with the current scripts while repeated formulas are centralized in
small reusable helpers.
"""


# =============================================================================
# Module Imports
# Future file: MechanisticAI.jl / package prelude
# =============================================================================
"""
Module Imports

Loads third-party dependencies used by the current helper surface. This section is intentionally broad during the transition phase so the refactored script copies can include one central entrypoint without changing behavior.
"""

using SimpleChains: SimpleChain, TurboDense, static, init_params
using SciMLBase: successful_retcode, ODEProblem, OptimizationSolution, OptimizationFunction, OptimizationProblem
using Random
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using DataFrames: DataFrame, DataFrameRow
using CSV, XLSX
using Statistics
using Dates
using JLD2
using StableRNGs, StatsBase
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches
using OrdinaryDiffEq: Tsit5
using Plots, CairoMakie

using ProgressMeter
using DifferentialEquations
using DiffEqBase
using Base.Threads


# =============================================================================
# Math Utilities And Constants
# Future file: math_utils.jl
# =============================================================================
"""
Math Utilities And Constants

Input: scalars, vectors, and log-scale parameter vectors.
Output: stable transforms, shared constants, metrics, and reusable loss terms.
"""

softplus(x) = log(1 + exp(x))
softplus_stable(x) = log1p(exp(-abs(x))) + max(x, 0)
relu_smooth(x; κ=0.05) = κ * softplus_stable(x / κ)

sigmoid(x) = 1 / (1 + exp(-x))

relu(x) = ifelse(x > 0, x, zero(x))

to_bounds(x, lb, ub; κ=0.05) = x +
                               κ * softplus_stable((lb - x) / κ) -
                               κ * softplus_stable((x - ub) / κ)

const DELTA = 1e-6
T_SCALE = 240.0
const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, T_SCALE];

smape(pred, obs) = 200 * mean(abs.(pred .- obs) ./ (abs.(pred) .+ abs.(obs)))
rmsle(y_true, y_pred) = sqrt(mean((log.(y_pred .+ 1) .- log.(y_true .+ 1)) .^ 2))

"""
    initial_conditions_from_log_params(θ)

Input: log-scale patient parameter vector with `θ[3] = log(Cs0)` and `θ[4] = log(Cc0)`.
Output: ODE initial condition vector `[Cs0, Cc0, 0.0]`.
"""
initial_conditions_from_log_params(θ) = [exp(θ[3]), exp(θ[4]), 0.0]

"""
    log_mse_loss(pred, obs; delta=DELTA)

Input: predicted and observed concentration vectors.
Output: mean squared error on `log(x + delta)`.
"""
log_mse_loss(pred, obs; delta=DELTA) = mean(abs2, log.(pred .+ delta) .- log.(obs .+ delta))

"""
    initial_condition_penalty(θ)

Input: log-scale patient parameter vector.
Output: smooth penalty used when `Cc0 > Cs0`; preserves the existing loss term.
"""
initial_condition_penalty(θ) = relu_smooth(θ[4] - θ[3])^2


# =============================================================================
# Core Types
# Future file: types.jl
# =============================================================================
"""
Core Types

Input: no runtime inputs.
Output: shared domain structs used by model, data, and diagnostic code.
"""

struct ctntUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

struct PatientData
    id::String
    timepoints::Vector{Float64}
    ctnt_data::Vector{Float64}
end


# =============================================================================
# Mechanistic And Neural ODE Models
# Future file: models.jl
# =============================================================================
"""
Mechanistic And Neural ODE Models

Input: log-scale ODE parameters, neural parameters, patient time grids.
Output: ODE right-hand sides, ODEProblem containers, and SimpleChain models.
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

    SimpleChain(static(input_dims), layers...)
end


# =============================================================================
# Patient Data IO
# Future file: data_io.jl
# =============================================================================
"""
Patient Data IO

Input: spreadsheet rows, DataFrames, CSV-compatible paths.
Output: `PatientData` vectors or long-format patient DataFrames.
"""

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

function fromPatientData2DataFrame(patients::Vector{PatientData}; save::Bool=false, save_path::String="pazienti_long.csv")
    ids = String[]
    times = Float64[]
    values = Float64[]

    for p in patients
        n = length(p.timepoints)
        append!(ids, fill(p.id, n))
        append!(times, p.timepoints)
        append!(values, p.ctnt_data)
    end

    df = DataFrame(patient_id=ids, time=times, troponin=values)

    if save
        CSV.write(save_path, df)
    end
    return df
end

function fromDataFrame2PatientData(df::DataFrame)
    patients_reloaded = PatientData[]

    for gdf in groupby(df, :patient_id)
        push!(patients_reloaded, PatientData(
            gdf.patient_id[1],
            gdf.time,
            gdf.troponin
        ))
    end

    return patients_reloaded
end


# =============================================================================
# Sampling And Pipeline Utilities
# Future file: fitting_utils.jl / pipeline_utils.jl
# =============================================================================
"""
Sampling And Pipeline Utilities

Input: run settings, path roots, datasets, bounds, and optimizer settings.
Output: reproducible initializations, resolved paths, preprocessed patients, metric tables, and fitting wrappers.
"""

function sample_initial_neural_parameters(n_initials::Int, chain::SimpleChain, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_parameters(n_patients::Int, n_initials::Int, lhs_lb::AbstractVector{T}, lhs_ub::AbstractVector{T}, rng::AbstractRNG) where T<:Real
    return sample(n_initials, repeat(lhs_lb, n_patients), repeat(lhs_ub, n_patients), LatinHypercubeSample(rng))
end


# =============================================================================
# Multi-Start Optimization
# Future file: optimization_utils.jl
# =============================================================================
"""
Multi-Start Optimization

Input: scalar loss functions, parameter bounds, optimizer settings, and random seeds.
Output: best optimization result plus all per-start results.
"""
module MultiStartOptimizer

using Optimization, OptimizationOptimJL
using Random: AbstractRNG
using StableRNGs: StableRNG
import QuasiMonteCarlo
using QuasiMonteCarlo: LatinHypercubeSample
import Base.Threads
using Base.Threads: @threads
using CSV, Tables
using ProgressMeter

export run_multistart

"""
    run_multistart(loss, N; lower, upper, optimizer, rng, maxiters, maxtime, prescreen, topk)

Input: objective `loss`, number of starts, optional bounds, optimizer, RNG, and run controls.
Output: `(best_solution, all_results)` from a Latin-hypercube multi-start optimization.
"""
function run_multistart(
    loss::Function,
    N::Int;
    lower::Union{Nothing,Vector{<:Real}}=nothing,
    upper::Union{Nothing,Vector{<:Real}}=nothing,
    optimizer=Fminbox(LBFGS()),
    verbose::Bool=true,
    callback::Function=(x, f) -> nothing,
    callback_every::Int=0,
    save_to_csv::Union{Nothing,String}=nothing,
    rng::AbstractRNG=StableRNG(42),
    maxiters::Int=1000,
    maxtime::Float64=80.0,
    prescreen::Bool=false,
    topk::Int=8
)
    lhs_starts = QuasiMonteCarlo.sample(N, lower, upper, LatinHypercubeSample(rng))
    starts = [Vector(lhs_starts[:, i]) for i in 1:N]

    selected_idx = collect(1:N)
    prescreen_losses = fill(NaN, N)

    if prescreen
        prescreen_losses = [loss(s) for s in starts]
        finite_idx = findall(isfinite, prescreen_losses)
        isempty(finite_idx) && error("Prescreen failed: no finite starting point found.")

        keep_n = min(topk, length(finite_idx))
        ord = sortperm(prescreen_losses[finite_idx])
        selected_idx = finite_idx[ord[1:keep_n]]
        starts = starts[selected_idx]
    end

    results = Vector{Any}(undef, length(starts))
    optf = OptimizationFunction((x, _) -> loss(x), AutoForwardDiff())

    done = Threads.Atomic{Int}(0)
    best_loss_atomic = Threads.Atomic{Float64}(Inf)
    progress = Progress(length(starts); desc="MultiStart", dt=0.1, showspeed=true)

    monitor = Threads.@spawn begin
        last = 0
        while true
            d = done[]
            if d > last
                ProgressMeter.update!(progress, d; showvalues=[(:best_loss, best_loss_atomic[])])
                last = d
            end
            d >= length(starts) && break
            sleep(0.1)
        end
    end

    best_lock = ReentrantLock()
    best_loss = Inf
    best_sol = nothing

    @threads for i in eachindex(starts)
        p0 = starts[i]

        optprob = isnothing(lower) ?
                  OptimizationProblem(optf, p0) :
                  OptimizationProblem(optf, p0; lb=lower, ub=upper)

        result = solve(optprob, optimizer; maxiters=maxiters, maxtime=maxtime)
        results[i] = result

        if result !== nothing
            lock(best_lock) do
                if result.minimum < best_loss
                    best_loss = result.minimum
                    best_sol = result
                    best_loss_atomic[] = best_loss
                    callback(result.u, result.minimum)
                end
            end
        end

        done[] += 1
    end

    wait(monitor)
    finish!(progress)

    if verbose
        nfail = count(r -> r === nothing, results)
        println("Finished MultiStart: best_loss=$(best_loss), failed=$(nfail)/$(length(starts))")
    end

    if save_to_csv !== nothing
        rows = [
            (
                start=selected_idx[i],
                prescreen_loss=prescreen ? prescreen_losses[selected_idx[i]] : NaN,
                loss=(r === nothing ? NaN : r.minimum),
                params=(r === nothing ? [] : r.u)
            )
            for (i, r) in enumerate(results)
        ]
        CSV.write(save_to_csv, Tables.columntable(rows))
    end

    best_sol === nothing && error("MultiStart failed: no valid solution found among $(length(starts)) starts.")
    return best_sol, results
end

end


"""
    bounded_patient_parameters(ode_params, patient_index, n_params, lb_param, ub_param, κ_bounds)

Input: concatenated log-scale ODE parameter vector and optional lower/upper bounds.
Output: the parameter slice for one patient, optionally passed through `to_bounds`.
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

function resolve_dataset_config(dataset_id::Integer)
    if dataset_id == 0
        return (dataset_name="MIMIC-IV", dataset_path="MIMIC-IV/NSTEMI_reorganized_skipped.xlsx", column_letter="B")
    elseif dataset_id == 1
        return (dataset_name="UMG", dataset_path="UMG_NSTEMI_Dataset.xlsx", column_letter="A")
    elseif dataset_id == 2
        return (dataset_name="UMG_STEMI", dataset_path="UMG_STEMI_Dataset.xlsx", column_letter="A")
    else
        error("Unsupported dataset_id=$(dataset_id). Use 0 (MIMIC-IV), 1 (UMG), or 2 (UMG_STEMI).")
    end
end

function build_experiment_paths(experiment::AbstractString; root::AbstractString="res")
    experiment_root = joinpath(root, experiment)
    fig_path = joinpath(experiment_root, "figs")
    models_path = joinpath(experiment_root, "models")
    mkpath(fig_path)
    mkpath(models_path)
    return (experiment_root=experiment_root, fig_path=fig_path, models_path=models_path)
end

function load_excel_patients(dataset_path::AbstractString, column_letter::AbstractString; data_root::AbstractString="data")
    file_path = joinpath(data_root, dataset_path)
    sheet_ids = "IDs"
    sheet_times = "times"
    sheet_values = "values"
    ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true))
    timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true))
    troponin_df = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true))
    patients = [row2Patient(ids[i, :], timepoints_df[i, :], troponin_df[i, :]) for i in 1:nrow(ids)]
    return patients, (file_path=file_path, ids=ids, timepoints_df=timepoints_df, troponin_df=troponin_df)
end

function preprocess_patients(patients::Vector{PatientData};
    t_scale::Real=T_SCALE,
    meas_min_number::Int=5,
    min_acq_time_before::Real=12.0,
    min_acq_n_before::Int=1,
    min_acq_time_after::Real=48.0,
    min_acq_n_after::Int=1,
    min_time::Real=72.0,
    max_gap::Union{Nothing,Real}=72.0,
    dup_tol::Real=0.0,
    dup_agg=mean,
    verbose::Bool=false)

    dup_counts = patient_duplicate_time_counts(patients; tol=dup_tol)
    duplicate_summary = collapse_duplicates!(patients, dup_counts; agg=dup_agg, tol=Float64(dup_tol))
    trimmed = trim_time(patients, t_scale)
    anoms = find_anomalies(
        trimmed,
        meas_min_number,
        min_acq_time_before, min_acq_n_before,
        min_acq_time_after, min_acq_n_after,
        min_time;
        max_gap_h=max_gap,
        verbose=verbose,
    )
    cleaned = filter(p -> !haskey(anoms, p.id), trimmed)
    return (patients=cleaned, trimmed=trimmed, anomalies=anoms, duplicate_summary=duplicate_summary)
end

function split_patients(patients::Vector{PatientData}; train_fraction::Real=0.8, seed::Integer=1234)
    shuffled = copy(patients)
    Random.seed!(seed)
    shuffle!(shuffled)
    n_train = Int(round(length(shuffled) * train_fraction))
    return (training=shuffled[1:n_train], test=shuffled[n_train+1:end])
end

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

function summarize_metrics(losses, smapes, rmsles)
    return (
        loss=summarize_metric_vector(losses),
        smape=summarize_metric_vector(smapes),
        rmsle=summarize_metric_vector(rmsles),
    )
end

function save_patient_metrics(path::AbstractString, ids, smapes, rmsles, losses)
    mkpath(dirname(path))
    df = DataFrame(patient_id=ids, smape=smapes, rmsle=rmsles, loss=losses)
    CSV.write(path, df)
    return df
end

function fit_patient_with_multistart(loss_fun::Function, lower::AbstractVector, upper::AbstractVector;
    n_multistart::Int=40,
    rng::AbstractRNG=StableRNG(1234),
    verbose::Bool=false,
    maxiters::Int=1000,
    maxtime::Real=80.0,
    prescreen::Bool=false,
    topk::Int=8)

    return MultiStartOptimizer.run_multistart(
        loss_fun,
        n_multistart;
        lower=lower,
        upper=upper,
        rng=rng,
        verbose=verbose,
        maxiters=maxiters,
        maxtime=Float64(maxtime),
        prescreen=prescreen,
        topk=topk,
    )
end

function predict_patient_curve(problem::ODEProblem, θ; saveat=1.0, abstol=1e-8, reltol=1e-6)
    u0 = initial_conditions_from_log_params(θ)
    prob = remake(problem; u0=u0, p=θ)
    sol = solve(prob, Tsit5(); p=θ, saveat=saveat, abstol=abstol, reltol=reltol)
    successful_retcode(sol) || error("Prediction solve failed with retcode=$(sol.retcode)")
    return sol
end

function plot_patient_fit(sol, patient::PatientData; title::AbstractString="Patient $(patient.id)", plasma_only::Bool=false)
    if plasma_only
        plt = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title=title)
    else
        plt = Plots.plot(sol[1, :]; lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title=title)
        Plots.plot!(plt, sol[2, :]; lw=2, label="Cytosol")
        Plots.plot!(plt, sol[3, :]; lw=2, label="Blood")
    end
    Plots.scatter!(plt, patient.timepoints, patient.ctnt_data; ms=5, label="Observed Data", legend=:best)
    return plt
end


# =============================================================================
# Loss Functions
# Future file: losses.jl
# =============================================================================
"""
Loss Functions

Input: model problems, patient observations, and log-scale parameter vectors.
Output: scalar objective values for patient-level or cohort-level optimization.
"""

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


# =============================================================================
# Patient Preprocessing
# Future file: preprocessing.jl
# =============================================================================
"""
Patient Preprocessing

Input: vectors of `PatientData` and filter thresholds.
Output: cleaned patient vectors, duplicate summaries, and anomaly reports.
"""

"""
    patient_duplicate_time_counts(patients::Vector{PatientData}; tol=0.0)

Return `Dict(id => n_duplicates)` after sorting each patient's timepoints.
"""
function patient_duplicate_time_counts(patients::Vector{PatientData}; tol::Real=0.0)
    counts = Dict{String,Int}()
    for p in patients
        t = sort(p.timepoints)
        length(t) < 2 && continue
        ndup = count(abs.(diff(t)) .<= tol)
        if ndup > 0
            counts[p.id] = ndup
        end
    end
    return counts
end

"""
    collapse_duplicate_times(tp, ct; agg=mean, tol=0.0)

Input: one patient's time and troponin vectors.
Output: sorted vectors with duplicate timepoints collapsed by `agg`.
"""
function collapse_duplicate_times(tp::Vector{Float64}, ct::Vector{Float64};
    agg=mean,
    tol::Float64=0.0
)
    @assert length(tp) == length(ct) "Lunghezze diverse"

    n = length(tp)
    n ≤ 1 && return (copy(tp), copy(ct))

    idx = sortperm(tp)
    t = tp[idx]
    y = ct[idx]

    t2 = Float64[]
    y2 = Float64[]

    i = 1
    while i ≤ n
        j = i
        while j < n && abs(t[j+1] - t[i]) ≤ tol
            j += 1
        end
        push!(t2, t[i])
        push!(y2, agg(@view y[i:j]))
        i = j + 1
    end

    return t2, y2
end

"""
    collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

Input: patient vector and duplicate counts from `patient_duplicate_time_counts`.
Output: `(n_patients_modified, n_points_removed_total)` after in-place collapse.
"""
function collapse_duplicates!(
    patients::Vector{PatientData},
    dup_counts::AbstractDict{String,Int};
    agg=mean,
    tol::Float64=0.0
)
    isempty(dup_counts) && return (0, 0)

    n_modified = 0
    n_removed_total = 0

    for p in patients
        haskey(dup_counts, p.id) || continue

        n_before = length(p.timepoints)
        tp2, ct2 = collapse_duplicate_times(p.timepoints, p.ctnt_data; agg=agg, tol=tol)
        n_after = length(tp2)

        empty!(p.timepoints)
        append!(p.timepoints, tp2)
        empty!(p.ctnt_data)
        append!(p.ctnt_data, ct2)

        n_modified += 1
        n_removed_total += (n_before - n_after)
    end

    return (n_modified, n_removed_total)
end

function trim_time(patients::AbstractVector{PatientData}, time_val)
    filtered_patients = PatientData[]

    for p in patients
        mask = p.timepoints .<= time_val

        if any(mask)
            tp = p.timepoints[mask]
            ct = p.ctnt_data[mask]
            push!(filtered_patients, PatientData(p.id, tp, ct))
        else
            @warn "Patient $(p.id) has no acquisitions ≤ $(time_val) h and will be excluded"
        end
    end

    @info "Kept $(length(filtered_patients)) patients out of $(length(patients))"

    return filtered_patients
end

"""
    count_acq_in_window_sorted(timepoints_sorted, h)

Input: sorted timepoint vector and time threshold.
Output: number of acquisitions with `t <= h`.
"""
@inline function count_acq_in_window_sorted(timepoints_sorted::AbstractVector{<:Real}, h::Real)
    return searchsortedlast(timepoints_sorted, h)
end

"""
    max_gap_sorted(timepoints_sorted)::Float64

Input: sorted timepoint vector.
Output: maximum consecutive time gap.
"""
@inline function max_gap_sorted(timepoints_sorted::AbstractVector{<:Real})
    n = length(timepoints_sorted)
    if n < 2
        return 0.0
    end
    gmax = 0.0
    @inbounds for i in 2:n
        g = timepoints_sorted[i] - timepoints_sorted[i-1]
        if g > gmax
            gmax = g
        end
    end
    return gmax
end

function find_anomalies(
    patients::Vector{PatientData},
    meas_min_number::Int=1,
    min_acq_time_before::Real=300.0,
    min_acq_n_before::Int=1,
    min_acq_time_after::Real=0.0,
    min_acq_n_after::Int=0,
    min_time::Real=0.0;
    max_gap_h::Union{Nothing,Real}=nothing,
    verbose::Bool=true
)
    anomalies = Dict{String,Vector{String}}()

    for p in patients
        issues = String[]

        tp = p.timepoints
        ct = p.ctnt_data
        n_tp = length(tp)
        n_ct = length(ct)

        if n_ct == 0
            push!(issues, "empty ctnt data")
        end
        if n_tp == 0
            push!(issues, "empty timepoints data")
        end
        if n_tp != n_ct
            push!(issues, "time ctnt mismatch")
        end

        if n_tp > 0
            tmax = maximum(tp)
            if tmax < min_time
                push!(issues, "less then $(min_time)h max time")
            end

            if any(<(0), tp)
                push!(issues, "negative time")
            end
            if n_ct > 0 && any(<(0), ct)
                push!(issues, "negative ctnt")
            end

            if n_tp < meas_min_number
                push!(issues, "n acquisizion < $meas_min_number")
            end

            if issorted(tp; lt=≤)
                n_before = count_acq_in_window_sorted(tp, min_acq_time_before)
                if n_before < min_acq_n_before
                    push!(issues, "less then $min_acq_n_before measurements in the first $(min_acq_time_before)h")
                end

                n_after = length(tp) - count_acq_in_window_sorted(tp, min_acq_time_after)
                if n_after < min_acq_n_after
                    push!(issues, "less then $min_acq_n_after measurements in the last $(min_acq_time_after)h")
                end

                if max_gap_h !== nothing
                    gmax = max_gap_sorted(tp)
                    if gmax > max_gap_h
                        push!(issues, "max gap > $(max_gap_h)h (max=$(round(gmax, digits=3))h)")
                    end
                end
            else
                push!(issues, "times not sorted")
            end
        end

        if !isempty(issues)
            anomalies[p.id] = issues
        end
    end
    if verbose
        if isempty(anomalies)
            println("No anomalies found")
        else
            for (id, issues) in anomalies
                @warn "Patient " * id * ": " * join(issues, ", ")
            end
        end
    end

    return anomalies
end


# =============================================================================
# Dataset Reporting And Preprocessing Pipeline
# Future file: preprocessing_reports.jl
# =============================================================================
"""
Dataset Reporting And Preprocessing Pipeline

Input: dataset configuration, raw Excel files, preprocessing thresholds, and output folders.
Output: text reports, all-eligible ID CSVs, JLD2 patient sets, and returned patient collections.
"""

"""
    DatasetReporter

Input: a dataset name and output directory through the constructor.
Output: mutable report state used by `report_step!` and `finalize_report`.
"""
mutable struct DatasetReporter
    dataset_name::String
    report_path::String
    io::IOStream
    step_count::Int
end

"""
    DatasetReporter(dataset_name; report_dir="res")

Input: dataset name and report directory.
Output: initialized report writer with a text header already written.
"""
function DatasetReporter(dataset_name::String; report_dir::String="res")
    mkpath(report_dir)
    ts = Dates.format(now(), "yyyymmdd_HHMMss")
    fname = "dataset_report_$(dataset_name)_$(ts).txt"
    report_path = joinpath(report_dir, fname)
    io = open(report_path, "w")

    header = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           DATASET PREPROCESSING REPORT                     ║
    ║  Dataset:  $(rpad(dataset_name, 46))║
    ║  Date:     $(rpad(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), 46))║
    ╚══════════════════════════════════════════════════════════════╝
    """

    _dual_print(io, header)
    return DatasetReporter(dataset_name, report_path, io, 0)
end

"""
    _summary_stats(v)

Input: numeric vector.
Output: NamedTuple with n, mean, std, median, quartiles, min, and max after dropping NaN.
"""
function _summary_stats(v::AbstractVector{<:Real})
    clean = filter(!isnan, v)
    n = length(clean)
    n == 0 && return (n=0, mean=NaN, std=NaN, median=NaN,
        q1=NaN, q3=NaN, min=NaN, max=NaN)
    return (
        n=n,
        mean=mean(clean),
        std=length(clean) > 1 ? std(clean) : 0.0,
        median=median(clean),
        q1=quantile(clean, 0.25),
        q3=quantile(clean, 0.75),
        min=minimum(clean),
        max=maximum(clean),
    )
end

"""
    _dual_print(io, text)

Input: open IO stream and text.
Output: writes the same text to the report file and stdout.
"""
function _dual_print(io::IOStream, text::String)
    print(io, text)
    print(stdout, text)
    flush(io)
end

"""
    _fmt_stat(s; digits=2)

Input: summary-stat NamedTuple.
Output: compact median/IQR/min-max string for reports.
"""
function _fmt_stat(s::NamedTuple; digits::Int=2)
    if isnan(s.median)
        return "N/A"
    end
    return string(
        round(s.median; digits),
        " [", round(s.q1; digits), "–", round(s.q3; digits), "]",
        " (", round(s.min; digits), "–", round(s.max; digits), ")"
    )
end

"""
    report_step!(reporter, step_name, patients; extra_info="")

Input: report state, step label, patient collection, and optional text.
Output: appends preprocessing statistics to both stdout and the report file.
"""
function report_step!(reporter::DatasetReporter,
    step_name::String,
    patients::AbstractVector{PatientData};
    extra_info::String="")

    reporter.step_count += 1
    step_n = reporter.step_count
    n_patients = length(patients)

    ctnt_counts = [count(!isnan, p.ctnt_data) for p in patients]
    ctnt_counts_stats = _summary_stats(Float64.(ctnt_counts))
    total_ctnt = sum(ctnt_counts)

    tp_counts = [count(!isnan, p.timepoints) for p in patients]
    total_tp = sum(tp_counts)

    first_times = Float64[]
    last_times = Float64[]
    obs_spans = Float64[]
    max_gaps = Float64[]

    for p in patients
        valid_tp = filter(!isnan, p.timepoints)
        isempty(valid_tp) && continue

        ft = first(valid_tp)
        lt = last(valid_tp)
        push!(first_times, ft)
        push!(last_times, lt)
        push!(obs_spans, lt - ft)

        if length(valid_tp) >= 2
            push!(max_gaps, maximum(diff(valid_tp)))
        end
    end

    first_stats = _summary_stats(first_times)
    last_stats = _summary_stats(last_times)
    span_stats = _summary_stats(obs_spans)
    gap_stats = _summary_stats(max_gaps)

    sep = "─"^62
    block = """

    $(sep)
      STEP $(step_n): $(step_name)
    $(sep)
      Numero pazienti:                     $(n_patients)
      Acquisizioni cTnT totali:            $(total_ctnt)
      Timepoints totali:                   $(total_tp)
    $(extra_info == "" ? "" : "  Info aggiuntive:                     $(extra_info)\n")
      Acquisizioni per paziente:           $(_fmt_stat(ctnt_counts_stats))
      Primo timepoint (h):                 $(_fmt_stat(first_stats))
      Ultimo timepoint (h):                $(_fmt_stat(last_stats))
      Observation span (h):                $(_fmt_stat(span_stats))
      Max gap consecutivo (h):             $(_fmt_stat(gap_stats))
    $(sep)

    """

    _dual_print(reporter.io, block)
    return nothing
end

"""
    finalize_report(reporter)

Input: active dataset reporter.
Output: closes the report file and returns its path.
"""
function finalize_report(reporter::DatasetReporter)
    footer = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  REPORT COMPLETATO                                         ║
    ║  Step totali registrati: $(lpad(reporter.step_count, 3))                                ║
    ║  File salvato in: $(rpad(reporter.report_path, 39))║
    ╚══════════════════════════════════════════════════════════════╝
    """
    _dual_print(reporter.io, footer)
    close(reporter.io)
    @info "Report chiuso: $(reporter.report_path)"
    return reporter.report_path
end

"""
    preprocessing_filter_description(...)

Input: filtering thresholds and removed-patient count.
Output: one-line description used in preprocessing reports.
"""
function preprocessing_filter_description(;
    meas_min_number::Int,
    min_acq_time_before::Real,
    min_acq_n_before::Int,
    min_acq_time_after::Real,
    min_acq_n_after::Int,
    min_time::Real,
    max_gap::Real,
    removed_count::Int)

    return string(
        "Filtro: min_meas=$(meas_min_number), ",
        "before=$(min_acq_n_before)×$(min_acq_time_before)h, ",
        "after=$(min_acq_n_after)×$(min_acq_time_after)h, ",
        "min_span=$(min_time)h, max_gap=$(max_gap)h. ",
        "Rimossi: $(removed_count)"
    )
end

"""
    save_preprocessed_dataset!(dataset_name, cleaned_patients, report_dir; train_fraction=0.8, seed=1234, reporter=nothing)

Input: cleaned patient collection and output settings.
Output: writes JLD2 train/test artifacts and returns the saved patient sets.
"""
function save_preprocessed_dataset!(
    dataset_name::String,
    cleaned_patients::Vector{PatientData},
    report_dir::String;
    train_fraction::Real=0.8,
    seed::Integer=1234,
    reporter::Union{Nothing,DatasetReporter}=nothing
)
    if dataset_name == "MIMIC-IV"
        Random.seed!(seed)
        shuffle!(cleaned_patients)
        n_train = Int(round(length(cleaned_patients) * train_fraction))
        training_dataset = cleaned_patients[1:n_train]
        test_dataset = cleaned_patients[n_train+1:end]

        if reporter !== nothing
            report_step!(reporter, "Training split ($(round(Int, train_fraction * 100))%)", training_dataset;
                extra_info="Split: $(n_train) training, $(length(test_dataset)) test")
            report_step!(reporter, "Test split ($(round(Int, (1 - train_fraction) * 100))%)", test_dataset)
        end

        @save "$(report_dir)/$(dataset_name)_trainingset.jld2" training_dataset
        @save "$(report_dir)/$(dataset_name)_testset.jld2" test_dataset

        return (cleaned_patients=cleaned_patients, training=training_dataset, test=test_dataset)
    end

    test_dataset = cleaned_patients
    @save "$(report_dir)/$(dataset_name)_testset.jld2" test_dataset cleaned_patients
    return (cleaned_patients=cleaned_patients, test=test_dataset)
end

"""
    run_dataset_report(; dataset_name, dataset_path, column_letter, ...)

Input: dataset identity, Excel location, preprocessing thresholds, and report/output directory.
Output: cleaned patients, saved split artifacts, and report path.
"""
function run_dataset_report(;
    dataset_name::String,
    dataset_path::String,
    column_letter::String,
    T_SCALE::Float64,
    meas_min_number::Int,
    min_acq_time_before::Float64,
    min_acq_n_before::Int,
    min_acq_time_after::Float64,
    min_acq_n_after::Int,
    min_time::Float64,
    max_gap::Float64,
    report_dir::String="res",
    train_fraction::Real=0.8,
    split_seed::Integer=1234,
    data_root::String="data"
)
    reporter = DatasetReporter(dataset_name; report_dir=report_dir)

    patients, load_info = load_excel_patients(dataset_path, column_letter; data_root=data_root)

    report_step!(reporter, "Raw dataset loaded", patients;
        extra_info="File: $(load_info.file_path)")

    dup_counts = patient_duplicate_time_counts(patients; tol=0.0)
    nmod, nrm = collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

    report_step!(reporter, "Duplicate timepoints collapsed", patients;
        extra_info="Pazienti modificati: $(nmod), punti rimossi: $(nrm)")

    trimmed_patients = trim_time(patients, T_SCALE)

    report_step!(reporter, "Trimmed to $(T_SCALE)h", trimmed_patients)

    anoms = find_anomalies(trimmed_patients,
        meas_min_number,
        min_acq_time_before, min_acq_n_before,
        min_acq_time_after, min_acq_n_after,
        min_time;
        max_gap_h=max_gap,
        verbose=false
    )

    cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_patients)

    filter_info = preprocessing_filter_description(
        meas_min_number=meas_min_number,
        min_acq_time_before=min_acq_time_before,
        min_acq_n_before=min_acq_n_before,
        min_acq_time_after=min_acq_time_after,
        min_acq_n_after=min_acq_n_after,
        min_time=min_time,
        max_gap=max_gap,
        removed_count=length(anoms),
    )

    report_step!(reporter, "Anomaly filtering (All Eligible)", cleaned_patients;
        extra_info=filter_info)

    df_ae_ids = DataFrame(patient=[p.id for p in cleaned_patients])
    CSV.write(joinpath(report_dir, "ids_all_eligible_$(dataset_name).csv"), df_ae_ids)

    saved_sets = save_preprocessed_dataset!(
        dataset_name,
        cleaned_patients,
        report_dir;
        train_fraction=train_fraction,
        seed=split_seed,
        reporter=reporter,
    )

    report_path = finalize_report(reporter)
    return merge(saved_sets, (report_path=report_path,))
end


# =============================================================================
# Dataset Plotting
# Future file: plotting.jl
# =============================================================================
"""
Dataset Plotting

Input: patient collections and model solutions.
Output: plot objects plus summary vectors needed by scripts.
"""

function plot_distribution(patients::AbstractVector{PatientData})
    all_times = vcat([p.timepoints for p in patients]...)
    all_ctnt = vcat([p.ctnt_data for p in patients]...)

    t_min = minimum(all_times)
    t_max = maximum(all_times)

    @info "Tempo  min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h"

    c_min = minimum(all_ctnt)
    c_max = maximum(all_ctnt)

    @info "CTnT   min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL"

    all_ctnt_log = log.(all_ctnt .+ DELTA)

    plt1 = Plots.histogram(all_times;
        bins=40,
        xlabel="Time (h)",
        ylabel="#",
        title="Time-points distribution",
        legend=false)

    plt2 = Plots.histogram(all_ctnt_log;
        bins=40,
        xlabel="CTnT (ng/mL)",
        ylabel="#",
        title="Troponin log distribution",
        legend=false)

    dist = Plots.plot(plt1, plt2; layout=(2, 1), size=(900, 600))

    return all_times, all_ctnt, t_min, t_max, c_min, c_max, dist
end

function scutter_patients(patients::AbstractVector{PatientData})
    plt = Plots.plot(; xlabel="Time (h)", ylabel="cTnT (ng/mL)",
        title="All patients: troponin vs time", legend=false)

    for p in patients
        Plots.scatter!(plt, p.timepoints, p.ctnt_data; lw=1, alpha=0.5)
    end
    return plt
end


# =============================================================================
# Residual Diagnostics And Parameter Summaries
# Future file: diagnostics.jl
# =============================================================================
"""
Residual Diagnostics And Parameter Summaries

Input: fitted parameters, patients, and prediction problems.
Output: residual DataFrames, diagnostic figures, metrics, and natural-scale parameter summaries.
"""

function patient_dims(patients::AbstractVector{PatientData})
    counts = [length(p.ctnt_data) for p in patients]

    i_max = argmax(counts)
    i_min = argmin(counts)

    p_max = patients[i_max]
    p_min = patients[i_min]

    println("Patient with MAX acquisitions: ", p_max.id,
        " -> ", counts[i_max], " samples; ctnt_data = ", length(p_max.ctnt_data))

    println("Patient with MIN acquisitions: ", p_min.id,
        " -> ", counts[i_min], " samples; ctnt_data = ", length(p_min.ctnt_data))

    return (length(p_min.ctnt_data), length(p_max.ctnt_data))
end

"""
    log_residuals(y, yhat; ϵ=DELTA)

Input: observed and predicted concentration vectors.
Output: residual vector `log(y + ϵ) - log(yhat + ϵ)`.
"""
function log_residuals(y, yhat; ϵ=DELTA)
    return log.(y .+ ϵ) .- log.(yhat .+ ϵ)
end

"""
    display_patient_prediction(problem, patient, p; abstol=1e-8, reltol=1e-6)

Input: an ODE problem, patient data, and parameter object.
Output: displayed CairoMakie figure for the plasma trajectory and observed points.
"""
function display_patient_prediction(problem::ODEProblem, patient::PatientData, p; abstol=1e-8, reltol=1e-6)
    sol = solve(problem, Tsit5(); p=p, abstol=abstol, reltol=reltol)

    fig = CairoMakie.Figure(size=(800, 500))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="Time (h)",
        ylabel="cTnT",
        title="cTnT simulation patient $(patient.id)")

    CairoMakie.lines!(ax, sol.t, sol[3, :], color=:blue, label="cTnT simulation")
    CairoMakie.scatter!(ax, patient.timepoints, patient.ctnt_data,
        color=:red, label="Data", markersize=8)

    axislegend(ax, position=:rt)
    display(fig)
    return fig
end

"""
    residual_tuple_from_prediction(pred, patient)

Input: solved prediction and patient observations.
Output: `(y, yhat, res)` using the diagnostics residual convention.
"""
function residual_tuple_from_prediction(pred, patient::PatientData)
    yhat = vec(pred[3, :])
    y = patient.ctnt_data
    res = log_residuals(y, yhat)
    return y, yhat, res
end

function compute_residuals_patient(model::ctntUDEModel, patient::PatientData, p::ComponentArray;
    plotting::Bool=false,
    abstol=1e-8,
    reltol=1e-6)

    pred = solve(model.problem, Tsit5(); p=p, saveat=patient.timepoints, abstol=abstol, reltol=reltol)

    if plotting
        display_patient_prediction(model.problem, patient, p; abstol=abstol, reltol=reltol)
    end

    return residual_tuple_from_prediction(pred, patient)
end

function compute_residuals_patient(problem::ODEProblem, patient::PatientData, p::Vector{Float64};
    plotting::Bool=false,
    abstol=1e-8,
    reltol=1e-6)

    pred = solve(problem, Tsit5(); p=p, saveat=patient.timepoints, abstol=abstol, reltol=reltol)

    if plotting
        display_patient_prediction(problem, patient, p; abstol=abstol, reltol=reltol)
    end

    return residual_tuple_from_prediction(pred, patient)
end

function add_time_bins!(df::DataFrame, edges::Vector{Float64})
    nb = length(edges) - 1
    b = similar(df.t, Int)
    for i in eachindex(df.t)
        k = searchsortedlast(edges, df.t[i])
        b[i] = clamp(k, 1, nb)
    end
    df.bin = b
    df.bin_center = [0.5 * (edges[k] + edges[k+1]) for k in b]
    return df
end

function bin_summary(df::DataFrame)
    g = groupby(df, :bin)
    centers = Float64[]
    q1 = Float64[]
    med = Float64[]
    q3 = Float64[]
    n = Int[]

    for sub in g
        push!(centers, first(sub.bin_center))
        push!(q1, quantile(sub.res, 0.25))
        push!(med, quantile(sub.res, 0.50))
        push!(q3, quantile(sub.res, 0.75))
        push!(n, nrow(sub))
    end

    ord = sortperm(centers)
    return (centers=centers[ord], q1=q1[ord], med=med[ord], q3=q3[ord], n=n[ord])
end

function plot_residuals_vs_time(df::DataFrame, edges::Vector{Float64}; title="Residuals vs time", TMAX=T_SCALE, nmin::Int=1)
    s = bin_summary(df)
    for i in eachindex(s.centers)
        @info "bin_center=$(s.centers[i])  n=$(s.n[i])"
    end

    med_mask = Float64[]
    q1_mask = Float64[]
    q3_mask = Float64[]
    for i in eachindex(s.n)
        if s.n[i] ≥ nmin
            push!(med_mask, s.med[i])
            push!(q1_mask, s.q1[i])
            push!(q3_mask, s.q3[i])
        else
            push!(med_mask, NaN)
            push!(q1_mask, NaN)
            push!(q3_mask, NaN)
        end
    end

    fig = CairoMakie.Figure(size=(950, 450))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="Time (h)",
        ylabel="log residual log(y) - log(ŷ)",
        title=title)

    CairoMakie.scatter!(ax, df.t, df.res; markersize=4, color=(:black, 0.25))
    CairoMakie.lines!(ax, s.centers, med_mask; linewidth=2, label="Median (per bin)", color=:blue)
    CairoMakie.band!(ax, s.centers, q1_mask, q3_mask; color=(:gray, 0.2), label="IQR (Q1-Q3)")

    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6), label="Zero line (horizontal)")

    CairoMakie.xlims!(ax, 0, TMAX)

    CairoMakie.vlines!(ax, edges[2:end-1];
        color=(:black, 0.35),
        linewidth=1.5,
        linestyle=:dash,
        label="Bins (vertical)")

    for i in eachindex(s.centers)
        x_rel = clamp(s.centers[i] / TMAX, 0.0, 1.0)
        CairoMakie.text!(ax, x_rel, 0.96;
            text="n=$(s.n[i])",
            space=:relative,
            align=(:center, :top),
        rotation=pi / 4,
        fontsize=12,
        color=(:black, 0.8))
    end

    CairoMakie.text!(ax, 0.99, 0.02;
        text="n = number of points in bin",
        space=:relative, align=(:right, :bottom),
        fontsize=12, color=(:black, 0.7))

    CairoMakie.Legend(fig[1, 2], ax;)

    fig
end

function plot_residuals_vs_fitted(df::DataFrame; title="Residuals vs fitted", ϵ=1e-10)
    fig = CairoMakie.Figure(size=(550, 450))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="log predicted ŷ ",
        ylabel="log residual log(y) - log(ŷ)",
        title=title)

    CairoMakie.scatter!(ax, log.(df.yhat .+ ϵ), df.res; markersize=5, color=(:black, 0.25), label="Residuals")
    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6), label="Zero line")

    fig
end

"""
    empty_natural_parameter_store()

Input: none.
Output: mutable vectors for natural-scale ODE/cUDE parameters.
"""
function empty_natural_parameter_store()
    return (a=Float64[], b=Float64[], Cs0=Float64[], Cc0=Float64[], β=Float64[])
end

"""
    append_natural_parameters!(store, log_params; n_params=5)

Input: log-scale parameter vector for one patient.
Output: mutates `store` by appending natural-scale parameters.
"""
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
    parameter_distribution_figure(params, par_names; title, show_outliers=true)

Input: parameter vectors and their display names.
Output: CairoMakie boxplot figure with the existing visual style.
"""
function parameter_distribution_figure(params, par_names; title::AbstractString, show_outliers::Bool=true)
    x = vcat([fill(1, length(params[1]))]...)

    fig = Figure(size=(1400, 700))
    Label(
        fig[0, 1:length(par_names)],
        title;
        fontsize=22,
        tellwidth=false
    )

    axes = []
    @showprogress desc = "Generating axes..." for p_name in par_names
        ax = Axis(fig[1, length(axes) + 1],
            title=p_name,
            xticklabelsvisible=false,
            xticksvisible=false,
        )
        push!(axes, ax)
    end

    my_colors = [:skyblue, :orange, :lightgreen, :pink, :violet]
    @showprogress desc = "Generating boxplots..." for (i, (ax, values)) in enumerate(zip(axes, params))
        current_color = my_colors[mod1(i, length(my_colors))]
        CairoMakie.boxplot!(
            ax, x, values;
            color=current_color,
            whiskerwidth=0.3,
            strokewidth=0.3,
            show_outliers=show_outliers
        )
    end

    return fig
end

function compute_plot_residuals(patients::Vector{PatientData}, ode_params_val::Vector{Float64}, best_nn::Vector{Float64},
    chain::SimpleChain; EDGES::Vector{Float64}=EDGES, N_params::Int=5, UDE::Bool=false,
    hi::Bool=false, show_plots::Bool=false, figsave_path::String="./", modelssave_path::String="./",
    dataset_label::String="")

    out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])
    smape_out = DataFrame(id=String[], smape=Float64[])

    param_store = empty_natural_parameter_store()

    @showprogress desc = "Computing residuals..." for (i, patient) in enumerate(patients)

        idx1 = N_params * (i - 1) + 1
        idx2 = N_params * i
        ode_p = ode_params_val[idx1:idx2]
        p = ComponentArray(ode=ode_p, neural=best_nn)

        append_natural_parameters!(param_store, p.ode; n_params=N_params)

        model = UDE ? ctntUDEModel(p, chain, patient.timepoints) : ctntCUDEModel(p, chain, patient.timepoints)

        y, yhat, res = compute_residuals_patient(model, patient, p; plotting=show_plots)

        append!(out, DataFrame(id=fill(patient.id, length(y)),
            t=patient.timepoints,
            y=y,
            yhat=yhat,
            res=res))

        smape_val = smape(yhat, y)
        append!(smape_out, DataFrame(id=patient.id, smape=smape_val))
    end

    params = parameter_vectors(param_store, UDE)

    @info "Saving residuals data to CSV and plotting"

    add_time_bins!(out, EDGES)

    fig_vs_time = plot_residuals_vs_time(
        out,
        EDGES;
        title="Residuals vs time - $dataset_label", TMAX=T_SCALE, nmin=1)

    fig_vs_fitted = plot_residuals_vs_fitted(
        out;
        title="Residuals vs fitted - $dataset_label")

    @info "Boxplotting params"

    par_names = parameter_names(UDE)
    f = parameter_distribution_figure(params, par_names;
        title="Parameter distributions — $dataset_label",
        show_outliers=true)

    if hi
        CSV.write("$(modelssave_path)/residuals_$(dataset_label)_hi.csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label)_hi.png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label)_hi.png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label)_hi.png", f)
    else
        CSV.write("$(modelssave_path)/residuals_$(dataset_label).csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label).png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label).png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label).png", f)
    end

    if show_plots
        display(fig_vs_time)
        display(fig_vs_fitted)
        display(f)
    end

    return out, smape_out
end

function compute_plot_residuals(patients::Vector{PatientData}, ode_params_val, ode_func::Function;
    EDGES::Vector{Float64}=EDGES, N_params::Int=5, UDE::Bool=false,
    hi::Bool=false, show_plots::Bool=false, figsave_path::String="./",
    modelssave_path::String="./", dataset_label::String="")

    out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])
    smape_out = DataFrame(id=String[], smape=Float64[])

    param_store = empty_natural_parameter_store()

    @showprogress desc = "Computing residuals..." for (i, patient) in enumerate(patients)

        p = vcat(ode_params_val[i]...)

        append_natural_parameters!(param_store, p; n_params=N_params)
        u0_init = initial_conditions_from_log_params(p)

        tspan = (0.0, patient.timepoints[end] + 10.0)

        problem = ODEProblem(ode_func, u0_init, tspan)

        y, yhat, res = compute_residuals_patient(problem, patient, p; plotting=show_plots)

        append!(out, DataFrame(id=fill(patient.id, length(y)),
            t=patient.timepoints,
            y=y,
            yhat=yhat,
            res=res))

        smape_val = smape(yhat, y)
        append!(smape_out, DataFrame(id=patient.id, smape=smape_val))
    end

    params = parameter_vectors(param_store, UDE)

    @info "Saving residuals data to CSV and plotting"

    add_time_bins!(out, EDGES)

    fig_vs_time = plot_residuals_vs_time(
        out,
        EDGES;
        title="Residuals vs time - $dataset_label", TMAX=T_SCALE, nmin=1)

    fig_vs_fitted = plot_residuals_vs_fitted(out; title="Residuals vs fitted - $dataset_label")

    @info "Boxplotting params"

    par_names = parameter_names(UDE)
    f = parameter_distribution_figure(params, par_names;
        title="Parameter distributions — $dataset_label",
        show_outliers=true)

    if hi
        CSV.write("$(modelssave_path)/residuals_$(dataset_label)_hi.csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label)_hi.png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label)_hi.png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label)_hi.png", f)
    else
        CSV.write("$(modelssave_path)/residuals_$(dataset_label).csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label).png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label).png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label).png", f)
    end

    if show_plots
        display(fig_vs_time)
        display(fig_vs_fitted)
        display(f)
    end

    return out, smape_out
end

function params_extraction(
    patients::Vector{PatientData},
    ode_params_val::Vector{Float64};
    UDE::Bool=false,
    N_params::Int=5,
    data_label::String="",
    dataset::String="",
    figsave_path::String="",
    show_outliers::Bool=false,
    savefigure::Bool=false
)

    param_store = empty_natural_parameter_store()

    @showprogress desc = "$(data_label) prams extraction..." for i in eachindex(patients)

        idx1 = N_params * (i - 1) + 1
        idx2 = N_params * i
        local ode_p = ode_params_val[idx1:idx2]
        append_natural_parameters!(param_store, ode_p; n_params=N_params)
    end

    a, b, Cs0, Cc0, β = param_store.a, param_store.b, param_store.Cs0, param_store.Cc0, param_store.β

    @info "Average, STD in $data_label param a: $(mean(a)) std: $(std(a))"
    @info "Average, STD in $data_label param b: $(mean(b)) std: $(std(b))"
    @info "Average, STD in $data_label param Cs0: $(mean(Cs0)) std: $(std(Cs0))"
    @info "Average, STD in $data_label param Cc0: $(mean(Cc0)) std: $(std(Cc0))"
    if N_params == 5
        @info "Average, STD in $data_label param β: $(mean(β)) std: $(std(β))"
    end

    @info "Median [Q1-Q3] in $data_label param a: $(median(a)) [$(quantile(a, 0.25)) - $(quantile(a, 0.75))]"
    @info "Median [Q1-Q3] in $data_label param b: $(median(b)) [$(quantile(b, 0.25)) - $(quantile(b, 0.75))]"
    @info "Median [Q1-Q3] in $data_label param Cs0: $(median(Cs0)) [$(quantile(Cs0, 0.25)) - $(quantile(Cs0, 0.75))]"
    @info "Median [Q1-Q3] in $data_label param Cc0: $(median(Cc0)) [$(quantile(Cc0, 0.25)) - $(quantile(Cc0, 0.75))]"
    if N_params == 5
        @info "Median [Q1-Q3] in $data_label param β: $(median(β)) [$(quantile(β, 0.25)) - $(quantile(β, 0.75))]"
    end

    params = parameter_vectors(param_store, UDE)

    @info "Boxplotting params"

    par_names = parameter_names(UDE)
    f = parameter_distribution_figure(params, par_names;
        title="Parameter distributions $data_label — $dataset dataset",
        show_outliers=show_outliers)

    if savefigure
        save("$(figsave_path)/$(data_label)_params_distribution_$(dataset).svg", f)
        @info "Figure saved at: $(figsave_path)/$(data_label)_params_distribution_$(dataset).svg"
    end
    return a, b, Cs0, Cc0, β, f
end

"""
    compute_residuals_long_unified(patients, params_df;
        model_type=:ode, chain=nothing, nn_params=nothing,
        tpad=10.0, abstol=1e-8, reltol=1e-6)

Unified residual computation for both ODE and cUDE models.

Returns `(residuals_df, metrics_df, params_extracted)`:
  - `residuals_df`: long-format DataFrame(id, t, y, yhat, res)
  - `metrics_df`: DataFrame(id, smape_val, rmsle_val)
  - `params_extracted`: DataFrame(patient_id, a, b, Cs0, Cc0, p5) where p5 is Td (ODE) or beta (cUDE)

Arguments:
  - `patients`: Dict{String, PatientData} or Vector{PatientData}
  - `params_df`: DataFrame with patient-specific parameters
      * For `:ode`: columns :patient, :p1…:p5 (log-scale)
      * For `:cude`: columns :patient_id, :a, :b, :Cs0, :Cc0, :beta (natural-scale)
  - `model_type`: `:ode` or `:cude`
  - `chain`: SimpleChain (required for :cude)
  - `nn_params`: Vector{Float64} of NN weights (required for :cude)
"""
function compute_residuals_long_unified(
    patients,
    params_df::DataFrame;
    model_type::Symbol=:ode,
    chain=nothing,
    nn_params=nothing,
    tpad::Real=10.0,
    abstol=1e-12,
    reltol=1e-10
)
    @assert model_type in (:ode, :cude) "model_type must be :ode or :cude"
    if model_type == :cude
        @assert chain !== nothing "chain required for cUDE"
        @assert nn_params !== nothing "nn_params required for cUDE"
    end

    patient_lookup = if patients isa Dict
        patients
    else
        Dict(p.id => p for p in patients)
    end

    residuals_out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])
    metrics_out = DataFrame(id=String[], smape_val=Float64[], rmsle_val=Float64[])
    params_out = DataFrame(patient_id=String[], a=Float64[], b=Float64[],
        Cs0=Float64[], Cc0=Float64[], p5=Float64[])

    id_col = model_type == :ode ? :patient : :patient_id

    for row in eachrow(params_df)
        pid = String(row[id_col])
        haskey(patient_lookup, pid) || continue
        p_data = patient_lookup[pid]

        tmax = maximum(p_data.timepoints) + tpad

        if model_type == :ode
            p_log = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]
            u0 = initial_conditions_from_log_params(p_log)
            prob = ODEProblem(troponin_ode!, u0, (0.0, tmax), p_log)
            pred = solve(prob, Tsit5(); saveat=p_data.timepoints, abstol=abstol, reltol=reltol)

            push!(params_out, (pid, exp(p_log[1]), exp(p_log[2]),
                exp(p_log[3]), exp(p_log[4]), exp(p_log[5])))
        else
            p_nat = Float64[row.a, row.b, row.Cs0, row.Cc0, row.beta]
            θ_log = log.(p_nat)
            u0 = initial_conditions_from_log_params(θ_log)

            cude_f!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
            prob = ODEProblem(cude_f!, u0, (0.0, tmax))
            p_full = ComponentArray(ode=θ_log, neural=nn_params)
            pred = solve(prob, Tsit5(); p=p_full, saveat=p_data.timepoints, abstol=abstol, reltol=reltol)

            push!(params_out, (pid, p_nat[1], p_nat[2], p_nat[3], p_nat[4], p_nat[5]))
        end

        if !successful_retcode(pred)
            @warn "Solve failed for patient $pid (model=$model_type)"
            continue
        end

        yhat = vec(pred[3, :])
        y = p_data.ctnt_data

        if length(yhat) != length(y)
            @warn "Length mismatch for patient $pid: yhat=$(length(yhat)), y=$(length(y))"
            continue
        end

        res = log_residuals(y, yhat)

        append!(residuals_out, DataFrame(
            id=fill(pid, length(y)),
            t=p_data.timepoints,
            y=y,
            yhat=yhat,
            res=res
        ))

        push!(metrics_out, (pid, smape(yhat, y), rmsle(y, yhat)))
    end

    return residuals_out, metrics_out, params_out
end


# =============================================================================
# Data-Only Objectives For PLA
# Future file: pla_objectives.jl
# =============================================================================
"""
Data-Only Objectives For PLA

Input: patient model tuples and log-scale parameter vectors.
Output: plasma predictions, log-residual vectors, RSS, or Gaussian negative log-likelihood values.
"""

"""
Solve the patient-specific cUDE/UDE model with fixed NN parameters and return plasma predictions.
Returns a Vector{Float64} on success, or `nothing` on failure.
"""
function patient_plasma_prediction(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode=θ, neural=fixed_nn_params)

    u0 = initial_conditions_from_log_params(θ)
    prob = remake(model.problem; u0=u0, p=p)

    sol = try
        solve(
            prob,
            Tsit5();
            p=p,
            saveat=timepoints,
            abstol=1e-8,
            reltol=1e-6
        )
    catch
        return nothing
    end

    if !successful_retcode(sol)
        return nothing
    end

    plasma = vec(sol[3, :])

    if length(plasma) != length(ctnt_data) || any(!isfinite, plasma)
        return nothing
    end

    return plasma
end

"""
Log-residual vector:
`log(pred_k + DELTA) - log(obs_k + DELTA)`.
"""
function patient_log_residuals(θ, data)
    ctnt_data = data[3]

    plasma = patient_plasma_prediction(θ, data)
    plasma === nothing && return nothing

    resid = log.(plasma .+ DELTA) .- log.(ctnt_data .+ DELTA)

    if any(!isfinite, resid)
        return nothing
    end

    return resid
end

"""
Residual sum of squares on the log scale.
This is the data-only part, with no physics-informed penalty.
"""
function patient_rss_log(θ, data)
    resid = patient_log_residuals(θ, data)
    resid === nothing && return Inf
    return sum(abs2, resid)
end

"""
Negative log-likelihood under a Gaussian error model on the log-transformed data.

If sigma2 === nothing:
    returns the profiled objective n * log(RSS/n)
    which is equivalent to -2 log L up to an additive constant.

If sigma2 is provided:
    returns the full NLL up to the constant n/2 * log(2π).
"""
function patient_nll_log_gaussian(θ, data; sigma2::Union{Nothing,Float64}=nothing)
    resid = patient_log_residuals(θ, data)
    resid === nothing && return Inf

    n = length(resid)
    n == 0 && return Inf

    rss = max(sum(abs2, resid), eps(Float64))

    if !isfinite(rss)
        return Inf
    end

    if sigma2 === nothing
        return n * log(rss / n)
    else
        if !(isfinite(sigma2) && sigma2 > 0.0)
            return Inf
        end
        return 0.5 * n * log(sigma2) + rss / (2 * sigma2)
    end
end


# =============================================================================
# Log Parsing Utilities
# Future file: log_parsing.jl
# =============================================================================
"""
Log Parsing Utilities

Input: historical plain-text optimizer logs.
Output: CSV files and a DataFrame with patient metrics and parameter columns.
"""

"""
    parse_log_to_csv(inpath::AbstractString;
                     out_csv::AbstractString = "params.csv",
                     meta_csv::Union{Nothing,AbstractString} = "meta.csv")

Parse a log like:
  "dataset: ...\nUB: [..]\nPatient: n1 | smape: ... | params: [a,b,c,...]"

Output:
  - `out_csv`: table with patient, smape, optional rmsle/loss, and p1..pK
  - `meta_csv`: optional one-row metadata table with dataset and UB
Returns the parameter DataFrame.
"""
function parse_log_to_csv(inpath::AbstractString; out_csv::AbstractString="params.csv",
    meta_csv::Union{Nothing,AbstractString}="meta.csv")

    txt = read(inpath, String)

    dataset = match(r"(?m)^\s*dataset:\s*(.+)$", txt)
    dataset = dataset === nothing ? missing : strip(dataset.captures[1])

    ub_m = match(r"(?m)^\s*UB:\s*\[([^\]]+)\]", txt)
    UB = ub_m === nothing ? missing :
         [parse(Float64, strip(x)) for x in split(ub_m.captures[1], ',')]

    num_re = raw"([+-]?(?:Inf|NaN|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))"
    pat_re = Regex(
        raw"(?m)^\s*Patient:\s*([^\|]+?)\s*\|\s*smape:\s*" * num_re *
        raw"(?:\s*\|\s*rmsle:\s*" * num_re *
        raw"\s*\|\s*loss:\s*" * num_re * raw")?" *
        raw"\s*\|\s*params:\s*\[([^\]]*)\]"
    )
    ms = collect(eachmatch(pat_re, txt))

    params_lists = Vector{Vector{Float64}}(undef, length(ms))
    patients = Vector{String}(undef, length(ms))
    smapes = Vector{Float64}(undef, length(ms))
    rmsles = Vector{Union{Missing,Float64}}(undef, length(ms))
    losses = Vector{Union{Missing,Float64}}(undef, length(ms))
    maxK = 0

    for (i, m) in enumerate(ms)
        patients[i] = strip(m.captures[1])
        smapes[i] = parse(Float64, m.captures[2])
        rmsles[i] = m.captures[3] === nothing ? missing : parse(Float64, m.captures[3])
        losses[i] = m.captures[4] === nothing ? missing : parse(Float64, m.captures[4])

        params_str = strip(m.captures[5])
        plist = isempty(params_str) ? Float64[] :
                [parse(Float64, strip(x)) for x in split(params_str, ',')]

        params_lists[i] = plist
        maxK = max(maxK, length(plist))
    end

    df = DataFrame(patient=patients, smape=smapes, rmsle=rmsles, loss=losses)
    for k in 1:maxK
        col = Vector{Union{Missing,Float64}}(undef, length(params_lists))
        @inbounds for i in eachindex(params_lists)
            col[i] = k <= length(params_lists[i]) ? params_lists[i][k] : missing
        end
        df[!, Symbol("p$k")] = col
    end

    CSV.write(out_csv, df)

    if meta_csv !== nothing
        meta = DataFrame(dataset=[dataset],
            UB=[UB === missing ? missing : join(UB, ",")])
        CSV.write(meta_csv, meta)
    end

    return df
end

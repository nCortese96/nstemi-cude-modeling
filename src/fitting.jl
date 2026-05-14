"""
fitting.jl

Optimization, patient-level fitting, metrics, and fit-output helpers.

Sections:
- Multi-Start Optimization: reproducible bounded optimization from multiple starts.
- Initialization Utilities: reusable neural and ODE parameter initialization.
- Fit Metrics And Tables: metric summaries and patient-level CSV builders.
- Prediction And Plotting: reusable ODE prediction and patient fit plots.
- ODE Td-Sigmoid Fitting: step 01 fitting and output helpers.
"""

using Base.Threads: @threads
using CSV
using DataFrames: DataFrame
using Logging
using Optimization, OptimizationOptimJL
using OrdinaryDiffEq: Tsit5
using Plots
using ProgressMeter
using QuasiMonteCarlo: LatinHypercubeSample, sample
using Random: AbstractRNG
using SciMLBase: ODEProblem, successful_retcode, solve, remake
using SimpleChains: SimpleChain, init_params
using StableRNGs: StableRNG
using Statistics: mean, median, quantile, std
using Tables
import Base.Threads

# =============================================================================
# Multi-Start Optimization
# =============================================================================

"""
    run_multistart(loss, N; lower, upper, optimizer, rng, maxiters, maxtime, prescreen, topk)

Run Latin-hypercube multi-start optimization and return `(best_solution,
all_results)`.
"""
# Used by: src/fitting.jl (fit_ode_patient). Planned use: scripts/02b_evaluate_cude_nn.jl, scripts/02d_evaluate_cude_nn_external_test.jl, scripts/05_run_systematic_truncation.jl, scripts/07_evaluate_symbolic_formula.jl.
function run_multistart(
    loss::Function,
    N::Int;
    lower::Union{Nothing,Vector{<:Real}}=nothing,
    upper::Union{Nothing,Vector{<:Real}}=nothing,
    optimizer=Fminbox(LBFGS()),
    verbose::Bool=true,
    callback::Function=(x, f) -> nothing,
    save_to_csv::Union{Nothing,String}=nothing,
    rng::AbstractRNG=StableRNG(42),
    maxiters::Int=1000,
    maxtime::Float64=80.0,
    prescreen::Bool=false,
    topk::Int=8
)
    lhs_starts = sample(N, lower, upper, LatinHypercubeSample(rng))
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

# =============================================================================
# Initialization Utilities
# =============================================================================

"""
    sample_initial_neural_parameters(n_initials, chain, rng)

Return reproducible neural-network parameter initializations for a SimpleChain.
"""
# Planned use: scripts/02a_run_cude_training.jl.
function sample_initial_neural_parameters(n_initials::Int, chain::SimpleChain, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

"""
    sample_initial_parameters(n_patients, n_initials, lhs_lb, lhs_ub, rng)

Return Latin-hypercube ODE parameter initializations for all patients.
"""
# Planned use: scripts/02a_run_cude_training.jl.
function sample_initial_parameters(n_patients::Int, n_initials::Int, lhs_lb::AbstractVector{T}, lhs_ub::AbstractVector{T}, rng::AbstractRNG) where T<:Real
    return sample(n_initials, repeat(lhs_lb, n_patients), repeat(lhs_ub, n_patients), LatinHypercubeSample(rng))
end

# =============================================================================
# Fit Metrics And Tables
# =============================================================================

"""
    summarize_metric_vector(values)

Return mean, standard deviation, median, quartiles, and IQR after dropping
non-finite values.
"""
# Planned use: scripts/02a_run_cude_training.jl and downstream model-comparison scripts.
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
# Planned use: scripts/02a_run_cude_training.jl and downstream model-comparison scripts.
function summarize_metrics(losses, smapes, rmsles)
    return (
        loss=summarize_metric_vector(losses),
        smape=summarize_metric_vector(smapes),
        rmsle=summarize_metric_vector(rmsles),
    )
end

"""
    save_patient_metrics(path, ids, smapes, rmsles, losses)

Write patient-level metrics to CSV and return the saved DataFrame.
"""
# Planned use: scripts/02b_evaluate_cude_nn.jl and downstream evaluation scripts.
function save_patient_metrics(path::AbstractString, ids, smapes, rmsles, losses)
    mkpath(dirname(path))
    df = DataFrame(patient_id=ids, smape=smapes, rmsle=rmsles, loss=losses)
    CSV.write(path, df)
    return df
end

# =============================================================================
# Prediction And Plotting
# =============================================================================

"""
    predict_patient_curve(problem, theta; saveat=1.0, abstol=1e-8, reltol=1e-6)

Solve a patient ODE problem after remaking its initial condition and parameters.
"""
# Planned use: scripts/05_run_systematic_truncation.jl and diagnostic scripts.
function predict_patient_curve(problem::ODEProblem, theta; saveat=1.0, abstol=1e-8, reltol=1e-6)
    u0 = initial_conditions_from_log_params(theta)
    prob = remake(problem; u0=u0, p=theta)
    sol = solve(prob, Tsit5(); p=theta, saveat=saveat, abstol=abstol, reltol=reltol)
    successful_retcode(sol) || error("Prediction solve failed with retcode=$(sol.retcode)")
    return sol
end

"""
    plot_patient_fit(sol, patient; title="Patient <id>", plasma_only=false)

Build a patient fit plot from an ODE solution and observed cTnT data.
"""
# Planned use: scripts/05_run_systematic_truncation.jl and diagnostic scripts.
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
# ODE Td-Sigmoid Fitting
# =============================================================================

"""
    ode_dataset_output_paths(output_root, dataset_name)

Return canonical step 01 output paths for one dataset.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function ode_dataset_output_paths(output_root::AbstractString, dataset_name::AbstractString)
    dataset_dir = joinpath(output_root, "$(dataset_name)_opt")
    return (
        dataset_dir=dataset_dir,
        fig_dir=joinpath(dataset_dir, "figs"),
        params_csv=joinpath(dataset_dir, "params_out.csv"),
        params_val_csv=joinpath(dataset_dir, "params_out_val.csv"),
    )
end

"""
    fit_ode_patient(patient, pguess, lower, upper; ...)

Fit one patient's Td-sigmoid ODE parameters with bounded multi-start
optimization.
"""
# Used by: src/fitting.jl (fit_ode_dataset).
function fit_ode_patient(patient::PatientData, pguess::AbstractVector, lower::AbstractVector, upper::AbstractVector;
    lambda_back::Real=1.0,
    n_multistart::Int=40,
    rng_seed::Integer=1234,
    maxiters::Int=1000,
    maxtime::Real=80.0,
    prescreen::Bool=false,
    topk::Int=8)

    t_data = patient.timepoints
    x_data = patient.ctnt_data
    tspan = (0.0, t_data[end] + 10.0)
    u0 = initial_conditions_from_log_params(pguess)
    prob = ODEProblem(troponin_ode!, u0, tspan, pguess)
    data = (prob, t_data, x_data)
    loss = theta -> patient_loss_formula(theta, data; λ_back=lambda_back)

    best_result, _ = run_multistart(
        loss,
        n_multistart;
        lower=lower,
        upper=upper,
        rng=StableRNG(rng_seed),
        verbose=false,
        maxiters=maxiters,
        maxtime=Float64(maxtime),
        prescreen=prescreen,
        topk=topk,
    )

    best_params = Vector(best_result.u)
    newprob = remake(prob; p=best_params, u0=initial_conditions_from_log_params(best_params))
    pred = solve(newprob, Tsit5(); saveat=t_data)
    sol = solve(newprob, Tsit5())

    return (
        patient=patient.id,
        smape=smape(pred[3, :], x_data),
        rmsle=rmsle(x_data, pred[3, :]),
        loss=best_result.minimum,
        params=best_params,
        sol=sol,
        pred=pred,
    )
end

"""
    save_ode_patient_plots(sol, patient, dataset_name, fig_dir; plotting=true)

Save full-state and plasma-only patient SVG plots for step 01.
"""
# Used by: src/fitting.jl (fit_ode_dataset).
function save_ode_patient_plots(sol, patient::PatientData, dataset_name::AbstractString, fig_dir::AbstractString; plotting::Bool=true)
    pl = Plots.plot(sol; idxs=1, lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title="ODE - Patient $(patient.id)")
    Plots.plot!(pl, sol; idxs=2, lw=2, label="Cytosol")
    Plots.plot!(pl, sol; idxs=3, lw=2, label="Blood")
    Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    pl_plasm = Plots.plot(sol; idxs=3, lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
    Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    if plotting
        display(pl)
        display(pl_plasm)
    end

    savefig(pl, joinpath(fig_dir, "patient_$(patient.id)_$(dataset_name).svg"))
    savefig(pl_plasm, joinpath(fig_dir, "patient_$(patient.id)_$(dataset_name)_plasm.svg"))

    return (full=pl, plasma=pl_plasm)
end

"""
    fit_ode_dataset(patients, settings; dataset_name, fig_dir)

Fit all patients for one step 01 dataset and save patient plots.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function fit_ode_dataset(patients::AbstractVector{PatientData}, settings; dataset_name::AbstractString, fig_dir::AbstractString)
    results = Vector{Any}(undef, length(patients))

    for (i, patient) in enumerate(patients)
        @info "Fitting ODE patient $(i)/$(length(patients)): $(patient.id)"
        result = fit_ode_patient(
            patient,
            settings.pguess,
            settings.lower,
            settings.upper;
            lambda_back=settings.lambda_back,
            n_multistart=settings.n_multistart,
            rng_seed=settings.rng_seed,
            maxiters=settings.maxiters,
            maxtime=settings.maxtime,
            prescreen=settings.prescreen,
            topk=settings.topk,
        )

        save_ode_patient_plots(result.sol, patient, dataset_name, fig_dir; plotting=settings.plotting)

        @info "Completed patient $(patient.id): SMAPE=$(result.smape), RMSLE=$(result.rmsle), loss=$(result.loss)"
        results[i] = result
    end

    return results
end

"""
    ode_fit_results_dataframe(results)

Build the step 01 parameter and metric DataFrame from patient fit results.
"""
# Used by: src/fitting.jl (save_ode_fit_results).
function ode_fit_results_dataframe(results)
    isempty(results) && return DataFrame(patient=String[], smape=Float64[], rmsle=Float64[], loss=Float64[])

    n_params = maximum(length(result.params) for result in results)
    df = DataFrame(
        patient=[result.patient for result in results],
        smape=[result.smape for result in results],
        rmsle=[result.rmsle for result in results],
        loss=[result.loss for result in results],
    )

    for k in 1:n_params
        df[!, Symbol("p$(k)")] = [k <= length(result.params) ? result.params[k] : NaN for result in results]
    end

    return df
end

"""
    subset_results_by_patient_ids(df, ids)

Return the subset of fit results whose patients are listed in `ids`, preserving
the original order of `df`.
"""
# Used by: src/fitting.jl (save_ode_fit_results).
function subset_results_by_patient_ids(df::DataFrame, ids::AbstractVector{<:AbstractString})
    requested_ids = Set(ids)
    fitted_ids = Set(string.(df.patient))
    missing_ids = setdiff(requested_ids, fitted_ids)
    isempty(missing_ids) || error("Cannot build validation subset. Missing fitted patient ID: $(first(missing_ids))")

    subset_idx = findall(patient -> string(patient) in requested_ids, df.patient)
    return df[subset_idx, :]
end

"""
    save_ode_fit_results(paths, results; validation_ids=String[])

Write step 01 full fit results and optional MIMIC-IV validation subset.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function save_ode_fit_results(paths, results; validation_ids::AbstractVector{<:AbstractString}=String[])
    df = ode_fit_results_dataframe(results)
    CSV.write(paths.params_csv, df)

    validation_df = nothing
    if !isempty(validation_ids)
        validation_df = subset_results_by_patient_ids(df, validation_ids)
        CSV.write(paths.params_val_csv, validation_df)
    end

    return (all=df, validation=validation_df)
end

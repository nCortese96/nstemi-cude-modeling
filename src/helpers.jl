"""
helpers.jl

Reorganized compatibility copy of `ctnt-ude-model.jl`.

This file now contains only helper sections that have not yet been split into
dedicated source files. Completed sections are removed from this file and
included directly by the scripts that need them.
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
using DataFrames: DataFrame, DataFrameRow, nrow
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

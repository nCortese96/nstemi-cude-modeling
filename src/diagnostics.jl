"""
diagnostics.jl

Reusable non-plotting helpers for model-comparison diagnostics.

Sections:
- Residual Tables: long-format residual and patient-metric computation.
- Summary Tables: parameter and metric summary builders.
- Comparison Tables: ODE/cUDE metric joins and delta-sMAPE reports.
- Profile Selection: deterministic patient selections for diagnostic profiles.
"""

using ComponentArrays: ComponentArray
using DataFrames
using OrdinaryDiffEq: Tsit5
using Random
using SciMLBase: ODEProblem, successful_retcode, solve
using Statistics: mean, quantile, std

# =============================================================================
# Residual Tables
# =============================================================================

"""
    log_residuals(y, yhat; eps=DELTA)

Return log-scale residuals `log(y + eps) - log(yhat + eps)`.
"""
# Used by: src/diagnostics.jl (compute_residuals_long_unified).
function log_residuals(y, yhat; eps=DELTA)
    return log.(y .+ eps) .- log.(yhat .+ eps)
end

"""
    add_time_bins!(df, edges)

Add integer time-bin IDs and bin centers to a residual DataFrame.
"""
# Used by: src/plotting.jl residual diagnostic plots.
function add_time_bins!(df::DataFrame, edges::AbstractVector{<:Real})
    nb = length(edges) - 1
    bins = similar(df.t, Int)

    for i in eachindex(df.t)
        k = searchsortedlast(edges, df.t[i])
        bins[i] = clamp(k, 1, nb)
    end

    df.bin = bins
    df.bin_center = [0.5 * (edges[k] + edges[k + 1]) for k in bins]
    return df
end

"""
    bin_summary(df)

Return residual quartiles and counts by time bin.
"""
# Used by: src/plotting.jl residual diagnostic plots.
function bin_summary(df::DataFrame)
    groups = groupby(df, :bin)
    centers = Float64[]
    q1 = Float64[]
    med = Float64[]
    q3 = Float64[]
    counts = Int[]

    for sub in groups
        push!(centers, first(sub.bin_center))
        push!(q1, quantile(sub.res, 0.25))
        push!(med, quantile(sub.res, 0.50))
        push!(q3, quantile(sub.res, 0.75))
        push!(counts, nrow(sub))
    end

    order = sortperm(centers)
    return (
        centers=centers[order],
        q1=q1[order],
        med=med[order],
        q3=q3[order],
        n=counts[order],
    )
end

"""
    compute_residuals_long_unified(patients, params_df; model_type, chain, nn_params, ...)

Compute long-format residuals, patient metrics, and natural-scale parameters
for either ODE or cUDE patient-level fits.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function compute_residuals_long_unified(
    patients,
    params_df::DataFrame;
    model_type::Symbol=:ode,
    chain=nothing,
    nn_params=nothing,
    tpad::Real=10.0,
    abstol::Real=1e-12,
    reltol::Real=1e-10,
)
    model_type in (:ode, :cude) || error("model_type must be :ode or :cude.")
    if model_type == :cude
        chain !== nothing || error("chain is required for cUDE residual computation.")
        nn_params !== nothing || error("nn_params is required for cUDE residual computation.")
    end

    patient_lookup = patients isa Dict ? patients : Dict(patient.id => patient for patient in patients)
    residuals_out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])
    metrics_out = DataFrame(id=String[], smape_val=Float64[], rmsle_val=Float64[])
    params_out = DataFrame(patient_id=String[], a=Float64[], b=Float64[], Cs0=Float64[], Cc0=Float64[], p5=Float64[])
    id_col = model_type == :ode ? :patient : :patient_id

    for row in eachrow(params_df)
        pid = String(row[id_col])
        haskey(patient_lookup, pid) || continue
        patient = patient_lookup[pid]
        tmax = maximum(patient.timepoints) + tpad

        if model_type == :ode
            params_log = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]
            u0 = initial_conditions_from_log_params(params_log)
            problem = ODEProblem(troponin_ode!, u0, (0.0, tmax), params_log)
            pred = solve(problem, Tsit5(); saveat=patient.timepoints, abstol=abstol, reltol=reltol)
            push!(params_out, (pid, exp(params_log[1]), exp(params_log[2]), exp(params_log[3]), exp(params_log[4]), exp(params_log[5])))
        else
            params_natural = Float64[row.a, row.b, row.Cs0, row.Cc0, row.beta]
            params_log = log.(params_natural)
            u0 = initial_conditions_from_log_params(params_log)
            cude_rhs!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
            problem = ODEProblem(cude_rhs!, u0, (0.0, tmax))
            full_params = ComponentArray(ode=params_log, neural=nn_params)
            pred = solve(problem, Tsit5(); p=full_params, saveat=patient.timepoints, abstol=abstol, reltol=reltol)
            push!(params_out, (pid, params_natural[1], params_natural[2], params_natural[3], params_natural[4], params_natural[5]))
        end

        if !successful_retcode(pred)
            @warn "Solve failed during residual computation." patient=pid model_type=model_type
            continue
        end

        yhat = vec(pred[3, :])
        y = patient.ctnt_data
        if length(yhat) != length(y)
            @warn "Skipping residual row because predicted and observed lengths differ." patient=pid yhat=length(yhat) y=length(y)
            continue
        end

        append!(
            residuals_out,
            DataFrame(
                id=fill(pid, length(y)),
                t=patient.timepoints,
                y=y,
                yhat=yhat,
                res=log_residuals(y, yhat),
            ),
        )

        push!(metrics_out, (pid, smape(yhat, y), rmsle(y, yhat)))
    end

    return residuals_out, metrics_out, params_out
end

"""
    diagnostic_parameter_table_from_fit_params(params_df; model_type)

Build the natural-scale parameter table used by diagnostic boxplots from
already-saved ODE or cUDE patient parameter CSVs, without solving trajectories.
"""
# Used by: scripts/03a_run_model_diagnostics.jl plot-only modes.
function diagnostic_parameter_table_from_fit_params(params_df::DataFrame; model_type::Symbol)
    model_type in (:ode, :cude) || error("model_type must be :ode or :cude.")

    if model_type == :ode
        required = [:patient, :p1, :p2, :p3, :p4, :p5]
        missing = setdiff(required, Symbol.(names(params_df)))
        isempty(missing) || error("Missing ODE parameter columns for diagnostics: $(missing)")

        return DataFrame(
            patient_id=string.(params_df.patient),
            a=exp.(Float64.(params_df.p1)),
            b=exp.(Float64.(params_df.p2)),
            Cs0=exp.(Float64.(params_df.p3)),
            Cc0=exp.(Float64.(params_df.p4)),
            p5=exp.(Float64.(params_df.p5)),
        )
    end

    required = [:patient_id, :a, :b, :Cs0, :Cc0, :beta]
    missing = setdiff(required, Symbol.(names(params_df)))
    isempty(missing) || error("Missing cUDE parameter columns for diagnostics: $(missing)")

    return DataFrame(
        patient_id=string.(params_df.patient_id),
        a=Float64.(params_df.a),
        b=Float64.(params_df.b),
        Cs0=Float64.(params_df.Cs0),
        Cc0=Float64.(params_df.Cc0),
        p5=Float64.(params_df.beta),
    )
end

"""
    compute_symbolic_formula_residuals(patients, params_list; edges=EDGES, ...)

Compute canonical step 04b residuals for the promoted symbolic surrogate.
Parameters are log-scale and ordered patient-major.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function compute_symbolic_formula_residuals(
    patients::AbstractVector{PatientData},
    params_list;
    edges=EDGES,
    n_params::Integer=5,
    tpad::Real=10.0,
    abstol::Real=1e-8,
    reltol::Real=1e-6,
)
    parameter_vectors = if params_list isa AbstractVector && all(p -> p isa AbstractVector, params_list)
        collect(params_list)
    else
        length(params_list) % n_params == 0 ||
            error("Formula parameter vector length $(length(params_list)) is not divisible by n_params=$(n_params).")
        [collect(params_list[(n_params * (i - 1) + 1):(n_params * i)]) for i in 1:(length(params_list) ÷ n_params)]
    end

    length(parameter_vectors) == length(patients) ||
        error("Formula residual computation received $(length(parameter_vectors)) parameter sets for $(length(patients)) patients.")

    residuals_out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])

    for (patient, params_log) in zip(patients, parameter_vectors)
        prob = symbolic_formula_problem(params_log, patient; tpad=tpad)
        pred = solve(prob, Tsit5(); p=params_log, saveat=patient.timepoints, abstol=abstol, reltol=reltol)

        if !successful_retcode(pred)
            @warn "Formula residual solve failed." patient=patient.id retcode=pred.retcode
            continue
        end

        yhat = vec(pred[3, :])
        y = patient.ctnt_data
        if length(yhat) != length(y)
            @warn "Skipping formula residuals because predicted and observed lengths differ." patient=patient.id yhat=length(yhat) y=length(y)
            continue
        end

        append!(
            residuals_out,
            DataFrame(
                id=fill(patient.id, length(y)),
                t=patient.timepoints,
                y=y,
                yhat=yhat,
                res=log_residuals(y, yhat),
            ),
        )
    end

    add_time_bins!(residuals_out, edges)
    return residuals_out
end

# =============================================================================
# Summary Tables
# =============================================================================

"""
    diagnostic_summary_stats(values)

Return mean, standard deviation, quartiles, median, and IQR after dropping
non-finite values.
"""
# Used by: src/diagnostics.jl summary table builders.
function diagnostic_summary_stats(values)
    finite_values = collect(filter(isfinite, values))
    isempty(finite_values) && return (mean=NaN, std=NaN, q1=NaN, median=NaN, q3=NaN, iqr=NaN)
    q1 = quantile(finite_values, 0.25)
    q3 = quantile(finite_values, 0.75)
    return (
        mean=mean(finite_values),
        std=length(finite_values) > 1 ? std(finite_values) : 0.0,
        q1=q1,
        median=quantile(finite_values, 0.50),
        q3=q3,
        iqr=q3 - q1,
    )
end

"""
    build_parameter_summary(parameter_sets)

Build the canonical parameter summary table for ODE and cUDE diagnostics.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function build_parameter_summary(parameter_sets)
    out = DataFrame(
        model=String[],
        dataset=String[],
        param=String[],
        mean=Float64[],
        std=Float64[],
        q1=Float64[],
        median=Float64[],
        q3=Float64[],
        iqr=Float64[],
    )

    for item in parameter_sets
        for (i, param_name) in enumerate(item.param_names)
            col = i <= 4 ? [:a, :b, :Cs0, :Cc0][i] : :p5
            stats = diagnostic_summary_stats(item.df[!, col])
            push!(out, (item.model, item.dataset, param_name, stats.mean, stats.std, stats.q1, stats.median, stats.q3, stats.iqr))
        end
    end

    return out
end

"""
    build_metrics_summary(metric_sets)

Build the canonical model/dataset metric summary table.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function build_metrics_summary(metric_sets)
    out = DataFrame(
        model=String[],
        dataset=String[],
        metric=String[],
        mean=Float64[],
        std=Float64[],
        q1=Float64[],
        median=Float64[],
        q3=Float64[],
        iqr=Float64[],
    )

    for item in metric_sets
        for (metric_name, col) in [("sMAPE", :smape_val), ("RMSLE", :rmsle_val)]
            stats = diagnostic_summary_stats(item.df[!, col])
            push!(out, (item.model, item.dataset, metric_name, stats.mean, stats.std, stats.q1, stats.median, stats.q3, stats.iqr))
        end
    end

    return out
end

# =============================================================================
# Comparison Tables
# =============================================================================

"""
    comparison_metrics_dataframe(ode_metrics, cude_metrics)

Join ODE and cUDE metrics by patient ID for subject-wise comparisons.
"""
# Used by: scripts/03a_run_model_diagnostics.jl and src/plotting.jl.
function comparison_metrics_dataframe(ode_metrics::DataFrame, cude_metrics::DataFrame)
    ode = DataFrame(
        patient_id=string.(ode_metrics.id),
        smape_ode=Float64.(ode_metrics.smape_val),
        rmsle_ode=Float64.(ode_metrics.rmsle_val),
    )
    cude = DataFrame(
        patient_id=string.(cude_metrics.id),
        smape_cude=Float64.(cude_metrics.smape_val),
        rmsle_cude=Float64.(cude_metrics.rmsle_val),
    )

    return innerjoin(ode, cude, on=:patient_id)
end

"""
    overlap_comparison_dataframe(ode_params, cude_metrics, cude_params)

Join ODE parameters, cUDE metrics, and cUDE parameters for overlap-profile
selection and plotting.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function overlap_comparison_dataframe(ode_params::DataFrame, cude_metrics::DataFrame, cude_params::DataFrame)
    ode = DataFrame(
        patient_id=string.(ode_params.patient),
        smape_ode=Float64.(ode_params.smape),
        rmsle_ode=Float64.(ode_params.rmsle),
        p1=Float64.(ode_params.p1),
        p2=Float64.(ode_params.p2),
        p3=Float64.(ode_params.p3),
        p4=Float64.(ode_params.p4),
        p5=Float64.(ode_params.p5),
    )
    cude_m = DataFrame(
        patient_id=string.(cude_metrics.patient_id),
        smape_cude=Float64.(cude_metrics.smape),
        rmsle_cude=Float64.(cude_metrics.rmsle),
    )

    merged = innerjoin(innerjoin(ode, cude_m, on=:patient_id), cude_params, on=:patient_id, makeunique=true)
    merged.delta_smape = merged.smape_cude .- merged.smape_ode
    return merged
end

"""
    write_delta_smape_report(path, comparisons; threshold=1.0)

Write the canonical delta-sMAPE report for ODE vs cUDE patient comparisons.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function write_delta_smape_report(path::AbstractString, comparisons; threshold::Real=1.0)
    mkpath(dirname(path))

    open(path, "w") do io
        for item in comparisons
            df = sort(copy(item.df), :delta_smape)
            total = nrow(df)
            cude_wins = count(x -> x < -threshold, df.delta_smape)
            ties = count(x -> abs(x) <= threshold, df.delta_smape)
            ode_wins = count(x -> x > threshold, df.delta_smape)

            println(io, "=== DELTA sMAPE REPORT ($(item.dataset_label)) ===")
            println(io, "Total Patients: ", total)
            println(io, "cUDE Wins (\u0394sMAPE < -$(threshold)%): ", round(100 * cude_wins / total, digits=2), "% (", cude_wins, " patients)")
            println(io, "Ties (|\u0394sMAPE| <= $(threshold)%):     ", round(100 * ties / total, digits=2), "% (", ties, " patients)")
            println(io, "ODE Wins (\u0394sMAPE > $(threshold)%):   ", round(100 * ode_wins / total, digits=2), "% (", ode_wins, " patients)")
            println(io, "-----------------------------------------")
            println(io, "Patient Breakdown:")

            for row in eachrow(df)
                outcome = if row.delta_smape < -threshold
                    "cUDE Wins"
                elseif row.delta_smape > threshold
                    "ODE Wins"
                else
                    "Tie"
                end

                println(
                    io,
                    "$(row.patient_id) | ODE: $(round(row.smape_ode, digits=2))% | cUDE: $(round(row.smape_cude, digits=2))% | Delta: $(round(row.delta_smape, digits=2))% | Outcome: $(outcome)",
                )
            end

            println(io, "=========================================\n")
        end
    end

    return path
end

# =============================================================================
# Profile Selection
# =============================================================================

"""
    select_metric_quartile_rows(df, metric_col; n_per_quartile=10, seed=42)

Select up to `n_per_quartile` rows from each metric quartile using a deterministic
global RNG reset pattern.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function select_metric_quartile_rows(
    df::DataFrame,
    metric_col::Symbol;
    n_per_quartile::Integer=10,
    seed::Integer=42,
)
    sorted_df = sort(dropmissing(copy(df), metric_col), metric_col)
    total = nrow(sorted_df)
    quartile_len = div(total, 4)
    selected = Dict{Int,DataFrame}()

    Random.seed!(seed)
    for (quartile, start_idx) in enumerate([1, quartile_len + 1, 2 * quartile_len + 1, 3 * quartile_len + 1])
        end_idx = quartile == 4 ? total : start_idx + quartile_len - 1
        chunk = sorted_df[start_idx:end_idx, :]
        if nrow(chunk) >= n_per_quartile
            selected[quartile] = chunk[shuffle(1:nrow(chunk))[1:n_per_quartile], :]
        else
            selected[quartile] = chunk
        end
    end

    return selected
end

"""
    cude_profile_selection_table(metrics, params)

Join cUDE metric and parameter tables for quartile profile selection.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function cude_profile_selection_table(metrics::DataFrame, params::DataFrame)
    return innerjoin(metrics, params, on=:patient_id, makeunique=true)
end

"""
    select_overlap_profile_rows(comparison; n_per_group=10)

Return cUDE-advantage, neutral, and ODE-advantage patient groups based on
delta sMAPE.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function select_overlap_profile_rows(comparison::DataFrame; n_per_group::Integer=10)
    ordered = sort(copy(comparison), :delta_smape)
    ordered.abs_delta = abs.(ordered.delta_smape)
    neutral_sorted = sort(ordered, :abs_delta)

    return (
        cude_advantage=nrow(ordered) >= n_per_group ? ordered[1:n_per_group, :] : ordered,
        neutral=nrow(neutral_sorted) >= n_per_group ? neutral_sorted[1:n_per_group, :] : neutral_sorted,
        ode_advantage=nrow(ordered) >= n_per_group ? ordered[end - n_per_group + 1:end, :] : ordered,
    )
end

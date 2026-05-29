"""
systematic_truncation.jl

Numerical and tabular helpers for workflow step 03c.

Sections:
- Model Runtime: ODE/cUDE carrier construction for fixed-NN truncation fits.
- Scenario Generation: deterministic systematic truncation cases.
- Fitting: patient-level full and truncated multi-start optimization.
- Tables: per-patient, aggregate, and paper-summary DataFrame builders.
- Overlay Inputs: deterministic ODE-vs-cUDE curve reconstruction.
"""

using ComponentArrays: ComponentArray
using CSV
using DataFrames: DataFrame, combine, groupby, nrow
using Logging
using OrdinaryDiffEq: Tsit5
using ProgressMeter
using Random: AbstractRNG
using SciMLBase: ODEProblem, remake, solve, successful_retcode
using StableRNGs: StableRNG
using Statistics: mean, median, quantile, std

# =============================================================================
# Model Runtime
# =============================================================================

"""
    build_systematic_truncation_model_runtime(model_key, settings; selected_model, cude_artifacts)

Build the fixed model runtime used by step 03c without changing legacy bounds,
guesses, or fixed-NN behavior.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function build_systematic_truncation_model_runtime(
    model_key::Symbol,
    settings;
    selected_model=nothing,
    cude_artifacts=nothing,
)
    if model_key == :ode
        return (
            model_key=:ode,
            model_name="ODE_TdSigmoid",
            model_tag="ode_tdsigmoid",
            curve_label="Model curve (ODE, saveat=1)",
            loss_label="patient_loss_formula",
            uses_fixed_nn=false,
            chain=nothing,
            fixed_nn=nothing,
            param_labels=["a", "b", "Cs0", "Cc0", "Td"],
            pguess=Vector{Float64}(settings.ode_pguess),
            lhs_lb=Vector{Float64}(settings.ode_lower),
            lhs_ub=Vector{Float64}(settings.ode_upper),
            lambda_back=Float64(settings.lambda_back),
        )
    elseif model_key == :cude
        selected_model === nothing && error("Selected cUDE model is required for systematic truncation.")
        cude_artifacts === nothing && error("cUDE training artifacts are required for systematic truncation.")

        chain = neural_network_model(selected_model.nn_depth, selected_model.nn_width; input_dims=settings.input_dim)
        1 <= selected_model.model_idx <= length(cude_artifacts.neural_network_parameters) ||
            error("Selected model_idx=$(selected_model.model_idx) is outside available cUDE candidates.")

        return (
            model_key=:cude,
            model_name="cUDE",
            model_tag="cude_fixednn",
            curve_label="Model curve (cUDE, saveat=1)",
            loss_label="patient_loss",
            uses_fixed_nn=true,
            chain=chain,
            fixed_nn=Vector{Float64}(cude_artifacts.neural_network_parameters[selected_model.model_idx]),
            param_labels=["a", "b", "Cs0", "Cc0", "beta"],
            pguess=Vector{Float64}(settings.cude_pguess),
            lhs_lb=Vector{Float64}(settings.cude_lower),
            lhs_ub=Vector{Float64}(settings.cude_upper),
            lambda_back=Float64(settings.lambda_back),
        )
    end

    error("Unsupported systematic truncation model key: $(model_key)")
end

"""
    build_systematic_truncation_carrier(base_patient, model_cfg)

Create the reusable ODEProblem/model carrier for one base patient.
"""
# Used by: src/systematic_truncation.jl.
function build_systematic_truncation_carrier(base_patient::PatientData, model_cfg)
    if model_cfg.model_key == :cude
        model = ctntCUDEModel(model_cfg.pguess, model_cfg.chain, base_patient.timepoints)
        return (kind=:cude, model=model, fixed_nn=model_cfg.fixed_nn)
    elseif model_cfg.model_key == :ode
        u0_init = [exp(model_cfg.pguess[3]), exp(model_cfg.pguess[4]), 0.0]
        problem = ODEProblem(troponin_ode!, u0_init, (0.0, base_patient.timepoints[end] + 10.0))
        return (kind=:ode, problem=problem)
    end

    error("Unsupported truncation carrier kind: $(model_cfg.model_key)")
end

"""
    systematic_truncation_patient_loss(theta, carrier, patient; lambda_back)

Evaluate the legacy patient loss for ODE or fixed-NN cUDE truncation fits.
"""
# Used by: src/systematic_truncation.jl.
function systematic_truncation_patient_loss(θ, carrier, patient::PatientData; lambda_back::Float64=1.0)
    if carrier.kind === :cude
        return patient_loss(θ, (carrier.model, patient.timepoints, patient.ctnt_data, carrier.fixed_nn); λ_back=lambda_back)
    elseif carrier.kind === :ode
        return patient_loss_formula(θ, (carrier.problem, patient.timepoints, patient.ctnt_data); λ_back=lambda_back)
    end

    error("Unsupported carrier kind: $(carrier.kind)")
end

# =============================================================================
# Scenario Generation
# =============================================================================

struct TruncationScenario
    patient::PatientData
    kept_idx::Vector{Int}
    removed_idx::Vector{Int}
    removed_start::Int
    removed_gap::Int
    removed_end::Int
    budget::Int
    trunc_section::String
    trunc_set::Int
end

"""
    init_trunc_meta_df()

Create the canonical step 03c truncation metadata table.
"""
# Used by: src/systematic_truncation.jl.
function init_trunc_meta_df()
    return DataFrame(
        base_patient=String[],
        synthetic_id=String[],
        trunc_section=String[],
        trunc_set=Int[],
        budget=Int[],
        kept_n=Int[],
        removed_n=Int[],
        removed_start=Int[],
        removed_gap=Int[],
        removed_end=Int[],
    )
end

"""
    init_trunc_metrics_df()

Create the canonical step 03c truncation metric table.
"""
# Used by: src/systematic_truncation.jl.
function init_trunc_metrics_df()
    return DataFrame(
        base_patient=String[],
        synthetic_id=String[],
        trunc_section=String[],
        trunc_set=Int[],
        removed_n=Int[],
        removed_start=Int[],
        removed_gap=Int[],
        removed_end=Int[],
        kept_n=Int[],
        loss=Float64[],
        smape_full=Float64[],
        rmsle_full=Float64[],
        smape_sparse=Float64[],
        rmsle_sparse=Float64[],
    )
end

"""
    init_trunc_params_df(param_labels)

Create the canonical step 03c truncation parameter table.
"""
# Used by: src/systematic_truncation.jl.
function init_trunc_params_df(param_labels::Vector{String})
    df = DataFrame(
        base_patient=String[],
        synthetic_id=String[],
        trunc_section=String[],
        trunc_set=Int[],
    )

    for label in param_labels
        df[!, Symbol(label)] = Float64[]
        df[!, Symbol("$(label)_ratio_vs_full")] = Float64[]
    end

    return df
end

"""
    push_trunc_param_row!(df, ...)

Append one natural-scale parameter row and ratios versus the full-data fit.
"""
# Used by: src/systematic_truncation.jl.
function push_trunc_param_row!(
    trunc_params::DataFrame,
    base_patient::String,
    synthetic_id::String,
    trunc_section::String,
    trunc_set::Int,
    par::Vector{Float64},
    full_par::Vector{Float64},
    param_labels::Vector{String},
)
    row = Dict{Symbol,Any}(
        :base_patient => base_patient,
        :synthetic_id => synthetic_id,
        :trunc_section => trunc_section,
        :trunc_set => trunc_set,
    )

    for (k, label) in enumerate(param_labels)
        row[Symbol(label)] = par[k]
        row[Symbol("$(label)_ratio_vs_full")] = par[k] / full_par[k]
    end

    push!(trunc_params, row)
    return trunc_params
end

"""
    truncation_budgets(n_obs; min_keep, levels)

Compute the two legacy truncation budgets for one patient.
"""
# Used by: src/systematic_truncation.jl.
function truncation_budgets(n_obs::Int; min_keep::Int, levels)
    max_remove = n_obs - min_keep
    max_remove >= 1 || return Int[]

    raw = [clamp(round(Int, level * max_remove), 1, max_remove) for level in levels]
    budgets = sort(unique(raw))

    if length(budgets) == 1
        max_remove >= 2 ||
            error("Patient with n_obs=$(n_obs) cannot generate 2 truncation sizes with min_keep=$(min_keep).")
        budgets = [budgets[1], min(max_remove, budgets[1] + 1)]
    end

    length(budgets) > 2 && (budgets = [first(budgets), last(budgets)])
    length(budgets) == 2 || error("Expected exactly 2 truncation budgets, got $(budgets).")
    return budgets
end

"""
    section_truncation_indices(n_obs, budget, section)

Return kept and removed indices for one start/middle/end truncation scenario.
"""
# Used by: src/systematic_truncation.jl.
function section_truncation_indices(n_obs::Int, budget::Int, section::Symbol)
    budget = clamp(budget, 1, n_obs - 1)

    if section === :start
        removed = collect(1:budget)
        removed_start, removed_gap, removed_end = budget, 0, 0
    elseif section === :end
        removed = collect(n_obs-budget+1:n_obs)
        removed_start, removed_gap, removed_end = 0, 0, budget
    elseif section === :middle
        lo_raw = round(Int, (n_obs - budget) / 2) + 1
        lo = clamp(lo_raw, 2, max(2, n_obs - budget))
        hi = lo + budget - 1

        if hi > n_obs - 1
            hi = n_obs - 1
            lo = hi - budget + 1
        end

        removed = collect(lo:hi)
        removed_start, removed_gap, removed_end = 0, budget, 0
    else
        error("Unsupported truncation section: $(section)")
    end

    kept = sort(setdiff(collect(1:n_obs), removed))
    return (
        kept=kept,
        removed=removed,
        removed_start=removed_start,
        removed_gap=removed_gap,
        removed_end=removed_end,
    )
end

"""
    generate_systematic_truncations(base_patient, base_patient_id; ...)

Generate the six legacy truncation scenarios for one base patient.
"""
# Used by: src/systematic_truncation.jl and test checks.
function generate_systematic_truncations(
    base_patient::PatientData,
    base_patient_id::String;
    min_keep::Int,
    levels,
    sections,
)
    n_obs = length(base_patient.timepoints)
    budgets = truncation_budgets(n_obs; min_keep=min_keep, levels=levels)

    isempty(budgets) && return TruncationScenario[], init_trunc_meta_df()

    scenarios = TruncationScenario[]
    trunc_meta = init_trunc_meta_df()

    for section in sections
        section_name = String(section)
        section_tag = section === :start ? "START" : section === :middle ? "MID" : "END"

        for (set_id, budget) in enumerate(budgets)
            idx = section_truncation_indices(n_obs, budget, section)
            length(idx.kept) >= min_keep ||
                error("Invalid truncation for patient $(base_patient.id): kept=$(length(idx.kept)) < min_keep=$(min_keep).")

            synthetic_id = "$(base_patient_id)_$(section_tag)_S$(set_id)_B$(lpad(string(budget), 2, "0"))"
            trunc_patient = PatientData(
                synthetic_id,
                base_patient.timepoints[idx.kept],
                base_patient.ctnt_data[idx.kept],
            )

            push!(scenarios, TruncationScenario(
                trunc_patient,
                idx.kept,
                idx.removed,
                idx.removed_start,
                idx.removed_gap,
                idx.removed_end,
                budget,
                section_name,
                set_id,
            ))

            push!(trunc_meta, (
                base_patient=base_patient.id,
                synthetic_id=synthetic_id,
                trunc_section=section_name,
                trunc_set=set_id,
                budget=budget,
                kept_n=length(idx.kept),
                removed_n=length(idx.removed),
                removed_start=idx.removed_start,
                removed_gap=idx.removed_gap,
                removed_end=idx.removed_end,
            ))
        end
    end

    length(scenarios) == 6 ||
        error("Expected 6 synthetic patients, got $(length(scenarios)) for $(base_patient.id).")

    return scenarios, trunc_meta
end

# =============================================================================
# Fitting
# =============================================================================

"""
    fit_truncation_patient_multistart(carrier, patient, model_cfg, settings; rng)

Fit one full or truncated patient using the legacy step 03c multi-start setup.
"""
# Used by: src/systematic_truncation.jl.
function fit_truncation_patient_multistart(
    carrier,
    patient::PatientData,
    model_cfg,
    settings;
    rng::AbstractRNG,
)
    loss_fun = θ -> systematic_truncation_patient_loss(
        θ,
        carrier,
        patient;
        lambda_back=model_cfg.lambda_back,
    )

    best_result, _ = run_multistart(
        loss_fun,
        settings.n_multistart;
        lower=model_cfg.lhs_lb,
        upper=model_cfg.lhs_ub,
        rng=rng,
        verbose=true,
        maxiters=settings.maxiters,
        maxtime=Float64(settings.maxtime),
        prescreen=false,
        topk=settings.topk,
        show_progress=settings.progress_bars,
    )

    best_result === nothing && error("Multi-start returned nothing for patient $(patient.id)")
    all(isfinite, best_result.u) || error("Non-finite parameters from multi-start for patient $(patient.id)")
    isfinite(best_result.minimum) || error("Non-finite objective from multi-start for patient $(patient.id)")

    return Vector{Float64}(best_result.u), Float64(best_result.minimum)
end

"""
    predict_truncation_plasma(carrier, theta, saveat)

Return plasma predictions at requested timepoints for a truncation carrier.
"""
# Used by: src/systematic_truncation.jl and overlay helpers.
function predict_truncation_plasma(carrier, θ_opt::Vector{Float64}, saveat)
    if carrier.kind === :cude
        p_opt = ComponentArray(ode=θ_opt, neural=carrier.fixed_nn)
        prob = remake(carrier.model.problem; u0=initial_conditions_from_log_params(θ_opt), p=p_opt)
        sol = solve(prob, Tsit5(); p=p_opt, saveat=saveat)
    elseif carrier.kind === :ode
        prob = remake(carrier.problem; u0=initial_conditions_from_log_params(θ_opt), p=θ_opt)
        sol = solve(prob, Tsit5(); p=θ_opt, saveat=saveat)
    else
        error("Unsupported carrier kind: $(carrier.kind)")
    end

    successful_retcode(sol) || error("Prediction solve failed with retcode=$(sol.retcode)")
    plasma = vec(sol[3, :])
    all(isfinite, plasma) || error("Non-finite plasma prediction.")
    return plasma
end

"""
    solve_truncation_curve(carrier, theta; saveat=1.0)

Return `(time, plasma)` for plotting a fitted truncation model.
"""
# Used by: src/systematic_truncation.jl and src/plotting.jl.
function solve_truncation_curve(carrier, θ_opt::Vector{Float64}; saveat::Real=1.0)
    if carrier.kind === :cude
        p_opt = ComponentArray(ode=θ_opt, neural=carrier.fixed_nn)
        prob = remake(carrier.model.problem; u0=initial_conditions_from_log_params(θ_opt), p=p_opt)
        sol = solve(prob, Tsit5(); p=p_opt, saveat=saveat)
    elseif carrier.kind === :ode
        prob = remake(carrier.problem; u0=initial_conditions_from_log_params(θ_opt), p=θ_opt)
        sol = solve(prob, Tsit5(); p=θ_opt, saveat=saveat)
    else
        error("Unsupported carrier kind: $(carrier.kind)")
    end

    successful_retcode(sol) || error("Curve solve failed with retcode=$(sol.retcode)")
    plasma = vec(sol[3, :])
    all(isfinite, plasma) || error("Non-finite plasma curve.")
    return collect(sol.t), plasma
end

"""
    fit_systematic_truncation_patient(base_patient, base_idx, model_cfg, settings)

Run the full-data fit and all truncation scenario fits for one base patient.
"""
# Used by: src/systematic_truncation.jl (run_systematic_truncation_target).
function fit_systematic_truncation_patient(
    base_patient::PatientData,
    base_idx::Int,
    model_cfg,
    settings,
)
    base_carrier = build_systematic_truncation_carrier(base_patient, model_cfg)

    θ_full, full_loss = fit_truncation_patient_multistart(
        base_carrier,
        base_patient,
        model_cfg,
        settings;
        rng=StableRNG(10_000 + base_idx),
    )
    full_par = exp.(θ_full)

    scenarios, trunc_meta = generate_systematic_truncations(
        base_patient,
        base_patient.id;
        min_keep=settings.min_keep_meas,
        levels=settings.truncation_levels,
        sections=settings.truncation_sections,
    )

    trunc_metrics = init_trunc_metrics_df()
    trunc_params = init_trunc_params_df(model_cfg.param_labels)
    successful_patients = PatientData[]
    validation_params = Vector{Vector{Float64}}()
    plot_records = Vector{Any}()

    progress = settings.progress_bars ?
               Progress(length(scenarios); desc="Validating truncations $(base_patient.id)", color=:cyan, showspeed=true) :
               nothing

    for (scenario_idx, scenario) in enumerate(scenarios)
        patient = scenario.patient

        θ_opt, best_objective = fit_truncation_patient_multistart(
            base_carrier,
            patient,
            model_cfg,
            settings;
            rng=StableRNG(100_000 * base_idx + scenario_idx),
        )

        pred_full = predict_truncation_plasma(base_carrier, θ_opt, base_patient.timepoints)
        pred_sparse = predict_truncation_plasma(base_carrier, θ_opt, patient.timepoints)
        curve_t, curve_plasma = solve_truncation_curve(base_carrier, θ_opt; saveat=1.0)

        smape_full = smape(pred_full, base_patient.ctnt_data)
        rmsle_full = rmsle(base_patient.ctnt_data, pred_full)
        smape_sparse = smape(pred_sparse, patient.ctnt_data)
        rmsle_sparse = rmsle(patient.ctnt_data, pred_sparse)

        all(isfinite, [best_objective, smape_full, rmsle_full, smape_sparse, rmsle_sparse]) ||
            error("Non-finite metrics for synthetic patient $(patient.id)")

        push!(trunc_metrics, (
            base_patient=base_patient.id,
            synthetic_id=patient.id,
            trunc_section=scenario.trunc_section,
            trunc_set=scenario.trunc_set,
            removed_n=length(scenario.removed_idx),
            removed_start=scenario.removed_start,
            removed_gap=scenario.removed_gap,
            removed_end=scenario.removed_end,
            kept_n=length(scenario.kept_idx),
            loss=best_objective,
            smape_full=smape_full,
            rmsle_full=rmsle_full,
            smape_sparse=smape_sparse,
            rmsle_sparse=rmsle_sparse,
        ))

        par = exp.(θ_opt)
        push_trunc_param_row!(
            trunc_params,
            base_patient.id,
            patient.id,
            scenario.trunc_section,
            scenario.trunc_set,
            par,
            full_par,
            model_cfg.param_labels,
        )

        push!(successful_patients, patient)
        push!(validation_params, θ_opt)
        push!(plot_records, (
            scenario=scenario,
            curve_t=curve_t,
            curve_plasma=curve_plasma,
        ))

        progress !== nothing && next!(progress)
    end

    progress !== nothing && finish!(progress)
    nrow(trunc_metrics) > 0 || error("No successful truncation fit for base patient $(base_patient.id)")

    return (
        full_loss=full_loss,
        trunc_meta=trunc_meta,
        trunc_metrics=trunc_metrics,
        trunc_params=trunc_params,
        synthetic_patients=[scenario.patient for scenario in scenarios],
        successful_patients=successful_patients,
        validation_params=validation_params,
        plot_records=plot_records,
    )
end

"""
    run_systematic_truncation_target(patients, patient_dataset, model_cfg, settings, paths; callbacks...)

Run step 03c for one model target and write all model-level CSV artifacts.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function run_systematic_truncation_target(
    gold_patients::AbstractVector{PatientData},
    patient_dataset::AbstractDict{String,String},
    model_cfg,
    settings,
    model_paths;
    initial_plot_callback=nothing,
    scenario_plot_callback=nothing,
    parameter_boxplot_callback=nothing,
)
    mkpath(model_paths.model_dir)

    trunc_meta_all = init_trunc_meta_df()
    trunc_metrics_all = init_trunc_metrics_df()
    trunc_params_all = init_trunc_params_df(model_cfg.param_labels)

    for base_idx in eachindex(gold_patients)
        base_patient = gold_patients[base_idx]
        haskey(patient_dataset, base_patient.id) ||
            error("Missing dataset mapping for patient $(base_patient.id)")

        @info "Starting $(model_cfg.model_name) truncation patient $(base_patient.id)" measurements=length(base_patient.timepoints)
        patient_paths = systematic_truncation_patient_output_paths(model_paths.model_dir, base_patient.id)
        mkpath(patient_paths.patient_dir)

        if initial_plot_callback !== nothing
            initial_plot_callback(base_patient, patient_paths.initial_scatter, model_cfg)
        end

        patient_result = fit_systematic_truncation_patient(base_patient, base_idx, model_cfg, settings)
        patient_df = fromPatientData2DataFrame(patient_result.synthetic_patients)

        save_systematic_truncation_patient_outputs(
            patient_paths;
            patient_dataframe=patient_df,
            meta=patient_result.trunc_meta,
            metrics=patient_result.trunc_metrics,
            params=patient_result.trunc_params,
        )

        append!(trunc_meta_all, patient_result.trunc_meta)
        append!(trunc_metrics_all, patient_result.trunc_metrics)
        append!(trunc_params_all, patient_result.trunc_params)

        if scenario_plot_callback !== nothing
            for record in patient_result.plot_records
                scenario_plot_callback(
                    base_patient,
                    record.scenario,
                    record.curve_t,
                    record.curve_plasma,
                    joinpath(patient_paths.patient_dir, "patient_$(record.scenario.patient.id).svg"),
                    model_cfg,
                )
            end
        end

        if parameter_boxplot_callback !== nothing && !isempty(patient_result.validation_params)
            parameter_boxplot_callback(
                patient_result.successful_patients,
                patient_result.validation_params,
                patient_paths.parameter_boxplot,
                model_cfg,
                patient_dataset[base_patient.id],
                base_patient.id,
            )
        end

        @info "Completed $(model_cfg.model_name) truncation patient $(base_patient.id)" scenarios=nrow(patient_result.trunc_metrics)
    end

    summaries = systematic_truncation_model_summaries(trunc_metrics_all, trunc_params_all, model_cfg.param_labels)
    save_systematic_truncation_model_outputs(
        model_paths;
        meta_all=trunc_meta_all,
        metrics_all=trunc_metrics_all,
        params_all=trunc_params_all,
        patient_summary=summaries.patient_summary,
        section_summary=summaries.section_summary,
        param_summary=summaries.param_summary,
    )

    return (
        meta_all=trunc_meta_all,
        metrics_all=trunc_metrics_all,
        params_all=trunc_params_all,
        summaries=summaries,
    )
end

# =============================================================================
# Tables
# =============================================================================

"""
    systematic_truncation_model_summaries(metrics, params, param_labels)

Build aggregate patient, section, and parameter summaries for one model target.
"""
# Used by: src/systematic_truncation.jl.
function systematic_truncation_model_summaries(metrics::DataFrame, params::DataFrame, param_labels::Vector{String})
    patient_summary = nrow(metrics) == 0 ? DataFrame() : combine(
        groupby(metrics, :base_patient),
        :removed_n => mean => :removed_n_mean,
        :smape_full => median => :smape_full_median,
        :rmsle_full => median => :rmsle_full_median,
        :loss => median => :loss_median,
    )

    section_summary = nrow(metrics) == 0 ? DataFrame() : combine(
        groupby(metrics, :trunc_section),
        :smape_full => median => :smape_full_median,
        :rmsle_full => median => :rmsle_full_median,
        :loss => median => :loss_median,
    )

    param_summary = DataFrame(param=String[], median=Float64[], q1=Float64[], q3=Float64[])
    if nrow(params) > 0
        for label in param_labels
            vals = params[!, Symbol(label)]
            push!(param_summary, (
                param=label,
                median=median(vals),
                q1=quantile(vals, 0.25),
                q3=quantile(vals, 0.75),
            ))
        end
    end

    return (patient_summary=patient_summary, section_summary=section_summary, param_summary=param_summary)
end

scenario_label(section, setnum) = string(uppercasefirst(string(section)), " (S", setnum, ")")

function median_iqr_string(values; digits::Int=2)
    v = collect(skipmissing(values))
    isempty(v) && return "NaN [NaN, NaN]"
    med = median(v)
    q1 = quantile(v, 0.25)
    q3 = quantile(v, 0.75)
    return string(round(med, digits=digits), " [", round(q1, digits=digits), ", ", round(q3, digits=digits), "]")
end

"""
    build_truncation_metrics_summary(ode_metrics, cude_metrics)

Build the integrated ODE/cUDE truncation metric summary table.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function build_truncation_metrics_summary(ode_metrics::DataFrame, cude_metrics::DataFrame)
    rows = DataFrame(
        Scenario=String[],
        ODE_sMAPE_full=String[],
        ODE_RMSLE_full=String[],
        cUDE_sMAPE_full=String[],
        cUDE_RMSLE_full=String[],
    )

    for section in ["start", "middle", "end"]
        for set_id in 1:2
            ode_sub = filter(row -> row.trunc_section == section && row.trunc_set == set_id, ode_metrics)
            cude_sub = filter(row -> row.trunc_section == section && row.trunc_set == set_id, cude_metrics)

            push!(rows, (
                scenario_label(section, set_id),
                median_iqr_string(ode_sub.smape_full),
                median_iqr_string(ode_sub.rmsle_full),
                median_iqr_string(cude_sub.smape_full),
                median_iqr_string(cude_sub.rmsle_full),
            ))
        end
    end

    return rows
end

"""
    build_truncation_params_summary(ode_params, cude_params)

Build the integrated ODE/cUDE truncation parameter-ratio summary table.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function build_truncation_params_summary(ode_params::DataFrame, cude_params::DataFrame)
    rows = DataFrame(
        Model=String[],
        Scenario=String[],
        a=String[],
        b=String[],
        Cs0=String[],
        Cc0=String[],
        Modulation_parameter=String[],
    )

    for section in ["start", "middle", "end"]
        for set_id in 1:2
            sub = filter(row -> row.trunc_section == section && row.trunc_set == set_id, ode_params)
            push!(rows, (
                "ODE",
                scenario_label(section, set_id),
                median_iqr_string(sub.a_ratio_vs_full),
                median_iqr_string(sub.b_ratio_vs_full),
                median_iqr_string(sub.Cs0_ratio_vs_full),
                median_iqr_string(sub.Cc0_ratio_vs_full),
                "Td: " * median_iqr_string(sub.Td_ratio_vs_full),
            ))
        end
    end

    for section in ["start", "middle", "end"]
        for set_id in 1:2
            sub = filter(row -> row.trunc_section == section && row.trunc_set == set_id, cude_params)
            push!(rows, (
                "cUDE",
                scenario_label(section, set_id),
                median_iqr_string(sub.a_ratio_vs_full),
                median_iqr_string(sub.b_ratio_vs_full),
                median_iqr_string(sub.Cs0_ratio_vs_full),
                median_iqr_string(sub.Cc0_ratio_vs_full),
                "β: " * median_iqr_string(sub.beta_ratio_vs_full),
            ))
        end
    end

    return rows
end

# =============================================================================
# Overlay Inputs
# =============================================================================

"""
    reconstruct_base_patient_from_truncation_df(df_trunc)

Reconstruct full base-patient observations from long-format truncation data.
"""
# Used by: src/systematic_truncation.jl.
function reconstruct_base_patient_from_truncation_df(df_trunc::DataFrame)
    all_times = sort(unique(Float64.(df_trunc.time)))
    troponin = Float64[]

    for time in all_times
        rows = df_trunc[df_trunc.time .== time, :]
        push!(troponin, Float64(rows.troponin[1]))
    end

    return all_times, troponin
end

"""
    identify_truncation_kept_removed(base_times, trunc_times)

Return kept and removed base-observation indices for an overlay scenario.
"""
# Used by: src/systematic_truncation.jl.
function identify_truncation_kept_removed(base_times::Vector{Float64}, trunc_times)
    kept = Int[]
    trunc_times_float = Float64.(trunc_times)

    for (idx, time) in enumerate(base_times)
        if any(abs.(trunc_times_float .- time) .< 1e-6)
            push!(kept, idx)
        end
    end

    removed = collect(setdiff(1:length(base_times), kept))
    return kept, removed
end

"""
    regenerate_systematic_truncation_model_plots(gold_patients, patient_dataset, model_cfg, settings, model_paths; callbacks...)

Regenerate step 03c patient-level plots from existing per-patient parameter CSVs
without rerunning any truncation optimization.
"""
# Used by: scripts/03c_run_systematic_truncation.jl (`plots` target).
function regenerate_systematic_truncation_model_plots(
    gold_patients::AbstractVector{PatientData},
    patient_dataset::AbstractDict{String,String},
    model_cfg,
    settings,
    model_paths;
    initial_plot_callback=nothing,
    scenario_plot_callback=nothing,
    parameter_boxplot_callback=nothing,
)
    for base_patient in gold_patients
        patient_paths = systematic_truncation_patient_output_paths(model_paths.model_dir, base_patient.id)
        validate_existing_paths(
            (
                params=patient_paths.params,
                metrics=patient_paths.metrics,
            );
            header="Required truncation plot-only files for $(model_cfg.model_name) patient $(base_patient.id)",
        )

        params_df = CSV.read(patient_paths.params, DataFrame)
        scenarios, _ = generate_systematic_truncations(
            base_patient,
            base_patient.id;
            min_keep=settings.min_keep_meas,
            levels=settings.truncation_levels,
            sections=settings.truncation_sections,
        )
        carrier = build_systematic_truncation_carrier(base_patient, model_cfg)

        if initial_plot_callback !== nothing
            initial_plot_callback(base_patient, patient_paths.initial_scatter, model_cfg)
        end

        successful_patients = PatientData[]
        validation_params = Vector{Vector{Float64}}()

        for scenario in scenarios
            match = params_df[params_df.synthetic_id .== scenario.patient.id, :]
            nrow(match) == 0 && continue
            nrow(match) == 1 ||
                error("Expected one saved parameter row for $(scenario.patient.id), got $(nrow(match)).")

            θ_opt = log.([Float64(match[1, Symbol(label)]) for label in model_cfg.param_labels])
            curve_t, curve_plasma = solve_truncation_curve(carrier, θ_opt; saveat=1.0)

            if scenario_plot_callback !== nothing
                scenario_plot_callback(
                    base_patient,
                    scenario,
                    curve_t,
                    curve_plasma,
                    joinpath(patient_paths.patient_dir, "patient_$(scenario.patient.id).svg"),
                    model_cfg,
                )
            end

            push!(successful_patients, scenario.patient)
            push!(validation_params, θ_opt)
        end

        if parameter_boxplot_callback !== nothing && !isempty(validation_params)
            haskey(patient_dataset, base_patient.id) ||
                error("Missing dataset mapping for patient $(base_patient.id)")
            parameter_boxplot_callback(
                successful_patients,
                validation_params,
                patient_paths.parameter_boxplot,
                model_cfg,
                patient_dataset[base_patient.id],
                base_patient.id,
            )
        end

        @info "Regenerated $(model_cfg.model_name) truncation plots for $(base_patient.id)." scenarios=length(validation_params)
    end

    return model_paths.model_dir
end

function _overlay_section_sort_key(section)
    order = Dict("start" => 1, "middle" => 2, "end" => 3)
    return get(order, string(section), 99)
end

function solve_overlay_ode_curve(params_natural::Vector{Float64}; tmax::Float64)
    θ = log.(params_natural)
    problem = ODEProblem(troponin_ode!, [params_natural[3], params_natural[4], 0.0], (0.0, tmax), θ)
    sol = solve(problem, Tsit5(); saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || error("ODE overlay solve failed with retcode=$(sol.retcode)")
    return collect(sol.t), vec(sol[3, :])
end

function solve_overlay_cude_curve(params_natural::Vector{Float64}, chain, nn_params::Vector{Float64}; tmax::Float64)
    θ = log.(params_natural)
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
    problem = ODEProblem(cude!, [params_natural[3], params_natural[4], 0.0], (0.0, tmax))
    p_full = ComponentArray(ode=θ, neural=nn_params)
    sol = solve(problem, Tsit5(); p=p_full, saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || error("cUDE overlay solve failed with retcode=$(sol.retcode)")
    return collect(sol.t), vec(sol[3, :])
end

"""
    build_truncation_overlay_records(ode_patient_dir, cude_patient_dir, chain, nn_params)

Build deterministic plot records for one patient's ODE-vs-cUDE overlay figures.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function build_truncation_overlay_records(
    ode_patient_dir::AbstractString,
    cude_patient_dir::AbstractString,
    chain,
    nn_params::Vector{Float64},
)
    patient_id = basename(ode_patient_dir)
    ode_params_df = CSV.read(joinpath(ode_patient_dir, "trunc_params.csv"), DataFrame)
    cude_params_df = CSV.read(joinpath(cude_patient_dir, "trunc_params.csv"), DataFrame)
    ode_metrics_df = CSV.read(joinpath(ode_patient_dir, "trunc_metrics.csv"), DataFrame)
    cude_metrics_df = CSV.read(joinpath(cude_patient_dir, "trunc_metrics.csv"), DataFrame)

    df_path = joinpath(ode_patient_dir, "df_$(patient_id).csv")
    isfile(df_path) || (df_path = joinpath(cude_patient_dir, "df_$(patient_id).csv"))
    isfile(df_path) || error("Missing truncation data CSV for overlay patient $(patient_id)")
    df_trunc = CSV.read(df_path, DataFrame)

    base_times, base_troponin = reconstruct_base_patient_from_truncation_df(df_trunc)
    tmax_solve = base_times[end] + 20.0

    keys = unique(vcat(
        [(string(row.trunc_section), Int(row.trunc_set)) for row in eachrow(ode_params_df)],
        [(string(row.trunc_section), Int(row.trunc_set)) for row in eachrow(cude_params_df)],
    ))
    sort!(keys; by=key -> (_overlay_section_sort_key(key[1]), key[2]))

    records = Vector{Any}()

    for (section, set_id) in keys
        ode_match = filter(row -> row.trunc_section == section && row.trunc_set == set_id, ode_params_df)
        cude_match = filter(row -> row.trunc_section == section && row.trunc_set == set_id, cude_params_df)
        nrow(ode_match) == 0 && continue
        nrow(cude_match) == 0 && continue

        ode_row = ode_match[1, :]
        cude_row = cude_match[1, :]
        ode_met = filter(row -> row.trunc_section == section && row.trunc_set == set_id, ode_metrics_df)
        cude_met = filter(row -> row.trunc_section == section && row.trunc_set == set_id, cude_metrics_df)

        ode_pars = Float64[ode_row.a, ode_row.b, ode_row.Cs0, ode_row.Cc0, ode_row.Td]
        cude_pars = Float64[cude_row.a, cude_row.b, cude_row.Cs0, cude_row.Cc0, cude_row.beta]

        ode_t, ode_plasma = solve_overlay_ode_curve(ode_pars; tmax=tmax_solve)
        cude_t, cude_plasma = solve_overlay_cude_curve(cude_pars, chain, nn_params; tmax=tmax_solve)

        trunc_rows = filter(row -> row.patient_id == string(ode_row.synthetic_id), df_trunc)
        kept_idx, removed_idx = identify_truncation_kept_removed(base_times, trunc_rows.time)

        push!(records, (
            patient_id=patient_id,
            section=section,
            set_id=set_id,
            base_times=base_times,
            base_troponin=base_troponin,
            kept_idx=kept_idx,
            removed_idx=removed_idx,
            ode_t=ode_t,
            ode_plasma=ode_plasma,
            cude_t=cude_t,
            cude_plasma=cude_plasma,
            ode_smape=nrow(ode_met) > 0 ? round(ode_met[1, :smape_full]; digits=2) : NaN,
            ode_rmsle=nrow(ode_met) > 0 ? round(ode_met[1, :rmsle_full]; digits=4) : NaN,
            cude_smape=nrow(cude_met) > 0 ? round(cude_met[1, :smape_full]; digits=2) : NaN,
            cude_rmsle=nrow(cude_met) > 0 ? round(cude_met[1, :rmsle_full]; digits=4) : NaN,
        ))
    end

    return records
end

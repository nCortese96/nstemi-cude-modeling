"""
run_systematic_truncation.jl

Refactored copy of `systematic_truncation_multiple.jl`.

Run systematic truncation validation for cUDE or ODE model configurations.

Pipeline:
1. Configure run settings.
2. Resolve input/output paths.
3. Load required data and model artifacts.
4. Run the main computation.
5. Save metrics, parameters, plots, and logs.

This copy uses `MechanisticAI.jl` as the shared helper entrypoint. The original
script is intentionally left untouched as the legacy baseline.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Shared dependencies and the central refactor entrypoint.
# =============================================================================
using DataFrames, XLSX, CSV
using Logging
using Plots, JLD2
using Random, StableRNGs
using ProgressMeter, Statistics, StatsBase
using CairoMakie
include("MechanisticAI.jl")
using .MultiStartOptimizer

# =============================================================================
# SCRIPT SETTINGS
# User-editable dataset/model/optimizer flags are preserved from the original
# script in the first executable block below.
# =============================================================================

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run are resolved near the settings that define
# dataset/model identity. Keep load paths explicit during this transition pass.
# =============================================================================

# =============================================================================
# OUTPUT PATHS
# Result directories and output files are created by the preserved pipeline below.
# Future cleanup should move path construction into `build_experiment_paths`.
# =============================================================================

# =============================================================================
# DERIVED SETTINGS
# Values computed from the settings above are kept inline for behavior parity.
# Future cleanup should collect them before the pipeline starts.
# =============================================================================

# =============================================================================
# HELPERS
# Script-local helper functions remain near their original location for now.
# Reusable candidates should migrate to helpers.jl after behavior is validated.
# =============================================================================

# =============================================================================
# PIPELINE
# Main execution flow copied from the original script. This first refactor pass
# changes includes and documentation only; numerical behavior is preserved.
# =============================================================================
# =============================================================================
# Systematic Truncation Validation Pipeline (Model-Agnostic)
#
# MODEL_ID = 1 -> cUDE with fixed NN params + patient_loss
# MODEL_ID = 2 -> ODE (troponin_ode!) + patient_loss_formula
# =============================================================================

# ---------------------- General truncation settings ---------------------------
const MIN_KEEP_MEAS = 4
const TRUNCATION_LEVELS = [0.35, 0.70]
const TRUNCATION_SECTIONS = (:start, :middle, :end)

const MS_N = 40
const MS_TOPK = 8
const MS_MAXITERS = 5000
const MS_MAXTIME = 80.0

# ---------------------- Model selection block ---------------------------------
# 1 = cUDE + fixed NN params
# 2 = ODE troponin_ode! (a,b,Cs0,Cc0,Td)
const MODEL_ID = 2
const N_params = 5

T_SCALE = 240.0
plotting = true
λ_back = 1.0

best_idx = 3                     # used only by MODEL_ID=1
run_dataset_name = "MIMIC-UMG"

# cUDE defaults
const CUDE_INPUT_DIM = 2
const CUDE_NN_DEPTH = 2
const CUDE_NN_WIDTH = 8
const CUDE_EXPERIMENT = "NSTEMI_cUDE_MIMIC-IV_MSE_2$(CUDE_NN_WIDTH)_sigmoid_regback"

# ODE defaults
const ODE_EXPERIMENT = "NSTEMI_ODE_TdSigmoid"

function choose_experiment(model_id::Int)
    if model_id == 1
        return CUDE_EXPERIMENT
    elseif model_id == 2
        return ODE_EXPERIMENT
    else
        error("Unsupported MODEL_ID=$(model_id). Use 1 (cUDE) or 2 (ODE).")
    end
end

function sanitize_tag(s::String)
    lowercase(replace(s, r"[^A-Za-z0-9]+" => "_"))
end

function configure_model_runtime(
    model_id::Int,
    experiment::String,
    models_path::String,
    best_idx::Int;
    λ_back::Float64=1.0
)
    if model_id == 1
        @info "MODEL_ID=1 selected: cUDE with fixed NN parameters"

        chain = neural_network_model(CUDE_NN_DEPTH, CUDE_NN_WIDTH; input_dims=CUDE_INPUT_DIM)

        nn_file = "$(models_path)/nnNSTEMI_$(experiment).jld2"
        isfile(nn_file) || error("Missing NN file for cUDE model: $(nn_file)")

        @load nn_file neural_network_parameters
        best_idx <= length(neural_network_parameters) ||
            error("best_idx=$(best_idx) out of range for loaded neural_network_parameters.")

        best_nn = Vector{Float64}(neural_network_parameters[best_idx])

        pguess = log.([0.005, 0.005, 0.1, 0.01, 0.5])
        lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001])
        lhs_ub = log.([10.0, 10.0, 500.0, 500.0, 1.0])

        return (
            model_id=model_id,
            model_name="cUDE",
            model_tag="cude_fixednn",
            curve_label="Model curve (cUDE, saveat=1)",
            loss_label="patient_loss",
            uses_fixed_nn=true,
            chain=chain,
            best_nn=best_nn,
            param_labels=["a", "b", "Cs0", "Cc0", "beta"],
            pguess=pguess,
            lhs_lb=lhs_lb,
            lhs_ub=lhs_ub,
            λ_back=λ_back
        )

    elseif model_id == 2
        @info "MODEL_ID=2 selected: ODE with sigmoid fraction and Td"

        pguess = log.([0.005, 0.005, 0.01, 0.01, 30.0]) # [a, b, Cs0, Cc0, Td]
        lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001])
        lhs_ub = log.([10.0, 10.0, 500.0, 500.0, 500.0])

        return (
            model_id=model_id,
            model_name="ODE_TdSigmoid",
            model_tag="ode_tdsigmoid",
            curve_label="Model curve (ODE, saveat=1)",
            loss_label="patient_loss_formula",
            uses_fixed_nn=false,
            chain=nothing,
            best_nn=nothing,
            param_labels=["a", "b", "Cs0", "Cc0", "Td"],
            pguess=pguess,
            lhs_lb=lhs_lb,
            lhs_ub=lhs_ub,
            λ_back=λ_back
        )
    else
        error("Unsupported MODEL_ID=$(model_id). Use 1 or 2.")
    end
end

function build_base_carrier(base_patient::PatientData, model_cfg)
    if model_cfg.model_id == 1
        model = ctntCUDEModel(model_cfg.pguess, model_cfg.chain, base_patient.timepoints)
        return (kind=:cude, model=model, fixed_nn=model_cfg.best_nn)
    elseif model_cfg.model_id == 2
        u0_init = [exp(model_cfg.pguess[3]), exp(model_cfg.pguess[4]), 0.0]
        tspan = (0.0, base_patient.timepoints[end] + 10.0)
        problem = ODEProblem(troponin_ode!, u0_init, tspan)
        return (kind=:ode, problem=problem)
    else
        error("Unknown model carrier for model_id=$(model_cfg.model_id)")
    end
end

function truncation_patient_loss(θ, carrier, patient::PatientData; λ_back::Float64=1.0)
    if carrier.kind === :cude
        return patient_loss(θ, (carrier.model, patient.timepoints, patient.ctnt_data, carrier.fixed_nn); λ_back=λ_back)
    elseif carrier.kind === :ode
        return patient_loss_formula(θ, (carrier.problem, patient.timepoints, patient.ctnt_data); λ_back=λ_back)
    else
        error("Unsupported carrier kind: $(carrier.kind)")
    end
end

"""
Fit patient parameters with strict multistart.
Works for both cUDE (fixed NN) and plain ODE.
"""
function fit_patient_multistart(
    carrier,
    patient::PatientData,
    lhs_lb::Vector{Float64},
    lhs_ub::Vector{Float64};
    λ_back::Float64=1.0,
    n_multistart::Int=MS_N,
    ms_topk::Int=MS_TOPK,
    ms_maxiters::Int=MS_MAXITERS,
    ms_maxtime::Float64=MS_MAXTIME,
    ms_rng::AbstractRNG=StableRNG(1234)
)
    loss_fun = θ -> truncation_patient_loss(θ, carrier, patient; λ_back=λ_back)

    best_result, _ = MultiStartOptimizer.run_multistart(
        loss_fun,
        n_multistart;
        lower=lhs_lb,
        upper=lhs_ub,
        rng=ms_rng,
        verbose=true,
        maxiters=ms_maxiters,
        maxtime=ms_maxtime,
        prescreen=false,
        topk=ms_topk
    )

    best_result === nothing && error("MultiStart returned nothing for patient $(patient.id)")
    all(isfinite, best_result.u) || error("Non-finite parameters from MultiStart for patient $(patient.id)")
    isfinite(best_result.minimum) || error("Non-finite objective from MultiStart for patient $(patient.id)")

    return Vector{Float64}(best_result.u), Float64(best_result.minimum)
end

function predict_plasma(carrier, θ_opt::Vector{Float64}, saveat::Vector{Float64})
    if carrier.kind === :cude
        p_opt = ComponentArray(ode=θ_opt, neural=carrier.fixed_nn)
        u0_new = [exp(θ_opt[3]), exp(θ_opt[4]), 0.0]
        prob = remake(carrier.model.problem; u0=u0_new, p=p_opt)

        pred = solve(prob, Tsit5(); p=p_opt, saveat=saveat)
        successful_retcode(pred) || error("ODE solve failed (cUDE) with retcode=$(pred.retcode)")

        plasma = vec(pred[3, :])
        all(isfinite, plasma) || error("Non-finite plasma prediction (cUDE).")
        return plasma

    elseif carrier.kind === :ode
        u0_new = [exp(θ_opt[3]), exp(θ_opt[4]), 0.0]
        prob = remake(carrier.problem; u0=u0_new, p=θ_opt)

        pred = solve(prob, Tsit5(); p=θ_opt, saveat=saveat)
        successful_retcode(pred) || error("ODE solve failed (ODE) with retcode=$(pred.retcode)")

        plasma = vec(pred[3, :])
        all(isfinite, plasma) || error("Non-finite plasma prediction (ODE).")
        return plasma
    else
        error("Unsupported carrier kind: $(carrier.kind)")
    end
end

"""
Solve full plasma trajectory for plotting with fixed saveat=1.
"""
function predict_plasma_curve(carrier, θ_opt::Vector{Float64}; saveat::Real=1.0)
    if carrier.kind === :cude
        p_opt = ComponentArray(ode=θ_opt, neural=carrier.fixed_nn)
        u0_new = [exp(θ_opt[3]), exp(θ_opt[4]), 0.0]
        prob = remake(carrier.model.problem; u0=u0_new, p=p_opt)

        sol = solve(prob, Tsit5(); p=p_opt, saveat=saveat)
        successful_retcode(sol) || error("ODE solve for plotting failed (cUDE) with retcode=$(sol.retcode)")

        curve_t = collect(sol.t)
        curve_plasma = vec(sol[3, :])
        all(isfinite, curve_plasma) || error("Non-finite plasma curve for plotting (cUDE).")

        return curve_t, curve_plasma

    elseif carrier.kind === :ode
        u0_new = [exp(θ_opt[3]), exp(θ_opt[4]), 0.0]
        prob = remake(carrier.problem; u0=u0_new, p=θ_opt)

        sol = solve(prob, Tsit5(); p=θ_opt, saveat=saveat)
        successful_retcode(sol) || error("ODE solve for plotting failed (ODE) with retcode=$(sol.retcode)")

        curve_t = collect(sol.t)
        curve_plasma = vec(sol[3, :])
        all(isfinite, curve_plasma) || error("Non-finite plasma curve for plotting (ODE).")

        return curve_t, curve_plasma
    else
        error("Unsupported carrier kind: $(carrier.kind)")
    end
end

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
Create an empty DataFrame for truncation metadata per synthetic scenario.
"""
function init_trunc_meta_df()
    DataFrame(
        base_patient=String[],
        synthetic_id=String[],
        trunc_section=String[],
        trunc_set=Int[],
        budget=Int[],
        kept_n=Int[],
        removed_n=Int[],
        removed_start=Int[],
        removed_gap=Int[],
        removed_end=Int[]
    )
end

"""
Create an empty DataFrame for fit quality metrics under truncation.
Includes full-timeline and sparse-timeline errors.
"""
function init_trunc_metrics_df()
    DataFrame(
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
        rmsle_sparse=Float64[]
    )
end

"""
Create an empty DataFrame for fitted parameters and ratios versus full-data fit.
Parameter columns are dynamic and depend on model_cfg.param_labels.
"""
function init_trunc_params_df(param_labels::Vector{String})
    df = DataFrame(
        base_patient=String[],
        synthetic_id=String[],
        trunc_section=String[],
        trunc_set=Int[]
    )

    for lbl in param_labels
        df[!, Symbol(lbl)] = Float64[]
        df[!, Symbol("$(lbl)_ratio_vs_full")] = Float64[]
    end

    return df
end

function push_trunc_param_row!(
    trunc_params::DataFrame,
    base_patient::String,
    synthetic_id::String,
    trunc_section::String,
    trunc_set::Int,
    par::Vector{Float64},
    full_par::Vector{Float64},
    param_labels::Vector{String}
)
    row = Dict{Symbol,Any}(
        :base_patient => base_patient,
        :synthetic_id => synthetic_id,
        :trunc_section => trunc_section,
        :trunc_set => trunc_set
    )

    for (k, lbl) in enumerate(param_labels)
        row[Symbol(lbl)] = par[k]
        row[Symbol("$(lbl)_ratio_vs_full")] = par[k] / full_par[k]
    end

    push!(trunc_params, row)
    return nothing
end

"""
Compute valid truncation budgets from the number of observations.
Budgets are derived from predefined levels and constrained by min_keep.
"""
function truncation_budgets(n_obs::Int; min_keep::Int=MIN_KEEP_MEAS, levels::Vector{Float64}=TRUNCATION_LEVELS)
    max_remove = n_obs - min_keep
    max_remove >= 1 || return Int[]

    raw = [clamp(round(Int, lv * max_remove), 1, max_remove) for lv in levels]
    budgets = sort(unique(raw))

    if length(budgets) == 1
        max_remove >= 2 || error(
            "Patient with n_obs=$(n_obs) cannot generate 2 truncation sizes with min_keep=$(min_keep)."
        )
        budgets = [budgets[1], min(max_remove, budgets[1] + 1)]
    end

    length(budgets) > 2 && (budgets = [first(budgets), last(budgets)])
    length(budgets) == 2 || error("Expected exactly 2 truncation budgets, got $(budgets).")

    return budgets
end

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
        error("Unsupported section $(section). Use :start, :middle, :end.")
    end

    kept = sort(setdiff(collect(1:n_obs), removed))
    return (
        kept=kept,
        removed=removed,
        removed_start=removed_start,
        removed_gap=removed_gap,
        removed_end=removed_end
    )
end

"""
Build structured truncation indices for a given budget.
Removals are distributed across start, central gap, and end segments.
"""
function mixed_truncation_indices(n_obs::Int, budget::Int)
    budget = clamp(budget, 1, n_obs - 1)

    n_start = max(1, floor(Int, 0.25 * budget))
    n_end = max(1, floor(Int, 0.25 * budget))
    n_gap = budget - n_start - n_end

    if n_gap < 1
        if n_start > n_end && n_start > 1
            n_start -= 1
        elseif n_end > 1
            n_end -= 1
        end
        n_gap = budget - n_start - n_end
    end
    n_gap = max(1, n_gap)

    start_idx = collect(1:n_start)
    end_idx = collect(n_obs-n_end+1:n_obs)

    mid_lo = n_start + 1
    mid_hi = n_obs - n_end
    gap_lo = clamp(round(Int, (mid_lo + mid_hi) / 2) - fld(n_gap, 2), mid_lo, mid_hi - n_gap + 1)
    gap_idx = collect(gap_lo:gap_lo+n_gap-1)

    removed = sort(unique(vcat(start_idx, end_idx, gap_idx)))
    kept = sort(setdiff(collect(1:n_obs), removed))
    return (kept=kept, removed=removed, start=start_idx, gap=gap_idx, finish=end_idx)
end

"""
Generate synthetic truncated patients and metadata from one base patient.
Each scenario stores kept and removed indices and removal composition.
"""
function generate_systematic_truncations(
    base_patient::PatientData,
    base_patient_id::String;
    min_keep::Int=MIN_KEEP_MEAS,
    levels::Vector{Float64}=TRUNCATION_LEVELS
)
    n_obs = length(base_patient.timepoints)
    budgets = truncation_budgets(n_obs; min_keep=min_keep, levels=levels)

    isempty(budgets) && return TruncationScenario[], init_trunc_meta_df()
    length(budgets) == 2 || error("Expected 2 truncation budgets for patient $(base_patient.id).")

    scenarios = TruncationScenario[]
    trunc_meta = init_trunc_meta_df()

    for section in TRUNCATION_SECTIONS
        section_name = String(section)
        section_tag = section === :start ? "START" : section === :middle ? "MID" : "END"

        for (set_id, b) in enumerate(budgets)
            idx = section_truncation_indices(n_obs, b, section)
            length(idx.kept) >= min_keep || error(
                "Invalid truncation for patient $(base_patient.id): kept=$(length(idx.kept)) < min_keep=$(min_keep)."
            )

            sid = "$(base_patient_id)_$(section_tag)_S$(set_id)_B$(lpad(string(b), 2, "0"))"
            trunc_patient = PatientData(sid, base_patient.timepoints[idx.kept], base_patient.ctnt_data[idx.kept])

            push!(scenarios, TruncationScenario(
                trunc_patient,
                idx.kept,
                idx.removed,
                idx.removed_start,
                idx.removed_gap,
                idx.removed_end,
                b,
                section_name,
                set_id
            ))

            push!(trunc_meta, (
                base_patient=base_patient.id,
                synthetic_id=sid,
                trunc_section=section_name,
                trunc_set=set_id,
                budget=b,
                kept_n=length(idx.kept),
                removed_n=length(idx.removed),
                removed_start=idx.removed_start,
                removed_gap=idx.removed_gap,
                removed_end=idx.removed_end
            ))
        end
    end

    length(scenarios) == 6 || error(
        "Expected 6 synthetic patients (3 sections x 2 sets), got $(length(scenarios)) for $(base_patient.id)."
    )

    return scenarios, trunc_meta
end

"""
Load one dataset, apply trimming and optional eligible-list filtering,
run high-information anomaly filtering, and return tagged gold patients.
"""
function load_gold_std_dataset(
    dataset_name::String,
    dataset_tag::String,
    dataset_path::String,
    column_letter::String;
    t_scale::Float64,
    apply_all_eligible::Bool=false,
    all_eligible_csv::Union{Nothing,String}=nothing,
    meas_min_number::Int,
    min_acq_time_before::Float64,
    min_acq_n_before::Int,
    min_acq_time_after::Float64,
    min_acq_n_after::Int,
    min_time::Float64,
    max_gap::Float64
)
    file_path = "data/$(dataset_path)"
    sheet_ids = "IDs"
    sheet_times = "times"
    sheet_values = "values"

    ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true))
    timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true))
    troponin_df = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true))

    patients = [row2Patient(ids[i, :], timepoints_df[i, :], troponin_df[i, :]) for i in 1:nrow(ids)]
    patients = trim_time(patients, t_scale)

    raw_n = length(patients)

    if apply_all_eligible
        all_eligible_csv === nothing && error("Missing all_eligible_csv for $(dataset_name)")
        isfile(all_eligible_csv) || error("Missing all eligible file for $(dataset_name): $(all_eligible_csv)")
        ae_ids = CSV.read(all_eligible_csv, DataFrame)
        all_eligible_idxs = findall(p -> p.id in ae_ids.patient, patients)
        patients = patients[all_eligible_idxs]
    end

    eligible_n = length(patients)

    anoms = find_anomalies(
        patients,
        meas_min_number,
        min_acq_time_before, min_acq_n_before,
        min_acq_time_after, min_acq_n_after,
        min_time;
        max_gap_h=max_gap,
        verbose=false
    )

    gold_std_patients = filter(p -> !haskey(anoms, p.id), patients)
    tagged_gold = [PatientData("$(dataset_tag)_$(p.id)", p.timepoints, p.ctnt_data) for p in gold_std_patients]

    @info "Dataset $(dataset_name): raw=$(raw_n), eligible=$(eligible_n), removed=$(length(anoms)), gold=$(length(tagged_gold))"

    return tagged_gold, raw_n, eligible_n, length(anoms)
end

"""
Plot one truncation fit against full timeline data.
Removed points are marked with X, used points with circles.
"""
function plot_truncation_fit(
    base_patient::PatientData,
    scenario::TruncationScenario,
    curve_t::Vector{Float64},
    curve_plasma::Vector{Float64},
    save_path::String;
    plotting::Bool=false,
    model_label::String="",
    curve_label::String="Model curve"
)
    ttl = isempty(model_label) ?
          "Base $(base_patient.id) - $(scenario.patient.id)" :
          "[$(model_label)] Base $(base_patient.id) - $(scenario.patient.id)"

    plt = Plots.plot(
        curve_t,
        curve_plasma;
        lw=2,
        label=curve_label,
        xlabel="Time (h)",
        ylabel="cTnT [ng/mL]",
        title=ttl
    )

    Plots.scatter!(
        plt,
        base_patient.timepoints[scenario.removed_idx],
        base_patient.ctnt_data[scenario.removed_idx];
        markershape=:x,
        markerstrokewidth=2,
        ms=7,
        color=:crimson,
        label="Removed measurements"
    )

    Plots.scatter!(
        plt,
        base_patient.timepoints[scenario.kept_idx],
        base_patient.ctnt_data[scenario.kept_idx];
        markershape=:circle,
        ms=5,
        color=:dodgerblue,
        label="Used measurements"
    )

    savefig(plt, save_path)
    plotting && display(plt)
    return plt
end

"""
Generic parameter extraction plot with model-specific labels.
Replaces hard-coded beta labeling in params_extraction.
"""
function params_extraction_generic(
    patients::Vector{PatientData},
    ode_params_val::Vector{Float64};
    param_labels::Vector{String},
    data_label::String="",
    dataset::String="",
    figsave_path::String="",
    show_outliers::Bool=false,
    savefigure::Bool=false
)
    n_params = length(param_labels)
    n_params == 5 || error("This workflow currently expects exactly 5 parameters.")

    values_by_param = [Float64[] for _ in 1:n_params]

    @showprogress desc = "$(data_label) params extraction..." for i in eachindex(patients)
        idx1 = n_params * (i - 1) + 1
        idx2 = n_params * i
        pars = exp.(ode_params_val[idx1:idx2])
        for k in 1:n_params
            push!(values_by_param[k], pars[k])
        end
    end

    for (k, lbl) in enumerate(param_labels)
        vals = values_by_param[k]
        @info "Average, STD in $(data_label) param $(lbl): $(mean(vals)) std: $(std(vals))"
        @info "Median [Q1-Q3] in $(data_label) param $(lbl): $(median(vals)) [$(quantile(vals, 0.25)) - $(quantile(vals, 0.75))]"
    end

    f = CairoMakie.Figure(size=(1400, 700))

    CairoMakie.Label(
        f[0, 1:length(param_labels)],
        "Parameter distributions — $(dataset) $(data_label)";
        fontsize=22,
        tellwidth=false
    )

    axes = CairoMakie.Axis[]
    for (i, lbl) in enumerate(param_labels)
        ax = CairoMakie.Axis(
            f[1, i],
            title=lbl,
            xticklabelsvisible=false,
            xticksvisible=false
        )
        push!(axes, ax)
    end

    x = fill(1, length(values_by_param[1]))
    my_colors = [:skyblue, :orange, :lightgreen, :pink, :violet]

    for (i, ax) in enumerate(axes)
        CairoMakie.boxplot!(
            ax,
            x,
            values_by_param[i];
            color=my_colors[mod1(i, length(my_colors))],
            whiskerwidth=0.3,
            strokewidth=0.3,
            show_outliers=show_outliers
        )
    end

    if savefigure
        CairoMakie.save("$(figsave_path)/boxplots_$(data_label).png", f)
    end

    return values_by_param, f
end

# ---------------------- Paths and runtime -------------------------------------
experiment = choose_experiment(MODEL_ID)

fig_path = "res/$(experiment)/figs"
models_path = "res/$(experiment)/models"
mkpath(fig_path)
mkpath(models_path)

model_cfg = configure_model_runtime(MODEL_ID, experiment, models_path, best_idx; λ_back=λ_back)
safe_model_tag = sanitize_tag(model_cfg.model_tag)

@assert N_params == 5 "This workflow is currently configured for exactly 5 parameters."
@assert length(model_cfg.pguess) == N_params "Invalid pguess length for selected model."
@assert length(model_cfg.lhs_lb) == N_params "Invalid lower bounds length for selected model."
@assert length(model_cfg.lhs_ub) == N_params "Invalid upper bounds length for selected model."

@info "Model selected: $(model_cfg.model_name)"
@info "Loss variant: $(model_cfg.loss_label)"
@info "Param labels: $(model_cfg.param_labels)"
@info "Initial pguess: $(exp.(model_cfg.pguess))"

figsave_path = "$(fig_path)/$(run_dataset_name)_systematic_truncation_$(safe_model_tag)"
modelssave_path = "$(models_path)/$(run_dataset_name)_systematic_truncation_$(safe_model_tag)"
mkpath(figsave_path)
mkpath(modelssave_path)

lhs_lb = model_cfg.lhs_lb
lhs_ub = model_cfg.lhs_ub
pguess = model_cfg.pguess

# ---------------------- Dataset filtering --------------------------------------
meas_min_number = 8
min_acq_time_before = 12.0
min_acq_n_before = 1
min_acq_time_after = 48.0
min_acq_n_after = 1
min_time = 72.0
max_gap = 24.0

mimic_gold, mimic_raw, mimic_eligible, mimic_removed = load_gold_std_dataset(
    "MIMIC-IV",
    "MIMIC",
    "MIMIC-IV/NSTEMI_reorganized_skipped.xlsx",
    "B";
    t_scale=T_SCALE,
    apply_all_eligible=true,
    all_eligible_csv="res/ids_all_eligible_MIMIC-IV_val.csv",
    meas_min_number=meas_min_number,
    min_acq_time_before=min_acq_time_before,
    min_acq_n_before=min_acq_n_before,
    min_acq_time_after=min_acq_time_after,
    min_acq_n_after=min_acq_n_after,
    min_time=min_time,
    max_gap=max_gap
)

umg_gold, umg_raw, umg_eligible, umg_removed = load_gold_std_dataset(
    "UMG",
    "UMG",
    "UMG_NSTEMI_Dataset.xlsx",
    "A";
    t_scale=T_SCALE,
    apply_all_eligible=false,
    all_eligible_csv=nothing,
    meas_min_number=meas_min_number,
    min_acq_time_before=min_acq_time_before,
    min_acq_n_before=min_acq_n_before,
    min_acq_time_after=min_acq_time_after,
    min_acq_n_after=min_acq_n_after,
    min_time=min_time,
    max_gap=max_gap
)

gold_std_patients = vcat(mimic_gold, umg_gold)
isempty(gold_std_patients) && error("No gold standard patients available after combined filtering.")
patient_dims(gold_std_patients)

patient_dataset = Dict{String,String}()
for p in mimic_gold
    patient_dataset[p.id] = "MIMIC-IV"
end
for p in umg_gold
    patient_dataset[p.id] = "UMG"
end

df_gold_ids = DataFrame(
    patient=[p.id for p in gold_std_patients],
    dataset=[patient_dataset[p.id] for p in gold_std_patients]
)
CSV.write(joinpath(models_path, "ids_gold_std_patients_$(run_dataset_name).csv"), df_gold_ids)

filter_report = DataFrame(
    dataset=["MIMIC-IV", "UMG"],
    raw_n=[mimic_raw, umg_raw],
    eligible_n=[mimic_eligible, umg_eligible],
    removed_anoms=[mimic_removed, umg_removed],
    gold_n=[length(mimic_gold), length(umg_gold)]
)
CSV.write(joinpath(models_path, "gold_std_filter_report_$(run_dataset_name).csv"), filter_report)

Random.seed!(42)

trunc_meta_all = init_trunc_meta_df()
trunc_metrics_all = init_trunc_metrics_df()
trunc_params_all = init_trunc_params_df(model_cfg.param_labels)

for base_choice_idx in eachindex(gold_std_patients)
    base_patient = gold_std_patients[base_choice_idx]
    haskey(patient_dataset, base_patient.id) || error("Missing dataset mapping for patient $(base_patient.id)")

    base_dataset = patient_dataset[base_patient.id]
    base_patient_id = base_patient.id
    @info "Selected base patient $(base_patient.id) with $(length(base_patient.timepoints)) measurements"

    # exp_run = "res/$(experiment)/SystematicTruncationValidation_no_pre_min4/model_$(safe_model_tag)/$(base_patient_id)"
    # exp_models = "$(exp_run)/models"
    # exp_figs = "$(exp_run)/figs"
    exp_models = "$(modelssave_path)/$(base_patient_id)"
    exp_figs = "$(figsave_path)/$(base_patient_id)"
    mkpath(exp_models)
    mkpath(exp_figs)

    init_title = "Initial measurements - $(base_patient.id) [$(model_cfg.model_name)]"
    plt_base_initial = Plots.scatter(
        base_patient.timepoints,
        base_patient.ctnt_data;
        ms=6,
        alpha=0.9,
        xlabel="Time (h)",
        ylabel="cTnT [ng/mL]",
        title=init_title,
        label="Initial measurements"
    )
    savefig(plt_base_initial, "$(exp_figs)/patient_$(base_patient.id)_initial_scatter.svg")
    plotting && display(plt_base_initial)

    base_carrier = build_base_carrier(base_patient, model_cfg)

    θ_full, full_loss = fit_patient_multistart(
        base_carrier,
        base_patient,
        lhs_lb,
        lhs_ub;
        λ_back=model_cfg.λ_back,
        n_multistart=MS_N,
        ms_topk=MS_TOPK,
        ms_maxiters=MS_MAXITERS,
        ms_maxtime=MS_MAXTIME,
        ms_rng=StableRNG(10_000 + base_choice_idx)
    )

    @info "Base fit objective for $(base_patient.id): $(full_loss)"
    full_par = exp.(θ_full)

    scenarios, trunc_meta = generate_systematic_truncations(base_patient, base_patient_id; min_keep=MIN_KEEP_MEAS)
    isempty(scenarios) && error("No valid truncation scenarios for patient $(base_patient.id)")

    CSV.write("$(exp_models)/truncation_meta.csv", trunc_meta)
    append!(trunc_meta_all, trunc_meta)

    rec_patients = [sc.patient for sc in scenarios]
    fromPatientData2DataFrame(rec_patients; save=true, save_path="$(exp_models)/df_$(base_patient_id).csv")

    trunc_metrics = init_trunc_metrics_df()
    trunc_params = init_trunc_params_df(model_cfg.param_labels)
    successful_patients = PatientData[]
    validation_params = Vector{Vector{Float64}}()

    ev_bar = Progress(length(scenarios); desc="Validating truncations", color=:cyan, showspeed=true)
    for (i, scenario) in enumerate(scenarios)
        patient = scenario.patient

        θ_opt, best_objective = fit_patient_multistart(
            base_carrier,
            patient,
            lhs_lb,
            lhs_ub;
            λ_back=model_cfg.λ_back,
            n_multistart=MS_N,
            ms_topk=MS_TOPK,
            ms_maxiters=MS_MAXITERS,
            ms_maxtime=MS_MAXTIME,
            ms_rng=StableRNG(100_000 * base_choice_idx + i)
        )

        pred_full = predict_plasma(base_carrier, θ_opt, base_patient.timepoints)
        pred_sparse = predict_plasma(base_carrier, θ_opt, patient.timepoints)
        curve_t, curve_plasma = predict_plasma_curve(base_carrier, θ_opt; saveat=1.0)

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
            rmsle_sparse=rmsle_sparse
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
            model_cfg.param_labels
        )

        push!(successful_patients, patient)
        push!(validation_params, θ_opt)

        plot_truncation_fit(
            base_patient,
            scenario,
            curve_t,
            curve_plasma,
            "$(exp_figs)/patient_$(patient.id).svg";
            plotting=plotting,
            model_label=model_cfg.model_name,
            curve_label=model_cfg.curve_label
        )

        next!(ev_bar)
    end
    finish!(ev_bar)

    nrow(trunc_metrics) == 0 && error("No successful truncation fit for base patient $(base_patient.id)")

    CSV.write("$(exp_models)/trunc_metrics.csv", trunc_metrics)
    CSV.write("$(exp_models)/trunc_params.csv", trunc_params)

    append!(trunc_metrics_all, trunc_metrics)
    append!(trunc_params_all, trunc_params)

    if !isempty(validation_params)
        ode_params_val_base = vcat(validation_params...)

        params_extraction_generic(
            successful_patients,
            ode_params_val_base;
            param_labels=model_cfg.param_labels,
            data_label="truncated_$(base_patient_id)",
            dataset=base_dataset,
            figsave_path=exp_figs,
            show_outliers=true,
            savefigure=true
        )
    end

    @info "Completed base patient $(base_patient.id): $(nrow(trunc_metrics))/$(length(scenarios)) successful truncations"
end

CSV.write("$(modelssave_path)/truncation_meta_all.csv", trunc_meta_all)
CSV.write("$(modelssave_path)/trunc_metrics_all.csv", trunc_metrics_all)
CSV.write("$(modelssave_path)/trunc_params_all.csv", trunc_params_all)

if nrow(trunc_metrics_all) > 0
    trunc_patient_summary = combine(
        groupby(trunc_metrics_all, :base_patient),
        :removed_n => mean => :removed_n_mean,
        :smape_full => median => :smape_full_median,
        :rmsle_full => median => :rmsle_full_median,
        :loss => median => :loss_median
    )
    CSV.write("$(modelssave_path)/truncation_patient_summary.csv", trunc_patient_summary)

    trunc_section_summary = combine(
        groupby(trunc_metrics_all, :trunc_section),
        :smape_full => median => :smape_full_median,
        :rmsle_full => median => :rmsle_full_median,
        :loss => median => :loss_median
    )
    CSV.write("$(modelssave_path)/truncation_section_summary.csv", trunc_section_summary)
end

if nrow(trunc_params_all) > 0
    param_summary = DataFrame(
        param=String[],
        median=Float64[],
        q1=Float64[],
        q3=Float64[]
    )

    for lbl in model_cfg.param_labels
        vals = trunc_params_all[!, Symbol(lbl)]
        push!(param_summary, (
            param=lbl,
            median=median(vals),
            q1=quantile(vals, 0.25),
            q3=quantile(vals, 0.75)
        ))
    end

    CSV.write("$(modelssave_path)/truncation_param_summary.csv", param_summary)
end

@info "SystematicTruncation process ended successfully ($(model_cfg.model_name), loss=$(model_cfg.loss_label))"

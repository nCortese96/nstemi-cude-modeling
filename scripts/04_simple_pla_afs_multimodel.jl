using LikelihoodProfiler, OptimizationLBFGSB, Distributions, Random
using DataFrames, CSV
using Plots, ProgressMeter, Logging, Dates
using JLD2

include("MechanisticAI.jl")

function patient_nll_log_gaussian(
    θ,
    data::Tuple{ctntUDEModel,<:AbstractVector,<:AbstractVector,<:AbstractVector}
)
    model, timepoints, ctnt_data, fixed_nn_params = data
    p = ComponentArray(ode=θ, neural=fixed_nn_params)

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]
    prob = remake(model.problem; u0=u0, p=p)

    sol = solve(prob, Tsit5(); p=p, saveat=timepoints, abstol=1e-8, reltol=1e-6)
    !successful_retcode(sol) && return Inf

    plasm = sol[3, :]
    resid = log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)
    rss = sum(abs2, resid)
    n = length(resid)

    return n * log(rss / n)
end

function patient_nll_log_gaussian(
    θ,
    data::Tuple{SciMLBase.ODEProblem,<:AbstractVector,<:AbstractVector}
)
    problem, timepoints, ctnt_data = data

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]
    prob = remake(problem; u0=u0, p=θ)

    sol = solve(prob, Tsit5(); p=θ, saveat=timepoints, abstol=1e-8, reltol=1e-6)
    !successful_retcode(sol) && return Inf

    plasm = sol[3, :]
    resid = log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)
    rss = sum(abs2, resid)
    n = length(resid)

    return n * log(rss / n)
end

safe_patient_id(pid) = replace(string(pid), r"[^\w\-]+" => "_")

function empty_profiles_df()
    DataFrame(
        patient_id=String[],
        patient_idx=Int[],
        param_idx=Int[],
        param_name=String[],
        class_label=String[],
        step_idx=Int[],
        branch_side=String[],
        x_theta=Float64[],
        x_exp=Float64[],
        delta_theta=Float64[],
        objective=Float64[],
        plr=Float64[],
    )
end

function empty_summary_df()
    DataFrame(
        patient_id=String[],
        patient_idx=Int[],
        param_idx=Int[],
        param_name=String[],
        class_label=String[],
        theta_hat=Float64[],
        theta_hat_exp=Float64[],
        threshold=Float64[],
        left_endpoint=Float64[],
        right_endpoint=Float64[],
        retcode_left=String[],
        retcode_right=String[],
    )
end

function classify_profile(ep, rc)
    left_raw = hasproperty(ep, :left) ? getproperty(ep, :left) : nothing
    right_raw = hasproperty(ep, :right) ? getproperty(ep, :right) : nothing

    left_ep = left_raw isa Number ? Float64(left_raw) :
              (left_raw !== nothing && hasproperty(left_raw, :value) && getproperty(left_raw, :value) isa Number ?
               Float64(getproperty(left_raw, :value)) : NaN)

    right_ep = right_raw isa Number ? Float64(right_raw) :
               (right_raw !== nothing && hasproperty(right_raw, :value) && getproperty(right_raw, :value) isa Number ?
                Float64(getproperty(right_raw, :value)) : NaN)

    rc_left = lowercase(string(hasproperty(rc, :left) ? getproperty(rc, :left) : rc))
    rc_right = lowercase(string(hasproperty(rc, :right) ? getproperty(rc, :right) : rc))

    left_identifiable = rc_left == "identifiable"
    right_identifiable = rc_right == "identifiable"

    class_label =
        if left_identifiable && right_identifiable
            "Identifiable"
        elseif left_identifiable || right_identifiable
            "Practically identifiable"
        else
            "Unidentifiable"
        end

    return class_label, left_ep, right_ep, rc_left, rc_right
end

function build_legend_panel()
    pleg = Plots.plot(
        xlim=(0, 1), ylim=(0, 1),
        framestyle=:none,
        xticks=false, yticks=false,
        grid=false,
        legend=false
    )

    Plots.annotate!(pleg, 0.50, 0.92, Plots.text("Legend", 16, :center))

    Plots.plot!(pleg, [0.18, 0.38], [0.78, 0.78], color=:blue, lw=2)
    Plots.scatter!(pleg, [0.28], [0.68], markercolor=:orange, markerstrokecolor=:black, ms=7)
    Plots.plot!(pleg, [0.18, 0.38], [0.58, 0.58], color=:green, ls=:dash, lw=2)

    Plots.annotate!(pleg, [
        (0.56, 0.78, Plots.text("profile", 12, :center)),
        (0.56, 0.68, Plots.text("profiler steps", 12, :center)),
        (0.56, 0.58, Plots.text("threshold", 12, :center)),
        (0.50, 0.24, Plots.text("Identifiable: both branches Identifiable", 10, :center)),
        (0.50, 0.17, Plots.text("Practically identifiable: at least one Identifiable branch", 10, :center)),
        (0.50, 0.10, Plots.text("Unidentifiable: no Identifiable branches", 10, :center)),
        # (0.50, 0.03, Plots.text("y-axis shows Δ objective", 10, :center)),
        (0.50, 0.35, Plots.text("y-axis: -2Δ profile log-likelihood", 11, :center)),
        (0.50, 0.42, Plots.text("x-axis: θ (natural scale, log10 axis)", 11, :center)),
        # (0.50, 0.24, Plots.text("Identifiable: 2 finite endpoints", 10, :center)),
        # (0.50, 0.17, Plots.text("Practically identifiable: 1 finite endpoint", 10, :center)),
        # (0.50, 0.10, Plots.text("Unidentifiable: 0 finite endpoints", 10, :center)),
        # (0.50, 0.03, Plots.text("Incomplete: branch stopped / maxiters / failure", 10, :center)),
    ])

    return pleg
end

function empirical_quantile(v::AbstractVector{<:Real}, q::Real)
    isempty(v) && return NaN
    s = sort(Float64.(v))
    idx = clamp(ceil(Int, q * length(s)), 1, length(s))
    return s[idx]
end

function profile_curve_data(cdf::DataFrame)
    keep = isfinite.(cdf.x_exp) .& isfinite.(cdf.delta_theta) .& isfinite.(cdf.plr)
    c = cdf[keep, [:x_exp, :delta_theta, :plr]]

    nrow(c) == 0 && return Float64[], Float64[], Float64[]

    ord = sortperm(c.x_exp)

    x_exp = Float64.(c.x_exp[ord])
    x_ctr = Float64.(c.delta_theta[ord])
    y_plr = Float64.(c.plr[ord])

    return x_exp, x_ctr, y_plr
end

function build_patient_composite_plot(pdf::DataFrame, sdf::DataFrame, patient_id::String, dataset_name::String)
    plist = Plots.Plot[]

    for j in 1:length(param_names)
        cdf = pdf[pdf.param_idx.==j, :]
        ssel = sdf[sdf.param_idx.==j, :]

        if nrow(cdf) == 0 || nrow(ssel) == 0
            pj = Plots.plot(
                title="$(pnames_plot[j]) | missing",
                legend=false,
                framestyle=:box
            )
            push!(plist, pj)
            continue
        end

        x_nat, _, y_plr = profile_curve_data(cdf)

        class_label = String(ssel.class_label[1])
        theta_hat_exp = Float64(ssel.theta_hat_exp[1])
        thr = Float64(ssel.threshold[1])

        pj = Plots.plot(
            x_nat,
            y_plr;
            xscale=:log10,
            legend=false,
            title="$(pnames_plot[j]) | $(class_label)",
            lw=2
        )

        Plots.scatter!(
            pj,
            x_nat,
            y_plr;
            ms=3,
            markercolor=:orange,
            markerstrokecolor=:black
        )

        Plots.vline!(pj, [theta_hat_exp], color=:black, ls=:dot, lw=1.5)
        Plots.hline!(pj, [thr], color=:green, ls=:dash, lw=2)
        Plots.scatter!(pj, [theta_hat_exp], [0.0], color=:black, ms=4)

        yvals = vcat(y_plr, [0.0, thr])
        Plots.ylims!(pj, (minimum(yvals), 1.05 * maximum(yvals)))

        push!(plist, pj)
    end

    push!(plist, build_legend_panel())

    return Plots.plot(
        plist...;
        layout=(2, 3),
        size=(1700, 1000),
        margins=2Plots.mm,
        plot_title="PLA patient $(patient_id) | dataset $(dataset_name)",
    )
end

const class_colors = Dict(
    "Identifiable" => :orange,
    "Practically identifiable" => :dodgerblue,
    "Unidentifiable" => :deeppink3,
)

const class_order = [
    "Identifiable",
    "Practically identifiable",
    "Unidentifiable",
]

function build_aggregate_legend_panel()
    pleg = Plots.plot(
        xlim=(0, 1), ylim=(0, 1),
        framestyle=:none,
        xticks=false, yticks=false,
        grid=false,
        legend=false
    )

    Plots.annotate!(pleg, 0.50, 0.92, Plots.text("Legend", 16, :center))

    Plots.plot!(pleg, [0.12, 0.30], [0.80, 0.80], color=class_colors["Identifiable"], lw=3)
    Plots.plot!(pleg, [0.12, 0.30], [0.68, 0.68], color=class_colors["Practically identifiable"], lw=3)
    Plots.plot!(pleg, [0.12, 0.30], [0.56, 0.56], color=class_colors["Unidentifiable"], lw=3)
    Plots.plot!(pleg, [0.12, 0.30], [0.42, 0.42], color=:green, ls=:dash, lw=2)

    Plots.annotate!(pleg, [
        (0.60, 0.80, Plots.text("Identifiable", 12, :center)),
        (0.60, 0.68, Plots.text("Practically identifiable", 12, :center)),
        (0.60, 0.56, Plots.text("Unidentifiable", 12, :center)),
        (0.60, 0.42, Plots.text("95% threshold", 12, :center)),
        (0.50, 0.24, Plots.text("x-axis: Δθ (log-scale parameter)", 11, :center)),
        (0.50, 0.16, Plots.text("y-axis: -2Δ profile log-likelihood", 11, :center)),
        # (0.50, 0.08, Plots.text("Aggregate panels show curves only (no profiler points)", 10, :center)),
    ])

    return pleg
end

function build_aggregate_parameter_plot(pdf::DataFrame, sdf::DataFrame, pname::String, pname_plot::String)
    sdfp = sdf[sdf.param_name.==pname, :]
    pdfp = pdf[pdf.param_name.==pname, :]

    nI = sum(sdfp.class_label .== "Identifiable")
    nPI = sum(sdfp.class_label .== "Practically identifiable")
    nUI = sum(sdfp.class_label .== "Unidentifiable")

    plt = Plots.plot(
        xlabel="Δ$(pname_plot)",
        ylabel="",
        legend=:best,
        gridalpha=0.15,
        title="", #"$(pname_plot)",
        lw=1.8,
        size=(900, 650)
    )

    # legenda con conteggi
    Plots.plot!(plt, [NaN], [NaN];
        color=:green, ls=:dash, lw=2,
        label="95% threshold"
    )
    Plots.plot!(plt, [NaN], [NaN];
        color=class_colors["Unidentifiable"], lw=5,
        label="Unidentifiable (n=$(nUI))"
    )
    Plots.plot!(plt, [NaN], [NaN];
        color=class_colors["Practically identifiable"], lw=5,
        label="Practically identifiable (n=$(nPI))"
    )
    Plots.plot!(plt, [NaN], [NaN];
        color=class_colors["Identifiable"], lw=5,
        label="Identifiable (n=$(nI))"
    )

    Plots.hline!(plt, [pla_thr], color=:green, ls=:dash, lw=2, label=nothing)

    x_focus = Float64[]
    y_focus = Float64[]

    for cls in class_order
        sdf_cls = sdfp[sdfp.class_label.==cls, :]
        nrow(sdf_cls) == 0 && continue

        for row in eachrow(sdf_cls)
            cdf = pdfp[
                (pdfp.patient_id.==row.patient_id).&(pdfp.param_name.==row.param_name),
                :
            ]

            nrow(cdf) < 2 && continue

            _, x_ctr, y_plr = profile_curve_data(cdf)
            length(x_ctr) < 2 && continue

            Plots.plot!(
                plt,
                x_ctr,
                y_plr;
                color=class_colors[cls],
                lw=1.6,
                alpha=0.75,
                label=nothing
            )

            # usa per il crop solo la parte interessante vicino alla soglia
            keep_focus = y_plr .<= 1.35 * pla_thr
            if any(keep_focus)
                append!(x_focus, x_ctr[keep_focus])
                append!(y_focus, y_plr[keep_focus])
            else
                append!(x_focus, x_ctr)
                append!(y_focus, y_plr)
            end
        end
    end

    # crop robusto, centrato su 0, come nella figura di esempio
    if !isempty(x_focus)
        xhalf = empirical_quantile(abs.(x_focus), 0.98)
        xhalf = max(xhalf, 1.0)
        Plots.xlims!(plt, (-1.05 * xhalf, 1.05 * xhalf))
    end

    if !isempty(y_focus)
        ytop = max(1.10 * pla_thr, empirical_quantile(y_focus, 0.98))
        ytop = max(ytop, 1.10 * pla_thr)
        Plots.ylims!(plt, (0.0, 1.05 * ytop))
    else
        Plots.ylims!(plt, (0.0, 1.10 * pla_thr))
    end

    return plt
end

length(ARGS) >= 2 || @info "REPL execution or default setting selected. 
        For CLI usage, provide at least <dataset_id> and <model_id>.
        \nUsage: JULIA_NUM_THREADS=<n> julia --project=. src/simple_pla_afs_multimodel.jl <dataset_id> <model_id> [run_compute=true|false] [run_plot_patients=true|false] [run_plot_aggregate=true|false]"

println("PLA script started $(now())")

dataset_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0
@info "Dataset ID: $dataset_id"
if dataset_id == 0
    dataset_name = "MIMIC-IV"
    UMG_data = false
elseif dataset_id == 1
    dataset_name = "UMG"
    UMG_data = true
else
    error("dataset_id must be 0 (MIMIC-IV) or 1 (UMG)")
end

# 1 = cUDE (fixed NN), 2 = ODE (troponin_ode!)
const MODEL_ID = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
@info "Model ID: $MODEL_ID"

input_dim = 2
nn_depth = 2
nn_width = 8
N_params = 5
best_idx = 3

RUN_COMPUTE = length(ARGS) >= 3 ? parse(Bool, ARGS[3]) : true
RUN_PLOT_PATIENTS = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true
RUN_PLOT_AGGREGATE = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : true

SEPARATE = false

if MODEL_ID == 1
    @info "Using cUDE fixed-NN model"
    experiment = "NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback"
    model_tag = "cUDE_NN_$(best_idx)"
elseif MODEL_ID == 2
    @info "Using ODE Td-sigmoid model"
    experiment = "NSTEMI_ODE_TdSigmoid/$(dataset_name)_opt_lambda1"
    model_tag = "ODE_TdSigmoid"
else
    error("Unsupported MODEL_ID=$(MODEL_ID). Use 1 (cUDE) or 2 (ODE).")
end

fig_path = "res/$(experiment)/figs"
models_path = "res/$(experiment)/models"

SOURCE_RUN_TAG = "run_20260323_113508"

run_tag = RUN_COMPUTE ?
          "$(dataset_name)_simplePLA_$(model_tag)_nll/run_afs_$(SEPARATE ? "sep_" : "")$(Dates.format(now(), "yyyymmdd_HHMMSS"))" :
          "$(dataset_name)_simplePLA_$(model_tag)_nll/$(SOURCE_RUN_TAG)"

modelssave_path = "$(models_path)/$(run_tag)"
csvsave_path = "$(modelssave_path)/csv"
patient_csv_path = "$(csvsave_path)/per_patient"

figsave_path = "$(fig_path)/$(run_tag)/plots_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
composite_fig_path = "$(figsave_path)/composite"
aggregate_fig_path = "$(figsave_path)/aggregate"

# source_modelssave_path = RUN_COMPUTE ? modelssave_path : "$(models_path)/$(SOURCE_RUN_TAG)"
# source_csvsave_path = "$(source_modelssave_path)/csv"
# source_patient_csv_path = "$(source_csvsave_path)/per_patient"

if RUN_COMPUTE
    mkpath(modelssave_path)
    mkpath(csvsave_path)
    mkpath(patient_csv_path)
end

mkpath(figsave_path)
mkpath(composite_fig_path)
mkpath(aggregate_fig_path)

chain = nothing
best_nn = nothing

if MODEL_ID == 1
    chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim)

    @load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters
    length(neural_network_parameters) >= best_idx || error("best_idx=$(best_idx) exceeds available NN models")
    best_nn = neural_network_parameters[best_idx]
end

@info "Loading dataset"
test_dataset = if UMG_data
    @load "res/UMG_testset.jld2" test_dataset
    test_dataset
else
    @load "res/MIMIC-IV_testset.jld2" test_dataset
    test_dataset
end

# fallback_cude_models = "res/NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback/models"

# if MODEL_ID == 1
#     dataset_candidates = UMG_data ?
#         ["$(models_path)/UMG_testset.jld2", "$(fallback_cude_models)/UMG_testset.jld2"] :
#         ["$(models_path)/testsetNSTEMI_$(experiment).jld2", "$(fallback_cude_models)/testsetNSTEMI_NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback.jld2"]
# else
#     dataset_candidates = UMG_data ?
#         ["$(fallback_cude_models)/UMG_testset.jld2"] :
#         ["$(fallback_cude_models)/testsetNSTEMI_NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback.jld2"]
# end

# dataset_file = nothing
# for f in dataset_candidates
#     if isfile(f)
#         dataset_file = f
#         break
#     end
# end
# dataset_file === nothing && error("No test dataset file found. Tried: $(dataset_candidates)")

# @load dataset_file test_dataset

@info "Dataset $(dataset_name) loaded with $(length(test_dataset)) patients"
patient_dims(test_dataset)

if MODEL_ID == 1
    params_path = "$(models_path)/$(dataset_name)_test_NN_$(best_idx)_ms_test/"
    @info "Loading initialization seeds from $(params_path)"
    @load "$(params_path)/best_params_val_$(dataset_name).jld2" ode_params_val

    params_val_df = CSV.read("$(params_path)/patients_params_val.csv", DataFrame)
    required_cols = [:patient_id, :a, :b, :Cs0, :Cc0, :beta]
    found_cols = Symbol.(names(params_val_df))
    missing_cols = setdiff(required_cols, found_cols)
    isempty(missing_cols) || error("Missing required columns in patients_params_val.csv. Missing=$(missing_cols), Found=$(names(params_val_df))")

    reshaped_params = permutedims(reshape(ode_params_val, N_params, :))
    all(isfinite, reshaped_params) || error("ode_params_val contains non-finite values")

    n_param_sets = size(reshaped_params, 1)
    n_patients = length(test_dataset)
    n_csv = nrow(params_val_df)

    n_param_sets == n_patients || error("Mismatch ode_params_val vs test_dataset: n_param_sets=$(n_param_sets), n_patients=$(n_patients)")
    n_csv == n_patients || error("Mismatch patients_params_val.csv vs test_dataset: n_csv=$(n_csv), n_patients=$(n_patients)")

    ids_dataset = [p.id for p in test_dataset]
    ids_csv = String.(params_val_df.patient_id)
    ids_dataset == ids_csv || error("patients_params_val.csv is not aligned with test_dataset ordering.")

    theta_from_csv = hcat(
        log.(params_val_df.a),
        log.(params_val_df.b),
        log.(params_val_df.Cs0),
        log.(params_val_df.Cc0),
        log.(params_val_df.beta)
    )

    max_abs_diff = maximum(abs.(theta_from_csv .- reshaped_params))
    max_abs_diff <= 1e-8 || error("Mismatch between ode_params_val and patients_params_val.csv (log-scale). max_abs_diff=$(max_abs_diff)")

else
    ode_params_candidates = [
        "res/NSTEMI_ODE_TdSigmoid/$(dataset_name)_opt_lambda1/models/params_out.csv",
        "res/NSTEMI_ODE_TdSigmoid/$(dataset_name)_opt/models/params_out.csv"
    ]

    # ode_params_csv = nothing
    ode_params_csv = "$(models_path)/params_out.csv"
    # for f in ode_params_candidates
    !isfile(ode_params_csv) && error("No ODE params_out.csv found. Tried: $(ode_params_candidates)")
    # end
    # ode_params_csv === nothing && error("No ODE params_out.csv found. Tried: $(ode_params_candidates)")

    @info "Loading ODE initialization seeds from $(ode_params_csv)"
    params_val_df = CSV.read(ode_params_csv, DataFrame)

    found_cols = Symbol.(names(params_val_df))
    required_cols = [:p1, :p2, :p3, :p4, :p5]
    missing_cols = setdiff(required_cols, found_cols)
    isempty(missing_cols) || error("Missing required columns in ODE params_out.csv. Missing=$(missing_cols), Found=$(names(params_val_df))")

    id_col = :patient in found_cols ? :patient :
             (:patient_id in found_cols ? :patient_id : error("Missing patient identifier column (patient/patient_id) in ODE params_out.csv"))

    ids_dataset = [p.id for p in test_dataset]

    rows_by_id = Dict{String,NTuple{5,Float64}}()
    for r in eachrow(params_val_df)
        pid = string(r[id_col])
        haskey(rows_by_id, pid) && error("Duplicate patient ID in ODE params_out.csv: $(pid)")
        rows_by_id[pid] = (
            Float64(r.p1),
            Float64(r.p2),
            Float64(r.p3),
            Float64(r.p4),
            Float64(r.p5),
        )
    end

    missing_ids = filter(id -> !haskey(rows_by_id, id), ids_dataset)
    isempty(missing_ids) || error("Missing ODE params for test_dataset IDs: $(missing_ids)")

    extra_ids = setdiff(collect(keys(rows_by_id)), ids_dataset)
    !isempty(extra_ids) && @warn "Extra IDs in ODE params_out.csv not used by test_dataset: $(extra_ids)"

    reshaped_params = Matrix{Float64}(undef, length(ids_dataset), 5)
    for (i, pid) in enumerate(ids_dataset)
        p = rows_by_id[pid]
        reshaped_params[i, 1] = p[1]
        reshaped_params[i, 2] = p[2]
        reshaped_params[i, 3] = p[3]
        reshaped_params[i, 4] = p[4]
        reshaped_params[i, 5] = p[5]
    end

    all(isfinite, reshaped_params) || error("ODE params_out.csv contains non-finite values")
end

n_param_sets = size(reshaped_params, 1)
n_patients = length(test_dataset)
n_param_sets == n_patients || error("Mismatch parameter starts vs test_dataset: n_param_sets=$(n_param_sets), n_patients=$(n_patients)")
@info "Loaded $(n_param_sets) patient-specific starts (log-scale)"

lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001])

if MODEL_ID == 1
    lhs_ub = log.([10.0, 10.0, 500.0, 500.0, 1.0])
    param_names = ["a", "b", "Cs0", "Cc0", "beta"]
    pnames_plot = ["a", "b", "Cs0", "Cc0", "β"]
else
    lhs_ub = log.([10.0, 10.0, 500.0, 500.0, 500.0])
    param_names = ["a", "b", "Cs0", "Cc0", "Td"]
    pnames_plot = ["a", "b", "Cs0", "Cc0", "Td"]
end
# θ_bounds = Tuple{Float64, Float64}[
#     (Float64(lhs_lb[k]), Float64(lhs_ub[k])) for k in 1:N_params
# ]

pla_thr = quantile(Chisq(1), 0.95)

optf = OptimizationFunction(
    (θ, data) -> patient_nll_log_gaussian(θ, data),
    AutoFiniteDiff()
)

# Better research
const PLA_REFIT_MAXITERS = 1000

const PLA_PROFILE_MAXITERS = 100_000
const PLA_STEP_SCALE = 0.2 #0.08

const PLA_SPAN = 1.25
const PLA_EXPAND_TRIES = 3
const PLA_EXPAND_FACTOR = 2.0

# CODEX
# const PLA_RETRY_MAXITERS_FACTOR = 2.0
# const PLA_RETRY_STEP_SHRINK = 0.5
# CODEX

const PLA_EPS_BOUND = 1e-7

function clamp_strictly_inside(theta, lb, ub; eps_bound=PLA_EPS_BOUND)
    θ = Float64.(copy(theta))
    for k in eachindex(θ)
        θ[k] = clamp(θ[k], lb[k] + eps_bound, ub[k] - eps_bound)
    end
    return θ
end

function extract_endpoint_value(x)
    x isa Number && return Float64(x)
    if x !== nothing && hasproperty(x, :value) && getproperty(x, :value) isa Number
        return Float64(getproperty(x, :value))
    end
    return NaN
end

function branch_failed_flag(rc_side)
    s = lowercase(string(rc_side))
    return occursin("max", s) || occursin("fail", s) || occursin("error", s)
end

function make_profile_window(theta_hat, lb, ub, j, span)
    plb = copy(lb)
    pub = copy(ub)

    lo = max(lb[j] + PLA_EPS_BOUND, theta_hat[j] - span)
    hi = min(ub[j] - PLA_EPS_BOUND, theta_hat[j] + span)

    if !(lo < theta_hat[j] < hi)
        lo = lb[j] + PLA_EPS_BOUND
        hi = ub[j] - PLA_EPS_BOUND
    end

    plb[j] = lo
    pub[j] = hi
    return plb, pub
end

function profile_hits_window_edge(pr, j, plb, pub; tol_frac=0.02)
    df = DataFrame(pr)
    x = Float64.(df[!, Symbol("x$j")])
    x = x[isfinite.(x)]

    isempty(x) && return false

    tol = tol_frac * max(pub[j] - plb[j], 1e-8)
    return (minimum(x) <= plb[j] + tol) || (maximum(x) >= pub[j] - tol)
end

function profile_score(pr)
    ep = endpoints(pr)
    rc = retcodes(pr)

    left_ep = extract_endpoint_value(hasproperty(ep, :left) ? getproperty(ep, :left) : nothing)
    right_ep = extract_endpoint_value(hasproperty(ep, :right) ? getproperty(ep, :right) : nothing)

    rc_left = hasproperty(rc, :left) ? getproperty(rc, :left) : rc
    rc_right = hasproperty(rc, :right) ? getproperty(rc, :right) : rc

    left_failed = branch_failed_flag(rc_left)
    right_failed = branch_failed_flag(rc_right)

    nfinite = Int(isfinite(left_ep)) + Int(isfinite(right_ep))
    nfailed = Int(left_failed) + Int(right_failed)
    npts = nrow(DataFrame(pr))

    return (nfinite, -nfailed, npts)
end

choose_better_profile(pr_old, pr_new) =
    profile_score(pr_new) > profile_score(pr_old) ? pr_new : pr_old

function profile_needs_retry(pr, j, plb, pub)
    rc = retcodes(pr)

    rc_left = hasproperty(rc, :left) ? getproperty(rc, :left) : rc
    rc_right = hasproperty(rc, :right) ? getproperty(rc, :right) : rc

    left_failed = branch_failed_flag(rc_left)
    right_failed = branch_failed_flag(rc_right)

    return left_failed || right_failed || profile_hits_window_edge(pr, j, plb, pub)
end

function solve_profile(optprob, theta_hat, plb, pub;
    step_scale=PLA_STEP_SCALE,
    maxiters=PLA_PROFILE_MAXITERS
)
    # plprob_j = ProfileLikelihoodProblem(
    #     optprob,
    #     theta_hat;
    #     idxs = [j],
    #     profile_lower = plb,
    #     profile_upper = pub,
    #     conf_level = 0.95,
    #     df = 1
    # )
    plprob = ProfileLikelihoodProblem(
        optprob,
        theta_hat;
        idxs=1:5,
        profile_lower=plb,
        profile_upper=pub,
        conf_level=0.95,
        df=1
    )

    profiler = OptimizationProfiler(
        optimizer=LBFGSB(),
        optimizer_opts=(maxiters=maxiters,),
        stepper=FixedStep(
            initial_step=(pars, idx) -> step_scale * max(abs(pars[idx]), 1e-3)
        )
    )

    sol = solve(plprob, profiler; parallel_type=:threads, verbose=false)
    return sol
end

function solve_profile_for_param(optprob, theta_hat, j, plb, pub;
    step_scale=PLA_STEP_SCALE,
    maxiters=PLA_PROFILE_MAXITERS
)
    plprob_j = ProfileLikelihoodProblem(
        optprob,
        theta_hat;
        idxs=[j],
        profile_lower=[plb[j]],
        profile_upper=[pub[j]],
        conf_level=0.95,
        df=1
    )

    profiler_j = OptimizationProfiler(
        optimizer=LBFGSB(),
        optimizer_opts=(maxiters=maxiters,),
        stepper=FixedStep(
            initial_step=(pars, idx) -> step_scale * max(abs(pars[idx]), 1e-3)
        )
    )

    sol_j = solve(plprob_j, profiler_j; parallel_type=:threads, verbose=false)
    return sol_j[1]
end

# CODEX
# function solve_profile_for_param(optprob, theta_hat, j, plb, pub;
#         step_scale = PLA_STEP_SCALE,
#         maxiters = PLA_PROFILE_MAXITERS,
#         attempt = 1
#     )
#     plprob_j = ProfileLikelihoodProblem(
#         optprob,
#         theta_hat;
#         idxs = [j],
#         profile_lower = plb,
#         profile_upper = pub,
#         conf_level = 0.95,
#         df = 1
#     )
#     effective_step = step_scale * (PLA_RETRY_STEP_SHRINK ^ max(attempt - 1, 0))

#     profiler_j = OptimizationProfiler(
#         optimizer = LBFGSB(),
#         optimizer_opts = (maxiters = maxiters,),
#         stepper = FixedStep(
#             initial_step = (pars, idx) -> effective_step * max(abs(pars[idx]), 1e-3)
#         )
#     )

#     sol_j = solve(plprob_j, profiler_j; parallel_type = :threads, verbose = false)
#     return sol_j[1]
# end
# CODEX

# Better research

profiles_long = empty_profiles_df()
summary_df = empty_summary_df()

if RUN_COMPUTE
    @info "Starting batch PLA compute and CSV export"

    @showprogress for i in eachindex(test_dataset)
        patient = test_dataset[i]
        patient_id = String(patient.id)
        patient_id_safe = safe_patient_id(patient_id)
        patient_tag = "patient_$(lpad(string(i), 4, '0'))_$(patient_id_safe)"

        # Better research
        θ0 = clamp_strictly_inside(reshaped_params[i, :], lhs_lb, lhs_ub)

        if MODEL_ID == 1
            θ_model = ComponentArray(ode=θ0, neural=best_nn)
            model = ctntCUDEModel(θ_model, chain, patient.timepoints)
            data_tpl = (model, patient.timepoints, patient.ctnt_data, best_nn)
        else
            u0_init = [exp(θ0[3]), exp(θ0[4]), 0.0]
            tspan = (0.0, patient.timepoints[end] + 10.0)
            problem = ODEProblem(troponin_ode!, u0_init, tspan)
            data_tpl = (problem, patient.timepoints, patient.ctnt_data)
        end

        optprob = OptimizationProblem(
            optf,
            θ0,
            data_tpl,
            lb=lhs_lb,
            ub=lhs_ub
        )

        opt_sol = Optimization.solve(
            optprob,
            LBFGS(linesearch=LineSearches.BackTracking()),
            maxiters=PLA_REFIT_MAXITERS
        )

        if !(isfinite(opt_sol.objective) && all(isfinite, opt_sol.u))
            @warn "Non-finite PLA refit for patient $(patient.id); falling back to clamped start"
            optpars = clamp_strictly_inside(θ0, lhs_lb, lhs_ub)
            Jhat = patient_nll_log_gaussian(optpars, data_tpl)
        else
            optpars = clamp_strictly_inside(opt_sol.u, lhs_lb, lhs_ub)
            Jhat = opt_sol.objective
        end

        sol = Vector{Any}(undef, 5)
        if SEPARATE
            for j in 1:5
                span = PLA_SPAN
                best_pr = nothing

                for attempt in 1:PLA_EXPAND_TRIES
                    plb, pub = make_profile_window(optpars, lhs_lb, lhs_ub, j, span)

                    pr = solve_profile_for_param(
                        optprob,
                        optpars,
                        j,
                        plb,
                        pub;
                        step_scale=PLA_STEP_SCALE,
                        maxiters=PLA_PROFILE_MAXITERS
                    )

                    best_pr = isnothing(best_pr) ? pr : choose_better_profile(best_pr, pr)

                    if !profile_needs_retry(pr, j, plb, pub)
                        break
                    end

                    full_window =
                        isapprox(plb[j], lhs_lb[j] + PLA_EPS_BOUND; atol=1e-10, rtol=0.0) &&
                        isapprox(pub[j], lhs_ub[j] - PLA_EPS_BOUND; atol=1e-10, rtol=0.0)

                    if full_window
                        break
                    end

                    span = min((lhs_ub[j] - lhs_lb[j]) / 2, span * PLA_EXPAND_FACTOR)
                end
                # CODEX
                # for attempt in 1:PLA_EXPAND_TRIES
                #     plb, pub = make_profile_window(optpars, lhs_lb, lhs_ub, j, span)
                #         pr = solve_profile_for_param(
                #             optprob,
                #             optpars,
                #             j,
                #             plb,
                #             pub;
                #             step_scale = PLA_STEP_SCALE,
                #             maxiters = PLA_PROFILE_MAXITERS,
                #             attempt = attempt
                #         )

                #     best_pr = isnothing(best_pr) ? pr : choose_better_profile(best_pr, pr)

                #     hit_edge = profile_hits_window_edge(pr, j, plb, pub)
                #     needs_retry = profile_needs_retry(pr, j, plb, pub)

                #     if needs_retry && !hit_edge
                #         pr_retry = solve_profile_for_param(
                #             optprob,
                #             optpars,
                #             j,
                #             plb,
                #             pub;
                #             # step_scale = PLA_STEP_SCALE * PLA_RETRY_STEP_SHRINK,
                #             step_scale = PLA_STEP_SCALE,
                #             maxiters = Int(round(PLA_PROFILE_MAXITERS * PLA_RETRY_MAXITERS_FACTOR)),
                #             attempt = attempt + 1
                #         )

                #         best_pr = choose_better_profile(best_pr, pr_retry)
                #         pr = choose_better_profile(pr, pr_retry)

                #         hit_edge = profile_hits_window_edge(pr, j, plb, pub)
                #         needs_retry = profile_needs_retry(pr, j, plb, pub)
                #     end

                #     if !needs_retry
                #         break
                #     end

                #     full_window =
                #         isapprox(plb[j], lhs_lb[j] + PLA_EPS_BOUND; atol = 1e-10, rtol = 0.0) &&
                #         isapprox(pub[j], lhs_ub[j] - PLA_EPS_BOUND; atol = 1e-10, rtol = 0.0)

                #     if full_window
                #         break
                #     end

                #     if hit_edge
                #         span = min((lhs_ub[j] - lhs_lb[j]) / 2, span * PLA_EXPAND_FACTOR)
                #     end
                # end
                # CODEX

                sol[j] = best_pr
            end
        else
            sol = solve_profile(optprob, optpars, lhs_lb, lhs_ub)
        end

        # Better research 

        patient_profiles = empty_profiles_df()
        patient_summary = empty_summary_df()

        for j in 1:5
            ep = endpoints(sol[j])
            rc = retcodes(sol[j])

            class_label, left_ep, right_ep, rc_left, rc_right = classify_profile(ep, rc)

            dfj = DataFrame(sol[j])

            x_theta = Float64.(dfj[!, Symbol("x$j")])
            y_obj = Float64.(dfj[!, :objective])

            keep = isfinite.(x_theta) .& isfinite.(y_obj)
            x_theta = x_theta[keep]
            y_obj = y_obj[keep]

            x_exp = exp.(x_theta)
            delta_theta = x_theta .- optpars[j]
            y_plr = y_obj .- Jhat

            branch_side = map(delta_theta) do dθ
                if isapprox(dθ, 0.0; atol=1e-10, rtol=0.0)
                    "center"
                elseif dθ < 0
                    "left"
                else
                    "right"
                end
            end

            ord = sortperm(delta_theta)
            x_theta = x_theta[ord]
            x_exp = x_exp[ord]
            delta_theta = delta_theta[ord]
            y_obj = y_obj[ord]
            y_plr = y_plr[ord]
            branch_side = branch_side[ord]

            for k in eachindex(x_theta)
                push!(patient_profiles, (
                    patient_id=patient_id,
                    patient_idx=i,
                    param_idx=j,
                    param_name=param_names[j],
                    class_label=class_label,
                    step_idx=k,
                    branch_side=branch_side[k],
                    x_theta=x_theta[k],
                    x_exp=x_exp[k],
                    delta_theta=delta_theta[k],
                    objective=y_obj[k],
                    plr=y_plr[k],
                ))
            end

            push!(patient_summary, (
                patient_id=patient_id,
                patient_idx=i,
                param_idx=j,
                param_name=param_names[j],
                class_label=class_label,
                theta_hat=optpars[j],
                theta_hat_exp=exp(optpars[j]),
                threshold=pla_thr,
                left_endpoint=left_ep,
                right_endpoint=right_ep,
                retcode_left=rc_left,
                retcode_right=rc_right,
            ))
        end

        append!(profiles_long, patient_profiles)
        append!(summary_df, patient_summary)

        CSV.write("$(patient_csv_path)/$(patient_tag)_profiles.csv", patient_profiles)
        CSV.write("$(patient_csv_path)/$(patient_tag)_summary.csv", patient_summary)
    end

    CSV.write("$(csvsave_path)/pla_profiles_long.csv", profiles_long)
    CSV.write("$(csvsave_path)/pla_profiles_summary.csv", summary_df)

    println("\nBatch PLA CSV saved to:")
    println("  $(csvsave_path)/pla_profiles_long.csv")
    println("  $(csvsave_path)/pla_profiles_summary.csv")
    println("  $(patient_csv_path)/*.csv")
end

if RUN_PLOT_PATIENTS
    @info "Starting batch plotting from saved CSVs"

    @showprogress for i in eachindex(test_dataset)
        patient_id = String(test_dataset[i].id)
        patient_id_safe = safe_patient_id(patient_id)
        patient_tag = "patient_$(lpad(string(i), 4, '0'))_$(patient_id_safe)"

        prof_csv = "$(patient_csv_path)/$(patient_tag)_profiles.csv"
        summ_csv = "$(patient_csv_path)/$(patient_tag)_summary.csv"

        if !isfile(prof_csv) || !isfile(summ_csv)
            @warn "Skipping $(patient_tag): missing CSV files"
            continue
        end

        pdf = CSV.read(prof_csv, DataFrame)
        sdf = CSV.read(summ_csv, DataFrame)

        pall = build_patient_composite_plot(pdf, sdf, patient_id_safe, dataset_name)

        # savefig(pall, "$(composite_fig_path)/$(patient_tag)_pla.png")
        savefig(pall, "$(composite_fig_path)/$(patient_tag)_pla.svg")
    end

    println("\nBatch PLA figures saved to:")
    # println("  $(composite_fig_path)/*.png")
    println("  $(composite_fig_path)/*.svg")
end

if RUN_PLOT_AGGREGATE
    @info "Starting aggregate plotting from saved global CSVs"

    profiles_long_csv = "$(csvsave_path)/pla_profiles_long.csv"
    summary_csv = "$(csvsave_path)/pla_profiles_summary.csv"

    if !isfile(profiles_long_csv) || !isfile(summary_csv)
        error("Missing aggregate CSVs: $(profiles_long_csv) or $(summary_csv)")
    end

    profiles_long_agg = CSV.read(profiles_long_csv, DataFrame)
    summary_df_agg = CSV.read(summary_csv, DataFrame)

    aggregate_panels = Plots.Plot[]

    for j in eachindex(param_names)
        pname = param_names[j]
        pname_plot = pnames_plot[j]

        plt = build_aggregate_parameter_plot(
            profiles_long_agg,
            summary_df_agg,
            pname,
            pname_plot
        )

        push!(aggregate_panels, plt)

        # savefig(plt, "$(aggregate_fig_path)/aggregate_$(pname)_delta_theta_plr.png")
        savefig(plt, "$(aggregate_fig_path)/aggregate_$(pname)_delta_theta_plr.svg")
    end

    push!(aggregate_panels, build_aggregate_legend_panel())

    pagg = Plots.plot(
        aggregate_panels...;
        layout=(2, 3),
        size=(1850, 1080),
        plot_title="Aggregate profile likelihood by parameter | dataset $(dataset_name)",
        plot_titlefontsize=18,
        left_margin=2Plots.mm,
        right_margin=2Plots.mm,
        bottom_margin=4Plots.mm,
        top_margin=4Plots.mm
    )

    # savefig(pagg, "$(aggregate_fig_path)/aggregate_profile_by_parameter.png")
    savefig(pagg, "$(aggregate_fig_path)/aggregate_profile_by_parameter.svg")

    println("\nAggregate PLA figures saved to:")
    # println("  $(aggregate_fig_path)/aggregate_*.png")
    println("  $(aggregate_fig_path)/aggregate_*.svg")
    # println("  $(aggregate_fig_path)/aggregate_profile_by_parameter.png")
    println("  $(aggregate_fig_path)/aggregate_profile_by_parameter.svg")
end

@info "PLA script finished $(now())"

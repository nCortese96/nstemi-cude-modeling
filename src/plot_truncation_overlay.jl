# =============================================================================
# Overlay Comparison Plots: cUDE vs ODE Systematic Truncation
#
# Reads pre-computed truncation results from both model runs, reconstructs
# model curves from saved parameters, and generates overlay SVG plots with
# both curves on the same figure for each patient × truncation scenario.
# =============================================================================

using DataFrames, CSV, JLD2
using Plots
using OrdinaryDiffEq: Tsit5
using SciMLBase: successful_retcode, ODEProblem, remake
using ComponentArrays: ComponentArray
using Revise
using Statistics

includet("ctnt-ude-model.jl")

# =========================== Configuration ====================================

const CUDE_EXPERIMENT = "NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback"
const ODE_EXPERIMENT = "NSTEMI_ODE_TdSigmoid"

const CUDE_BASE = "res/$(CUDE_EXPERIMENT)/models/MIMIC-UMG_systematic_truncation_cude_fixednn"
const ODE_BASE = "res/$(ODE_EXPERIMENT)/models/MIMIC-UMG_systematic_truncation_ode_tdsigmoid"

# cUDE neural network settings
const CUDE_NN_DEPTH = 2
const CUDE_NN_WIDTH = 8
const CUDE_INPUT_DIM = 2
const BEST_NN_IDX = 3

const NN_FILE = "res/$(CUDE_EXPERIMENT)/models/nnNSTEMI_$(CUDE_EXPERIMENT).jld2"

# Output root — combined overlays are saved here
const OVERLAY_ROOT = "res/truncation_overlay_combined_legend_off"

# Original figure roots (for saving bar plots as requested)
const CUDE_FIGS_BASE = "res/$(CUDE_EXPERIMENT)/figs/MIMIC-UMG_systematic_truncation_cude_fixednn"
const ODE_FIGS_BASE = "res/$(ODE_EXPERIMENT)/figs/MIMIC-UMG_systematic_truncation_ode_tdsigmoid"

# Optional: generate parameter bar plots
const PLOT_PARAM_BARS = false

# Optional: toggle legend in overlap plots
const PLOT_LEGEND = false

# =========================== Load cUDE neural network =========================

@info "Loading cUDE neural network from $(NN_FILE) ..."
isfile(NN_FILE) || error("NN file not found: $(NN_FILE)")

chain = neural_network_model(CUDE_NN_DEPTH, CUDE_NN_WIDTH; input_dims=CUDE_INPUT_DIM)
@load NN_FILE neural_network_parameters
BEST_NN_IDX <= length(neural_network_parameters) ||
    error("BEST_NN_IDX=$(BEST_NN_IDX) out of range (have $(length(neural_network_parameters)) sets)")
best_nn = Vector{Float64}(neural_network_parameters[BEST_NN_IDX])
@info "Loaded NN parameters set #$(BEST_NN_IDX), length=$(length(best_nn))"

# =========================== Curve solvers ====================================

"""
Solve the ODE model (`troponin_ode!`) and return (t, plasma) with saveat=1.0.
`params_natural` = [a, b, Cs0, Cc0, Td] in natural scale.
"""
function solve_ode_curve(params_natural::Vector{Float64}; tmax::Float64=250.0)
    θ = log.(params_natural)
    u0 = [params_natural[3], params_natural[4], 0.0]
    prob = ODEProblem(troponin_ode!, u0, (0.0, tmax), θ)
    sol = solve(prob, Tsit5(); saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || error("ODE solve failed (retcode=$(sol.retcode))")
    return collect(sol.t), vec(sol[3, :])
end

"""
Solve the cUDE model (`ctnt_cude!`) with fixed NN and return (t, plasma) with saveat=1.0.
`params_natural` = [a, b, Cs0, Cc0, beta] in natural scale.
"""
function solve_cude_curve(params_natural::Vector{Float64}, chain, nn_params::Vector{Float64}; tmax::Float64=250.0)
    θ = log.(params_natural)
    u0 = [params_natural[3], params_natural[4], 0.0]

    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
    prob = ODEProblem(cude!, u0, (0.0, tmax))

    p_full = ComponentArray(ode=θ, neural=nn_params)
    sol = solve(prob, Tsit5(); p=p_full, saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || error("cUDE solve failed (retcode=$(sol.retcode))")
    return collect(sol.t), vec(sol[3, :])
end

# =========================== Patient data reconstruction ======================

"""
Reconstruct the full (un-truncated) base patient data by taking the union of
all truncated patient timepoints within the given long-format DataFrame.
"""
function reconstruct_base_patient(df_trunc::DataFrame)
    all_times = sort(unique(df_trunc.time))
    troponin = Float64[]
    for t in all_times
        rows = df_trunc[df_trunc.time.==t, :]
        push!(troponin, rows.troponin[1])
    end
    return all_times, troponin
end

"""
Determine kept and removed indices for a truncation scenario by matching
the truncated patient timepoints against the full base patient timepoints.
"""
function identify_kept_removed(base_times::Vector{Float64}, trunc_times::Vector{Float64})
    kept = Int[]
    for (i, t) in enumerate(base_times)
        if any(abs.(trunc_times .- t) .< 1e-6)
            push!(kept, i)
        end
    end
    removed = collect(setdiff(1:length(base_times), kept))
    return kept, removed
end

# =========================== Main processing loop =============================

# Discover patient directories present in both model runs
ode_patients = filter(d -> isdir(joinpath(ODE_BASE, d)), readdir(ODE_BASE))
cude_patients = filter(d -> isdir(joinpath(CUDE_BASE, d)), readdir(CUDE_BASE))
common_patients = sort(intersect(ode_patients, cude_patients))

isempty(common_patients) && error("No common patients found between ODE and cUDE results!")
@info "Found $(length(common_patients)) common patients: $(common_patients)"

mkpath(OVERLAY_ROOT)

n_plots_created = 0
n_skipped = 0

for patient_id in common_patients
    global n_plots_created, n_skipped
    @info "Processing patient: $(patient_id) ..."

    ode_models_dir = joinpath(ODE_BASE, patient_id)
    cude_models_dir = joinpath(CUDE_BASE, patient_id)

    # Per-patient output directory
    overlay_dir = joinpath(OVERLAY_ROOT, patient_id)
    mkpath(overlay_dir)

    # --- Load truncation params ---
    ode_params_csv = joinpath(ode_models_dir, "trunc_params.csv")
    cude_params_csv = joinpath(cude_models_dir, "trunc_params.csv")

    if !isfile(ode_params_csv)
        @warn "Missing ODE trunc_params for $(patient_id), skipping"
        continue
    end
    if !isfile(cude_params_csv)
        @warn "Missing cUDE trunc_params for $(patient_id), skipping"
        continue
    end

    ode_params_df = CSV.read(ode_params_csv, DataFrame)
    cude_params_df = CSV.read(cude_params_csv, DataFrame)

    # --- Load truncation metrics (sMAPE and RMSLE) ---
    ode_metrics_csv = joinpath(ode_models_dir, "trunc_metrics.csv")
    cude_metrics_csv = joinpath(cude_models_dir, "trunc_metrics.csv")

    if !isfile(ode_metrics_csv)
        @warn "Missing ODE trunc_metrics for $(patient_id), skipping"
        continue
    end
    if !isfile(cude_metrics_csv)
        @warn "Missing cUDE trunc_metrics for $(patient_id), skipping"
        continue
    end

    ode_metrics_df = CSV.read(ode_metrics_csv, DataFrame)
    cude_metrics_df = CSV.read(cude_metrics_csv, DataFrame)

    # --- Load truncated patient data to reconstruct base patient ---
    # Try ODE first, fall back to cUDE (both should have the same data)
    df_csv = joinpath(ode_models_dir, "df_$(patient_id).csv")
    if !isfile(df_csv)
        df_csv = joinpath(cude_models_dir, "df_$(patient_id).csv")
    end
    if !isfile(df_csv)
        @warn "Missing df CSV for $(patient_id), skipping"
        continue
    end
    df_trunc = CSV.read(df_csv, DataFrame)

    base_times, base_troponin = reconstruct_base_patient(df_trunc)
    tmax_solve = base_times[end] + 20.0

    # --- Iterate over all unique (trunc_section, trunc_set) pairs found in either model ---
    all_keys = unique(vcat(
        [(r.trunc_section, r.trunc_set) for r in eachrow(ode_params_df)],
        [(r.trunc_section, r.trunc_set) for r in eachrow(cude_params_df)]
    ))

    for (section, set_id) in all_keys
        # --- Find matching ODE row ---
        ode_match = filter(r -> r.trunc_section == section && r.trunc_set == set_id, ode_params_df)
        if nrow(ode_match) == 0
            @warn "No ODE params for $(patient_id) section=$(section) set=$(set_id), skipping scenario"
            n_skipped += 1
            continue
        end
        ode_row = ode_match[1, :]

        # --- Find matching cUDE row ---
        cude_match = filter(r -> r.trunc_section == section && r.trunc_set == set_id, cude_params_df)
        if nrow(cude_match) == 0
            @warn "No cUDE params for $(patient_id) section=$(section) set=$(set_id), skipping scenario"
            n_skipped += 1
            continue
        end
        cude_row = cude_match[1, :]

        # --- Find matching metrics rows ---
        ode_met = filter(r -> r.trunc_section == section && r.trunc_set == set_id, ode_metrics_df)
        cude_met = filter(r -> r.trunc_section == section && r.trunc_set == set_id, cude_metrics_df)

        ode_smape = nrow(ode_met) > 0 ? round(ode_met[1, :smape_full]; digits=2) : NaN
        ode_rmsle = nrow(ode_met) > 0 ? round(ode_met[1, :rmsle_full]; digits=4) : NaN
        cude_smape = nrow(cude_met) > 0 ? round(cude_met[1, :smape_full]; digits=2) : NaN
        cude_rmsle = nrow(cude_met) > 0 ? round(cude_met[1, :rmsle_full]; digits=4) : NaN

        # --- Extract params in natural scale ---
        ode_pars = Float64[ode_row.a, ode_row.b, ode_row.Cs0, ode_row.Cc0, ode_row.Td]
        cude_pars = Float64[cude_row.a, cude_row.b, cude_row.Cs0, cude_row.Cc0, cude_row.beta]

        # --- Solve curves ---
        ode_t, ode_plasma = try
            solve_ode_curve(ode_pars; tmax=tmax_solve)
        catch e
            @warn "ODE solve failed for $(patient_id) $(section) S$(set_id): $(e)"
            n_skipped += 1
            continue
        end

        cude_t, cude_plasma = try
            solve_cude_curve(cude_pars, chain, best_nn; tmax=tmax_solve)
        catch e
            @warn "cUDE solve failed for $(patient_id) $(section) S$(set_id): $(e)"
            n_skipped += 1
            continue
        end

        # --- Determine kept/removed from the ODE synthetic ID ---
        sid_ode = ode_row.synthetic_id
        trunc_patient_rows = filter(r -> r.patient_id == sid_ode, df_trunc)
        trunc_times = trunc_patient_rows.time

        kept_idx, removed_idx = identify_kept_removed(base_times, trunc_times)

        # --- Build overlay plot ---
        section_upper = uppercase(section)
        budget_str = lpad(string(length(removed_idx)), 2, "0")
        title_str = "$(patient_id) — $(section_upper) S$(set_id) B$(budget_str)"

        ode_label = "ODE  (sMAPE=$(ode_smape)%, RMSLE=$(ode_rmsle))"
        cude_label = "cUDE (sMAPE=$(cude_smape)%, RMSLE=$(cude_rmsle))"

        plt = Plots.plot(
            ode_t, ode_plasma;
            lw=2, color=:royalblue, linestyle=:solid,
            label=ode_label,
            xlabel="Time (h)",
            ylabel="cTnT [ng/mL]",
            # title=title_str,
            legend=PLOT_LEGEND ? :best : false,
            legendfontsize=8,
            grid=true,
            size=(900, 550)
        )

        Plots.plot!(
            plt,
            cude_t, cude_plasma;
            lw=2, color=:darkorange, linestyle=:dash,
            label=cude_label
        )

        # Removed measurements (X markers)
        if !isempty(removed_idx)
            Plots.scatter!(
                plt,
                base_times[removed_idx],
                base_troponin[removed_idx];
                markershape=:x,
                markerstrokewidth=2,
                ms=7,
                color=:crimson,
                label="Removed (n=$(length(removed_idx)))"
            )
        end

        # Used measurements (circle markers)
        if !isempty(kept_idx)
            Plots.scatter!(
                plt,
                base_times[kept_idx],
                base_troponin[kept_idx];
                markershape=:circle,
                ms=5,
                color=:dodgerblue,
                label="Kept (n=$(length(kept_idx)))"
            )
        end

        # --- Save overlay ---
        fname = "overlay_$(patient_id)_$(section_upper)_S$(set_id)_B$(budget_str).svg"
        save_path = joinpath(overlay_dir, fname)
        Plots.savefig(plt, save_path)
        n_plots_created += 1

        @info "  → Saved overlay: $(save_path)"

        # --- Optional: Parameter Bar Plots ---
        if PLOT_PARAM_BARS
            # Labels for the 5 parameters
            param_labels = ["a", "b", "Cs0", "Cc0", "Td/β"]

            # Extract ratios (ODE uses Td, cUDE uses beta at index 5)
            ode_ratios = [
                ode_row.a_ratio_vs_full,
                ode_row.b_ratio_vs_full,
                ode_row.Cs0_ratio_vs_full,
                ode_row.Cc0_ratio_vs_full,
                ode_row.Td_ratio_vs_full
            ]
            cude_ratios = [
                cude_row.a_ratio_vs_full,
                cude_row.b_ratio_vs_full,
                cude_row.Cs0_ratio_vs_full,
                cude_row.Cc0_ratio_vs_full,
                cude_row.beta_ratio_vs_full
            ]

            # Create grouped bar plot
            # We use a matrix for grouped bars: each row is a parameter, each column a model
            bar_data = hcat(ode_ratios, cude_ratios)

            p_bar = Plots.bar(
                param_labels,
                bar_data;
                label=["ODE" "cUDE"],
                color=[:royalblue :darkorange], # Match curve colors
                alpha=0.8,
                ylabel="Ratio vs Full-Data Fit",
                title="Param Stability — $(patient_id) $(section_upper) S$(set_id)",
                ylim=(0, 2.5), # Cap for readability, common in these plots
                grid=true,
                size=(700, 450)
            )

            # Add baseline line at 1.0
            Plots.hline!(p_bar, [1.0], lw=1.5, color=:black, linestyle=:dash, label="Baseline (1.0)")

            # Save in combined folder
            bname = "params_$(patient_id)_$(section_upper)_S$(set_id)_B$(budget_str).svg"
            Plots.savefig(p_bar, joinpath(overlay_dir, bname))

            # Save in original ODE figs folder
            ode_fig_dir = joinpath(ODE_FIGS_BASE, patient_id)
            if isdir(ode_fig_dir)
                Plots.savefig(p_bar, joinpath(ode_fig_dir, bname))
            end

            # Save in original cUDE figs folder
            cude_fig_dir = joinpath(CUDE_FIGS_BASE, patient_id)
            if isdir(cude_fig_dir)
                Plots.savefig(p_bar, joinpath(cude_fig_dir, bname))
            end

            @info "  → Saved bar plot: $(bname)"
        end
    end
end

@info "Overlay plot generation complete. Total plots created: $(n_plots_created), skipped: $(n_skipped)"

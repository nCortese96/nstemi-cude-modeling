"""
run_diagnostics.jl

Refactored copy of `plot_diagnostics.jl`.

Compute and plot residual and parameter diagnostics for ODE and cUDE outputs.

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
using DataFrames, CSV, JLD2, Statistics
using CairoMakie
using OrdinaryDiffEq: Tsit5
using SciMLBase: successful_retcode, ODEProblem, remake
using ComponentArrays: ComponentArray
include("MechanisticAI.jl")

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
# Unified Diagnostics: Residuals + Parameter Boxplots (ODE & cUDE)
# =============================================================================
#
# Replaces and unifies:
#   - boxplotting_cude.jl
#   - residuals_calculations.jl
#   - residual_calculation refactor.jl
#   - statistics_cairo refactor.jl
#
# Fixed experiment configurations:
#   cUDE: NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback, NN #3, ms_test
#   ODE:  NSTEMI_ODE_TdSigmoid, opt_lambda1
# =============================================================================

# ========================= Section 1: Configuration ==========================

const DIAG_OUT = "res/diagnostics"
const DIAG_FIGS = joinpath(DIAG_OUT, "figs")
const DIAG_MODELS = joinpath(DIAG_OUT, "models")
mkpath(DIAG_FIGS)
mkpath(DIAG_MODELS)

# ── cUDE paths ──
const CUDE_BASE = "res/NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback/models"
const CUDE_NN_JLD2 = joinpath(CUDE_BASE, "nnNSTEMI_NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback.jld2")
const CUDE_BEST_IDX = 3
const CUDE_NN_DEPTH = 2
const CUDE_NN_WIDTH = 8
const CUDE_NN_IN = 2  # cUDE input: (τ, β)

const CUDE_MIMIC_PARAMS = joinpath(CUDE_BASE, "MIMIC-IV_test_NN_3_ms_test", "patients_params_val.csv")
const CUDE_MIMIC_METRICS = joinpath(CUDE_BASE, "MIMIC-IV_test_NN_3_ms_test", "patients_metrics_val.csv")
const CUDE_UMG_PARAMS = joinpath(CUDE_BASE, "UMG_test_NN_ab10_3_ms_test", "patients_params_val.csv")
const CUDE_UMG_METRICS = joinpath(CUDE_BASE, "UMG_test_NN_ab10_3_ms_test", "patients_metrics_val.csv")
# const CUDE_MIMIC_PARAMS_LOG = joinpath(CUDE_BASE, "MIMIC-IV_test_NN_3_ms_test", "best_params_val_MIMIC-IV.jld2")
# const CUDE_UMG_PARAMS_LOG = joinpath(CUDE_BASE, "UMG_test_NN_ab10_3_ms_test", "best_params_val_UMG.jld2")

# ── ODE paths ──
const ODE_MIMIC_CSV = "res/NSTEMI_ODE_TdSigmoid/MIMIC-IV_opt_lambda1/models/params_out_val.csv"
const ODE_UMG_CSV = "res/NSTEMI_ODE_TdSigmoid/UMG_opt_lambda1/models/params_out.csv"

# ── Patient data ──
# const DS_MIMIC_TRAIN = "res/MIMIC-IV_trainingset.jld2"
const DS_MIMIC_TEST = "res/MIMIC-IV_testset.jld2"
const DS_UMG_TEST = "res/UMG_testset.jld2"

# ── Plotting constants ──
const TMAX_DIAG = 240.0
const EDGES_DIAG = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, TMAX_DIAG]

# ======================== Section 2: Data Loading ============================

@info "Loading Neural Network parameters..."
@load CUDE_NN_JLD2 neural_network_parameters
const BEST_NN = Vector{Float64}(neural_network_parameters[CUDE_BEST_IDX])
const CHAIN = neural_network_model(CUDE_NN_DEPTH, CUDE_NN_WIDTH; input_dims=CUDE_NN_IN)

@info "Loading patient datasets..."
PATIENTS_MIMIC = Dict{String,PatientData}()
PATIENTS_UMG = Dict{String,PatientData}()

function _load_ds!(dict, path, key)
    if isfile(path)
        ds = JLD2.load(path, key)
        for p in ds
            dict[p.id] = p
        end
    else
        @warn "File not found: $path"
    end
end

# _load_ds!(PATIENTS, DS_MIMIC_TRAIN, "training_dataset")
_load_ds!(PATIENTS_MIMIC, DS_MIMIC_TEST, "test_dataset")
_load_ds!(PATIENTS_UMG, DS_UMG_TEST, "test_dataset")
@info "Loaded $(length(PATIENTS_MIMIC)) distinct patient profiles for MIMIC-IV."
@info "Loaded $(length(PATIENTS_UMG)) distinct patient profiles for UMG."

@info "Loading parameter CSVs..."
df_ode_mimic = CSV.read(ODE_MIMIC_CSV, DataFrame)
df_ode_umg = CSV.read(ODE_UMG_CSV, DataFrame)
df_cude_mimic = CSV.read(CUDE_MIMIC_PARAMS, DataFrame)
df_cude_umg = CSV.read(CUDE_UMG_PARAMS, DataFrame)
# df_cude_mimic_log = permutedims(reshape(JLD2.load(CUDE_MIMIC_PARAMS_LOG, "ode_params_val"), 5, :))
# df_cude_umg_log = permutedims(reshape(JLD2.load(CUDE_UMG_PARAMS_LOG, "ode_params_val"), 5, :))

@info "Patient counts:"
@info "  ODE  MIMIC-IV (val): $(nrow(df_ode_mimic))"
@info "  ODE  UMG:            $(nrow(df_ode_umg))"
@info "  cUDE MIMIC-IV (val): $(nrow(df_cude_mimic))"
@info "  cUDE UMG (ab10):     $(nrow(df_cude_umg))"

# ==================== Section 3: Residual Computation ========================

@info "Computing residuals for all 4 model×dataset combinations..."

res_ode_mimic, met_ode_mimic, par_ode_mimic = compute_residuals_long_unified(
    PATIENTS_MIMIC, df_ode_mimic; model_type=:ode)
@info "  ✓ ODE × MIMIC-IV: $(nrow(res_ode_mimic)) residual points"
@info mean(met_ode_mimic.smape_val)

res_ode_umg, met_ode_umg, par_ode_umg = compute_residuals_long_unified(
    PATIENTS_UMG, df_ode_umg; model_type=:ode)
@info "  ✓ ODE × UMG: $(nrow(res_ode_umg)) residual points"
@info mean(met_ode_umg.smape_val)

res_cude_mimic, met_cude_mimic, par_cude_mimic = compute_residuals_long_unified(
    PATIENTS_MIMIC, df_cude_mimic; model_type=:cude, chain=CHAIN, nn_params=BEST_NN)
@info "  ✓ cUDE × MIMIC-IV: $(nrow(res_cude_mimic)) residual points"
@info mean(met_cude_mimic.smape_val)

res_cude_umg, met_cude_umg, par_cude_umg = compute_residuals_long_unified(
    PATIENTS_UMG, df_cude_umg; model_type=:cude, chain=CHAIN, nn_params=BEST_NN)
@info "  ✓ cUDE × UMG: $(nrow(res_cude_umg)) residual points"
@info mean(met_cude_umg.smape_val)

# Save residuals CSVs
for (label, df) in [("ODE_MIMIC", res_ode_mimic), ("ODE_UMG", res_ode_umg),
    ("cUDE_MIMIC", res_cude_mimic), ("cUDE_UMG", res_cude_umg)]
    CSV.write(joinpath(DIAG_MODELS, "residuals_$(label).csv"), df)
end
@info "Residual CSVs saved."

# Save metrics CSVs
for (label, df) in [("ODE_MIMIC", met_ode_mimic), ("ODE_UMG", met_ode_umg),
    ("cUDE_MIMIC", met_cude_mimic), ("cUDE_UMG", met_cude_umg)]
    CSV.write(joinpath(DIAG_MODELS, "metrics_$(label).csv"), df)
end
@info "Metrics CSVs saved."

# =================== Section 4: Residuals vs Fitted ==========================

@info "Generating Residuals vs Fitted plot..."

function plot_res_vs_fitted!(ax, df; title="")
    ϵ = 1e-10
    CairoMakie.scatter!(ax, log.(df.yhat .+ ϵ), df.res;
        markersize=5, color=(:black, 0.25), label="Residuals")
    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6))
    ax.title = title
    ax.xlabel = "log predicted ŷ"
    ax.ylabel = "log residual"
end

fig_fitted = CairoMakie.Figure(size=(1200, 900), fontsize=14)

ax_ff = [
    Axis(fig_fitted[1, 1]),  # ODE MIMIC
    Axis(fig_fitted[1, 2]),  # cUDE MIMIC
    Axis(fig_fitted[2, 1]),  # ODE UMG
    Axis(fig_fitted[2, 2]),  # cUDE UMG
]

plot_res_vs_fitted!(ax_ff[1], res_ode_mimic; title="ODE — MIMIC-IV")
plot_res_vs_fitted!(ax_ff[2], res_cude_mimic; title="cUDE — MIMIC-IV")
plot_res_vs_fitted!(ax_ff[3], res_ode_umg; title="ODE — UMG")
plot_res_vs_fitted!(ax_ff[4], res_cude_umg; title="cUDE — UMG")

fig_fitted_mimic = CairoMakie.Figure(size=(1200, 500), fontsize=14)

ax_ff_mimic = [
    Axis(fig_fitted_mimic[1, 1]),  # ODE MIMIC
    Axis(fig_fitted_mimic[1, 2]),  # cUDE MIMIC
]

fig_fitted_umg = CairoMakie.Figure(size=(1200, 500), fontsize=14)

ax_ff_umg = [
    Axis(fig_fitted_umg[1, 1]),  # ODE UMG
    Axis(fig_fitted_umg[1, 2]),  # cUDE UMG
]

plot_res_vs_fitted!(ax_ff_mimic[1], res_ode_mimic; title="ODE — MIMIC-IV")
plot_res_vs_fitted!(ax_ff_mimic[2], res_cude_mimic; title="cUDE — MIMIC-IV")
plot_res_vs_fitted!(ax_ff_umg[1], res_ode_umg; title="ODE — UMG")
plot_res_vs_fitted!(ax_ff_umg[2], res_cude_umg; title="cUDE — UMG")

# Label(fig_fitted[0, :], "Residuals vs Fitted Values"; fontsize=20, tellwidth=false)

CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_fitted.svg"), fig_fitted)
CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_fitted.png"), fig_fitted, px_per_unit=3)

CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_fitted_mimic.svg"), fig_fitted_mimic)
CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_fitted_mimic.png"), fig_fitted_mimic, px_per_unit=3)

CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_fitted_umg.svg"), fig_fitted_umg)
CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_fitted_umg.png"), fig_fitted_umg, px_per_unit=3)

@info "  ✓ Residuals vs Fitted saved."

# =================== Section 5: Residuals vs Time ===========================

@info "Generating Residuals vs Time plot..."

function plot_res_vs_time!(ax, df, edges; title="", tmax=TMAX_DIAG, nmin=1)
    add_time_bins!(df, edges)
    s = bin_summary(df)

    # Mask bins with too few points
    med_m = [s.n[i] >= nmin ? s.med[i] : NaN for i in eachindex(s.n)]
    q1_m = [s.n[i] >= nmin ? s.q1[i] : NaN for i in eachindex(s.n)]
    q3_m = [s.n[i] >= nmin ? s.q3[i] : NaN for i in eachindex(s.n)]

    CairoMakie.scatter!(ax, df.t, df.res; markersize=4, color=(:black, 0.2))
    CairoMakie.lines!(ax, s.centers, med_m; linewidth=2, color=:blue, label="Median")
    CairoMakie.band!(ax, s.centers, q1_m, q3_m; color=(:gray, 0.2), label="IQR")
    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6))
    CairoMakie.vlines!(ax, edges[2:end-1]; color=(:black, 0.3), linewidth=1, linestyle=:dash)
    CairoMakie.xlims!(ax, 0, tmax)

    # Bin counts
    for i in eachindex(s.centers)
        x_rel = clamp(s.centers[i] / tmax, 0.0, 1.0)
        CairoMakie.text!(ax, x_rel, 0.96; text="n=$(s.n[i])", space=:relative,
            align=(:center, :top), rotation=pi / 4, fontsize=10, color=(:black, 0.7))
    end

    ax.title = title
    ax.xlabel = "Time (h)"
    ax.ylabel = "log residual"
end

fig_time = CairoMakie.Figure(size=(1400, 900), fontsize=14)

ax_ft = [
    Axis(fig_time[1, 1]),
    Axis(fig_time[1, 2]),
    Axis(fig_time[2, 1]),
    Axis(fig_time[2, 2]),
]

plot_res_vs_time!(ax_ft[1], res_ode_mimic, EDGES_DIAG; title="ODE — MIMIC-IV")
plot_res_vs_time!(ax_ft[2], res_cude_mimic, EDGES_DIAG; title="cUDE — MIMIC-IV")
plot_res_vs_time!(ax_ft[3], res_ode_umg, EDGES_DIAG; title="ODE — UMG")
plot_res_vs_time!(ax_ft[4], res_cude_umg, EDGES_DIAG; title="cUDE — UMG")

fig_time_mimic = CairoMakie.Figure(size=(1400, 500), fontsize=14)

ax_ft_mimic = [
    Axis(fig_time_mimic[1, 1]),  # ODE MIMIC
    Axis(fig_time_mimic[1, 2]),  # cUDE MIMIC
]

fig_time_umg = CairoMakie.Figure(size=(1400, 500), fontsize=14)

ax_ft_umg = [
    Axis(fig_time_umg[1, 1]),  # ODE UMG
    Axis(fig_time_umg[1, 2]),  # cUDE UMG
]

plot_res_vs_time!(ax_ft_mimic[1], res_ode_mimic, EDGES_DIAG; title="ODE — MIMIC-IV")
plot_res_vs_time!(ax_ft_mimic[2], res_cude_mimic, EDGES_DIAG; title="cUDE — MIMIC-IV")
plot_res_vs_time!(ax_ft_umg[1], res_ode_umg, EDGES_DIAG; title="ODE — UMG")
plot_res_vs_time!(ax_ft_umg[2], res_cude_umg, EDGES_DIAG; title="cUDE — UMG")

# Label(fig_time[0, :], "Residuals vs Time"; fontsize=20, tellwidth=false)

CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_time.svg"), fig_time)
CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_time.png"), fig_time, px_per_unit=3)

CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_time_mimic.svg"), fig_time_mimic)
CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_time_mimic.png"), fig_time_mimic, px_per_unit=3)

CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_time_umg.svg"), fig_time_umg)
CairoMakie.save(joinpath(DIAG_FIGS, "residuals_vs_time_umg.png"), fig_time_umg, px_per_unit=3)

@info "  ✓ Residuals vs Time saved."

# ============ Section 6: Parameter Boxplots per Model ========================
# For each model, compare parameter distributions across datasets

@info "Generating per-model parameter boxplots..."

const COLORS_DATASET = Dict("MIMIC-IV" => :steelblue, "UMG" => :darkorange)

function make_param_boxplot_per_model(par_df_ds1, par_df_ds2, model_name, par_names)
    n_params = length(par_names)
    fig = CairoMakie.Figure(size=(300 * n_params, 550), fontsize=14)
    # Label(fig[0, 1:n_params], "$(model_name) — Parameter Distributions by Dataset";
    # fontsize=20, tellwidth=false)

    for (i, pname) in enumerate(par_names)
        col_sym = (i <= 4) ? [:a, :b, :Cs0, :Cc0][i] : :p5
        vals = vcat(par_df_ds1[!, col_sym], par_df_ds2[!, col_sym])
        groups = vcat(fill(1, nrow(par_df_ds1)), fill(2, nrow(par_df_ds2)))
        colors = [g == 1 ? COLORS_DATASET["MIMIC-IV"] : COLORS_DATASET["UMG"] for g in groups]

        ax = Axis(fig[1, i], title=pname)
        CairoMakie.boxplot!(ax, groups, vals; color=colors, whiskerwidth=0.4, strokewidth=0.5)
        ax.xticks = (1:2, ["MIMIC-IV", "UMG"])
        ax.xticklabelrotation = pi / 5
    end

    return fig
end

# ── cUDE boxplots ──
cude_par_names = ["a", "b", "Cs0", "Cc0", "β"]
fig_cude_box = make_param_boxplot_per_model(
    par_cude_mimic, par_cude_umg, "cUDE", cude_par_names)

CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cUDE_by_dataset.svg"), fig_cude_box)
CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cUDE_by_dataset.png"), fig_cude_box, px_per_unit=3)
@info "  ✓ cUDE parameter boxplots saved."

# ── ODE boxplots ──
ode_par_names = ["a", "b", "Cs0", "Cc0", "Td"]
fig_ode_box = make_param_boxplot_per_model(
    par_ode_mimic, par_ode_umg, "ODE", ode_par_names)

CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_ODE_by_dataset.svg"), fig_ode_box)
CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_ODE_by_dataset.png"), fig_ode_box, px_per_unit=3)
@info "  ✓ ODE parameter boxplots saved."

# ============ Section 7: Parameter Boxplots Cross-Model ======================
# For each dataset, compare shared params (a, b, Cs0, Cc0) between ODE and cUDE,
# plus separate panels for model-specific params (Td, β)

@info "Generating cross-model parameter boxplots..."

const COLORS_MODEL = Dict("ODE" => :royalblue, "cUDE" => :darkorange)
const SHARED_PARAMS = ["a", "b", "Cs0", "Cc0"]

function make_param_boxplot_cross_model(par_ode, par_cude, dataset_name; complete_plot::Bool=true)

    figs = []
    if complete_plot
        # 4 shared + Td standalone + β standalone = 6 columns
        fig = CairoMakie.Figure(size=(1800, 550), fontsize=14)
        # Label(fig[0, 1:6], "$(dataset_name) — Parameter Distributions by Model";
        # fontsize=20, tellwidth=false)

        # Shared parameters (a, b, Cs0, Cc0) — side-by-side ODE vs cUDE
        for (i, pname) in enumerate(SHARED_PARAMS)
            col_sym = [:a, :b, :Cs0, :Cc0][i]
            vals = vcat(par_ode[!, col_sym], par_cude[!, col_sym])
            groups = vcat(fill(1, nrow(par_ode)), fill(2, nrow(par_cude)))
            colors = [g == 1 ? COLORS_MODEL["ODE"] : COLORS_MODEL["cUDE"] for g in groups]

            ax = Axis(fig[1, i], title=pname)
            CairoMakie.boxplot!(ax, groups, vals; color=colors, whiskerwidth=0.4, strokewidth=0.5)
            ax.xticks = (1:2, ["ODE", "cUDE"])
            ax.xticklabelrotation = pi / 5
        end

        # Td (ODE only) — standalone
        ax_td = Axis(fig[1, 5], title="Td (ODE)")
        CairoMakie.boxplot!(ax_td, fill(1, nrow(par_ode)), par_ode.p5;
            color=COLORS_MODEL["ODE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_td.xticks = ([1], ["ODE"])
        ax_td.xticklabelrotation = pi / 5

        # β (cUDE only) — standalone
        ax_beta = Axis(fig[1, 6], title="β (cUDE)")
        CairoMakie.boxplot!(ax_beta, fill(1, nrow(par_cude)), par_cude.p5;
            color=COLORS_MODEL["cUDE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_beta.xticks = ([1], ["cUDE"])
        ax_beta.xticklabelrotation = pi / 5

        push!(figs, fig)
    else
        # 4 shared parameters only
        fig = CairoMakie.Figure(size=(1200, 550), fontsize=14)
        # Label(fig[0, 1:4], "$(dataset_name) — Parameter Distributions by Model";
        # fontsize=20, tellwidth=false)

        for (i, pname) in enumerate(SHARED_PARAMS)
            col_sym = [:a, :b, :Cs0, :Cc0][i]
            vals = vcat(par_ode[!, col_sym], par_cude[!, col_sym])
            groups = vcat(fill(1, nrow(par_ode)), fill(2, nrow(par_cude)))
            colors = [g == 1 ? COLORS_MODEL["ODE"] : COLORS_MODEL["cUDE"] for g in groups]

            ax = Axis(fig[1, i], title=pname)
            CairoMakie.boxplot!(ax, groups, vals; color=colors, whiskerwidth=0.4, strokewidth=0.5)
            ax.xticks = (1:2, ["ODE", "cUDE"])
            ax.xticklabelrotation = pi / 5
        end
        push!(figs, fig)

        # Td and β boxplots
        fig = CairoMakie.Figure(size=(600, 550), fontsize=14)
        ax_td = Axis(fig[1, 1], title="Td (ODE)")
        CairoMakie.boxplot!(ax_td, fill(1, nrow(par_ode)), par_ode.p5;
            color=COLORS_MODEL["ODE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_td.xticks = ([1], ["ODE"])
        ax_td.xticklabelrotation = pi / 5
        ax_beta = Axis(fig[1, 2], title="β (cUDE)")
        CairoMakie.boxplot!(ax_beta, fill(1, nrow(par_cude)), par_cude.p5;
            color=COLORS_MODEL["cUDE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_beta.xticks = ([1], ["cUDE"])
        ax_beta.xticklabelrotation = pi / 5
        push!(figs, fig)

        # Td (ODE only) — standalone
        fig = CairoMakie.Figure(size=(300, 550), fontsize=14)
        ax_td = Axis(fig[1, 1], title="Td (ODE)")
        CairoMakie.boxplot!(ax_td, fill(1, nrow(par_ode)), par_ode.p5;
            color=COLORS_MODEL["ODE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_td.xticks = ([1], ["ODE"])
        ax_td.xticklabelrotation = pi / 5
        push!(figs, fig)

        # β (cUDE only) — standalone
        fig = CairoMakie.Figure(size=(300, 550), fontsize=14)
        ax_beta = Axis(fig[1, 1], title="β (cUDE)")
        CairoMakie.boxplot!(ax_beta, fill(1, nrow(par_cude)), par_cude.p5;
            color=COLORS_MODEL["cUDE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_beta.xticks = ([1], ["cUDE"])
        ax_beta.xticklabelrotation = pi / 5
        push!(figs, fig)
    end

    return figs
end

# ── MIMIC-IV cross-model ──
fig_cross_mimic = make_param_boxplot_cross_model(par_ode_mimic, par_cude_mimic, "MIMIC-IV")
CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_MIMIC.svg"), fig_cross_mimic[1])
CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_MIMIC.png"), fig_cross_mimic[1], px_per_unit=3)
@info "  ✓ MIMIC-IV cross-model boxplots saved."

# ── UMG cross-model ──
fig_cross_umg = make_param_boxplot_cross_model(par_ode_umg, par_cude_umg, "UMG")
CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_UMG.svg"), fig_cross_umg[1])
CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_UMG.png"), fig_cross_umg[1], px_per_unit=3)
@info "  ✓ UMG cross-model boxplots saved."

separated_figs = make_param_boxplot_cross_model(par_ode_mimic, par_cude_mimic, "MIMIC-IV"; complete_plot=false)
for (i, fig) in enumerate(separated_figs)
    CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_MIMIC_separated_$(i).svg"), fig)
    CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_MIMIC_separated_$(i).png"), fig, px_per_unit=3)
end
@info "  ✓ MIMIC-IV cross-model boxplots saved."

separated_figs = make_param_boxplot_cross_model(par_ode_umg, par_cude_umg, "UMG"; complete_plot=false)
for (i, fig) in enumerate(separated_figs)
    CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_UMG_separated_$(i).svg"), fig)
    CairoMakie.save(joinpath(DIAG_FIGS, "boxplot_params_cross_model_UMG_separated_$(i).png"), fig, px_per_unit=3)
end
@info "  ✓ UMG cross-model boxplots saved."

# ==================== Section 8: Summary Statistics ==========================

@info "Computing summary statistics..."

function summary_stats(vals::AbstractVector)
    mu = mean(vals)
    s = std(vals)
    q1, med, q3 = quantile(vals, (0.25, 0.5, 0.75))
    iqr = q3 - q1
    return (mean=mu, std=s, q1=q1, median=med, q3=q3, iqr=iqr)
end

# Parameter summary
param_summary = DataFrame(
    model=String[], dataset=String[], param=String[],
    mean=Float64[], std=Float64[], q1=Float64[], median=Float64[], q3=Float64[], iqr=Float64[]
)

for (model, dataset, pdf, names) in [
    ("ODE", "MIMIC-IV", par_ode_mimic, ["a", "b", "Cs0", "Cc0", "Td"]),
    ("ODE", "UMG", par_ode_umg, ["a", "b", "Cs0", "Cc0", "Td"]),
    ("cUDE", "MIMIC-IV", par_cude_mimic, ["a", "b", "Cs0", "Cc0", "beta"]),
    ("cUDE", "UMG", par_cude_umg, ["a", "b", "Cs0", "Cc0", "beta"])
]
    for (i, pname) in enumerate(names)
        col = (i <= 4) ? [:a, :b, :Cs0, :Cc0][i] : :p5
        ss = summary_stats(pdf[!, col])
        push!(param_summary, (model, dataset, pname, ss.mean, ss.std, ss.q1, ss.median, ss.q3, ss.iqr))
    end
end

CSV.write(joinpath(DIAG_MODELS, "parameter_summary.csv"), param_summary)

# Metrics summary
metrics_summary = DataFrame(
    model=String[], dataset=String[], metric=String[],
    mean=Float64[], std=Float64[], q1=Float64[], median=Float64[], q3=Float64[], iqr=Float64[]
)

for (model, dataset, mdf) in [
    ("ODE", "MIMIC-IV", met_ode_mimic),
    ("ODE", "UMG", met_ode_umg),
    ("cUDE", "MIMIC-IV", met_cude_mimic),
    ("cUDE", "UMG", met_cude_umg)
]
    for (mname, col) in [("sMAPE", :smape_val), ("RMSLE", :rmsle_val)]
        ss = summary_stats(mdf[!, col])
        push!(metrics_summary, (model, dataset, mname, ss.mean, ss.std, ss.q1, ss.median, ss.q3, ss.iqr))
    end
end

CSV.write(joinpath(DIAG_MODELS, "metrics_summary.csv"), metrics_summary)
@info "  ✓ Summary statistics saved."

@info "================================================================"
@info "  Diagnostics script completed successfully!"
@info "  Figures: $(DIAG_FIGS)"
@info "  CSVs:   $(DIAG_MODELS)"
@info "================================================================"

# ============================================================
# MODULE OVERVIEW
# ============================================================
#
# PURPOSE
# -------
# Fair, reproducible selection of the best cUDE candidate model
# from per-patient validation metrics produced by multiple training
# runs (different widths and random initialisations).
#
# PIPELINE SUMMARY
# ----------------
#   1. read_candidate_metrics          — load one CSV, attach metadata
#   2. align_candidates_on_common_patients — restrict to the shared
#        patient subset so every metric is computed on identical data
#   3. build_metric_matrices           — stack per-patient vectors into
#        (n_patients × n_candidates) matrices for vectorised ops
#   4. count_smape_wins                — per-patient best-model count
#   5. pairwise_smape_tables           — head-to-head W_ij matrices
#   6. compute_dominance_scores        — average net pairwise advantage
#   7. bootstrap_ranking_stability     — non-parametric confidence on
#        the ranking via patient resampling
#   8. summarize_and_rank_candidates   — combine all metrics, sort by
#        the 5-criterion lexicographic rule, store alternative ranks
#   9. summarize_configs               — best candidate per width
#  10. Plot functions (see below)
#  11. run_validation_model_selection  — single entry point; calls
#        steps 1-10 and writes all CSV + PNG outputs
#
# RANKING RULE (lexicographic, in priority order)
# ------------------------------------------------
#   1. wins_smape      ↓   per-patient win count (higher = better)
#   2. dominance_score ↓   mean(W_ij - W_ji)/N over all opponents
#   3. median_smape    ↑   central tendency
#   4. q90_smape       ↑   tail robustness (worst-case patients)
#   5. median_rmsle    ↑   secondary metric tie-break
#
# WHY DOMINANCE SCORE?
# --------------------
# Win count alone can produce a paradox: the global winner may
# lose the direct head-to-head against the runner-up (non-transitive
# rankings are common with small N).  The dominance score D_k
# measures how consistently a model outperforms across *all* opponents,
# not just globally.  A positive D_k means the model wins more patients
# than it loses on average.
#
# BOOTSTRAP STABILITY
# -------------------
# With N=56 patients the win margin between top candidates can be small.
# bootstrap_ranking_stability() resamples patients with replacement
# 1000 times and records how often each candidate ranks first.
# A top-1 frequency ≥ 0.50 indicates an unambiguous winner.
# The 95% CI on win rate is saved in win_rate_ci_lo / win_rate_ci_hi.
#
# PLOTS PRODUCED
# --------------
#   win_rate_bar.png
#       Horizontal bar chart of win rate (%) with 95% bootstrap CI.
#       Bars coloured by NN width; dominance score D annotated.
#
#   pareto_median_vs_q90_smape.png
#       Scatter of median sMAPE (x) vs 90th-pct sMAPE (y).
#       Point size ∝ win rate; Pareto-front models circled.
#       Ideal model is bottom-left with large marker.
#
#   pairwise_net_dominance_heatmap.png
#       Heatmap of (W_ij - W_ji)/N with diverging colormap.
#       Blue = row model dominates; red = column model dominates.
#       More informative than raw win counts because it is
#       normalised and symmetric around zero.
#
#   topk_smape_ecdf.png
#       ECDF of per-patient sMAPE for the top-k candidates.
#       Vertical dashed lines = median; dotted lines = 90th pct.
#       A left-shifted, steep ECDF is better.
#
#   delta_smape_best_vs_runnerup.png
#       Per-patient signed difference sMAPE(runner-up) - sMAPE(best),
#       sorted and shown as a bar chart.  Blue = best model wins,
#       red = runner-up wins.  Directly answers: "for which patients
#       does the selection matter, and by how much?"
#
#   bootstrap_top1_frequency.png
#       Frequency with which each candidate ranks first across 1000
#       bootstrap replicates.  Use to judge whether the selection
#       is stable or whether multiple candidates are equivalent.
#
#   all_smape_ecdf_supplementary.png
#       All 16 candidates — intended for supplementary material only.
#
# TYPICAL USAGE
# -------------
#   candidate_specs = [
#       (path = "res/EXP_w4/models/MIMIC-IV_test_NN_1_ms_test/patients_metrics_val.csv",
#        config_code = "24", model_index = 1),
#       ...
#   ]
#   results = run_validation_model_selection(candidate_specs;
#       output_dir  = "res/model_selection",
#       eps_smape   = 0.0,     # tie tolerance (sMAPE %)
#       tie_mode    = :split,  # :split | :all | :first
#       top_k_ecdf  = 4,       # candidates shown in ECDF plot
#       n_bootstrap = 1000,    # bootstrap replicates
#   )
#   # results.ranking         — full ranked DataFrame
#   # results.config_summary  — best model per width
#   # results.smape_mat       — (N_patients × K_candidates) sMAPE matrix
#   # results.pairwise_wins   — W_ij matrix
# ============================================================

using CSV
using DataFrames
using Statistics
using Printf
using Plots
using Random

# ============================================================
# Validation-set model selection from per-patient CSV metrics
# ------------------------------------------------------------
# Expected CSV columns:
#   patient_id, smape, rmsle, loss
#
# Typical usage from the REPL:
#
# results = run_validation_model_selection(candidate_specs;
#     output_dir = "res/model_selection",
#     eps_smape = 0.0,
#     tie_mode = :split,
#     top_k_ecdf = 4,
#     n_bootstrap = 1000,
# )
# ============================================================

const REQUIRED_COLUMNS = [:patient_id, :smape, :rmsle, :loss]

"""
Parse config/model metadata from filenames such as:
- model_26_patients_metrics_val_tesn_NN_1.csv
- model_28_patients_metrics_val_tesn_NN_2.csv.csv
- model_216_patients_metrics_val_tesn_NN_3.csv
"""
function parse_candidate_metadata(path::AbstractString)
    filename = basename(path)
    m = match(r"model_(\d+)_patients_metrics.*?_(\d+)\.csv(?:\.csv)?$", filename)
    m === nothing && error("Cannot parse config/model metadata from filename: $filename")

    config_code = m.captures[1]
    model_index = parse(Int, m.captures[2])

    width_label = startswith(config_code, "2") && length(config_code) >= 2 ? config_code[2:end] : config_code
    candidate_id = "cfg$(config_code)_m$(model_index)"

    return (
        filename = filename,
        config_code = config_code,
        width_label = width_label,
        model_index = model_index,
        candidate_id = candidate_id,
    )
end

"""
Read one CSV and attach metadata.
"""
function read_candidate_metrics(path::AbstractString; meta_override = nothing)
    df = CSV.read(path, DataFrame)

    missing_cols = setdiff(REQUIRED_COLUMNS, Symbol.(names(df)))
    isempty(missing_cols) || error("File $(basename(path)) is missing required columns: $missing_cols")

    meta = isnothing(meta_override) ? parse_candidate_metadata(path) : meta_override

    out = select(df, REQUIRED_COLUMNS)
    out.patient_id = string.(out.patient_id)
    out.smape = Float64.(out.smape)
    out.rmsle = Float64.(out.rmsle)
    out.loss = Float64.(out.loss)

    out[!, :config_code] = fill(meta.config_code, nrow(out))
    out[!, :width_label] = fill(meta.width_label, nrow(out))
    out[!, :model_index] = fill(meta.model_index, nrow(out))
    out[!, :candidate_id] = fill(meta.candidate_id, nrow(out))
    out[!, :source_file] = fill(path, nrow(out))

    sort!(out, :patient_id)
    return out
end

"""
Keep only the intersection of patient IDs across all candidates.
"""
function align_candidates_on_common_patients(candidate_dfs::Vector{DataFrame})
    isempty(candidate_dfs) && error("No candidate DataFrames were provided.")

    patient_sets = [Set(df.patient_id) for df in candidate_dfs]
    common_patients = sort!(collect(reduce(intersect, patient_sets)))
    isempty(common_patients) && error("No common patient IDs across the provided CSV files.")

    aligned = DataFrame[]
    common_set = Set(common_patients)

    for df in candidate_dfs
        sub = filter(row -> row.patient_id in common_set, df)
        sort!(sub, :patient_id)
        push!(aligned, sub)
    end

    ref_ids = aligned[1].patient_id
    for (k, df) in enumerate(aligned)
        df.patient_id == ref_ids || error("Patient alignment failed for candidate index $k")
    end

    return aligned, ref_ids
end

"""
Build metric matrices with rows=patients and cols=candidates.
"""
function build_metric_matrices(candidate_dfs::Vector{DataFrame})
    patients = candidate_dfs[1].patient_id
    candidate_ids = [string(first(df.candidate_id)) for df in candidate_dfs]
    config_codes = [string(first(df.config_code)) for df in candidate_dfs]
    width_labels = [string(first(df.width_label)) for df in candidate_dfs]
    model_indices = [Int(first(df.model_index)) for df in candidate_dfs]

    smape_mat = hcat([df.smape for df in candidate_dfs]...)
    rmsle_mat = hcat([df.rmsle for df in candidate_dfs]...)
    loss_mat  = hcat([df.loss for df in candidate_dfs]...)

    return (
        patients = patients,
        candidate_ids = candidate_ids,
        config_codes = config_codes,
        width_labels = width_labels,
        model_indices = model_indices,
        smape_mat = smape_mat,
        rmsle_mat = rmsle_mat,
        loss_mat = loss_mat,
    )
end

"""
Count per-patient sMAPE wins.
- eps_smape: tolerance for declaring a tie.
- tie_mode: :split | :all | :first
"""
function count_smape_wins(smape_mat::AbstractMatrix; eps_smape::Real = 0.0, tie_mode::Symbol = :split)
    n_patients, n_candidates = size(smape_mat)
    wins = zeros(Float64, n_candidates)
    best_candidate_per_patient = Vector{Vector{Int}}(undef, n_patients)

    for i in 1:n_patients
        row = vec(smape_mat[i, :])
        min_val = minimum(row)
        winners = findall(x -> x <= min_val + eps_smape, row)
        best_candidate_per_patient[i] = winners

        if tie_mode == :split
            inc = 1.0 / length(winners)
            for w in winners; wins[w] += inc; end
        elseif tie_mode == :all
            for w in winners; wins[w] += 1.0; end
        elseif tie_mode == :first
            wins[first(winners)] += 1.0
        else
            error("Unsupported tie_mode = $tie_mode. Use :split, :all or :first")
        end
    end

    return wins, best_candidate_per_patient
end

"""
Create pairwise head-to-head matrices using sMAPE.
pairwise_wins[i,j] = number of patients where candidate i beats candidate j by more than eps_smape.
"""
function pairwise_smape_tables(smape_mat::AbstractMatrix; eps_smape::Real = 0.0)
    n_patients, n_candidates = size(smape_mat)
    pairwise_wins = zeros(Int, n_candidates, n_candidates)
    pairwise_ties = zeros(Int, n_candidates, n_candidates)

    for i in 1:n_candidates, j in 1:n_candidates
        if i == j
            pairwise_ties[i, j] = n_patients
            continue
        end
        Δ = vec(smape_mat[:, i] .- smape_mat[:, j])
        pairwise_wins[i, j] = count(<(-eps_smape), Δ)
        pairwise_ties[i, j] = count(x -> abs(x) <= eps_smape, Δ)
    end

    return pairwise_wins, pairwise_ties
end

"""
Compute dominance score for each candidate:
D_i = mean over j≠i of (W_ij - W_ji) / N_patients.
A positive score means candidate i wins more patients than it loses, on average.
"""
function compute_dominance_scores(pairwise_wins::AbstractMatrix, n_patients::Int)
    n = size(pairwise_wins, 1)
    scores = zeros(Float64, n)
    for i in 1:n
        total = 0.0
        for j in 1:n
            i == j && continue
            total += (pairwise_wins[i, j] - pairwise_wins[j, i]) / n_patients
        end
        scores[i] = total / (n - 1)
    end
    return scores
end

"""
Bootstrap stability of the ranking: resample patients with replacement and
recompute rank-1 candidate. Returns top-1 frequency and CI for win rate.
"""
function bootstrap_ranking_stability(smape_mat::AbstractMatrix; n_bootstrap::Int = 1000,
        eps_smape::Real = 0.0, tie_mode::Symbol = :split, rng::AbstractRNG = Random.GLOBAL_RNG)
    n_patients, n_candidates = size(smape_mat)
    top1_counts = zeros(Int, n_candidates)
    win_boot = zeros(Float64, n_bootstrap, n_candidates)

    for b in 1:n_bootstrap
        idx = rand(rng, 1:n_patients, n_patients)
        wins_b, _ = count_smape_wins(smape_mat[idx, :]; eps_smape = eps_smape, tie_mode = tie_mode)
        win_boot[b, :] = wins_b ./ n_patients
        top1_counts[argmax(wins_b)] += 1
    end

    top1_freq = top1_counts ./ n_bootstrap
    win_rate_ci_lo = [quantile(win_boot[:, j], 0.025) for j in 1:n_candidates]
    win_rate_ci_hi = [quantile(win_boot[:, j], 0.975) for j in 1:n_candidates]

    return top1_freq, win_rate_ci_lo, win_rate_ci_hi
end

"""
Summarize and rank candidates.
Primary ranking rule:
  1) wins_smape descending (per-patient best-model count)
  2) dominance_score descending (average pairwise net advantage)
  3) median_smape ascending
  4) q90_smape ascending (tail behaviour)
  5) median_rmsle ascending (secondary metric tie-break)
"""
function summarize_and_rank_candidates(mats; eps_smape::Real = 0.0, tie_mode::Symbol = :split,
        n_bootstrap::Int = 1000, rng::AbstractRNG = Random.GLOBAL_RNG)
    wins_smape, best_per_patient = count_smape_wins(mats.smape_mat; eps_smape = eps_smape, tie_mode = tie_mode)
    pairwise_wins, pairwise_ties = pairwise_smape_tables(mats.smape_mat; eps_smape = eps_smape)
    dominance = compute_dominance_scores(pairwise_wins, size(mats.smape_mat, 1))
    top1_freq, win_ci_lo, win_ci_hi = bootstrap_ranking_stability(mats.smape_mat;
        n_bootstrap = n_bootstrap, eps_smape = eps_smape, tie_mode = tie_mode, rng = rng)

    n_patients = size(mats.smape_mat, 1)
    n_candidates = length(mats.candidate_ids)

    ranking = DataFrame(
        candidate_id     = mats.candidate_ids,
        config_code      = mats.config_codes,
        width_label      = mats.width_labels,
        model_index      = mats.model_indices,
        wins_smape       = wins_smape,
        win_rate         = wins_smape ./ n_patients,
        win_rate_ci_lo   = win_ci_lo,
        win_rate_ci_hi   = win_ci_hi,
        top1_boot_freq   = top1_freq,
        dominance_score  = dominance,
        median_smape     = [median(mats.smape_mat[:, j]) for j in 1:n_candidates],
        q90_smape        = [quantile(mats.smape_mat[:, j], 0.90) for j in 1:n_candidates],
        q1_smape         = [quantile(mats.smape_mat[:, j], 0.25) for j in 1:n_candidates],
        q3_smape         = [quantile(mats.smape_mat[:, j], 0.75) for j in 1:n_candidates],
        mean_smape       = [mean(mats.smape_mat[:, j]) for j in 1:n_candidates],
        median_rmsle     = [median(mats.rmsle_mat[:, j]) for j in 1:n_candidates],
        mean_rmsle       = [mean(mats.rmsle_mat[:, j]) for j in 1:n_candidates],
        median_loss      = [median(mats.loss_mat[:, j]) for j in 1:n_candidates],
        mean_loss        = [mean(mats.loss_mat[:, j]) for j in 1:n_candidates],
        q1_loss          = [quantile(mats.loss_mat[:, j], 0.25) for j in 1:n_candidates],
        q3_loss          = [quantile(mats.loss_mat[:, j], 0.75) for j in 1:n_candidates],
    )

    sort!(ranking, [:wins_smape, :dominance_score, :median_smape, :q90_smape, :median_rmsle],
          rev = [true, true, false, false, false])
    ranking[!, :rank] = 1:nrow(ranking)

    # Alternative rankings for sensitivity analysis
    ranking[!, :rank_by_median_smape]   = invperm(sortperm(ranking.median_smape))
    ranking[!, :rank_by_dominance]      = invperm(sortperm(ranking.dominance_score, rev = true))
    ranking[!, :rank_by_median_loss]    = invperm(sortperm(ranking.median_loss))

    return ranking, best_per_patient, pairwise_wins, pairwise_ties
end

"""
Summarize at the configuration level, keeping the best candidate per config.
"""
function summarize_configs(candidate_ranking::DataFrame)
    grouped = groupby(candidate_ranking, :config_code)
    rows = DataFrame[]

    for g in grouped
        gs = sort(copy(g), :rank)
        best = gs[1, :]

        push!(rows, DataFrame(
            config_code               = [best.config_code],
            width_label               = [best.width_label],
            best_candidate_id         = [best.candidate_id],
            best_model_index          = [best.model_index],
            best_rank                 = [best.rank],
            best_win_rate             = [best.win_rate],
            best_dominance_score      = [best.dominance_score],
            best_top1_boot_freq       = [best.top1_boot_freq],
            best_median_smape         = [best.median_smape],
            best_q90_smape            = [best.q90_smape],
            best_median_rmsle         = [best.median_rmsle],
            mean_win_rate_in_config   = [mean(gs.win_rate)],
            mean_dominance_in_config  = [mean(gs.dominance_score)],
            n_models_in_config        = [nrow(gs)],
        ))
    end

    out = vcat(rows...)
    sort!(out, :best_rank)
    return out
end

"""
Save long-format combined DataFrame.
"""
function combined_long_dataframe(candidate_dfs::Vector{DataFrame})
    out = vcat(candidate_dfs...)
    sort!(out, [:config_code, :model_index, :patient_id])
    return out
end

"""
Per-patient table: best candidate by sMAPE and margin to second best.
"""
function best_model_per_patient_df(patients, candidate_ids, smape_mat; eps_smape::Real = 0.0)
    n_patients, _ = size(smape_mat)

    best_candidate       = Vector{String}(undef, n_patients)
    best_smape           = Vector{Float64}(undef, n_patients)
    second_best_smape    = Vector{Float64}(undef, n_patients)
    delta_to_second      = Vector{Float64}(undef, n_patients)
    margin_is_meaningful = Vector{Bool}(undef, n_patients)

    for i in 1:n_patients
        row = vec(smape_mat[i, :])
        order = sortperm(row)
        best_idx   = order[1]
        second_idx = order[min(2, length(order))]
        best_candidate[i]       = candidate_ids[best_idx]
        best_smape[i]           = row[best_idx]
        second_best_smape[i]    = row[second_idx]
        delta_to_second[i]      = row[second_idx] - row[best_idx]
        margin_is_meaningful[i] = delta_to_second[i] > eps_smape
    end

    return DataFrame(
        patient_id           = patients,
        best_candidate_id    = best_candidate,
        best_smape           = best_smape,
        second_best_smape    = second_best_smape,
        delta_to_second      = delta_to_second,
        margin_is_meaningful = margin_is_meaningful,
    )
end


# ============================================================
# Plot helpers
# ============================================================

const _WIDTH_PALETTE = [
    "#1f77b4", "#d95f02", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22"
]

const _MODEL_MARKERS = Dict(1=>:circle, 2=>:rect, 3=>:diamond, 4=>:utriangle, 5=>:dtriangle, 6=>:star5)
const _MODEL_LINESTYLES = Dict(1=>:solid, 2=>:dash, 3=>:dot, 4=>:dashdot, 5=>:dashdotdot, 6=>:solid)

_parse_int_or_inf(x) = something(tryparse(Int, string(x)), typemax(Int))

function _sorted_unique_strings(v)
    out = unique(string.(v))
    sort!(out, by = _parse_int_or_inf)
    return out
end

function _width_color_map(width_labels)
    widths = _sorted_unique_strings(width_labels)
    return Dict(w => _WIDTH_PALETTE[mod1(i, length(_WIDTH_PALETTE))] for (i, w) in enumerate(widths))
end

_marker_for_model(model_index::Integer) = get(_MODEL_MARKERS, Int(model_index), :circle)
_linestyle_for_model(model_index::Integer) = get(_MODEL_LINESTYLES, Int(model_index), :solid)

# ------------------------------------------------------------------
# 1. Win-rate bar chart with 95% bootstrap CI
# ------------------------------------------------------------------
"""
Horizontal bar chart of win rates (%) with 95% bootstrap confidence intervals.
Bars are coloured by NN width; rank number annotated on each bar.
"""
function plot_wins_bar(candidate_ranking::DataFrame; output_path::AbstractString)
    df = sort(copy(candidate_ranking), :rank)
    n  = nrow(df)

    yvals  = collect(n:-1:1)   # top-ranked at top
    labels = string.(df.candidate_id)
    rates  = 100.0 .* Float64.(df.win_rate)
    ci_lo  = 100.0 .* Float64.(df.win_rate_ci_lo)
    ci_hi  = 100.0 .* Float64.(df.win_rate_ci_hi)
    xerr_lo = rates .- ci_lo
    xerr_hi = ci_hi .- rates

    cmap   = _width_color_map(df.width_label)
    colors = [cmap[string(w)] for w in df.width_label]

    p = bar(
        yvals, rates;
        orientation   = :h,
        yticks        = (yvals, labels),
        xlabel        = "Validation win rate (%, 95% bootstrap CI)",
        ylabel        = "Candidate model",
        title         = "Per-patient sMAPE win rate across candidate models",
        label         = false,
        color         = colors,
        size          = (1200, 900),
        left_margin   = 28Plots.mm,
        right_margin  = 14Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin    = 6Plots.mm,
    )

    xmax = maximum(ci_hi)
    Plots.xlims!(p, 0, xmax + max(1.5, 0.14 * xmax))

    # Error bars (manual, horizontal)
    for (y, r, elo, ehi) in zip(yvals, rates, xerr_lo, xerr_hi)
        Plots.plot!(p, [r - elo, r + ehi], [y, y];
            linewidth = 2, linecolor = :black, label = false)
        Plots.scatter!(p, [r - elo, r + ehi], [y, y];
            markersize = 4, markercolor = :black, markerstrokewidth = 0, label = false)
    end

    # Annotate rank and dominance score
    for (y, r, d) in zip(yvals, rates, df.dominance_score)
        Plots.annotate!(p, r + 0.08 * max(1.0, xmax),  y,
            Plots.text(@sprintf("%.1f%%  D=%.2f", r, d), 9, :left))
    end

    # Width colour legend
    for width in _sorted_unique_strings(df.width_label)
        Plots.scatter!(p, [NaN], [NaN];
            color = cmap[width], markershape = :square,
            markersize = 7, markerstrokewidth = 0.5,
            label = "width=$(width)")
    end

    Plots.plot!(p, legend = :bottomright)
    savefig(p, output_path)
    return p
end

# ------------------------------------------------------------------
# 2. Pareto plot: median sMAPE vs q90 sMAPE, size = win rate
# ------------------------------------------------------------------
"""
Pareto-style scatter: x = median sMAPE, y = 90th-percentile sMAPE.
Point area proportional to win rate.  Best-in-class (Pareto-front) models
are encircled.  Top-ranked model labelled explicitly.
"""
function plot_pareto_smape(candidate_ranking::DataFrame; output_path::AbstractString)
    df   = copy(candidate_ranking)
    cmap = _width_color_map(df.width_label)

    # Pareto front (non-dominated in median and q90, both to minimise)
    on_front = Bool[true for _ in 1:nrow(df)]
    for i in 1:nrow(df)
        for j in 1:nrow(df)
            i == j && continue
            if df.median_smape[j] <= df.median_smape[i] && df.q90_smape[j] <= df.q90_smape[i] &&
               (df.median_smape[j] < df.median_smape[i] || df.q90_smape[j] < df.q90_smape[i])
                on_front[i] = false
                break
            end
        end
    end

    p = Plots.plot(
        xlabel        = "Median validation sMAPE (%)",
        ylabel        = "90th-percentile validation sMAPE (%)",
        title         = "Model selection: median vs tail error (Pareto front)",
        legend        = :outerright,
        size          = (1350, 900),
        left_margin   = 14Plots.mm,
        right_margin  = 48Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin    = 6Plots.mm,
    )

    max_rate = maximum(df.win_rate)
    for row in eachrow(df)
        sz = 6 + 14 * row.win_rate / max(max_rate, 0.01)
        Plots.scatter!(p,
            [row.median_smape], [row.q90_smape];
            color            = cmap[string(row.width_label)],
            markershape      = _marker_for_model(row.model_index),
            markersize       = sz,
            markerstrokewidth = on_front[row.rank] ? 2.5 : 0.5,
            markerstrokecolor = on_front[row.rank] ? :black : :auto,
            label            = string(row.candidate_id),
        )
    end

    # Label the overall top-ranked model
    top = df[df.rank .== 1, :]
    if nrow(top) > 0
        Plots.annotate!(p, top.median_smape[1],
            top.q90_smape[1] + 0.5,
            Plots.text("★ " * string(top.candidate_id[1]), 9, :center))
    end

    Plots.annotate!(p,
        minimum(df.median_smape) + 0.01 * (maximum(df.median_smape) - minimum(df.median_smape) + eps()),
        maximum(df.q90_smape) - 0.5,
        Plots.text("circle border = Pareto-front  |  point size ∝ win rate", 8, :left))

    savefig(p, output_path)
    return p
end

# ------------------------------------------------------------------
# 3. Pairwise net-dominance heatmap
# ------------------------------------------------------------------
"""
Heatmap of the net pairwise advantage: (W_ij - W_ji) / N_patients.
Diverging colour scale centred at 0: blue = row model dominates,
red = column model dominates.  Diagonal is 0 by convention.
"""
function plot_pairwise_heatmap(candidate_ranking::DataFrame, candidate_ids::Vector{String},
        pairwise_wins::AbstractMatrix, n_patients::Int; output_path::AbstractString)
    ordered_ids = string.(candidate_ranking.candidate_id)
    id_to_col = Dict(id => i for (i, id) in enumerate(candidate_ids))
    idx = [id_to_col[id] for id in ordered_ids]
    raw = pairwise_wins[idx, idx]
    n   = size(raw, 1)

    # Net advantage matrix, diagonal = 0
    net = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        net[i, j] = (raw[i, j] - raw[j, i]) / n_patients
    end

    # Best-ranked at top of y-axis
    ylabels  = reverse(ordered_ids)
    net_plot = reverse(net, dims = 1)

    clim_abs = max(maximum(abs.(net_plot)), 0.01)

    p = Plots.heatmap(
        ordered_ids, ylabels, net_plot;
        xlabel        = "Opponent model",
        ylabel        = "Reference model",
        title         = "Net pairwise win advantage  (positive = row model wins more patients)",
        xrotation     = 45,
        size          = (1250, 1050),
        left_margin   = 24Plots.mm,
        right_margin  = 10Plots.mm,
        bottom_margin = 18Plots.mm,
        top_margin    = 6Plots.mm,
        color         = :RdBu,
        clims         = (-clim_abs, clim_abs),
        colorbar_title = "Net advantage",
    )

    for i in 1:n, j in 1:n
        val = net_plot[i, j]
        txt_color = abs(val) > 0.5 * clim_abs ? :white : :black
        Plots.annotate!(p, j, i, Plots.text(@sprintf("%+.2f", val), 7, txt_color))
    end

    savefig(p, output_path)
    return p
end

# ------------------------------------------------------------------
# 4. Top-k sMAPE ECDF with median/q90 markers
# ------------------------------------------------------------------
"""
ECDF of per-patient sMAPE for the top-k candidates.
Vertical dashed lines show median; dotted lines show 90th percentile.
"""
function plot_topk_ecdf_smape(candidate_ranking::DataFrame, candidate_ids::Vector{String},
        smape_mat::AbstractMatrix; top_k::Int = 4, output_path::AbstractString)
    k = min(top_k, nrow(candidate_ranking))
    selected_ids = string.(candidate_ranking.candidate_id[1:k])

    id_to_col = Dict(id => i for (i, id) in enumerate(candidate_ids))
    id_to_row = Dict(string(row.candidate_id) => row for row in eachrow(candidate_ranking))
    cmap      = _width_color_map(candidate_ranking.width_label)

    p = Plots.plot(
        xlabel        = "Validation sMAPE (%)",
        ylabel        = "Empirical CDF",
        title         = "Top $(k) candidate models — validation sMAPE ECDF",
        size          = (1150, 850),
        left_margin   = 14Plots.mm,
        right_margin  = 32Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin    = 6Plots.mm,
        legend        = :outerright,
    )

    for id in selected_ids
        row = id_to_row[id]
        col = id_to_col[id]
        xs  = sort(vec(smape_mat[:, col]))
        ys  = collect(1:length(xs)) ./ length(xs)
        lc  = cmap[string(row.width_label)]
        ls  = _linestyle_for_model(row.model_index)

        Plots.plot!(p, xs, ys;
            linewidth = 2.2, linecolor = lc, linestyle = ls, label = string(id))

        med = median(xs)
        q90 = quantile(xs, 0.90)
        Plots.vline!(p, [med]; linecolor = lc, linestyle = :dash,
            linewidth = 1.2, label = false)
        Plots.vline!(p, [q90]; linecolor = lc, linestyle = :dot,
            linewidth = 1.2, label = false)
    end

    # Legend proxies for the dashed/dotted lines
    Plots.plot!(p, [NaN], [NaN]; linecolor = :gray, linestyle = :dash,
        linewidth = 1.2, label = "median")
    Plots.plot!(p, [NaN], [NaN]; linecolor = :gray, linestyle = :dot,
        linewidth = 1.2, label = "90th pct")

    savefig(p, output_path)
    return p
end

# (retained for supplementary use)
function plot_all_ecdf_smape(candidate_ranking::DataFrame, candidate_ids::Vector{String},
        smape_mat::AbstractMatrix; output_path::AbstractString)
    return plot_topk_ecdf_smape(candidate_ranking, candidate_ids, smape_mat;
        top_k = nrow(candidate_ranking), output_path = output_path)
end

# ------------------------------------------------------------------
# 5. Paired Δ-sMAPE plot: best vs runner-up
# ------------------------------------------------------------------
"""
For each validation patient, plot the signed difference
  Δ = sMAPE(runner-up) − sMAPE(best)
sorted in ascending order.  Patients where Δ > 0 (best improves) are
shown in blue; Δ ≤ 0 (runner-up better) in red.
Horizontal zero line and the share of patients in each zone are annotated.
"""
function plot_delta_smape_sorted(candidate_ranking::DataFrame, candidate_ids::Vector{String},
        smape_mat::AbstractMatrix; eps_smape::Real = 0.0, output_path::AbstractString)
    nrow(candidate_ranking) >= 2 || return nothing

    id_to_col = Dict(id => i for (i, id) in enumerate(candidate_ids))
    best_id   = string(candidate_ranking.candidate_id[1])
    runner_id = string(candidate_ranking.candidate_id[2])

    smape_best   = vec(smape_mat[:, id_to_col[best_id]])
    smape_runner = vec(smape_mat[:, id_to_col[runner_id]])
    Δ = smape_runner .- smape_best          # positive = best model is better
    Δ_sorted = sort(Δ)
    n = length(Δ_sorted)

    n_better = count(x ->  x >  eps_smape, Δ)
    n_tie    = count(x -> abs(x) <= eps_smape, Δ)
    n_worse  = count(x ->  x < -eps_smape, Δ)
    pct_better = 100.0 * n_better / n
    pct_worse  = 100.0 * n_worse  / n

    colors = [d > eps_smape ? "#1f77b4" : (d < -eps_smape ? "#d62728" : "#7f7f7f") for d in Δ_sorted]

    p = Plots.bar(
        1:n, Δ_sorted;
        color         = colors,
        xlabel        = "Patient (sorted by Δ)",
        ylabel        = "sMAPE($(runner_id)) − sMAPE($(best_id))  (%)",
        title         = "Per-patient sMAPE difference: $(best_id) vs $(runner_id)",
        label         = false,
        size          = (1150, 700),
        left_margin   = 16Plots.mm,
        right_margin  = 10Plots.mm,
        bottom_margin = 12Plots.mm,
        top_margin    = 6Plots.mm,
        linewidth     = 0,
    )

    Plots.hline!(p, [0.0]; linecolor = :black, linewidth = 1.5, label = false)

    Plots.annotate!(p, n * 0.02, maximum(Δ_sorted) * 0.88,
        Plots.text(@sprintf("best model lower sMAPE: %d/%d (%.0f%%)", n_better, n, pct_better), 10, :left, :blue))
    Plots.annotate!(p, n * 0.02, maximum(Δ_sorted) * 0.72,
        Plots.text(@sprintf("runner-up lower sMAPE: %d/%d (%.0f%%)", n_worse, n, pct_worse), 10, :left, :red))

    savefig(p, output_path)
    return p
end

# ------------------------------------------------------------------
# 6. Bootstrap top-1 frequency bar
# ------------------------------------------------------------------
"""
Bar chart of bootstrap top-1 frequency for each candidate.
A frequency ≥ 0.5 indicates a clearly stable winner.
"""
function plot_bootstrap_stability(candidate_ranking::DataFrame; output_path::AbstractString)
    df = sort(copy(candidate_ranking), :rank)
    n  = nrow(df)
    yvals = collect(n:-1:1)
    labels = string.(df.candidate_id)
    freqs  = Float64.(df.top1_boot_freq) .* 100

    cmap   = _width_color_map(df.width_label)
    colors = [cmap[string(w)] for w in df.width_label]

    p = bar(
        yvals, freqs;
        orientation   = :h,
        yticks        = (yvals, labels),
        xlabel        = "Bootstrap top-1 frequency (%)",
        ylabel        = "Candidate model",
        title         = "Ranking stability: how often each model ranks first (n=1000 bootstrap)",
        label         = false,
        color         = colors,
        size          = (1200, 900),
        left_margin   = 28Plots.mm,
        right_margin  = 14Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin    = 6Plots.mm,
    )

    Plots.vline!(p, [50.0]; linecolor = :black, linestyle = :dash,
        linewidth = 1.5, label = "50% threshold")

    xmax = max(maximum(freqs), 60.0)
    Plots.xlims!(p, 0, xmax + 6)

    for (y, f) in zip(yvals, freqs)
        Plots.annotate!(p, f + 0.8, y, Plots.text(@sprintf("%.1f%%", f), 9, :left))
    end

    for width in _sorted_unique_strings(df.width_label)
        Plots.scatter!(p, [NaN], [NaN];
            color = cmap[width], markershape = :square,
            markersize = 7, markerstrokewidth = 0.5,
            label = "width=$(width)")
    end

    Plots.plot!(p, legend = :bottomright)
    savefig(p, output_path)
    return p
end


# ============================================================
# Main entry point
# ============================================================

function _run_shared_workflow(candidate_dfs, csv_paths, output_dir, eps_smape, tie_mode,
        top_k_ecdf, n_bootstrap, rng)
    mkpath(output_dir)

    sort!(candidate_dfs, by = df -> (parse(Int, first(df.config_code)), Int(first(df.model_index))))
    aligned_dfs, common_patients = align_candidates_on_common_patients(candidate_dfs)
    mats = build_metric_matrices(aligned_dfs)
    n_patients = length(common_patients)

    ranking, best_per_patient, pairwise_wins, pairwise_ties =
        summarize_and_rank_candidates(mats;
            eps_smape = eps_smape, tie_mode = tie_mode,
            n_bootstrap = n_bootstrap, rng = rng)

    config_summary  = summarize_configs(ranking)
    combined_long   = combined_long_dataframe(aligned_dfs)
    best_patient_df = best_model_per_patient_df(mats.patients, mats.candidate_ids,
                        mats.smape_mat; eps_smape = eps_smape)

    # --- CSV output ---
    CSV.write(joinpath(output_dir, "combined_validation_metrics_long.csv"), combined_long)
    CSV.write(joinpath(output_dir, "candidate_ranking.csv"), ranking)
    CSV.write(joinpath(output_dir, "config_summary_best_model.csv"), config_summary)
    CSV.write(joinpath(output_dir, "best_model_per_patient.csv"), best_patient_df)

    pairwise_wins_df = DataFrame(pairwise_wins, :auto)
    rename!(pairwise_wins_df, Symbol.(mats.candidate_ids))
    insertcols!(pairwise_wins_df, 1, :winner_candidate => mats.candidate_ids)
    CSV.write(joinpath(output_dir, "pairwise_smape_wins.csv"), pairwise_wins_df)

    pairwise_ties_df = DataFrame(pairwise_ties, :auto)
    rename!(pairwise_ties_df, Symbol.(mats.candidate_ids))
    insertcols!(pairwise_ties_df, 1, :candidate_row => mats.candidate_ids)
    CSV.write(joinpath(output_dir, "pairwise_smape_ties.csv"), pairwise_ties_df)

    # --- Plots ---
    plot_wins_bar(ranking;
        output_path = joinpath(output_dir, "win_rate_bar.png"))
    plot_pareto_smape(ranking;
        output_path = joinpath(output_dir, "pareto_median_vs_q90_smape.png"))
    plot_pairwise_heatmap(ranking, mats.candidate_ids, pairwise_wins, n_patients;
        output_path = joinpath(output_dir, "pairwise_net_dominance_heatmap.png"))
    plot_topk_ecdf_smape(ranking, mats.candidate_ids, mats.smape_mat; top_k = top_k_ecdf,
        output_path = joinpath(output_dir, "topk_smape_ecdf.png"))
    plot_delta_smape_sorted(ranking, mats.candidate_ids, mats.smape_mat; eps_smape = eps_smape,
        output_path = joinpath(output_dir, "delta_smape_best_vs_runnerup.png"))
    plot_bootstrap_stability(ranking;
        output_path = joinpath(output_dir, "bootstrap_top1_frequency.png"))
    # Supplementary: all-candidate ECDF (less prominent in paper)
    plot_all_ecdf_smape(ranking, mats.candidate_ids, mats.smape_mat;
        output_path = joinpath(output_dir, "all_smape_ecdf_supplementary.png"))

    # --- Text report ---
    top1_stable = ranking.top1_boot_freq[1] >= 0.50
    open(joinpath(output_dir, "selection_report.txt"), "w") do io
        println(io, "Validation model selection report")
        println(io, "================================")
        println(io, "Candidates: $(length(csv_paths))")
        println(io, "Common patients: $(n_patients)")
        println(io, "Tie tolerance (eps_smape): $(eps_smape)")
        println(io, "Tie mode: $(tie_mode)")
        println(io, "Bootstrap samples: $(n_bootstrap)")
        println(io)
        println(io, "Ranking rule:")
        println(io, "  1) wins_smape (desc)        — per-patient best-model count")
        println(io, "  2) dominance_score (desc)   — average pairwise net advantage over all opponents")
        println(io, "  3) median_smape (asc)")
        println(io, "  4) q90_smape (asc)           — tail robustness")
        println(io, "  5) median_rmsle (asc)        — secondary metric tie-break")
        println(io)
        println(io, "Top-ranked candidate:")
        show(io, MIME"text/plain"(), ranking[1:1, :])
        println(io)
        println(io)
        if nrow(ranking) >= 2
            println(io, "Runner-up candidate:")
            show(io, MIME"text/plain"(), ranking[2:2, :])
            println(io)
            println(io)
        end
        println(io, "Ranking stability (bootstrap top-1 freq ≥ 50%): $(top1_stable ? "STABLE ✓" : "UNSTABLE — consider co-winner set")")
        println(io)
        println(io, "Configuration-level summary:")
        show(io, MIME"text/plain"(), config_summary)
        println(io)
    end

    return (
        ranking         = ranking,
        config_summary  = config_summary,
        combined_long   = combined_long,
        best_patient_df = best_patient_df,
        candidate_ids   = mats.candidate_ids,
        patients        = mats.patients,
        smape_mat       = mats.smape_mat,
        rmsle_mat       = mats.rmsle_mat,
        loss_mat        = mats.loss_mat,
        pairwise_wins   = pairwise_wins,
        pairwise_ties   = pairwise_ties,
    )
end

"""
Run selection from a vector of CSV file paths (names must encode metadata).
"""
function run_validation_model_selection(csv_paths::Vector{String};
        output_dir::AbstractString  = "res/model_selection",
        eps_smape::Real              = 0.0,
        tie_mode::Symbol             = :split,
        top_k_ecdf::Int              = 4,
        n_bootstrap::Int             = 1000,
        rng::AbstractRNG             = Random.GLOBAL_RNG,
    )
    isempty(csv_paths) && error("No CSV paths provided.")
    candidate_dfs = [read_candidate_metrics(path) for path in csv_paths]
    return _run_shared_workflow(candidate_dfs, csv_paths, output_dir,
        eps_smape, tie_mode, top_k_ecdf, n_bootstrap, rng)
end

"""
Run selection from an explicit candidate specification vector.

candidate_specs = [
    (path = "...", config_code = "26", model_index = 1),
    (path = "...", config_code = "28", model_index = 2),
    ...
]
"""
function run_validation_model_selection(candidate_specs::Vector{<:NamedTuple};
        output_dir::AbstractString  = "res/model_selection",
        eps_smape::Real              = 0.0,
        tie_mode::Symbol             = :split,
        top_k_ecdf::Int              = 4,
        n_bootstrap::Int             = 1000,
        rng::AbstractRNG             = Random.GLOBAL_RNG,
    )
    csv_paths     = [String(spec.path) for spec in candidate_specs]
    meta_overrides = map(candidate_specs) do spec
        cfg = String(spec.config_code)
        mid = Int(spec.model_index)
        width_label  = haskey(spec, :width_label)  ? String(spec.width_label)  :
                       (startswith(cfg, "2") && length(cfg) >= 2 ? cfg[2:end] : cfg)
        candidate_id = haskey(spec, :candidate_id) ? String(spec.candidate_id) :
                       "cfg$(cfg)_m$(mid)"
        (filename = basename(String(spec.path)), config_code = cfg,
         width_label = width_label, model_index = mid, candidate_id = candidate_id)
    end

    candidate_dfs = [read_candidate_metrics(path; meta_override = meta)
                     for (path, meta) in zip(csv_paths, meta_overrides)]
    return _run_shared_workflow(candidate_dfs, csv_paths, output_dir,
        eps_smape, tie_mode, top_k_ecdf, n_bootstrap, rng)
end

# Optional CLI
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("Usage: julia model_selection_clean.jl OUTPUT_DIR CSV1 CSV2 ...")
    run_validation_model_selection(ARGS[2:end]; output_dir = ARGS[1])
end
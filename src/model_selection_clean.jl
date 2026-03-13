using CSV
using DataFrames
using Statistics
using Printf
using Plots

# ============================================================
# Validation-set model selection from per-patient CSV metrics
# ------------------------------------------------------------
# Expected CSV columns:
#   patient_id, smape, rmsle, loss
#
# Typical usage from the REPL:
#
# csv_paths = [
#     "/path/to/model_26_patients_metrics_val_tesn_NN_1.csv",
#     "/path/to/model_26_patients_metrics_val_tesn_NN_2.csv",
#     "/path/to/model_28_patients_metrics_val_tesn_NN_1.csv",
#     "/path/to/model_28_patients_metrics_val_tesn_NN_2.csv",
#     "/path/to/model_216_patients_metrics_val_tesn_NN_1.csv",
#     "/path/to/model_216_patients_metrics_val_tesn_NN_2.csv",
# ]
#
# results = run_validation_model_selection(csv_paths;
#     output_dir = "res/model_selection",
#     eps_smape = 0.0,
#     tie_mode = :split,
#     top_k_ecdf = 6,
# )
#
# Optional CLI usage:
# julia model_selection_from_validation_csvs.jl outdir csv1 csv2 csv3 ...
# ============================================================

const REQUIRED_COLUMNS = [:patient_id, :smape, :rmsle, :loss]

"""
Parse config/model metadata from filenames such as:
- model_26_patients_metrics_val_tesn_NN_1.csv
- model_28_patients_metrics_val_tesn_NN_2.csv.csv
- model_216_patients_metrics_val_tesn_NN_3.csv

Returns a named tuple with:
- config_code  -> e.g. "26", "28", "216"
- width_label  -> e.g. "6", "8", "16" if config starts with depth=2
- model_index  -> e.g. 1, 2, 3, 4
- candidate_id -> e.g. "cfg26_m1"
"""
function parse_candidate_metadata(path::AbstractString)
    filename = basename(path)
    m = match(r"model_(\d+)_patients_metrics.*?_(\d+)\.csv(?:\.csv)?$", filename)
    m === nothing && error("Cannot parse config/model metadata from filename: $filename")

    config_code = m.captures[1]
    model_index = parse(Int, m.captures[2])

    # If your experiment code is depth+width (e.g. 26 => depth=2,width=6 ; 216 => depth=2,width=16)
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
Returns the filtered/sorted DataFrames and the common patient IDs.
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
- eps_smape = tolerance for declaring a tie.
- tie_mode = :split | :all | :first
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
            for w in winners
                wins[w] += inc
            end
        elseif tie_mode == :all
            for w in winners
                wins[w] += 1.0
            end
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
pairwise_ties[i,j] = number of patients where |candidate i - candidate j| <= eps_smape.
"""
function pairwise_smape_tables(smape_mat::AbstractMatrix; eps_smape::Real = 0.0)
    _, n_candidates = size(smape_mat)
    pairwise_wins = zeros(Int, n_candidates, n_candidates)
    pairwise_ties = zeros(Int, n_candidates, n_candidates)

    for i in 1:n_candidates, j in 1:n_candidates
        if i == j
            pairwise_ties[i, j] = size(smape_mat, 1)
            continue
        end
        Δ = vec(smape_mat[:, i] .- smape_mat[:, j])
        pairwise_wins[i, j] = count(<(-eps_smape), Δ)
        pairwise_ties[i, j] = count(x -> abs(x) <= eps_smape, Δ)
    end

    return pairwise_wins, pairwise_ties
end

"""
Summarize each candidate model and rank with the requested rule:
1) wins_smape descending
2) median_smape ascending
3) median_rmsle ascending
4) median_loss ascending (final deterministic tie-break)
"""
function summarize_and_rank_candidates(mats; eps_smape::Real = 0.0, tie_mode::Symbol = :split)
    wins_smape, best_per_patient = count_smape_wins(mats.smape_mat; eps_smape = eps_smape, tie_mode = tie_mode)

    n_candidates = length(mats.candidate_ids)
    ranking = DataFrame(
        candidate_id = mats.candidate_ids,
        config_code = mats.config_codes,
        width_label = mats.width_labels,
        model_index = mats.model_indices,
        wins_smape = wins_smape,
        mean_smape = [mean(mats.smape_mat[:, j]) for j in 1:n_candidates],
        median_smape = [median(mats.smape_mat[:, j]) for j in 1:n_candidates],
        q1_smape = [quantile(mats.smape_mat[:, j], 0.25) for j in 1:n_candidates],
        q3_smape = [quantile(mats.smape_mat[:, j], 0.75) for j in 1:n_candidates],
        mean_rmsle = [mean(mats.rmsle_mat[:, j]) for j in 1:n_candidates],
        median_rmsle = [median(mats.rmsle_mat[:, j]) for j in 1:n_candidates],
        mean_loss = [mean(mats.loss_mat[:, j]) for j in 1:n_candidates],
        median_loss = [median(mats.loss_mat[:, j]) for j in 1:n_candidates],
        q1_loss = [quantile(mats.loss_mat[:, j], 0.25) for j in 1:n_candidates],
        q3_loss = [quantile(mats.loss_mat[:, j], 0.75) for j in 1:n_candidates],
    )

    sort!(ranking, [:wins_smape, :median_smape, :median_rmsle, :median_loss], rev = [true, false, false, false])
    ranking[!, :rank_smape_rule] = 1:nrow(ranking)

    # Additional alternative rankings for sensitivity analysis
    ranking[!, :rank_median_loss] = invperm(sortperm(1:nrow(ranking), by = i -> ranking.median_loss[i]))
    ranking[!, :rank_mean_loss]   = invperm(sortperm(1:nrow(ranking), by = i -> ranking.mean_loss[i]))
    ranking[!, :rank_median_smape] = invperm(sortperm(1:nrow(ranking), by = i -> ranking.median_smape[i]))

    return ranking, best_per_patient
end

"""
Summarize at the configuration level (e.g. width), keeping the best candidate for each config.
This is useful when you have 4 models per configuration.
"""
function summarize_configs(candidate_ranking::DataFrame)
    grouped = groupby(candidate_ranking, :config_code)
    rows = DataFrame[]

    for g in grouped
        gs = sort(copy(g), :rank_smape_rule)
        best = gs[1, :]

        push!(rows, DataFrame(
            config_code = [best.config_code],
            width_label = [best.width_label],
            best_candidate_id = [best.candidate_id],
            best_model_index = [best.model_index],
            best_rank_smape_rule = [best.rank_smape_rule],
            best_wins_smape = [best.wins_smape],
            best_median_smape = [best.median_smape],
            best_median_rmsle = [best.median_rmsle],
            best_median_loss = [best.median_loss],
            mean_rank_within_config = [mean(gs.rank_smape_rule)],
            median_rank_within_config = [median(gs.rank_smape_rule)],
            mean_wins_within_config = [mean(gs.wins_smape)],
            n_models_in_config = [nrow(gs)],
        ))
    end

    out = vcat(rows...)
    sort!(out, [:best_rank_smape_rule, :best_median_smape], rev = [false, false])
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
Save per-patient table with the best candidate according to sMAPE.
"""
function best_model_per_patient_df(patients, candidate_ids, smape_mat, rmsle_mat, loss_mat; eps_smape::Real = 0.0)
    n_patients, _ = size(smape_mat)

    best_candidate = Vector{String}(undef, n_patients)
    best_smape = Vector{Float64}(undef, n_patients)
    second_best_smape = Vector{Float64}(undef, n_patients)
    delta_second_minus_best = Vector{Float64}(undef, n_patients)

    for i in 1:n_patients
        row = vec(smape_mat[i, :])
        order = sortperm(row)
        best_idx = order[1]
        second_idx = order[min(2, length(order))]
        best_candidate[i] = candidate_ids[best_idx]
        best_smape[i] = row[best_idx]
        second_best_smape[i] = row[second_idx]
        delta_second_minus_best[i] = row[second_idx] - row[best_idx]
    end

    return DataFrame(
        patient_id = patients,
        best_candidate_id = best_candidate,
        best_smape = best_smape,
        second_best_smape = second_best_smape,
        delta_second_minus_best = delta_second_minus_best,
    )
end


# --------------------------
# Plot helpers
# --------------------------

const _WIDTH_PALETTE = [
    "#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
    "#17becf", "#bcbd22"
]

const _MODEL_MARKERS = Dict(
    1 => :circle,
    2 => :rect,
    3 => :diamond,
    4 => :utriangle,
    5 => :dtriangle,
    6 => :star5,
)

const _MODEL_LINESTYLES = Dict(
    1 => :solid,
    2 => :dash,
    3 => :dot,
    4 => :dashdot,
    5 => :dashdotdot,
    6 => :solid,
)

_parse_int_or_inf(x) = something(tryparse(Int, string(x)), typemax(Int))

function _sorted_unique_strings(v)
    out = unique(string.(v))
    sort!(out, by = _parse_int_or_inf)
    return out
end

function _width_color_map(width_labels)
    widths = _sorted_unique_strings(width_labels)
    return Dict(width => _WIDTH_PALETTE[mod1(i, length(_WIDTH_PALETTE))] for (i, width) in enumerate(widths))
end

_marker_for_model(model_index::Integer) = get(_MODEL_MARKERS, Int(model_index), :circle)
_linestyle_for_model(model_index::Integer) = get(_MODEL_LINESTYLES, Int(model_index), :solid)

function plot_wins_bar(candidate_ranking::DataFrame; output_path::AbstractString)
    df = sort(copy(candidate_ranking), :rank_smape_rule)
    n = nrow(df)

    # Keep top-ranked model at the top.
    yvals = collect(n:-1:1)
    labels = string.(df.candidate_id)
    wins = Float64.(df.wins_smape)

    cmap = _width_color_map(df.width_label)
    colors = [cmap[string(w)] for w in df.width_label]

    p = bar(
        yvals,
        wins;
        orientation = :h,
        yticks = (yvals, labels),
        xlabel = "Validation sMAPE wins (count)",
        ylabel = "Candidate model (ranked by selection rule)",
        title = "Validation selection by sMAPE wins",
        label = false,
        color = colors,
        size = (1250, 950),
        left_margin = 28Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )

    xmax = maximum(wins)
    Plots.xlims!(p, 0, xmax + max(0.8, 0.12 * xmax))

    for (y, v) in zip(yvals, wins)
        annotate!(p, v + 0.08 * max(1.0, xmax), y, Plots.text(@sprintf("%.1f", v), 10, :left))
    end

    # Width legend
    for width in _sorted_unique_strings(df.width_label)
        Plots.scatter!(p, [NaN], [NaN];
            color = cmap[width],
            markershape = :square,
            markersize = 7,
            markerstrokewidth = 0.5,
            label = "width=$(width)")
    end

    Plots.plot!(p, legend = :bottomright)
    savefig(p, output_path)
    return p
end

function plot_median_smape_vs_wins(candidate_ranking::DataFrame; output_path::AbstractString)
    df = sort(copy(candidate_ranking), :rank_smape_rule)
    cmap = _width_color_map(df.width_label)

    p = Plots.plot(
        xlabel = "Median validation sMAPE (%)",
        ylabel = "Validation sMAPE wins (count)",
        title = "Candidate ranking: wins vs median validation sMAPE",
        legend = :outerright,
        size = (1400, 900),
        left_margin = 16Plots.mm,
        right_margin = 42Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )

    for row in eachrow(df)
        Plots.scatter!(p,
            [row.median_smape],
            [row.wins_smape];
            color = cmap[string(row.width_label)],
            markershape = _marker_for_model(row.model_index),
            markersize = 8,
            markerstrokewidth = 0.6,
            label = string(row.candidate_id),
        )
    end

    # Add a small guide inside the figure.
    annotate!(
        p,
        minimum(df.median_smape) + 0.02 * (maximum(df.median_smape) - minimum(df.median_smape) + eps()),
        maximum(df.wins_smape) - 0.35,
        Plots.text("color = width, marker shape = model index", 9, :left)
    )

    savefig(p, output_path)
    return p
end

function plot_pairwise_heatmap(candidate_ranking::DataFrame, candidate_ids::Vector{String}, pairwise_wins::AbstractMatrix; output_path::AbstractString)
    ordered_ids = string.(candidate_ranking.candidate_id)
    id_to_col = Dict(id => i for (i, id) in enumerate(candidate_ids))
    idx = [id_to_col[id] for id in ordered_ids]
    mat = pairwise_wins[idx, idx]

    # Put the best-ranked model at the top of the y-axis.
    ylabels = reverse(ordered_ids)
    mat_plot = reverse(mat, dims = 1)

    p = Plots.heatmap(
        ordered_ids,
        ylabels,
        mat_plot;
        xlabel = "Beaten model",
        ylabel = "Winning model",
        title = "Pairwise head-to-head validation sMAPE wins",
        xrotation = 45,
        size = (1250, 1050),
        left_margin = 24Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 18Plots.mm,
        top_margin = 6Plots.mm,
        colorbar_title = "Win count",
    )

    vmax = maximum(mat_plot)
    for i in 1:size(mat_plot, 1), j in 1:size(mat_plot, 2)
        val = mat_plot[i, j]
        txt_color = val > 0.55 * vmax ? :white : :black
        annotate!(p, j, i, Plots.text(string(val), 7, txt_color))
    end

    savefig(p, output_path)
    return p
end

function _plot_smape_ecdf(candidate_ranking::DataFrame, candidate_ids::Vector{String}, smape_mat::AbstractMatrix,
        selected_ids::Vector{String}; title::AbstractString, output_path::AbstractString)
    id_to_col = Dict(id => i for (i, id) in enumerate(candidate_ids))
    id_to_row = Dict(string(row.candidate_id) => row for row in eachrow(candidate_ranking))
    cmap = _width_color_map(candidate_ranking.width_label)

    p = Plots.plot(
        xlabel = "Validation sMAPE (%)",
        ylabel = "Empirical CDF",
        title = title,
        size = (1250, 900),
        left_margin = 14Plots.mm,
        right_margin = 32Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
        legend = :outerright,
    )

    for id in selected_ids
        row = id_to_row[id]
        col = id_to_col[id]
        xs = sort(vec(smape_mat[:, col]))
        ys = collect(1:length(xs)) ./ length(xs)
        Plots.plot!(p, xs, ys;
            linewidth = 2.2,
            linecolor = cmap[string(row.width_label)],
            linestyle = _linestyle_for_model(row.model_index),
            label = string(id))
    end

    savefig(p, output_path)
    return p
end

function plot_topk_ecdf_smape(candidate_ranking::DataFrame, candidate_ids::Vector{String}, smape_mat::AbstractMatrix;
        top_k::Int = 6, output_path::AbstractString)
    k = min(top_k, nrow(candidate_ranking))
    selected_ids = string.(candidate_ranking.candidate_id[1:k])
    title = k < nrow(candidate_ranking) ?
        "Top $(k) candidate models: validation sMAPE ECDF" :
        "All candidate models: validation sMAPE ECDF"

    return _plot_smape_ecdf(candidate_ranking, candidate_ids, smape_mat, selected_ids;
        title = title,
        output_path = output_path)
end

function plot_all_ecdf_smape(candidate_ranking::DataFrame, candidate_ids::Vector{String}, smape_mat::AbstractMatrix;
        output_path::AbstractString)
    selected_ids = string.(candidate_ranking.candidate_id)
    return _plot_smape_ecdf(candidate_ranking, candidate_ids, smape_mat, selected_ids;
        title = "All candidate models: validation sMAPE ECDF",
        output_path = output_path)
end

function plot_best_vs_runnerup(candidate_ranking::DataFrame, candidate_ids::Vector{String}, smape_mat::AbstractMatrix;
        output_path::AbstractString)
    nrow(candidate_ranking) >= 2 || return nothing

    id_to_col = Dict(id => i for (i, id) in enumerate(candidate_ids))
    best_id = string(candidate_ranking.candidate_id[1])
    runner_id = string(candidate_ranking.candidate_id[2])

    # Convention: x = runner-up, y = best-ranked.
    x = vec(smape_mat[:, id_to_col[runner_id]])
    y = vec(smape_mat[:, id_to_col[best_id]])

    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    span = hi - lo + eps()

    below = findall(i -> y[i] < x[i], eachindex(x))  # best-ranked lower sMAPE
    above = findall(i -> y[i] > x[i], eachindex(x))  # runner-up lower sMAPE
    equal = findall(i -> y[i] == x[i], eachindex(x))

    p = Plots.plot(
        xlabel = "$(runner_id) validation sMAPE (%)",
        ylabel = "$(best_id) validation sMAPE (%)",
        title = "Paired validation comparison (each point = one validation patient)",
        size = (980, 980),
        left_margin = 18Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
        legend = :topleft,
    )

    if !isempty(below)
        Plots.scatter!(p, x[below], y[below];
            label = "$(best_id) lower sMAPE (n=$(length(below)))",
            markersize = 5,
            markerstrokewidth = 0.5,
        )
    end
    if !isempty(above)
        Plots.scatter!(p, x[above], y[above];
            label = "$(runner_id) lower sMAPE (n=$(length(above)))",
            markersize = 5,
            markerstrokewidth = 0.5,
        )
    end
    if !isempty(equal)
        Plots.scatter!(p, x[equal], y[equal];
            label = "equal sMAPE (n=$(length(equal)))",
            markersize = 5,
            markerstrokewidth = 0.5,
        )
    end

    Plots.plot!(p, [lo, hi], [lo, hi];
        linewidth = 2,
        linecolor = :black,
        linestyle = :dash,
        label = "equal performance")

    annotate!(
        p,
        lo + 0.03 * span,
        hi - 0.08 * span,
        Plots.text("Points below the diagonal: $(best_id) has lower validation sMAPE", 9, :left)
    )

    savefig(p, output_path)
    return p
end
# --------------------------
# Main entry point
# --------------------------

"""
Run the full automatic model-selection workflow.

Inputs
------
- csv_paths: vector of paths to CSV files produced by validation runs.

Keyword args
------------
- output_dir: where to save CSV summaries and plots.
- eps_smape: tolerance for declaring per-patient sMAPE ties.
- tie_mode: :split | :all | :first.
- top_k_ecdf: how many top-ranked models to show in the ECDF plot.

Outputs
-------
Returns a named tuple with ranking DataFrames and metric matrices.
"""
function run_validation_model_selection(csv_paths::Vector{String};
        output_dir::AbstractString = "res/model_selection",
        eps_smape::Real = 0.0,
        tie_mode::Symbol = :split,
        top_k_ecdf::Int = 6,
    )

    isempty(csv_paths) && error("No CSV paths were provided.")
    mkpath(output_dir)

    candidate_dfs = [read_candidate_metrics(path) for path in csv_paths]
    sort!(candidate_dfs, by = df -> (parse(Int, first(df.config_code)), Int(first(df.model_index))))
    aligned_dfs, common_patients = align_candidates_on_common_patients(candidate_dfs)
    mats = build_metric_matrices(aligned_dfs)

    ranking, best_per_patient = summarize_and_rank_candidates(mats; eps_smape = eps_smape, tie_mode = tie_mode)
    config_summary = summarize_configs(ranking)
    pairwise_wins, pairwise_ties = pairwise_smape_tables(mats.smape_mat; eps_smape = eps_smape)

    combined_long = combined_long_dataframe(aligned_dfs)
    best_patient_df = best_model_per_patient_df(mats.patients, mats.candidate_ids, mats.smape_mat, mats.rmsle_mat, mats.loss_mat; eps_smape = eps_smape)

    CSV.write(joinpath(output_dir, "combined_validation_metrics_long.csv"), combined_long)
    CSV.write(joinpath(output_dir, "candidate_ranking_smape_rule.csv"), ranking)
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

    # Plots
    plot_wins_bar(ranking; output_path = joinpath(output_dir, "wins_bar.png"))
    plot_median_smape_vs_wins(ranking; output_path = joinpath(output_dir, "wins_vs_median_smape.png"))
    plot_pairwise_heatmap(ranking, mats.candidate_ids, pairwise_wins; output_path = joinpath(output_dir, "pairwise_smape_wins_heatmap.png"))
    plot_all_ecdf_smape(ranking, mats.candidate_ids, mats.smape_mat; output_path = joinpath(output_dir, "all_smape_ecdf.png"))
    plot_topk_ecdf_smape(ranking, mats.candidate_ids, mats.smape_mat; top_k = top_k_ecdf, output_path = joinpath(output_dir, "topk_smape_ecdf.png"))
    plot_best_vs_runnerup(ranking, mats.candidate_ids, mats.smape_mat; output_path = joinpath(output_dir, "best_vs_runnerup_smape_scatter.png"))

    open(joinpath(output_dir, "selection_report.txt"), "w") do io
        println(io, "Validation model selection report")
        println(io, "================================")
        println(io, "Candidates read: $(length(csv_paths))")
        println(io, "Common patients used for fair comparison: $(length(common_patients))")
        println(io, "Tie handling mode: $(tie_mode)")
        println(io, "sMAPE tie tolerance: $(eps_smape)")
        println(io)
        println(io, "Primary rule: wins_smape (desc) -> median_smape (asc) -> median_rmsle (asc) -> median_loss (asc)")
        println(io, "ECDF plots:")
        println(io, " - all_smape_ecdf.png : all $(nrow(ranking)) ranked candidates")
        println(io, " - topk_smape_ecdf.png : top $(min(top_k_ecdf, nrow(ranking))) candidates by the primary rule")
        println(io)
        println(io, "Interpretation note:")
        println(io, "The top-ranked candidate is selected by a lexicographic rule. Therefore, a model with lower median sMAPE can rank below another candidate if it achieves fewer per-patient sMAPE wins.")
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
        println(io, "Best configuration-level summary:")
        show(io, MIME"text/plain"(), config_summary)
        println(io)
    end

    return (
        ranking = ranking,
        config_summary = config_summary,
        combined_long = combined_long,
        best_patient_df = best_patient_df,
        candidate_ids = mats.candidate_ids,
        patients = mats.patients,
        smape_mat = mats.smape_mat,
        rmsle_mat = mats.rmsle_mat,
        loss_mat = mats.loss_mat,
        pairwise_wins = pairwise_wins,
        pairwise_ties = pairwise_ties,
    )
end



"""
Run the workflow starting from an explicit candidate specification vector, e.g.

candidate_specs = [
    (path = "res/EXP26/models/test_NN_1/patients_metrics_val.csv", config_code = "26", model_index = 1),
    (path = "res/EXP26/models/test_NN_2/patients_metrics_val.csv", config_code = "26", model_index = 2),
    (path = "res/EXP28/models/test_NN_1/patients_metrics_val.csv", config_code = "28", model_index = 1),
]

This is the most robust option when the CSV filenames themselves do not encode the width/model.
"""
function run_validation_model_selection(candidate_specs::Vector{<:NamedTuple};
        output_dir::AbstractString = "res/model_selection",
        eps_smape::Real = 0.0,
        tie_mode::Symbol = :split,
        top_k_ecdf::Int = 6,
    )

    csv_paths = [String(spec.path) for spec in candidate_specs]
    meta_overrides = map(candidate_specs) do spec
        cfg = String(spec.config_code)
        mid = Int(spec.model_index)
        width_label = haskey(spec, :width_label) ? String(spec.width_label) : (startswith(cfg, "2") && length(cfg) >= 2 ? cfg[2:end] : cfg)
        candidate_id = haskey(spec, :candidate_id) ? String(spec.candidate_id) : "cfg$(cfg)_m$(mid)"
        (
            filename = basename(String(spec.path)),
            config_code = cfg,
            width_label = width_label,
            model_index = mid,
            candidate_id = candidate_id,
        )
    end

    mkpath(output_dir)
    candidate_dfs = [read_candidate_metrics(path; meta_override = meta) for (path, meta) in zip(csv_paths, meta_overrides)]
    sort!(candidate_dfs, by = df -> (parse(Int, first(df.config_code)), Int(first(df.model_index))))
    aligned_dfs, common_patients = align_candidates_on_common_patients(candidate_dfs)
    mats = build_metric_matrices(aligned_dfs)

    ranking, best_per_patient = summarize_and_rank_candidates(mats; eps_smape = eps_smape, tie_mode = tie_mode)
    config_summary = summarize_configs(ranking)
    pairwise_wins, pairwise_ties = pairwise_smape_tables(mats.smape_mat; eps_smape = eps_smape)

    combined_long = combined_long_dataframe(aligned_dfs)
    best_patient_df = best_model_per_patient_df(mats.patients, mats.candidate_ids, mats.smape_mat, mats.rmsle_mat, mats.loss_mat; eps_smape = eps_smape)

    CSV.write(joinpath(output_dir, "combined_validation_metrics_long.csv"), combined_long)
    CSV.write(joinpath(output_dir, "candidate_ranking_smape_rule.csv"), ranking)
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

    plot_wins_bar(ranking; output_path = joinpath(output_dir, "wins_bar.png"))
    plot_median_smape_vs_wins(ranking; output_path = joinpath(output_dir, "wins_vs_median_smape.png"))
    plot_pairwise_heatmap(ranking, mats.candidate_ids, pairwise_wins; output_path = joinpath(output_dir, "pairwise_smape_wins_heatmap.png"))
    plot_all_ecdf_smape(ranking, mats.candidate_ids, mats.smape_mat; output_path = joinpath(output_dir, "all_smape_ecdf.png"))
    plot_topk_ecdf_smape(ranking, mats.candidate_ids, mats.smape_mat; top_k = top_k_ecdf, output_path = joinpath(output_dir, "topk_smape_ecdf.png"))
    plot_best_vs_runnerup(ranking, mats.candidate_ids, mats.smape_mat; output_path = joinpath(output_dir, "best_vs_runnerup_smape_scatter.png"))

    open(joinpath(output_dir, "selection_report.txt"), "w") do io
        println(io, "Validation model selection report")
        println(io, "================================")
        println(io, "Candidates read: $(length(csv_paths))")
        println(io, "Common patients used for fair comparison: $(length(common_patients))")
        println(io, "Tie handling mode: $(tie_mode)")
        println(io, "sMAPE tie tolerance: $(eps_smape)")
        println(io)
        println(io, "Primary rule: wins_smape (desc) -> median_smape (asc) -> median_rmsle (asc) -> median_loss (asc)")
        println(io, "ECDF plots:")
        println(io, " - all_smape_ecdf.png : all $(nrow(ranking)) ranked candidates")
        println(io, " - topk_smape_ecdf.png : top $(min(top_k_ecdf, nrow(ranking))) candidates by the primary rule")
        println(io)
        println(io, "Interpretation note:")
        println(io, "The top-ranked candidate is selected by a lexicographic rule. Therefore, a model with lower median sMAPE can rank below another candidate if it achieves fewer per-patient sMAPE wins.")
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
        println(io, "Best configuration-level summary:")
        show(io, MIME"text/plain"(), config_summary)
        println(io)
    end

    return (
        ranking = ranking,
        config_summary = config_summary,
        combined_long = combined_long,
        best_patient_df = best_patient_df,
        candidate_ids = mats.candidate_ids,
        patients = mats.patients,
        smape_mat = mats.smape_mat,
        rmsle_mat = mats.rmsle_mat,
        loss_mat = mats.loss_mat,
        pairwise_wins = pairwise_wins,
        pairwise_ties = pairwise_ties,
    )
end

# Optional CLI
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("Usage: julia model_selection_from_validation_csvs.jl OUTPUT_DIR CSV1 CSV2 ...")
    output_dir = ARGS[1]
    csv_paths = ARGS[2:end]
    run_validation_model_selection(csv_paths; output_dir = output_dir)
end

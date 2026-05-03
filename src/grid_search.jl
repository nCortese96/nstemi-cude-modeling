using Printf
using DataFrames, CSV
using Statistics
using Plots
using Dates

# ---------------------------------------------
# Robust selection from models_summary CSV only
# ---------------------------------------------
widths = [4, 6, 8]
dataset_name = "MIMIC-IV"

run_ts = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
output_dir = joinpath("res", "model_selection", "robust_summary_$run_ts")
mkpath(output_dir)

required_cols = [
    :model_id, :model_idx, :nn_depth, :nn_width, :n_patients,
    :loss_mean, :loss_std, :loss_median, :loss_q1, :loss_q3, :loss_iqr,
    :smape_mean, :smape_std, :smape_median, :smape_q1, :smape_q3, :smape_iqr,
    :rmsle_mean, :rmsle_std, :rmsle_median, :rmsle_q1, :rmsle_q3, :rmsle_iqr
]

function _add_width_legend_inside!(p, cmap::Dict, width_vals)
    for w in sort(unique(width_vals))
        Plots.scatter!(p, [NaN], [NaN];
            label = "width=$(w)",
            color = cmap[w],
            markersize = 7,
            markershape = :circle,
            markerstrokewidth = 0.5)
    end
    Plots.plot!(p, legend = :best)
    return p
end

function robust_select(df::DataFrame)
    isempty(df) && error("robust_select received an empty DataFrame.")

    ranked = sort(copy(df),
        [:loss_mean, :loss_std, :smape_mean, :smape_std, :rmsle_mean, :rmsle_std],
        rev = [false, false, false, false, false, false])

    best = copy(ranked[1:1, :])
    best[!, :selection_primary] = fill("loss_mean", 1)
    best[!, :selection_tiebreak_1] = fill("loss_std", 1)
    best[!, :selection_tiebreak_2] = fill("smape_mean", 1)
    best[!, :selection_tiebreak_3] = fill("smape_std", 1)
    best[!, :selection_tiebreak_4] = fill("rmsle_mean", 1)
    best[!, :selection_tiebreak_5] = fill("rmsle_std", 1)

    best_loss_mean = best.loss_mean[1]
    return best, ranked, best_loss_mean
end

# function consensus_ranking(df::DataFrame; top_k::Int=3, w_median::Float64=2.0, w_mean::Float64=1.0)
#     d = copy(df)
#     d[!, :topk_hits] = zeros(Int, nrow(d))
#     d[!, :borda_weighted] = zeros(Float64, nrow(d))
#     d[!, :worst_rank] = zeros(Int, nrow(d))

#     metrics = [:smape, :rmsle, :loss]
#     boards = [(:median, w_median), (:mean, w_mean)]

#     for m in metrics
#         for (s, w) in boards
#             col = Symbol("$(m)_$(s)")
#             ord = sortperm(d[!, col])  # lower is better
#             rank = similar(ord)

#             for (r, idx) in enumerate(ord)
#                 rank[idx] = r
#             end

#             d[!, Symbol("rank_$(m)_$(s)")] = rank
#             d[!, :topk_hits] .+= rank .<= top_k
#             d[!, :borda_weighted] .+= w .* max.(0, top_k + 1 .- rank)
#             d[!, :worst_rank] = max.(d[!, :worst_rank], rank)
#         end
#     end

#     sort!(d,
#         [:topk_hits, :borda_weighted, :worst_rank, :smape_median, :rmsle_median, :loss_median],
#         rev = [true, true, false, false, false, false])

#     return d
# end

# USAGE
# consensus = consensus_ranking(general_summary; top_k=3, w_median=2.0, w_mean=1.0)
# CSV.write(joinpath(output_dir, "consensus_ranking.csv"), consensus)

function _width_color_map(width_vals)
    palette = [:steelblue, :darkorange, :forestgreen, :purple, :brown, :deeppink, :teal, :olive]
    u = sort(unique(width_vals))
    return Dict(w => palette[mod1(i, length(palette))] for (i, w) in enumerate(u))
end

function _interval_sort_cols(metric::Symbol)
    if metric == :smape
        return [:smape_median, :smape_iqr, :rmsle_median, :loss_median]
    elseif metric == :rmsle
        return [:rmsle_median, :rmsle_iqr, :smape_median, :loss_median]
    elseif metric == :loss
        return [:loss_median, :loss_iqr, :smape_median, :rmsle_median]
    else
        error("Unsupported metric for interval-ranked plot: $(metric)")
    end
end

function plot_metric_interval_ranked(df::DataFrame, metric::Symbol, selected_model_id::AbstractString; output_path::AbstractString)
    med_col = Symbol("$(metric)_median")
    q1_col = Symbol("$(metric)_q1")
    q3_col = Symbol("$(metric)_q3")

    sort_cols = _interval_sort_cols(metric)
    d = sort(copy(df), sort_cols, rev = fill(false, length(sort_cols)))

    x = collect(1:nrow(d))
    labels = String.(d.model_id)
    y = Float64.(d[!, med_col])
    yerr_low = y .- Float64.(d[!, q1_col])
    yerr_high = Float64.(d[!, q3_col]) .- y

    cmap = _width_color_map(d.nn_width)
    point_colors = [cmap[w] for w in d.nn_width]
    metric_label = uppercase(String(metric))

    p = Plots.scatter(
        x, y;
        yerror = (yerr_low, yerr_high),
        markercolor = point_colors,
        xlabel = "Model (ranked by robust rule on $(metric_label) median)",
        ylabel = "$(metric_label) median with IQR interval",
        xticks = (x, labels),
        xrotation = 45,
        title = "$(metric_label) model ranking with robust uncertainty intervals",
        legend = false,
        markersize = 6,
        markerstrokewidth = 0.4,
        size = (1400, 760),
        left_margin = 12Plots.mm,
        right_margin = 8Plots.mm,
        bottom_margin = 14Plots.mm,
        top_margin = 6Plots.mm,
        label = false
    )

    sel_idx = findfirst(==(selected_model_id), String.(d.model_id))
    if sel_idx !== nothing
        Plots.scatter!(p, [sel_idx], [y[sel_idx]];
            markershape = :star5, markersize = 12, color = :black, label = "selected")
    end

    _add_width_legend_inside!(p, cmap, d.nn_width)

    savefig(p, output_path)
    return p
end

# function plot_smape_interval_ranked(df::DataFrame, selected_model_id::AbstractString; output_path::AbstractString)
#     return plot_metric_interval_ranked(df, :smape, selected_model_id; output_path = output_path)
# end

function plot_smape_median_vs_iqr(df::DataFrame, selected_model_id::AbstractString; output_path::AbstractString)
    d = copy(df)
    cmap = _width_color_map(d.nn_width)

    p = Plots.plot(
        xlabel = "Median sMAPE",
        ylabel = "sMAPE IQR",
        title = "Robust selection space: median vs IQR (lower is better)",
        # legend = :outerright,
        legend = :best,
        size = (1150, 820),
        left_margin = 12Plots.mm,
        right_margin = 32Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )

    for w in sort(unique(d.nn_width))
        idx = findall(==(w), d.nn_width)
        Plots.scatter!(p, d.smape_median[idx], d.smape_iqr[idx];
            label = "width=$(w)",
            color = cmap[w],
            markersize = 8,
            markerstrokewidth = 0.5,
        )
    end

    sel = filter(:model_id => ==(selected_model_id), d)
    if nrow(sel) > 0
        Plots.scatter!(p, sel.smape_median, sel.smape_iqr;
            label = "selected",
            markershape = :star5,
            markersize = 14,
            color = :black)
        Plots.annotate!(p, sel.smape_median[1], sel.smape_iqr[1],
            Plots.text("  " * selected_model_id, 9, :left))
    end

    savefig(p, output_path)
    return p
end

function plot_metric_mean_std_ranked(df::DataFrame, metric::Symbol, selected_model_id::AbstractString; output_path::AbstractString)
    mean_col = Symbol("$(metric)_mean")
    std_col = Symbol("$(metric)_std")

    sort_cols = if metric == :smape
        [:smape_mean, :smape_std, :rmsle_mean, :loss_mean]
    elseif metric == :rmsle
        [:rmsle_mean, :rmsle_std, :smape_mean, :loss_mean]
    elseif metric == :loss
        [:loss_mean, :loss_std, :smape_mean, :rmsle_mean]
    else
        error("Unsupported metric: $(metric)")
    end

    d = sort(copy(df), sort_cols, rev = fill(false, length(sort_cols)))

    x = collect(1:nrow(d))
    labels = String.(d.model_id)
    y = Float64.(d[!, mean_col])
    ystd = Float64.(d[!, std_col])

    cmap = _width_color_map(d.nn_width)
    point_colors = [cmap[w] for w in d.nn_width]

    p = Plots.scatter(
        x, y;
        yerror = (ystd, ystd),
        markercolor = point_colors,
        xlabel = "Model (ranked by $(metric) mean + std)",
        ylabel = "$(uppercase(String(metric))) mean ± std",
        xticks = (x, labels),
        xrotation = 45,
        title = "$(uppercase(String(metric))) comparison (mean ± std)",
        legend = false,
        markersize = 6,
        markerstrokewidth = 0.4,
        size = (1400, 760),
        left_margin = 12Plots.mm,
        right_margin = 8Plots.mm,
        bottom_margin = 14Plots.mm,
        top_margin = 6Plots.mm,
        label = false
    )

    sel_idx = findfirst(==(selected_model_id), String.(d.model_id))
    if sel_idx !== nothing
        Plots.scatter!(p, [sel_idx], [y[sel_idx]];
            markershape = :star5, markersize = 12, color = :black, label = "selected")
    end

    _add_width_legend_inside!(p, cmap, d.nn_width)

    savefig(p, output_path)
    return p
end

function plot_tiebreak_rmsle_vs_loss(df::DataFrame, selected_model_id::AbstractString; output_path::AbstractString)
    d = copy(df)
    cmap = _width_color_map(d.nn_width)

    p = Plots.plot(
        xlabel = "RMSLE median",
        ylabel = "Loss median",
        title = "Tie-break view: RMSLE vs Loss (lower is better)",
        # legend = :outerright,
        legend = :best,
        size = (1100, 780),
        left_margin = 12Plots.mm,
        right_margin = 30Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )

    for w in sort(unique(d.nn_width))
        idx = findall(==(w), d.nn_width)
        Plots.scatter!(p, d.rmsle_median[idx], d.loss_median[idx];
            label = "width=$(w)",
            color = cmap[w],
            markersize = 8,
            markerstrokewidth = 0.5,
        )
    end

    sel = filter(:model_id => ==(selected_model_id), d)
    if nrow(sel) > 0
        Plots.scatter!(p, sel.rmsle_median, sel.loss_median;
            label = "selected",
            markershape = :star5,
            markersize = 14,
            color = :black)
    end

    savefig(p, output_path)
    return p
end

function plot_mean_vs_median_dumbbell(df::DataFrame; output_path::AbstractString)
    metrics = [:smape, :rmsle, :loss]
    cmap = _width_color_map(df.nn_width)

    p = Plots.plot(
        layout = (3, 1),
        size = (1450, 1250),
        left_margin = 12Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 8Plots.mm
    )

    for (k, metric) in enumerate(metrics)
        mean_col = Symbol("$(metric)_mean")
        med_col = Symbol("$(metric)_median")

        d = sort(copy(df), med_col)
        x = collect(1:nrow(d))
        labels = String.(d.model_id)
        y_mean = Float64.(d[!, mean_col])
        y_med = Float64.(d[!, med_col])
        cols = [cmap[w] for w in d.nn_width]

        for i in eachindex(x)
            Plots.plot!(p[k], [x[i], x[i]], [y_med[i], y_mean[i]];
                color = :gray60, alpha = 0.5, lw = 1.3, label = false)
        end

        Plots.scatter!(p[k], x, y_med;
            markercolor = cols, markershape = :circle, markersize = 5.5,
            markerstrokewidth = 0.3, label = k == 1 ? "median" : false)

        Plots.scatter!(p[k], x, y_mean;
            markercolor = cols, markershape = :diamond, markersize = 5.5,
            markerstrokewidth = 0.3, label = k == 1 ? "mean" : false)

        Plots.plot!(p[k];
            xticks = (x, labels), xrotation = 45,
            xlabel = "Models (ordered by $(uppercase(String(metric))) median)",
            ylabel = uppercase(String(metric)),
            title = "$(uppercase(String(metric))): mean vs median",
            legend = k == 1 ? :outertopright : false)
    end

    for w in sort(unique(df.nn_width))
        Plots.scatter!(p[1], [NaN], [NaN];
            color = cmap[w], markershape = :circle, markersize = 7,
            markerstrokewidth = 0.3, label = "width=$(w)")
    end

    savefig(p, output_path)
    return p
end

summary_chunks = DataFrame[]
missing_summary_files = String[]

for w in widths
    println("Processing width = $w")
    exp_path = "NSTEMI_cUDE_$(dataset_name)_MSE_2$(w)_sigmoid_regback"
    summary_csv_path = "res/$(exp_path)/models/models_summary_$(dataset_name).csv"

    if !isfile(summary_csv_path)
        push!(missing_summary_files, summary_csv_path)
        continue
    end

    df = CSV.read(summary_csv_path, DataFrame)

    miss = setdiff(required_cols, Symbol.(names(df)))
    isempty(miss) || error("Missing columns in $(summary_csv_path): $(miss)")

    df[!, :experiment_path] = fill(exp_path, nrow(df))
    push!(summary_chunks, df)
end

isempty(summary_chunks) && error("No models_summary files found for selected widths.")

general_summary = vcat(summary_chunks...; cols = :union)
sort!(general_summary,
      [:loss_mean, :loss_std, :smape_mean, :smape_std, :rmsle_mean, :rmsle_std],
      rev = fill(false, 6))

CSV.write(joinpath(output_dir, "general_summary_$(dataset_name).csv"), general_summary)

if !isempty(missing_summary_files)
    @warn("Missing models_summary files:\n" * join(missing_summary_files, "\n"))
end

# Global robust selection across all models
best_global, pool_global, best_loss = robust_select(general_summary)
selected_model_id = String(best_global.model_id[1])

CSV.write(joinpath(output_dir, "robust_selected_model_$(dataset_name).csv"), best_global)
CSV.write(joinpath(output_dir, "robust_candidate_pool_$(dataset_name).csv"), pool_global)

# Best model per width (same robust criterion)
rows = DataFrame[]
for g in groupby(general_summary, :nn_width)
    gw = copy(g)
    best_w, pool_w, best_loss_w = robust_select(gw)

    push!(rows, DataFrame(
        nn_width = [best_w.nn_width[1]],
        best_model_id = [best_w.model_id[1]],
        best_model_idx = [best_w.model_idx[1]],

        # criteri effettivi di selezione
        best_loss_mean = [best_w.loss_mean[1]],
        best_loss_std = [best_w.loss_std[1]],
        best_smape_mean = [best_w.smape_mean[1]],
        best_smape_std = [best_w.smape_std[1]],
        best_rmsle_mean = [best_w.rmsle_mean[1]],
        best_rmsle_std = [best_w.rmsle_std[1]],

        # contesto descrittivo (opzionale)
        best_loss_median = [best_w.loss_median[1]],
        best_loss_iqr = [best_w.loss_iqr[1]],
        best_smape_median = [best_w.smape_median[1]],
        best_smape_iqr = [best_w.smape_iqr[1]],
        best_rmsle_median = [best_w.rmsle_median[1]],
        best_rmsle_iqr = [best_w.rmsle_iqr[1]],

        # uguale a best_loss_mean, lo tieni solo come colonna esplicativa
        # best_loss_for_width = [best_loss_w],
        pool_size = [nrow(pool_w)],
    ))
end

best_by_width = vcat(rows...)
sort!(best_by_width, :nn_width)
CSV.write(joinpath(output_dir, "robust_best_by_width_$(dataset_name).csv"), best_by_width)

# Plots
plot_smape_median_vs_iqr(general_summary, selected_model_id;
    output_path = joinpath(output_dir, "plot_smape_median_vs_iqr.png"))
plot_tiebreak_rmsle_vs_loss(general_summary, selected_model_id;
    output_path = joinpath(output_dir, "plot_tiebreak_rmsle_vs_loss.png"))
plot_metric_mean_std_ranked(general_summary, :smape, selected_model_id;
    output_path = joinpath(output_dir, "plot_smape_mean_std_ranked.png"))
plot_metric_mean_std_ranked(general_summary, :rmsle, selected_model_id;
    output_path = joinpath(output_dir, "plot_rmsle_mean_std_ranked.png"))
plot_metric_mean_std_ranked(general_summary, :loss, selected_model_id;
    output_path = joinpath(output_dir, "plot_loss_mean_std_ranked.png"))
plot_metric_interval_ranked(general_summary, :smape, selected_model_id;
    output_path = joinpath(output_dir, "plot_smape_interval_ranked.png"))
plot_metric_interval_ranked(general_summary, :rmsle, selected_model_id;
    output_path = joinpath(output_dir, "plot_rmsle_interval_ranked.png"))
plot_metric_interval_ranked(general_summary, :loss, selected_model_id;
    output_path = joinpath(output_dir, "plot_loss_interval_ranked.png"))
plot_mean_vs_median_dumbbell(general_summary;
    output_path = joinpath(output_dir, "plot_mean_vs_median_dumbbell.png"))

open(joinpath(output_dir, "robust_selection_report.txt"), "w") do io
    println(io, "Robust selection from models_summary")
    println(io, "====================================")
    println(io, "Run timestamp: $(run_ts)")
    println(io, "Output directory: $(output_dir)")
    println(io, "Dataset: $(dataset_name)")
    println(io, "Widths considered: $(widths)")
    println(io)
    println(io, "Rule:")
    println(io, "1) choose minimum loss_mean")
    println(io, "2) tie-break on loss_std")
    println(io, "3) tie-break on smape_mean, then smape_std")
    println(io, "4) tie-break on rmsle_mean, then rmsle_std")
    println(io)
    println(io, "Selected model:")
    show(io, MIME"text/plain"(), best_global)
    println(io)
    println(io)
    println(io, "Best model by width:")
    show(io, MIME"text/plain"(), best_by_width)
    println(io)
end

println("Done. Selected model: ", selected_model_id)
println("Outputs in: ", output_dir)

# using Printf
# using DataFrames, CSV
# # using Revise
# # includet("model_selection_clean.jl")
# # includet("model_selection_minimal.jl")


# widths = [4, 6, 8, 16]
# n_models = 4
# dataset_name = "MIMIC-IV"

# use_multistart = true

# candidate_specs = NamedTuple{(:path, :config_code, :model_index), Tuple{String, String, Int}}[]
# missing_files = String[]

# # Added logic: collect per-width models summaries without replacing current selection flow
# summary_chunks = DataFrame[]
# missing_summary_files = String[]

# for w in widths
#     println("Processing width = $w")
#     exp_path = "NSTEMI_cUDE_$(dataset_name)_MSE_2$(w)_sigmoid_regback"

#     summary_csv_path = "res/$(exp_path)/models/models_summary_$(dataset_name).csv"
#     if isfile(summary_csv_path)
#         df_summary = CSV.read(summary_csv_path, DataFrame)
#         df_summary[!, :experiment_path] = fill(exp_path, nrow(df_summary))
#         push!(summary_chunks, df_summary)
#     else
#         push!(missing_summary_files, summary_csv_path)
#     end

#     for j in 1:n_models
#         println("  Processing model index = $j")

#         csv_path = "res/$(exp_path)/models/$(dataset_name)_test_NN_$(j)$(use_multistart ? "_ms_test" : "")/patients_metrics_val.csv"

#         # Fallback for runs saved under *_ms_test
#         if isfile(csv_path)
#             csv_path_ms = "res/$(exp_path)/models/$(dataset_name)_test_NN_$(j)_ms_test/patients_metrics_val.csv"
#         else
#             push!(missing_files, csv_path * " OR " * csv_path_ms)
#             continue
#         end

#         push!(candidate_specs, (
#             path = csv_path,
#             config_code = "2$(w)",
#             model_index = j,
#         ))
#     end
# end

# # Added logic: write global summary sorted by smape_median
# if !isempty(summary_chunks)
#     general_summary = vcat(summary_chunks...; cols = :union)
#     sort!(general_summary, :smape_median)
#     CSV.write("res/general_summary_$(dataset_name).csv", general_summary)
# else
#     @warn "No models_summary CSV files found. general summary was not written."
# end

# if !isempty(missing_summary_files)
#     @warn(
#         "The following models_summary CSV files were not found:\n" *
#         join(missing_summary_files, "\n")
#     )
# end

# if !isempty(missing_files)
#     error(
#         "The following validation CSV files were not found:\n" *
#         join(missing_files, "\n")
#     )
# end

# expected_n = length(widths) * n_models
# found_n = length(candidate_specs)

# if found_n != expected_n
#     error("Expected $expected_n CSV files, but found $found_n.")
# end

# # results = run_validation_model_selection(
# #     candidate_specs;
# #     output_dir = "res/model_selection/test",
# #     eps_smape = 0.0,
# #     tie_mode = :split,
# #     top_k_ecdf = 6,
# # )

# # results_minimal = run_validation_model_selection_minimal(
# #     candidate_specs;
# #     output_dir = "res/model_selection/minimal",
# #     delta_tol = 0.0,
# # )
using Printf
using DataFrames, CSV
using Statistics
using Plots

# ---------------------------------------------
# Robust selection from models_summary CSV only
# ---------------------------------------------
widths = [4, 6, 8, 16]
dataset_name = "MIMIC-IV"

# Robustness margin on sMAPE median:
# keep models with smape_median <= best_smape_median + smape_margin
smape_margin = 0.30

output_dir = "res/model_selection/robust_summary"
mkpath(output_dir)

required_cols = [
    :model_id, :model_idx, :nn_depth, :nn_width, :n_patients,
    :loss_mean, :loss_std, :loss_median, :loss_q1, :loss_q3, :loss_iqr,
    :smape_mean, :smape_std, :smape_median, :smape_q1, :smape_q3, :smape_iqr,
    :rmsle_mean, :rmsle_std, :rmsle_median, :rmsle_q1, :rmsle_q3, :rmsle_iqr
]

function robust_select(df::DataFrame; smape_margin::Real = 0.30)
    isempty(df) && error("robust_select received an empty DataFrame.")

    best_smape = minimum(df.smape_median)
    pool = filter(row -> row.smape_median <= best_smape + smape_margin, df)

    sort!(pool, [:smape_iqr, :rmsle_median, :loss_median, :smape_median],
          rev = [false, false, false, false])

    best = copy(pool[1:1, :])
    best[!, :selection_smape_margin] = fill(smape_margin, 1)
    best[!, :selection_best_smape_median] = fill(best_smape, 1)

    return best, pool, best_smape
end

function _width_color_map(width_vals)
    palette = [:steelblue, :darkorange, :forestgreen, :purple, :brown, :deeppink, :teal, :olive]
    u = sort(unique(width_vals))
    return Dict(w => palette[mod1(i, length(palette))] for (i, w) in enumerate(u))
end

function plot_smape_median_vs_iqr(df::DataFrame, selected_model_id::AbstractString; output_path::AbstractString)
    d = copy(df)
    cmap = _width_color_map(d.nn_width)

    p = plot(
        xlabel = "Median sMAPE",
        ylabel = "sMAPE IQR",
        title = "Robust selection space: median vs IQR (lower is better)",
        legend = :outerright,
        size = (1150, 820),
        left_margin = 12Plots.mm,
        right_margin = 32Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )

    for w in sort(unique(d.nn_width))
        idx = findall(==(w), d.nn_width)
        scatter!(p, d.smape_median[idx], d.smape_iqr[idx];
            label = "width=$(w)",
            color = cmap[w],
            markersize = 8,
            markerstrokewidth = 0.5,
        )
    end

    sel = filter(:model_id => ==(selected_model_id), d)
    if nrow(sel) > 0
        scatter!(p, sel.smape_median, sel.smape_iqr;
            label = "selected",
            markershape = :star5,
            markersize = 14,
            color = :black)
        annotate!(p, sel.smape_median[1], sel.smape_iqr[1],
            Plots.text("  " * selected_model_id, 9, :left))
    end

    savefig(p, output_path)
    return p
end

function plot_smape_interval_ranked(df::DataFrame, selected_model_id::AbstractString; output_path::AbstractString)
    d = sort(copy(df), [:smape_median, :smape_iqr, :rmsle_median, :loss_median],
             rev = [false, false, false, false])

    x = collect(1:nrow(d))
    labels = String.(d.model_id)
    y = Float64.(d.smape_median)
    yerr_low = y .- Float64.(d.smape_q1)
    yerr_high = Float64.(d.smape_q3) .- y

    p = scatter(
        x, y;
        yerror = (yerr_low, yerr_high),
        xlabel = "Model (ranked by robust rule)",
        ylabel = "sMAPE median with IQR interval",
        xticks = (x, labels),
        xrotation = 45,
        title = "Model ranking with robust uncertainty intervals",
        legend = false,
        markersize = 6,
        markerstrokewidth = 0.4,
        size = (1400, 760),
        left_margin = 12Plots.mm,
        right_margin = 8Plots.mm,
        bottom_margin = 14Plots.mm,
        top_margin = 6Plots.mm,
    )

    sel_idx = findfirst(==(selected_model_id), String.(d.model_id))
    if sel_idx !== nothing
        scatter!(p, [sel_idx], [y[sel_idx]];
            markershape = :star5, markersize = 12, color = :black, label = false)
    end

    savefig(p, output_path)
    return p
end

function plot_tiebreak_rmsle_vs_loss(df::DataFrame, selected_model_id::AbstractString; output_path::AbstractString)
    d = copy(df)
    cmap = _width_color_map(d.nn_width)

    p = plot(
        xlabel = "RMSLE median",
        ylabel = "Loss median",
        title = "Tie-break view: RMSLE vs Loss (lower is better)",
        legend = :outerright,
        size = (1100, 780),
        left_margin = 12Plots.mm,
        right_margin = 30Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin = 6Plots.mm,
    )

    for w in sort(unique(d.nn_width))
        idx = findall(==(w), d.nn_width)
        scatter!(p, d.rmsle_median[idx], d.loss_median[idx];
            label = "width=$(w)",
            color = cmap[w],
            markersize = 8,
            markerstrokewidth = 0.5,
        )
    end

    sel = filter(:model_id => ==(selected_model_id), d)
    if nrow(sel) > 0
        scatter!(p, sel.rmsle_median, sel.loss_median;
            label = "selected",
            markershape = :star5,
            markersize = 14,
            color = :black)
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
sort!(general_summary, [:smape_median, :smape_iqr, :rmsle_median, :loss_median],
      rev = [false, false, false, false])

CSV.write("res/general_summary_$(dataset_name).csv", general_summary)

if !isempty(missing_summary_files)
    @warn("Missing models_summary files:\n" * join(missing_summary_files, "\n"))
end

# Global robust selection across all models
best_global, pool_global, best_smape = robust_select(general_summary; smape_margin = smape_margin)
selected_model_id = String(best_global.model_id[1])

CSV.write(joinpath(output_dir, "robust_selected_model_$(dataset_name).csv"), best_global)
CSV.write(joinpath(output_dir, "robust_candidate_pool_$(dataset_name).csv"), pool_global)

# Best model per width (same robust criterion)
rows = DataFrame[]
for g in groupby(general_summary, :nn_width)
    gw = copy(g)
    best_w, pool_w, best_smape_w = robust_select(gw; smape_margin = smape_margin)

    push!(rows, DataFrame(
        nn_width = [best_w.nn_width[1]],
        best_model_id = [best_w.model_id[1]],
        best_model_idx = [best_w.model_idx[1]],
        best_smape_median = [best_w.smape_median[1]],
        best_smape_iqr = [best_w.smape_iqr[1]],
        best_rmsle_median = [best_w.rmsle_median[1]],
        best_loss_median = [best_w.loss_median[1]],
        best_smape_for_width = [best_smape_w],
        pool_size = [nrow(pool_w)],
    ))
end

best_by_width = vcat(rows...)
sort!(best_by_width, :nn_width)
CSV.write(joinpath(output_dir, "robust_best_by_width_$(dataset_name).csv"), best_by_width)

# Plots
plot_smape_median_vs_iqr(general_summary, selected_model_id;
    output_path = joinpath(output_dir, "plot_smape_median_vs_iqr.png"))
plot_smape_interval_ranked(general_summary, selected_model_id;
    output_path = joinpath(output_dir, "plot_smape_interval_ranked.png"))
plot_tiebreak_rmsle_vs_loss(general_summary, selected_model_id;
    output_path = joinpath(output_dir, "plot_tiebreak_rmsle_vs_loss.png"))

open(joinpath(output_dir, "robust_selection_report.txt"), "w") do io
    println(io, "Robust selection from models_summary")
    println(io, "====================================")
    println(io, "Dataset: $(dataset_name)")
    println(io, "Widths considered: $(widths)")
    println(io, "Selection margin on smape_median: $(smape_margin)")
    println(io)
    println(io, "Rule:")
    println(io, "1) keep models with smape_median <= best_smape + margin")
    println(io, "2) choose minimum smape_iqr")
    println(io, "3) tie-break on rmsle_median, then loss_median, then smape_median")
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
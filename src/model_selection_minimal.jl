# ==========================
# Minimal model selection
# ==========================

_rank_ascending(v::AbstractVector{<:Real}) = invperm(sortperm(collect(v)))

function summarize_and_rank_candidates_minimal(mats)
    n_candidates = length(mats.candidate_ids)

    smape_mean   = [mean(mats.smape_mat[:, j]) for j in 1:n_candidates]
    smape_std    = [std(mats.smape_mat[:, j]) for j in 1:n_candidates]
    smape_median = [median(mats.smape_mat[:, j]) for j in 1:n_candidates]
    smape_q1     = [quantile(mats.smape_mat[:, j], 0.25) for j in 1:n_candidates]
    smape_q3     = [quantile(mats.smape_mat[:, j], 0.75) for j in 1:n_candidates]

    loss_mean    = [mean(mats.loss_mat[:, j]) for j in 1:n_candidates]
    loss_std     = [std(mats.loss_mat[:, j]) for j in 1:n_candidates]
    loss_median  = [median(mats.loss_mat[:, j]) for j in 1:n_candidates]
    loss_q1      = [quantile(mats.loss_mat[:, j], 0.25) for j in 1:n_candidates]
    loss_q3      = [quantile(mats.loss_mat[:, j], 0.75) for j in 1:n_candidates]

    rmsle_mean   = [mean(mats.rmsle_mat[:, j]) for j in 1:n_candidates]
    rmsle_median = [median(mats.rmsle_mat[:, j]) for j in 1:n_candidates]

    ranking = DataFrame(
        candidate_id   = mats.candidate_ids,
        config_code    = mats.config_codes,
        width_label    = mats.width_labels,
        model_index    = mats.model_indices,

        smape_mean     = smape_mean,
        smape_std      = smape_std,
        smape_median   = smape_median,
        smape_q1       = smape_q1,
        smape_q3       = smape_q3,
        smape_iqr      = smape_q3 .- smape_q1,

        loss_mean      = loss_mean,
        loss_std       = loss_std,
        loss_median    = loss_median,
        loss_q1        = loss_q1,
        loss_q3        = loss_q3,
        loss_iqr       = loss_q3 .- loss_q1,

        rmsle_mean     = rmsle_mean,
        rmsle_median   = rmsle_median,
    )

    score_metrics = [
        :smape_mean, :smape_std, :smape_median, :smape_iqr,
        :loss_mean,  :loss_std,  :loss_median,  :loss_iqr
    ]

    rank_cols = Symbol[]
    for m in score_metrics
        c = Symbol("rank_" * String(m))
        ranking[!, c] = _rank_ascending(ranking[!, m])
        push!(rank_cols, c)
    end

    ranking[!, :composite_rank_score] = [
        mean(Float64[ranking[i, c] for c in rank_cols]) for i in 1:nrow(ranking)
    ]

    # Minimal deterministic rule:
    # 1) composite rank score (lower better)
    # 2) rmsle median (lower better)
    # 3) rmsle mean (lower better)
    sort!(ranking, [:composite_rank_score, :rmsle_median, :rmsle_mean],
        rev = [false, false, false])

    ranking[!, :rank_minimal] = 1:nrow(ranking)
    return ranking
end

function summarize_configs_minimal(candidate_ranking::DataFrame)
    grouped = groupby(candidate_ranking, :config_code)
    rows = DataFrame[]

    for g in grouped
        gs = sort(copy(g), :rank_minimal)
        best = gs[1, :]

        push!(rows, DataFrame(
            config_code            = [best.config_code],
            width_label            = [best.width_label],
            best_candidate_id      = [best.candidate_id],
            best_model_index       = [best.model_index],
            best_rank_minimal      = [best.rank_minimal],
            best_composite_score   = [best.composite_rank_score],
            best_smape_median      = [best.smape_median],
            best_smape_iqr         = [best.smape_iqr],
            best_loss_median       = [best.loss_median],
            best_loss_iqr          = [best.loss_iqr],
            best_rmsle_median      = [best.rmsle_median],
            n_models_in_config     = [nrow(gs)],
        ))
    end

    out = vcat(rows...)
    sort!(out, :best_rank_minimal)
    return out
end

function sign_test_pvalue(delta::AbstractVector{<:Real}; tol::Real = 0.0)
    n_pos = count(x -> x > tol, delta)
    n_neg = count(x -> x < -tol, delta)
    n = n_pos + n_neg
    n == 0 && return 1.0

    k = min(n_pos, n_neg)
    p_lower = zero(BigFloat)
    for i in 0:k
        p_lower += BigFloat(binomial(n, i))
    end
    p = 2 * p_lower / (BigFloat(2)^n)
    return Float64(min(p, BigFloat(1.0)))
end

function plot_minimal_summary_panel(candidate_ranking::DataFrame; output_path::AbstractString)
    df = sort(copy(candidate_ranking), :rank_minimal)
    n = nrow(df)
    x = collect(1:n)
    labels = string.(df.candidate_id)

    p1 = plot(
        title = "sMAPE summary (ranked models)",
        xlabel = "Candidate model",
        ylabel = "sMAPE",
        xticks = (x, labels),
        xrotation = 45,
        legend = :topright,
        left_margin = 12Plots.mm,
        right_margin = 8Plots.mm,
        bottom_margin = 8Plots.mm,
    )

    for i in x
        plot!(p1, [i, i], [df.smape_q1[i], df.smape_q3[i]];
            linewidth = 6, linecolor = :steelblue, label = false)
        plot!(p1, [i, i], [df.smape_mean[i] - df.smape_std[i], df.smape_mean[i] + df.smape_std[i]];
            linewidth = 2.5, linecolor = :firebrick, label = false)
        scatter!(p1, [i], [df.smape_median[i]];
            markersize = 6, markershape = :circle, markercolor = :black, label = false)
        scatter!(p1, [i], [df.smape_mean[i]];
            markersize = 6, markershape = :xcross, markercolor = :firebrick, label = false)
    end

    plot!(p1, [NaN, NaN], [NaN, NaN]; linewidth = 6, linecolor = :steelblue, label = "IQR")
    plot!(p1, [NaN, NaN], [NaN, NaN]; linewidth = 2.5, linecolor = :firebrick, label = "mean ± std")
    scatter!(p1, [NaN], [NaN]; markershape = :circle, markercolor = :black, label = "median")
    scatter!(p1, [NaN], [NaN]; markershape = :xcross, markercolor = :firebrick, label = "mean")

    p2 = plot(
        title = "Loss summary (ranked models)",
        xlabel = "Candidate model",
        ylabel = "Loss",
        xticks = (x, labels),
        xrotation = 45,
        legend = :topright,
        left_margin = 12Plots.mm,
        right_margin = 8Plots.mm,
        bottom_margin = 10Plots.mm,
    )

    for i in x
        plot!(p2, [i, i], [df.loss_q1[i], df.loss_q3[i]];
            linewidth = 6, linecolor = :seagreen, label = false)
        plot!(p2, [i, i], [df.loss_mean[i] - df.loss_std[i], df.loss_mean[i] + df.loss_std[i]];
            linewidth = 2.5, linecolor = :darkorange, label = false)
        scatter!(p2, [i], [df.loss_median[i]];
            markersize = 6, markershape = :circle, markercolor = :black, label = false)
        scatter!(p2, [i], [df.loss_mean[i]];
            markersize = 6, markershape = :xcross, markercolor = :darkorange, label = false)
    end

    plot!(p2, [NaN, NaN], [NaN, NaN]; linewidth = 6, linecolor = :seagreen, label = "IQR")
    plot!(p2, [NaN, NaN], [NaN, NaN]; linewidth = 2.5, linecolor = :darkorange, label = "mean ± std")
    scatter!(p2, [NaN], [NaN]; markershape = :circle, markercolor = :black, label = "median")
    scatter!(p2, [NaN], [NaN]; markershape = :xcross, markercolor = :darkorange, label = "mean")

    fig = plot(p1, p2; layout = (2, 1), size = (1500, 1100))
    savefig(fig, output_path)
    return fig
end

function plot_minimal_rank_heatmap(candidate_ranking::DataFrame; output_path::AbstractString)
    df = sort(copy(candidate_ranking), :rank_minimal)

    rank_cols = [
        :rank_smape_mean, :rank_smape_std, :rank_smape_median, :rank_smape_iqr,
        :rank_loss_mean,  :rank_loss_std,  :rank_loss_median,  :rank_loss_iqr
    ]
    xlabels = ["sMAPE mean", "sMAPE std", "sMAPE median", "sMAPE IQR",
               "Loss mean", "Loss std", "Loss median", "Loss IQR"]

    mat = hcat([Float64.(df[!, c]) for c in rank_cols]...)
    ylabels = reverse(string.(df.candidate_id))
    mat_plot = reverse(mat, dims = 1)

    p = heatmap(
        xlabels, ylabels, mat_plot;
        xlabel = "Component rank",
        ylabel = "Candidate model",
        title = "Minimal rank decomposition (1 = best)",
        xrotation = 35,
        color = cgrad(:viridis, rev = true),
        colorbar_title = "Rank",
        size = (1250, 900),
        left_margin = 16Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 16Plots.mm,
    )

    for i in 1:size(mat_plot, 1), j in 1:size(mat_plot, 2)
        annotate!(p, j, i, Plots.text(string(Int(round(mat_plot[i, j]))), 8, :white))
    end

    savefig(p, output_path)
    return p
end

function plot_minimal_top2_delta(candidate_ranking::DataFrame, candidate_ids::Vector{String},
        metric_mat::AbstractMatrix; metric_name::AbstractString = "sMAPE",
        tol::Real = 0.0, output_path::AbstractString)
    nrow(candidate_ranking) >= 2 || return nothing

    df = sort(copy(candidate_ranking), :rank_minimal)
    best_id = string(df.candidate_id[1])
    runner_id = string(df.candidate_id[2])

    id_to_col = Dict(id => i for (i, id) in enumerate(candidate_ids))
    best = vec(metric_mat[:, id_to_col[best_id]])
    runner = vec(metric_mat[:, id_to_col[runner_id]])

    delta = runner .- best
    ord = sortperm(delta)
    delta_sorted = delta[ord]

    colors = [d > tol ? "#1f77b4" : (d < -tol ? "#d62728" : "#7f7f7f") for d in delta_sorted]

    p = bar(
        1:length(delta_sorted), delta_sorted;
        color = colors,
        label = false,
        xlabel = "Patient (sorted)",
        ylabel = "$(metric_name)(runner-up) - $(metric_name)(best)",
        title = "Top-2 paired difference on $(metric_name): $(best_id) vs $(runner_id)",
        size = (1250, 700),
        left_margin = 14Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
    )

    hline!(p, [0.0]; linecolor = :black, linestyle = :dash, linewidth = 1.8, label = false)

    n = length(delta)
    n_best = count(x -> x > tol, delta)
    n_worse = count(x -> x < -tol, delta)
    n_tie = n - n_best - n_worse
    pval = sign_test_pvalue(delta; tol = tol)

    ymax = maximum(delta_sorted)
    ymin = minimum(delta_sorted)
    span = ymax - ymin + eps()
    ytxt = ymax - 0.08 * span

    annotate!(p, length(delta_sorted) * 0.02, ytxt,
        Plots.text("best better: $(n_best)/$(n), tie: $(n_tie), runner-up better: $(n_worse), sign-test p=$( @sprintf("%.3g", pval) )", 9, :left))

    savefig(p, output_path)

    return p, (
        best_id = best_id,
        runner_id = runner_id,
        n_best = n_best,
        n_worse = n_worse,
        n_tie = n_tie,
        p_value = pval,
    )
end

function _run_minimal_selection_workflow(candidate_dfs::Vector{DataFrame}, csv_paths::Vector{String};
        output_dir::AbstractString = "res/model_selection/minimal",
        delta_tol::Real = 0.0)
    isempty(candidate_dfs) && error("No candidate dataframes provided.")
    mkpath(output_dir)

    sort!(candidate_dfs, by = df -> (parse(Int, first(df.config_code)), Int(first(df.model_index))))
    aligned_dfs, common_patients = align_candidates_on_common_patients(candidate_dfs)
    mats = build_metric_matrices(aligned_dfs)

    ranking = summarize_and_rank_candidates_minimal(mats)
    config_summary = summarize_configs_minimal(ranking)
    combined_long = combined_long_dataframe(aligned_dfs)

    CSV.write(joinpath(output_dir, "candidate_ranking_minimal.csv"), ranking)
    CSV.write(joinpath(output_dir, "config_summary_minimal.csv"), config_summary)
    CSV.write(joinpath(output_dir, "combined_validation_metrics_long.csv"), combined_long)

    plot_minimal_summary_panel(ranking;
        output_path = joinpath(output_dir, "minimal_summary_panel.png"))
    plot_minimal_rank_heatmap(ranking;
        output_path = joinpath(output_dir, "minimal_rank_heatmap.png"))

    cmp_smape = nothing
    cmp_loss = nothing
    if nrow(ranking) >= 2
        _, cmp_smape = plot_minimal_top2_delta(ranking, mats.candidate_ids, mats.smape_mat;
            metric_name = "sMAPE", tol = delta_tol,
            output_path = joinpath(output_dir, "top2_delta_smape.png"))
        _, cmp_loss = plot_minimal_top2_delta(ranking, mats.candidate_ids, mats.loss_mat;
            metric_name = "Loss", tol = delta_tol,
            output_path = joinpath(output_dir, "top2_delta_loss.png"))
    end

    open(joinpath(output_dir, "selection_report_minimal.txt"), "w") do io
        println(io, "Minimal model-selection report")
        println(io, "==============================")
        println(io, "Candidates: $(length(csv_paths))")
        println(io, "Common patients: $(length(common_patients))")
        println(io, "Composite score = average rank across:")
        println(io, "  sMAPE(mean,std,median,iqr) + Loss(mean,std,median,iqr)")
        println(io, "Tie-break: RMSLE median, then RMSLE mean")
        println(io)
        println(io, "Top-ranked model:")
        show(io, MIME"text/plain"(), ranking[1:1, :])
        println(io)
        println(io)
        if cmp_smape !== nothing
            println(io, "Top-2 sMAPE paired sign-test p-value: $(cmp_smape.p_value)")
            println(io, "Top-2 Loss paired sign-test p-value: $(cmp_loss.p_value)")
            println(io)
        end
        println(io, "Config-level summary:")
        show(io, MIME"text/plain"(), config_summary)
        println(io)
    end

    return (
        ranking = ranking,
        config_summary = config_summary,
        combined_long = combined_long,
        candidate_ids = mats.candidate_ids,
        patients = mats.patients,
        smape_mat = mats.smape_mat,
        rmsle_mat = mats.rmsle_mat,
        loss_mat = mats.loss_mat,
    )
end

function run_validation_model_selection_minimal(csv_paths::Vector{String};
        output_dir::AbstractString = "res/model_selection/minimal",
        delta_tol::Real = 0.0)
    candidate_dfs = [read_candidate_metrics(path) for path in csv_paths]
    return _run_minimal_selection_workflow(candidate_dfs, csv_paths;
        output_dir = output_dir, delta_tol = delta_tol)
end

function run_validation_model_selection_minimal(candidate_specs::Vector{<:NamedTuple};
        output_dir::AbstractString = "res/model_selection/minimal",
        delta_tol::Real = 0.0)
    csv_paths = [String(spec.path) for spec in candidate_specs]
    meta_overrides = map(candidate_specs) do spec
        cfg = String(spec.config_code)
        mid = Int(spec.model_index)
        width_label = haskey(spec, :width_label) ? String(spec.width_label) :
            (startswith(cfg, "2") && length(cfg) >= 2 ? cfg[2:end] : cfg)
        candidate_id = haskey(spec, :candidate_id) ? String(spec.candidate_id) : "cfg$(cfg)_m$(mid)"
        (
            filename = basename(String(spec.path)),
            config_code = cfg,
            width_label = width_label,
            model_index = mid,
            candidate_id = candidate_id,
        )
    end

    candidate_dfs = [read_candidate_metrics(path; meta_override = meta)
                     for (path, meta) in zip(csv_paths, meta_overrides)]

    return _run_minimal_selection_workflow(candidate_dfs, csv_paths;
        output_dir = output_dir, delta_tol = delta_tol)
end
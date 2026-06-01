"""
plotting.jl

Reusable plotting helpers for refactored workflow scripts.

Sections:
- Patient Fit Plots: ODE/cUDE patient trajectory visualizations.
- Training Loss Plots: optimization loss-curve plots.
- Model Diagnostic Plots: residual, parameter, metric, and profile plots.
- Profile Likelihood Plots: step 03b patient and aggregate PLA plots.
- Systematic Truncation Plots: step 03c patient, parameter, and overlay plots.
- Symbolic Regression Plots: step 04a teacher and surrogate plots.
"""

using CairoMakie
using ComponentArrays: ComponentArray
using DataFrames: DataFrame, nrow
using Logging
using OrdinaryDiffEq: Tsit5
using Plots
using ProgressMeter
using SciMLBase: ODEProblem, successful_retcode, solve
using Statistics: mean, median, quantile, std

_style_value(style, key::Symbol, default) =
    style !== nothing && hasproperty(style, key) ? getproperty(style, key) : default

_plots_margin(style, key::Symbol, default_mm::Real) =
    _style_value(style, key, default_mm) * Plots.mm

# =============================================================================
# Patient Fit Plots
# =============================================================================

"""
    plot_patient_fit(sol, patient; title="Patient <id>", plasma_only=false)

Build a patient fit plot from an ODE solution and observed cTnT data.
"""
# Planned use: scripts/05_run_systematic_truncation.jl and diagnostic scripts.
function plot_patient_fit(sol, patient::PatientData; title::AbstractString="Patient $(patient.id)", plasma_only::Bool=false)
    if plasma_only
        plt = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title=title)
    else
        plt = Plots.plot(sol[1, :]; lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title=title)
        Plots.plot!(plt, sol[2, :]; lw=2, label="Cytosol")
        Plots.plot!(plt, sol[3, :]; lw=2, label="Blood")
    end
    Plots.scatter!(plt, patient.timepoints, patient.ctnt_data; ms=5, label="Observed Data", legend=:best)
    return plt
end

"""
    save_ode_patient_plots(sol, patient, dataset_name, fig_dir; plotting=true)

Save full-state and plasma-only patient SVG plots for step 01.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function save_ode_patient_plots(sol, patient::PatientData, dataset_name::AbstractString, fig_dir::AbstractString; plotting::Bool=true)
    pl = Plots.plot(sol; idxs=1, lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title="ODE - Patient $(patient.id)")
    Plots.plot!(pl, sol; idxs=2, lw=2, label="Cytosol")
    Plots.plot!(pl, sol; idxs=3, lw=2, label="Blood")
    Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    pl_plasm = Plots.plot(sol; idxs=3, lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
    Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    if plotting
        display(pl)
        display(pl_plasm)
    end

    savefig(pl, joinpath(fig_dir, "patient_$(patient.id)_$(dataset_name).svg"))
    savefig(pl_plasm, joinpath(fig_dir, "patient_$(patient.id)_$(dataset_name)_plasm.svg"))

    return (full=pl, plasma=pl_plasm)
end

"""
    save_cude_patient_plots(sol, patient, dataset_name, profiles_dir; plotting=true, display_plots=false)

Save full-state and plasma-only cUDE patient SVG plots for step 02b.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl, scripts/02d_evaluate_cude_nn_external_test.jl.
function save_cude_patient_plots(
    sol,
    patient::PatientData,
    dataset_name::AbstractString,
    profiles_dir::AbstractString;
    plotting::Bool=true,
    display_plots::Bool=false,
)
    plotting || return nothing
    mkpath(profiles_dir)

    pl = Plots.plot(sol[1, :]; lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title="cUDE NN Patient $(patient.id)")
    Plots.plot!(pl, sol[2, :]; lw=2, label="Cytosol")
    Plots.plot!(pl, sol[3, :]; lw=2, label="Blood")
    Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    pl_plasm = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
    Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    if display_plots
        display(pl)
        display(pl_plasm)
    end

    savefig(pl, joinpath(profiles_dir, "patient_$(patient.id)$(dataset_name).svg"))
    savefig(pl_plasm, joinpath(profiles_dir, "patient_$(patient.id)$(dataset_name)_plasm.svg"))

    return (full=pl, plasma=pl_plasm)
end

"""
    save_symbolic_formula_patient_plots(sol, patient, dataset_name, profiles_dir; plotting=true, display_plots=false)

Save full-state and plasma-only symbolic-surrogate patient SVG plots for step
04b.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function save_symbolic_formula_patient_plots(
    sol,
    patient::PatientData,
    dataset_name::AbstractString,
    profiles_dir::AbstractString;
    plotting::Bool=true,
    display_plots::Bool=false,
)
    plotting || return nothing
    mkpath(profiles_dir)

    pl = Plots.plot(sol[1, :]; lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title="Surrogate - Patient $(patient.id)")
    Plots.plot!(pl, sol[2, :]; lw=2, label="Cytosol")
    Plots.plot!(pl, sol[3, :]; lw=2, label="Blood")
    Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    pl_plasm = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
    Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    if display_plots
        display(pl)
        display(pl_plasm)
    end

    savefig(pl, joinpath(profiles_dir, "patient_$(patient.id)$(dataset_name).svg"))
    savefig(pl_plasm, joinpath(profiles_dir, "patient_$(patient.id)$(dataset_name)_plasm.svg"))

    return (full=pl, plasma=pl_plasm)
end

# =============================================================================
# Model Selection Plots
# =============================================================================

function _model_selection_width_color_map(width_vals)
    palette = [:steelblue, :darkorange, :forestgreen, :purple, :brown, :deeppink, :teal, :olive]
    widths = sort(unique(width_vals))
    return Dict(width => palette[mod1(i, length(palette))] for (i, width) in enumerate(widths))
end

function _add_model_selection_width_legend!(plot_obj, color_map::Dict, width_vals)
    for width in sort(unique(width_vals))
        Plots.scatter!(
            plot_obj,
            [NaN],
            [NaN];
            label="width=$(width)",
            color=color_map[width],
            markersize=7,
            markershape=:circle,
            markerstrokewidth=0.5,
        )
    end
    Plots.plot!(plot_obj, legend=:best)
    return plot_obj
end

function _model_selection_interval_sort_cols(metric::Symbol)
    if metric == :smape
        return [:smape_median, :smape_iqr, :rmsle_median, :loss_median]
    elseif metric == :rmsle
        return [:rmsle_median, :rmsle_iqr, :smape_median, :loss_median]
    elseif metric == :loss
        return [:loss_median, :loss_iqr, :smape_median, :rmsle_median]
    else
        error("Unsupported model-selection interval metric: $(metric)")
    end
end

function _model_selection_mean_std_sort_cols(metric::Symbol)
    if metric == :smape
        return [:smape_mean, :smape_std, :rmsle_mean, :loss_mean]
    elseif metric == :rmsle
        return [:rmsle_mean, :rmsle_std, :smape_mean, :loss_mean]
    elseif metric == :loss
        return [:loss_mean, :loss_std, :smape_mean, :rmsle_mean]
    else
        error("Unsupported model-selection mean/std metric: $(metric)")
    end
end

"""
    plot_model_selection_metric_interval_ranked(df, metric, selected_model_id; output_path)

Save a ranked median/IQR plot for one model-selection metric.
"""
# Used by: src/plotting.jl (save_model_selection_plots).
function plot_model_selection_metric_interval_ranked(
    df::DataFrame,
    metric::Symbol,
    selected_model_id::AbstractString;
    output_path::AbstractString,
)
    med_col = Symbol("$(metric)_median")
    q1_col = Symbol("$(metric)_q1")
    q3_col = Symbol("$(metric)_q3")

    sorted_df = sort(copy(df), _model_selection_interval_sort_cols(metric), rev=fill(false, 4))
    x = collect(1:nrow(sorted_df))
    labels = String.(sorted_df.model_id)
    y = Float64.(sorted_df[!, med_col])
    yerr_low = y .- Float64.(sorted_df[!, q1_col])
    yerr_high = Float64.(sorted_df[!, q3_col]) .- y

    color_map = _model_selection_width_color_map(sorted_df.nn_width)
    point_colors = [color_map[width] for width in sorted_df.nn_width]
    metric_label = uppercase(String(metric))

    plot_obj = Plots.scatter(
        x,
        y;
        yerror=(yerr_low, yerr_high),
        markercolor=point_colors,
        xlabel="Model ranked by $(metric_label) median",
        ylabel="$(metric_label) median with IQR interval",
        xticks=(x, labels),
        xrotation=45,
        title="$(metric_label) model ranking with robust uncertainty intervals",
        legend=false,
        markersize=6,
        markerstrokewidth=0.4,
        size=(1400, 760),
        left_margin=12 * Plots.mm,
        right_margin=8 * Plots.mm,
        bottom_margin=14 * Plots.mm,
        top_margin=6 * Plots.mm,
        label=false,
    )

    selected_idx = findfirst(==(selected_model_id), String.(sorted_df.model_id))
    if selected_idx !== nothing
        Plots.scatter!(plot_obj, [selected_idx], [y[selected_idx]]; markershape=:star5, markersize=12, color=:black, label="selected")
    end

    _add_model_selection_width_legend!(plot_obj, color_map, sorted_df.nn_width)
    savefig(plot_obj, output_path)
    return plot_obj
end

"""
    plot_model_selection_smape_median_vs_iqr(df, selected_model_id; output_path)

Save the model-selection sMAPE median/IQR scatter plot.
"""
# Used by: src/plotting.jl (save_model_selection_plots).
function plot_model_selection_smape_median_vs_iqr(
    df::DataFrame,
    selected_model_id::AbstractString;
    output_path::AbstractString,
)
    color_map = _model_selection_width_color_map(df.nn_width)
    plot_obj = Plots.plot(
        xlabel="Median sMAPE",
        ylabel="sMAPE IQR",
        title="Robust selection space: median vs IQR (lower is better)",
        legend=:best,
        size=(1150, 820),
        left_margin=12 * Plots.mm,
        right_margin=32 * Plots.mm,
        bottom_margin=10 * Plots.mm,
        top_margin=6 * Plots.mm,
    )

    for width in sort(unique(df.nn_width))
        idx = findall(==(width), df.nn_width)
        Plots.scatter!(
            plot_obj,
            df.smape_median[idx],
            df.smape_iqr[idx];
            label="width=$(width)",
            color=color_map[width],
            markersize=8,
            markerstrokewidth=0.5,
        )
    end

    selected = filter(:model_id => ==(selected_model_id), df)
    if nrow(selected) > 0
        Plots.scatter!(plot_obj, selected.smape_median, selected.smape_iqr; label="selected", markershape=:star5, markersize=14, color=:black)
        Plots.annotate!(plot_obj, selected.smape_median[1], selected.smape_iqr[1], Plots.text("  " * selected_model_id, 9, :left))
    end

    savefig(plot_obj, output_path)
    return plot_obj
end

"""
    plot_model_selection_metric_mean_std_ranked(df, metric, selected_model_id; output_path)

Save a ranked mean/std plot for one model-selection metric.
"""
# Used by: src/plotting.jl (save_model_selection_plots).
function plot_model_selection_metric_mean_std_ranked(
    df::DataFrame,
    metric::Symbol,
    selected_model_id::AbstractString;
    output_path::AbstractString,
)
    mean_col = Symbol("$(metric)_mean")
    std_col = Symbol("$(metric)_std")
    sorted_df = sort(copy(df), _model_selection_mean_std_sort_cols(metric), rev=fill(false, 4))

    x = collect(1:nrow(sorted_df))
    labels = String.(sorted_df.model_id)
    y = Float64.(sorted_df[!, mean_col])
    ystd = Float64.(sorted_df[!, std_col])
    color_map = _model_selection_width_color_map(sorted_df.nn_width)
    point_colors = [color_map[width] for width in sorted_df.nn_width]
    metric_label = uppercase(String(metric))

    plot_obj = Plots.scatter(
        x,
        y;
        yerror=(ystd, ystd),
        markercolor=point_colors,
        xlabel="Model ranked by $(metric) mean + std",
        ylabel="$(metric_label) mean +/- std",
        xticks=(x, labels),
        xrotation=45,
        title="$(metric_label) comparison (mean +/- std)",
        legend=false,
        markersize=6,
        markerstrokewidth=0.4,
        size=(1400, 760),
        left_margin=12 * Plots.mm,
        right_margin=8 * Plots.mm,
        bottom_margin=14 * Plots.mm,
        top_margin=6 * Plots.mm,
        label=false,
    )

    selected_idx = findfirst(==(selected_model_id), String.(sorted_df.model_id))
    if selected_idx !== nothing
        Plots.scatter!(plot_obj, [selected_idx], [y[selected_idx]]; markershape=:star5, markersize=12, color=:black, label="selected")
    end

    _add_model_selection_width_legend!(plot_obj, color_map, sorted_df.nn_width)
    savefig(plot_obj, output_path)
    return plot_obj
end

"""
    plot_model_selection_tiebreak_rmsle_vs_loss(df, selected_model_id; output_path)

Save the RMSLE-vs-loss tie-break scatter plot.
"""
# Used by: src/plotting.jl (save_model_selection_plots).
function plot_model_selection_tiebreak_rmsle_vs_loss(
    df::DataFrame,
    selected_model_id::AbstractString;
    output_path::AbstractString,
)
    color_map = _model_selection_width_color_map(df.nn_width)
    plot_obj = Plots.plot(
        xlabel="RMSLE median",
        ylabel="Loss median",
        title="Tie-break view: RMSLE vs Loss (lower is better)",
        legend=:best,
        size=(1100, 780),
        left_margin=12 * Plots.mm,
        right_margin=30 * Plots.mm,
        bottom_margin=10 * Plots.mm,
        top_margin=6 * Plots.mm,
    )

    for width in sort(unique(df.nn_width))
        idx = findall(==(width), df.nn_width)
        Plots.scatter!(plot_obj, df.rmsle_median[idx], df.loss_median[idx]; label="width=$(width)", color=color_map[width], markersize=8, markerstrokewidth=0.5)
    end

    selected = filter(:model_id => ==(selected_model_id), df)
    if nrow(selected) > 0
        Plots.scatter!(plot_obj, selected.rmsle_median, selected.loss_median; label="selected", markershape=:star5, markersize=14, color=:black)
    end

    savefig(plot_obj, output_path)
    return plot_obj
end

"""
    plot_model_selection_mean_vs_median_dumbbell(df; output_path)

Save the model-selection mean-vs-median dumbbell plot for all metrics.
"""
# Used by: src/plotting.jl (save_model_selection_plots).
function plot_model_selection_mean_vs_median_dumbbell(df::DataFrame; output_path::AbstractString)
    metrics = [:smape, :rmsle, :loss]
    color_map = _model_selection_width_color_map(df.nn_width)

    plot_obj = Plots.plot(
        layout=(3, 1),
        size=(1450, 1250),
        left_margin=12 * Plots.mm,
        right_margin=10 * Plots.mm,
        bottom_margin=10 * Plots.mm,
        top_margin=8 * Plots.mm,
    )

    for (panel_idx, metric) in enumerate(metrics)
        mean_col = Symbol("$(metric)_mean")
        med_col = Symbol("$(metric)_median")
        sorted_df = sort(copy(df), med_col)
        x = collect(1:nrow(sorted_df))
        labels = String.(sorted_df.model_id)
        y_mean = Float64.(sorted_df[!, mean_col])
        y_median = Float64.(sorted_df[!, med_col])
        colors = [color_map[width] for width in sorted_df.nn_width]

        for i in eachindex(x)
            Plots.plot!(plot_obj[panel_idx], [x[i], x[i]], [y_median[i], y_mean[i]]; color=:gray60, alpha=0.5, lw=1.3, label=false)
        end

        Plots.scatter!(plot_obj[panel_idx], x, y_median; markercolor=colors, markershape=:circle, markersize=5.5, markerstrokewidth=0.3, label=panel_idx == 1 ? "median" : false)
        Plots.scatter!(plot_obj[panel_idx], x, y_mean; markercolor=colors, markershape=:diamond, markersize=5.5, markerstrokewidth=0.3, label=panel_idx == 1 ? "mean" : false)
        Plots.plot!(
            plot_obj[panel_idx];
            xticks=(x, labels),
            xrotation=45,
            xlabel="Models ordered by $(uppercase(String(metric))) median",
            ylabel=uppercase(String(metric)),
            title="$(uppercase(String(metric))): mean vs median",
            legend=panel_idx == 1 ? :outertopright : false,
        )
    end

    for width in sort(unique(df.nn_width))
        Plots.scatter!(plot_obj[1], [NaN], [NaN]; color=color_map[width], markershape=:circle, markersize=7, markerstrokewidth=0.3, label="width=$(width)")
    end

    savefig(plot_obj, output_path)
    return plot_obj
end

"""
    save_model_selection_plots(paths, selection; plotting=true)

Save the canonical step 02c model-selection diagnostic plots.
"""
# Used by: scripts/02c_grid_search.jl.
function save_model_selection_plots(paths, selection; plotting::Bool=true)
    plotting || return nothing
    mkpath(paths.fig_dir)

    selected_model_id = String(selection.selected_model.model_id[1])
    df = selection.general_summary

    plot_model_selection_smape_median_vs_iqr(df, selected_model_id; output_path=paths.plots.smape_median_vs_iqr)
    plot_model_selection_tiebreak_rmsle_vs_loss(df, selected_model_id; output_path=paths.plots.tiebreak_rmsle_vs_loss)
    plot_model_selection_metric_mean_std_ranked(df, :smape, selected_model_id; output_path=paths.plots.smape_mean_std_ranked)
    plot_model_selection_metric_mean_std_ranked(df, :rmsle, selected_model_id; output_path=paths.plots.rmsle_mean_std_ranked)
    plot_model_selection_metric_mean_std_ranked(df, :loss, selected_model_id; output_path=paths.plots.loss_mean_std_ranked)
    plot_model_selection_metric_interval_ranked(df, :smape, selected_model_id; output_path=paths.plots.smape_interval_ranked)
    plot_model_selection_metric_interval_ranked(df, :rmsle, selected_model_id; output_path=paths.plots.rmsle_interval_ranked)
    plot_model_selection_metric_interval_ranked(df, :loss, selected_model_id; output_path=paths.plots.loss_interval_ranked)
    plot_model_selection_mean_vs_median_dumbbell(df; output_path=paths.plots.mean_vs_median_dumbbell)

    return paths.plots
end

# =============================================================================
# Parameter And Correction Plots
# =============================================================================

"""
    parameter_distribution_figure(params, par_names; title, show_outliers=true, show_progress=true)

Build a CairoMakie boxplot figure with natural-scale parameter distributions.
"""
# Used by: src/plotting.jl (params_extraction).
function parameter_distribution_figure(
    params,
    par_names;
    title::AbstractString,
    show_outliers::Bool=true,
    show_progress::Bool=true,
)
    fig = CairoMakie.Figure(size=(1400, 700))
    CairoMakie.Label(
        fig[0, 1:length(par_names)],
        title;
        fontsize=22,
        tellwidth=false,
    )

    axes = []
    axes_progress = show_progress ? Progress(length(par_names); desc="Generating parameter axes", showspeed=true) : nothing
    for p_name in par_names
        ax = CairoMakie.Axis(
            fig[1, length(axes) + 1],
            title=p_name,
            xticklabelsvisible=false,
            xticksvisible=false,
        )
        push!(axes, ax)
        axes_progress !== nothing && next!(axes_progress)
    end
    axes_progress !== nothing && finish!(axes_progress)

    colors = [:skyblue, :orange, :lightgreen, :pink, :violet]
    box_progress = show_progress ? Progress(length(params); desc="Generating parameter boxplots", showspeed=true) : nothing
    for (i, (ax, values)) in enumerate(zip(axes, params))
        CairoMakie.boxplot!(
            ax,
            fill(1, length(values)),
            values;
            color=colors[mod1(i, length(colors))],
            whiskerwidth=0.3,
            strokewidth=0.3,
            show_outliers=show_outliers,
        )
        box_progress !== nothing && next!(box_progress)
    end
    box_progress !== nothing && finish!(box_progress)

    return fig
end

"""
    params_extraction(patients, flat_log_params; ...)

Legacy-compatible parameter extraction helper returning natural-scale vectors
and an optional saved distribution figure.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl and legacy-compatible scripts through MechanisticAI.jl.
function params_extraction(
    patients::Vector{PatientData},
    ode_params_val::Vector{Float64};
    UDE::Bool=false,
    N_params::Int=5,
    data_label::String="",
    dataset::String="",
    figsave_path::String="",
    show_outliers::Bool=false,
    savefigure::Bool=false,
    show_progress::Bool=true,
)
    length(ode_params_val) == length(patients) * N_params ||
        error("Expected $(length(patients) * N_params) parameters for $(length(patients)) patients, got $(length(ode_params_val)).")

    param_store = extract_natural_parameters(ode_params_val; n_params=N_params)
    a, b, Cs0, Cc0, β = param_store.a, param_store.b, param_store.Cs0, param_store.Cc0, param_store.β

    @info "Average, STD in $(data_label) param a: $(mean(a)) std: $(std(a))"
    @info "Average, STD in $(data_label) param b: $(mean(b)) std: $(std(b))"
    @info "Average, STD in $(data_label) param Cs0: $(mean(Cs0)) std: $(std(Cs0))"
    @info "Average, STD in $(data_label) param Cc0: $(mean(Cc0)) std: $(std(Cc0))"
    if N_params == 5
        @info "Average, STD in $(data_label) param β: $(mean(β)) std: $(std(β))"
    end

    @info "Median [Q1-Q3] in $(data_label) param a: $(median(a)) [$(quantile(a, 0.25)) - $(quantile(a, 0.75))]"
    @info "Median [Q1-Q3] in $(data_label) param b: $(median(b)) [$(quantile(b, 0.25)) - $(quantile(b, 0.75))]"
    @info "Median [Q1-Q3] in $(data_label) param Cs0: $(median(Cs0)) [$(quantile(Cs0, 0.25)) - $(quantile(Cs0, 0.75))]"
    @info "Median [Q1-Q3] in $(data_label) param Cc0: $(median(Cc0)) [$(quantile(Cc0, 0.25)) - $(quantile(Cc0, 0.75))]"
    if N_params == 5
        @info "Median [Q1-Q3] in $(data_label) param β: $(median(β)) [$(quantile(β, 0.25)) - $(quantile(β, 0.75))]"
    end

    params = parameter_vectors(param_store, UDE)
    par_names = parameter_names(UDE)
    fig = parameter_distribution_figure(
        params,
        par_names;
        title="Parameter distributions $(data_label) — $(dataset) dataset",
        show_outliers=show_outliers,
        show_progress=show_progress,
    )

    if savefigure
        mkpath(figsave_path)
        output_path = joinpath(figsave_path, "$(data_label)_params_distribution_$(dataset).svg")
        CairoMakie.save(output_path, fig)
        @info "Figure saved at: $(output_path)"
    end

    return a, b, Cs0, Cc0, β, fig
end

"""
    save_cude_correction_function_plot(path, chain, nn_params; t_scale, plotting=true, display_plot=false)

Save the learned cUDE correction function over the legacy time/beta grid.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl.
function save_cude_correction_function_plot(
    path::AbstractString,
    chain,
    nn_params;
    t_scale::Real,
    t_grid=0.1:0.1:2500,
    beta_values=0.1:0.1:1.0,
    plotting::Bool=true,
    display_plot::Bool=false,
)
    plotting || return nothing
    mkpath(dirname(path))

    plt = Plots.plot()
    for β in beta_values
        y = [chain([t / t_scale, β], nn_params)[1] for t in t_grid]
        Plots.plot!(plt, t_grid, y; label="β = $(β)", linewidth=2)
    end
    Plots.plot!(
        plt;
        xlabel="Time (h)",
        ylabel="rupture f(t_norm,β)",
        title="Learned sarcomere rupture function",
    )

    display_plot && display(plt)
    savefig(plt, path)
    return plt
end

"""
    save_symbolic_formula_correction_plots(paths; t_grid, beta_values, plotting=true, display_plots=false)

Save the official symbolic-surrogate correction curves used by step 04b.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function save_symbolic_formula_correction_plots(
    paths;
    t_grid=0.1:0.1:2400.0,
    beta_values=0.1:0.1:1.0,
    plotting::Bool=true,
    display_plots::Bool=false,
)
    plotting || return nothing
    mkpath(paths.dataset_dir)

    t_values = Float64.(collect(t_grid))
    beta_grid = Float64.(collect(beta_values))
    t_eff_values = [symbolic_surrogate_effective_time(beta) for beta in beta_grid]

    with_title = Plots.plot()
    for (beta, t_eff) in zip(beta_grid, t_eff_values)
        y = [symbolic_surrogate_correction(t / T_SCALE, beta) for t in t_values]
        Plots.plot!(with_title, t_values, y; label="T_eff = $(round(t_eff, digits=2))", linewidth=2)
    end
    Plots.plot!(
        with_title;
        xlabel="Time (h)",
        ylabel="SR(τ,T_eff)",
        title="Learned sarcomere rupture function",
        legend=:bottomright,
    )

    no_title = Plots.plot()
    for (beta, t_eff) in zip(beta_grid, t_eff_values)
        y = [symbolic_surrogate_correction(t / T_SCALE, beta) for t in t_values]
        Plots.plot!(no_title, t_values, y; label="T_eff = $(round(t_eff, digits=2))", linewidth=2)
    end
    Plots.plot!(no_title; xlabel="Time (h)", legend=false)

    display_plots && display(with_title)
    display_plots && display(no_title)

    savefig(with_title, paths.correction_surrogate_with_title)
    savefig(no_title, paths.correction_surrogate)

    return (with_title=with_title, no_title=no_title)
end

"""
    save_symbolic_formula_parameter_boxplot(path, flat_log_params; n_params, dataset_label, plotting=true)

Save the natural-scale parameter distribution boxplot for step 04b.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function save_symbolic_formula_parameter_boxplot(
    path::AbstractString,
    flat_log_params::AbstractVector;
    n_params::Integer=5,
    dataset_label::AbstractString,
    plotting::Bool=true,
    display_plots::Bool=false,
)
    plotting || return nothing
    mkpath(dirname(path))

    param_store = extract_natural_parameters(flat_log_params; n_params=n_params)
    params = parameter_vectors(param_store, n_params == 4)
    par_names = parameter_names(n_params == 4)
    fig = parameter_distribution_figure(
        params,
        par_names;
        title="Parameter distributions — $(dataset_label)",
        show_outliers=true,
        show_progress=false,
    )

    CairoMakie.save(path, fig)
    display_plots && display(fig)
    return fig
end

"""
    save_symbolic_formula_residual_plots(paths, residuals; dataset_label, edges, tmax, plotting=true)

Save step 04b residual-vs-time and residual-vs-fitted plots.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function save_symbolic_formula_residual_plots(
    paths,
    residuals::DataFrame;
    dataset_label::AbstractString,
    edges=EDGES,
    tmax::Real=T_SCALE,
    plotting::Bool=true,
    display_plots::Bool=false,
)
    plotting || return nothing
    mkpath(paths.residuals_dir)

    fig_time = CairoMakie.Figure(size=(550, 450))
    ax_time = CairoMakie.Axis(fig_time[1, 1])
    plot_residuals_vs_time_panel!(
        ax_time,
        residuals,
        edges;
        title="Residuals vs time - $(dataset_label)",
        tmax=tmax,
        nmin=1,
    )

    fig_fitted = CairoMakie.Figure(size=(550, 450))
    ax_fitted = CairoMakie.Axis(fig_fitted[1, 1])
    plot_residuals_vs_fitted_panel!(ax_fitted, residuals; title="Residuals vs fitted - $(dataset_label)")

    CairoMakie.save(paths.residuals_vs_time, fig_time)
    CairoMakie.save(paths.residuals_vs_fitted, fig_fitted)

    display_plots && display(fig_time)
    display_plots && display(fig_fitted)

    return (time=fig_time, fitted=fig_fitted)
end

# =============================================================================
# Model Diagnostic Plots
# =============================================================================

"""
    plot_residuals_vs_fitted_panel!(ax, df; title="")

Draw one residuals-vs-fitted diagnostic panel.
"""
# Used by: src/plotting.jl (save_residual_diagnostic_plots).
function plot_residuals_vs_fitted_panel!(ax, df::DataFrame; title::AbstractString="", style=nothing)
    eps = 1e-10
    markersize = _style_value(style, :scatter_markersize, 5)
    CairoMakie.scatter!(ax, log.(df.yhat .+ eps), df.res; markersize=markersize, color=(:black, 0.25), label="Residuals")
    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6))
    ax.title = title
    ax.xlabel = "log predicted yhat"
    ax.ylabel = "log residual"
    return ax
end

"""
    plot_residuals_vs_time_panel!(ax, df, edges; title, tmax, nmin=1)

Draw one residuals-vs-time diagnostic panel with median/IQR bin summaries.
"""
# Used by: src/plotting.jl (save_residual_diagnostic_plots).
function plot_residuals_vs_time_panel!(
    ax,
    df::DataFrame,
    edges;
    title::AbstractString="",
    tmax::Real=T_SCALE,
    nmin::Integer=1,
    style=nothing,
)
    plot_df = copy(df)
    add_time_bins!(plot_df, edges)
    summary = bin_summary(plot_df)

    med = [summary.n[i] >= nmin ? summary.med[i] : NaN for i in eachindex(summary.n)]
    q1 = [summary.n[i] >= nmin ? summary.q1[i] : NaN for i in eachindex(summary.n)]
    q3 = [summary.n[i] >= nmin ? summary.q3[i] : NaN for i in eachindex(summary.n)]

    markersize = _style_value(style, :scatter_markersize, 4)
    linewidth = _style_value(style, :summary_linewidth, 2)
    bin_label_fontsize = _style_value(style, :bin_label_fontsize, 10)

    CairoMakie.scatter!(ax, plot_df.t, plot_df.res; markersize=markersize, color=(:black, 0.2))
    CairoMakie.lines!(ax, summary.centers, med; linewidth=linewidth, color=:blue, label="Median")
    CairoMakie.band!(ax, summary.centers, q1, q3; color=(:gray, 0.2), label="IQR")
    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6))
    CairoMakie.vlines!(ax, edges[2:end-1]; color=(:black, 0.3), linewidth=1, linestyle=:dash)
    CairoMakie.xlims!(ax, 0, tmax)

    for i in eachindex(summary.centers)
        x_rel = clamp(summary.centers[i] / tmax, 0.0, 1.0)
        CairoMakie.text!(
            ax,
            x_rel,
            0.96;
            text="n=$(summary.n[i])",
            space=:relative,
            align=(:center, :top),
            rotation=pi / 4,
            fontsize=bin_label_fontsize,
            color=(:black, 0.7),
        )
    end

    ax.title = title
    ax.xlabel = "Time (h)"
    ax.ylabel = "log residual"
    return ax
end

"""
    save_residual_diagnostic_plots(paths; residual sets, edges, tmax, plotting=true)

Save the canonical residual diagnostic plots for step 03a.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function save_residual_diagnostic_plots(
    paths;
    residuals_ode_mimic,
    residuals_ode_umg,
    residuals_cude_mimic,
    residuals_cude_umg,
    edges,
    tmax::Real,
    plotting::Bool=true,
    style=nothing,
)
    plotting || return nothing
    mkpath(paths.residuals_fig_dir)

    figure_fontsize = _style_value(style, :figure_fontsize, 14)

    fig_fitted = CairoMakie.Figure(size=(1200, 900), fontsize=figure_fontsize)
    fitted_axes = [Axis(fig_fitted[1, 1]), Axis(fig_fitted[1, 2]), Axis(fig_fitted[2, 1]), Axis(fig_fitted[2, 2])]
    plot_residuals_vs_fitted_panel!(fitted_axes[1], residuals_ode_mimic; title="ODE - MIMIC-IV", style=style)
    plot_residuals_vs_fitted_panel!(fitted_axes[2], residuals_cude_mimic; title="cUDE - MIMIC-IV", style=style)
    plot_residuals_vs_fitted_panel!(fitted_axes[3], residuals_ode_umg; title="ODE - UMG", style=style)
    plot_residuals_vs_fitted_panel!(fitted_axes[4], residuals_cude_umg; title="cUDE - UMG", style=style)

    fig_fitted_mimic = CairoMakie.Figure(size=(1200, 500), fontsize=figure_fontsize)
    fitted_mimic_axes = [Axis(fig_fitted_mimic[1, 1]), Axis(fig_fitted_mimic[1, 2])]
    plot_residuals_vs_fitted_panel!(fitted_mimic_axes[1], residuals_ode_mimic; title="ODE - MIMIC-IV", style=style)
    plot_residuals_vs_fitted_panel!(fitted_mimic_axes[2], residuals_cude_mimic; title="cUDE - MIMIC-IV", style=style)

    fig_fitted_umg = CairoMakie.Figure(size=(1200, 500), fontsize=figure_fontsize)
    fitted_umg_axes = [Axis(fig_fitted_umg[1, 1]), Axis(fig_fitted_umg[1, 2])]
    plot_residuals_vs_fitted_panel!(fitted_umg_axes[1], residuals_ode_umg; title="ODE - UMG", style=style)
    plot_residuals_vs_fitted_panel!(fitted_umg_axes[2], residuals_cude_umg; title="cUDE - UMG", style=style)

    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_fitted.svg"), fig_fitted)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_fitted.png"), fig_fitted, px_per_unit=3)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_fitted_mimic.svg"), fig_fitted_mimic)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_fitted_mimic.png"), fig_fitted_mimic, px_per_unit=3)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_fitted_umg.svg"), fig_fitted_umg)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_fitted_umg.png"), fig_fitted_umg, px_per_unit=3)

    fig_time = CairoMakie.Figure(size=(1400, 900), fontsize=figure_fontsize)
    time_axes = [Axis(fig_time[1, 1]), Axis(fig_time[1, 2]), Axis(fig_time[2, 1]), Axis(fig_time[2, 2])]
    plot_residuals_vs_time_panel!(time_axes[1], residuals_ode_mimic, edges; title="ODE - MIMIC-IV", tmax=tmax, style=style)
    plot_residuals_vs_time_panel!(time_axes[2], residuals_cude_mimic, edges; title="cUDE - MIMIC-IV", tmax=tmax, style=style)
    plot_residuals_vs_time_panel!(time_axes[3], residuals_ode_umg, edges; title="ODE - UMG", tmax=tmax, style=style)
    plot_residuals_vs_time_panel!(time_axes[4], residuals_cude_umg, edges; title="cUDE - UMG", tmax=tmax, style=style)

    fig_time_mimic = CairoMakie.Figure(size=(1400, 500), fontsize=figure_fontsize)
    time_mimic_axes = [Axis(fig_time_mimic[1, 1]), Axis(fig_time_mimic[1, 2])]
    plot_residuals_vs_time_panel!(time_mimic_axes[1], residuals_ode_mimic, edges; title="ODE - MIMIC-IV", tmax=tmax, style=style)
    plot_residuals_vs_time_panel!(time_mimic_axes[2], residuals_cude_mimic, edges; title="cUDE - MIMIC-IV", tmax=tmax, style=style)

    fig_time_umg = CairoMakie.Figure(size=(1400, 500), fontsize=figure_fontsize)
    time_umg_axes = [Axis(fig_time_umg[1, 1]), Axis(fig_time_umg[1, 2])]
    plot_residuals_vs_time_panel!(time_umg_axes[1], residuals_ode_umg, edges; title="ODE - UMG", tmax=tmax, style=style)
    plot_residuals_vs_time_panel!(time_umg_axes[2], residuals_cude_umg, edges; title="cUDE - UMG", tmax=tmax, style=style)

    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_time.svg"), fig_time)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_time.png"), fig_time, px_per_unit=3)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_time_mimic.svg"), fig_time_mimic)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_time_mimic.png"), fig_time_mimic, px_per_unit=3)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_time_umg.svg"), fig_time_umg)
    CairoMakie.save(joinpath(paths.residuals_fig_dir, "residuals_vs_time_umg.png"), fig_time_umg, px_per_unit=3)

    return paths.residuals_fig_dir
end

function _diagnostic_param_boxplot_per_model(par_df_ds1, par_df_ds2, par_names)
    n_params = length(par_names)
    fig = CairoMakie.Figure(size=(300 * n_params, 550), fontsize=14)
    colors_dataset = Dict("MIMIC-IV" => :steelblue, "UMG" => :darkorange)

    for (i, pname) in enumerate(par_names)
        col = i <= 4 ? [:a, :b, :Cs0, :Cc0][i] : :p5
        vals = vcat(par_df_ds1[!, col], par_df_ds2[!, col])
        groups = vcat(fill(1, nrow(par_df_ds1)), fill(2, nrow(par_df_ds2)))
        colors = [g == 1 ? colors_dataset["MIMIC-IV"] : colors_dataset["UMG"] for g in groups]

        ax = Axis(fig[1, i], title=pname)
        CairoMakie.boxplot!(ax, groups, vals; color=colors, whiskerwidth=0.4, strokewidth=0.5)
        ax.xticks = (1:2, ["MIMIC-IV", "UMG"])
        ax.xticklabelrotation = pi / 5
    end

    return fig
end

function _diagnostic_param_boxplot_cross_model(par_ode, par_cude; complete_plot::Bool=true)
    colors_model = Dict("ODE" => :royalblue, "cUDE" => :darkorange)
    shared_params = ["a", "b", "Cs0", "Cc0"]
    figs = Any[]

    if complete_plot
        fig = CairoMakie.Figure(size=(1800, 550), fontsize=14)
        for (i, pname) in enumerate(shared_params)
            col = [:a, :b, :Cs0, :Cc0][i]
            vals = vcat(par_ode[!, col], par_cude[!, col])
            groups = vcat(fill(1, nrow(par_ode)), fill(2, nrow(par_cude)))
            colors = [g == 1 ? colors_model["ODE"] : colors_model["cUDE"] for g in groups]
            ax = Axis(fig[1, i], title=pname)
            CairoMakie.boxplot!(ax, groups, vals; color=colors, whiskerwidth=0.4, strokewidth=0.5)
            ax.xticks = (1:2, ["ODE", "cUDE"])
            ax.xticklabelrotation = pi / 5
        end

        ax_td = Axis(fig[1, 5], title="Td (ODE)")
        CairoMakie.boxplot!(ax_td, fill(1, nrow(par_ode)), par_ode.p5; color=colors_model["ODE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_td.xticks = ([1], ["ODE"])
        ax_td.xticklabelrotation = pi / 5

        ax_beta = Axis(fig[1, 6], title="beta (cUDE)")
        CairoMakie.boxplot!(ax_beta, fill(1, nrow(par_cude)), par_cude.p5; color=colors_model["cUDE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_beta.xticks = ([1], ["cUDE"])
        ax_beta.xticklabelrotation = pi / 5
        push!(figs, fig)
    else
        fig_shared = CairoMakie.Figure(size=(1200, 550), fontsize=14)
        for (i, pname) in enumerate(shared_params)
            col = [:a, :b, :Cs0, :Cc0][i]
            vals = vcat(par_ode[!, col], par_cude[!, col])
            groups = vcat(fill(1, nrow(par_ode)), fill(2, nrow(par_cude)))
            colors = [g == 1 ? colors_model["ODE"] : colors_model["cUDE"] for g in groups]
            ax = Axis(fig_shared[1, i], title=pname)
            CairoMakie.boxplot!(ax, groups, vals; color=colors, whiskerwidth=0.4, strokewidth=0.5)
            ax.xticks = (1:2, ["ODE", "cUDE"])
            ax.xticklabelrotation = pi / 5
        end
        push!(figs, fig_shared)

        fig_pair = CairoMakie.Figure(size=(600, 550), fontsize=14)
        ax_td_pair = Axis(fig_pair[1, 1], title="Td (ODE)")
        CairoMakie.boxplot!(ax_td_pair, fill(1, nrow(par_ode)), par_ode.p5; color=colors_model["ODE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_td_pair.xticks = ([1], ["ODE"])
        ax_td_pair.xticklabelrotation = pi / 5
        ax_beta_pair = Axis(fig_pair[1, 2], title="beta (cUDE)")
        CairoMakie.boxplot!(ax_beta_pair, fill(1, nrow(par_cude)), par_cude.p5; color=colors_model["cUDE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_beta_pair.xticks = ([1], ["cUDE"])
        ax_beta_pair.xticklabelrotation = pi / 5
        push!(figs, fig_pair)

        fig_td = CairoMakie.Figure(size=(300, 550), fontsize=14)
        ax_td = Axis(fig_td[1, 1], title="Td (ODE)")
        CairoMakie.boxplot!(ax_td, fill(1, nrow(par_ode)), par_ode.p5; color=colors_model["ODE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_td.xticks = ([1], ["ODE"])
        ax_td.xticklabelrotation = pi / 5
        push!(figs, fig_td)

        fig_beta = CairoMakie.Figure(size=(300, 550), fontsize=14)
        ax_beta = Axis(fig_beta[1, 1], title="beta (cUDE)")
        CairoMakie.boxplot!(ax_beta, fill(1, nrow(par_cude)), par_cude.p5; color=colors_model["cUDE"], whiskerwidth=0.4, strokewidth=0.5)
        ax_beta.xticks = ([1], ["cUDE"])
        ax_beta.xticklabelrotation = pi / 5
        push!(figs, fig_beta)
    end

    return figs
end

"""
    save_parameter_diagnostic_boxplots(paths; parameter sets, plotting=true)

Save canonical parameter boxplots for step 03a.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function save_parameter_diagnostic_boxplots(
    paths;
    par_ode_mimic,
    par_ode_umg,
    par_cude_mimic,
    par_cude_umg,
    plotting::Bool=true,
)
    plotting || return nothing
    mkpath(paths.boxplots_fig_dir)

    fig_cude = _diagnostic_param_boxplot_per_model(par_cude_mimic, par_cude_umg, ["a", "b", "Cs0", "Cc0", "beta"])
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cUDE_by_dataset.svg"), fig_cude)
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cUDE_by_dataset.png"), fig_cude, px_per_unit=3)

    fig_ode = _diagnostic_param_boxplot_per_model(par_ode_mimic, par_ode_umg, ["a", "b", "Cs0", "Cc0", "Td"])
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_ODE_by_dataset.svg"), fig_ode)
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_ODE_by_dataset.png"), fig_ode, px_per_unit=3)

    mimic_full = _diagnostic_param_boxplot_cross_model(par_ode_mimic, par_cude_mimic)
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cross_model_MIMIC.svg"), mimic_full[1])
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cross_model_MIMIC.png"), mimic_full[1], px_per_unit=3)

    umg_full = _diagnostic_param_boxplot_cross_model(par_ode_umg, par_cude_umg)
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cross_model_UMG.svg"), umg_full[1])
    CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cross_model_UMG.png"), umg_full[1], px_per_unit=3)

    for (dataset_label, par_ode, par_cude) in [("MIMIC", par_ode_mimic, par_cude_mimic), ("UMG", par_ode_umg, par_cude_umg)]
        separated = _diagnostic_param_boxplot_cross_model(par_ode, par_cude; complete_plot=false)
        for (i, fig) in enumerate(separated)
            CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cross_model_$(dataset_label)_separated_$(i).svg"), fig)
            CairoMakie.save(joinpath(paths.boxplots_fig_dir, "boxplot_params_cross_model_$(dataset_label)_separated_$(i).png"), fig, px_per_unit=3)
        end
    end

    return paths.boxplots_fig_dir
end

function _plot_subjectwise_metric!(ax, x, y, title, xlabel, ylabel)
    max_val = max(maximum(x), maximum(y))
    min_val = min(minimum(x), minimum(y))
    total = length(x)
    better_cude = sum(y .< x)
    better_ode = sum(y .> x)
    perc_cude = round(100 * better_cude / total, digits=1)
    perc_ode = round(100 * better_ode / total, digits=1)
    pad = (max_val - min_val) * 0.05 + 1e-4
    lim_min = max(0.0, min_val - pad)
    lim_max = max_val + pad

    CairoMakie.band!(ax, [lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max], color=(:royalblue, 0.08))
    CairoMakie.band!(ax, [lim_min, lim_max], [lim_min, lim_min], [lim_min, lim_max], color=(:darkorange, 0.08))
    CairoMakie.scatter!(ax, x, y, color=:black, markersize=8, alpha=0.6)
    CairoMakie.ablines!(ax, 0, 1, color=:red, linestyle=:dash, linewidth=2)

    ax.title = title
    ax.xlabel = xlabel
    ax.ylabel = ylabel
    CairoMakie.xlims!(ax, lim_min, lim_max)
    CairoMakie.ylims!(ax, lim_min, lim_max)

    CairoMakie.text!(
        ax,
        lim_min + (lim_max - lim_min) * 0.05,
        lim_max - (lim_max - lim_min) * 0.05;
        text="ODE better:\n$(perc_ode)%",
        align=(:left, :top),
        color=:royalblue,
        fontsize=16,
        font=:bold,
    )
    CairoMakie.text!(
        ax,
        lim_max - (lim_max - lim_min) * 0.05,
        lim_min + (lim_max - lim_min) * 0.05;
        text="cUDE better:\n$(perc_cude)%",
        align=(:right, :bottom),
        color=:darkorange,
        fontsize=16,
        font=:bold,
    )

    return ax
end

function _append_metric_distribution!(datasets, models, smapes, rmsles, df, dataset_id, model_id)
    n = nrow(df)
    append!(datasets, fill(dataset_id, n))
    append!(models, fill(model_id, n))
    append!(smapes, df.smape_val)
    append!(rmsles, df.rmsle_val)
    return nothing
end

"""
    save_metric_comparison_paper_plots(paths; comparison and metric tables, plotting=true)

Save subject-wise scatter, violin, and mean/std metric comparison plots.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function save_metric_comparison_paper_plots(
    paths;
    comparison_mimic,
    comparison_umg,
    metrics_ode_mimic,
    metrics_ode_umg,
    metrics_cude_mimic,
    metrics_cude_umg,
    plotting::Bool=true,
)
    plotting || return nothing
    mkpath(paths.metrics_comparison_fig_dir)

    fig_scatter = CairoMakie.Figure(size=(1000, 1000), fontsize=18)
    ax1 = Axis(fig_scatter[1, 1], aspect=1)
    ax2 = Axis(fig_scatter[1, 2], aspect=1)
    ax3 = Axis(fig_scatter[2, 1], aspect=1)
    ax4 = Axis(fig_scatter[2, 2], aspect=1)
    _plot_subjectwise_metric!(ax1, comparison_mimic.smape_ode, comparison_mimic.smape_cude, "MIMIC-IV: sMAPE", "ODE sMAPE (%)", "cUDE sMAPE (%)")
    _plot_subjectwise_metric!(ax2, comparison_mimic.rmsle_ode, comparison_mimic.rmsle_cude, "MIMIC-IV: RMSLE", "ODE RMSLE", "cUDE RMSLE")
    _plot_subjectwise_metric!(ax3, comparison_umg.smape_ode, comparison_umg.smape_cude, "UMG: sMAPE", "ODE sMAPE (%)", "cUDE sMAPE (%)")
    _plot_subjectwise_metric!(ax4, comparison_umg.rmsle_ode, comparison_umg.rmsle_cude, "UMG: RMSLE", "ODE RMSLE", "cUDE RMSLE")
    CairoMakie.save(joinpath(paths.metrics_comparison_fig_dir, "scatter_subjectwise_cUDE_vs_ODE.svg"), fig_scatter)
    CairoMakie.save(joinpath(paths.metrics_comparison_fig_dir, "scatter_subjectwise_cUDE_vs_ODE.png"), fig_scatter, px_per_unit=3)

    datasets = Int[]
    models = Int[]
    smapes = Float64[]
    rmsles = Float64[]
    _append_metric_distribution!(datasets, models, smapes, rmsles, metrics_cude_mimic, 1, 1)
    _append_metric_distribution!(datasets, models, smapes, rmsles, metrics_ode_mimic, 1, 2)
    _append_metric_distribution!(datasets, models, smapes, rmsles, metrics_cude_umg, 2, 1)
    _append_metric_distribution!(datasets, models, smapes, rmsles, metrics_ode_umg, 2, 2)

    fig_violin = CairoMakie.Figure(size=(1000, 500), fontsize=18)
    ax_smape = Axis(fig_violin[1, 1], title="sMAPE Distribution", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="sMAPE (%)")
    ax_rmsle = Axis(fig_violin[1, 2], title="RMSLE Distribution", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="RMSLE")
    colors = [model_id == 1 ? :darkorange : :royalblue for model_id in models]
    CairoMakie.violin!(ax_smape, datasets, smapes, dodge=models, color=colors, show_median=true, mediancolor=:black)
    CairoMakie.violin!(ax_rmsle, datasets, rmsles, dodge=models, color=colors, show_median=true, mediancolor=:black)
    elem_cude = PolyElement(color=:darkorange, strokecolor=:transparent)
    elem_ode = PolyElement(color=:royalblue, strokecolor=:transparent)
    Legend(fig_violin[1, 3], [elem_cude, elem_ode], ["cUDE", "ODE"], "Models", framevisible=false)
    CairoMakie.save(joinpath(paths.metrics_comparison_fig_dir, "violin_metrics_cUDE_vs_ODE.svg"), fig_violin)

    groups = [metrics_cude_mimic, metrics_ode_mimic, metrics_cude_umg, metrics_ode_umg]
    group_datasets = [1, 1, 2, 2]
    group_models = [1, 2, 1, 2]
    smape_means = [mean(df.smape_val) for df in groups]
    smape_stds = [std(df.smape_val) for df in groups]
    rmsle_means = [mean(df.rmsle_val) for df in groups]
    rmsle_stds = [std(df.rmsle_val) for df in groups]

    fig_bar = CairoMakie.Figure(size=(1000, 500), fontsize=18)
    ax_bar_smape = Axis(fig_bar[1, 1], title="sMAPE (Mean +/- STD)", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="sMAPE (%)")
    ax_bar_rmsle = Axis(fig_bar[1, 2], title="RMSLE (Mean +/- STD)", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="RMSLE")
    group_colors = [model_id == 1 ? :darkorange : :royalblue for model_id in group_models]
    x_dodged = [Float64(dataset_id) + (model_id == 1 ? -0.2 : 0.2) for (dataset_id, model_id) in zip(group_datasets, group_models)]
    CairoMakie.barplot!(ax_bar_smape, x_dodged, smape_means, color=group_colors, width=0.35)
    CairoMakie.errorbars!(ax_bar_smape, x_dodged, smape_means, smape_stds, color=:black, whiskerwidth=10)
    CairoMakie.barplot!(ax_bar_rmsle, x_dodged, rmsle_means, color=group_colors, width=0.35)
    CairoMakie.errorbars!(ax_bar_rmsle, x_dodged, rmsle_means, rmsle_stds, color=:black, whiskerwidth=10)
    Legend(fig_bar[1, 3], [elem_cude, elem_ode], ["cUDE", "ODE"], "Models", framevisible=false)
    CairoMakie.save(joinpath(paths.metrics_comparison_fig_dir, "barplot_mean_std_metrics_cUDE_vs_ODE.svg"), fig_bar)
    CairoMakie.save(joinpath(paths.metrics_comparison_fig_dir, "barplot_mean_std_metrics_cUDE_vs_ODE.png"), fig_bar, px_per_unit=3)

    return paths.metrics_comparison_fig_dir
end

function _diagnostic_profile_base(patient::PatientData)
    plt = Plots.plot(
        size=(800, 500),
        xlabel="Time (h)",
        ylabel="cTnT [ng/mL]",
        legend=false,
        margin=5 * Plots.mm,
        grid=true,
    )
    Plots.scatter!(
        plt,
        patient.timepoints,
        patient.ctnt_data;
        markershape=:circle,
        color=:red,
        ms=6,
        markerstrokewidth=1.5,
        label="Observation",
    )
    return plt
end

function _diagnostic_solve_ode(params_log::Vector{Float64}, tmax::Real)
    u0 = [exp(params_log[3]), exp(params_log[4]), 0.0]
    problem = ODEProblem(troponin_ode!, u0, (0.0, tmax + 10.0), params_log)
    sol = solve(problem, Tsit5(); saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || @warn "ODE diagnostic profile solve failed."
    return sol.t, sol[3, :]
end

function _diagnostic_solve_cude(params_natural::Vector{Float64}, nn_params, chain, tmax::Real)
    params_log = log.(params_natural .+ 1e-15)
    u0 = [params_natural[3], params_natural[4], 0.0]
    cude_rhs!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
    problem = ODEProblem(cude_rhs!, u0, (0.0, tmax + 10.0))
    full_params = ComponentArray(ode=params_log, neural=nn_params)
    sol = solve(problem, Tsit5(); p=full_params, saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || @warn "cUDE diagnostic profile solve failed."
    return sol.t, sol[3, :]
end

"""
    save_ode_quartile_profile_plots(selections, patient_lookup, out_dir; plotting=true)

Save selected ODE profile plots by sMAPE quartile.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function save_ode_quartile_profile_plots(selections, patient_lookup, out_dir::AbstractString; plotting::Bool=true)
    plotting || return nothing
    mkpath(out_dir)

    for quartile in sort(collect(keys(selections)))
        for row in eachrow(selections[quartile])
            patient_id = String(row.patient)
            haskey(patient_lookup, patient_id) || continue
            patient = patient_lookup[patient_id]
            tmax = maximum(patient.timepoints) + 10.0
            params_log = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]
            t_ode, y_ode = _diagnostic_solve_ode(params_log, tmax)
            plt = _diagnostic_profile_base(patient)
            Plots.plot!(plt, t_ode, y_ode; lw=2, color=:blue)
            Plots.savefig(plt, joinpath(out_dir, "ODE_Q$(quartile)_$(patient_id).svg"))
        end
    end

    return out_dir
end

"""
    save_cude_quartile_profile_plots(selections, patient_lookup, nn_params, chain, out_dir; plotting=true)

Save selected cUDE profile plots by sMAPE quartile.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function save_cude_quartile_profile_plots(selections, patient_lookup, nn_params, chain, out_dir::AbstractString; plotting::Bool=true)
    plotting || return nothing
    mkpath(out_dir)

    for quartile in sort(collect(keys(selections)))
        for row in eachrow(selections[quartile])
            patient_id = String(row.patient_id)
            haskey(patient_lookup, patient_id) || continue
            patient = patient_lookup[patient_id]
            tmax = maximum(patient.timepoints) + 10.0
            params_natural = Float64[row.a, row.b, row.Cs0, row.Cc0, row.beta]
            t_cude, y_cude = _diagnostic_solve_cude(params_natural, nn_params, chain, tmax)
            plt = _diagnostic_profile_base(patient)
            Plots.plot!(plt, t_cude, y_cude; lw=2, color=:darkorange, linestyle=:dash)
            Plots.savefig(plt, joinpath(out_dir, "cUDE_Q$(quartile)_$(patient_id).svg"))
        end
    end

    return out_dir
end

"""
    save_overlap_profile_plots(groups, patient_lookup, nn_params, chain, out_dir; plotting=true)

Save selected overlap profile plots for cUDE advantage, neutral, and ODE
advantage groups.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function save_overlap_profile_plots(groups, patient_lookup, nn_params, chain, out_dir::AbstractString; plotting::Bool=true)
    plotting || return nothing
    mkpath(out_dir)

    group_specs = [
        ("CUDE_Advantage", groups.cude_advantage),
        ("Neutral_Advantage", groups.neutral),
        ("ODE_Advantage", groups.ode_advantage),
    ]

    for (group_prefix, group_df) in group_specs
        for row in eachrow(group_df)
            patient_id = String(row.patient_id)
            haskey(patient_lookup, patient_id) || continue
            patient = patient_lookup[patient_id]
            tmax = maximum(patient.timepoints) + 10.0
            params_log_ode = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]
            params_natural_cude = Float64[row.a, row.b, row.Cs0, row.Cc0, row.beta]
            t_ode, y_ode = _diagnostic_solve_ode(params_log_ode, tmax)
            t_cude, y_cude = _diagnostic_solve_cude(params_natural_cude, nn_params, chain, tmax)

            plt = _diagnostic_profile_base(patient)
            Plots.plot!(plt, t_ode, y_ode; lw=2, color=:blue, label="ODE")
            Plots.plot!(plt, t_cude, y_cude; lw=2, color=:darkorange, linestyle=:dash, label="cUDE")
            delta_label = "Delta sMAPE: $(round(row.delta_smape, digits=2))%"
            Plots.plot!(plt; legend=:best, legendtitle=delta_label, legendtitlefontsize=9)
            Plots.savefig(plt, joinpath(out_dir, "Overlap_$(group_prefix)_$(patient_id).svg"))
        end
    end

    return out_dir
end

# =============================================================================
# Profile Likelihood Plots
# =============================================================================

const PROFILE_LIKELIHOOD_CLASS_COLORS = Dict(
    "Identifiable" => :orange,
    "Practically identifiable" => :dodgerblue,
    "Unidentifiable" => :deeppink3,
)

const PROFILE_LIKELIHOOD_CLASS_ORDER = [
    "Identifiable",
    "Practically identifiable",
    "Unidentifiable",
]

"""
    profile_likelihood_empirical_quantile(values, q)

Return the empirical quantile used by legacy PLA plot cropping.
"""
# Used by: src/plotting.jl PLA plotting helpers.
function profile_likelihood_empirical_quantile(values::AbstractVector{<:Real}, q::Real)
    isempty(values) && return NaN
    sorted_values = sort(Float64.(values))
    idx = clamp(ceil(Int, q * length(sorted_values)), 1, length(sorted_values))
    return sorted_values[idx]
end

"""
    profile_likelihood_curve_data(df)

Extract sorted natural-scale x, centered log-scale x, and PLR values from a
single profile curve.
"""
# Used by: src/plotting.jl PLA plotting helpers.
function profile_likelihood_curve_data(df::DataFrame)
    keep = isfinite.(df.x_exp) .& isfinite.(df.delta_theta) .& isfinite.(df.plr)
    curve = df[keep, [:x_exp, :delta_theta, :plr]]

    nrow(curve) == 0 && return Float64[], Float64[], Float64[]

    order = sortperm(curve.x_exp)
    return (
        Float64.(curve.x_exp[order]),
        Float64.(curve.delta_theta[order]),
        Float64.(curve.plr[order]),
    )
end

"""
    build_profile_likelihood_legend_panel()

Build the patient-level PLA legend panel.
"""
# Used by: src/plotting.jl (build_profile_likelihood_patient_composite_plot).
function build_profile_likelihood_legend_panel()
    legend_panel = Plots.plot(
        xlim=(0, 1), ylim=(0, 1),
        framestyle=:none,
        xticks=false, yticks=false,
        grid=false,
        legend=false,
    )

    Plots.annotate!(legend_panel, 0.50, 0.92, Plots.text("Legend", 16, :center))
    Plots.plot!(legend_panel, [0.18, 0.38], [0.78, 0.78], color=:blue, lw=2)
    Plots.scatter!(legend_panel, [0.28], [0.68], markercolor=:orange, markerstrokecolor=:black, ms=7)
    Plots.plot!(legend_panel, [0.18, 0.38], [0.58, 0.58], color=:green, ls=:dash, lw=2)

    Plots.annotate!(legend_panel, [
        (0.56, 0.78, Plots.text("profile", 12, :center)),
        (0.56, 0.68, Plots.text("profiler steps", 12, :center)),
        (0.56, 0.58, Plots.text("threshold", 12, :center)),
        (0.50, 0.24, Plots.text("Identifiable: both branches Identifiable", 10, :center)),
        (0.50, 0.17, Plots.text("Practically identifiable: at least one Identifiable branch", 10, :center)),
        (0.50, 0.10, Plots.text("Unidentifiable: no Identifiable branches", 10, :center)),
        (0.50, 0.35, Plots.text("y-axis: -2Δ profile log-likelihood", 11, :center)),
        (0.50, 0.42, Plots.text("x-axis: θ (natural scale, log10 axis)", 11, :center)),
    ])

    return legend_panel
end

"""
    build_profile_likelihood_patient_composite_plot(profiles, summary, patient_id, dataset_name; ...)

Build the six-panel patient-level PLA plot.
"""
# Used by: src/plotting.jl (save_profile_likelihood_patient_plots).
function build_profile_likelihood_patient_composite_plot(
    profiles::DataFrame,
    summary::DataFrame,
    patient_id::AbstractString,
    dataset_name::AbstractString;
    param_names,
    pnames_plot,
)
    panels = Plots.Plot[]

    for j in 1:length(param_names)
        curve_df = profiles[profiles.param_idx.==j, :]
        summary_row = summary[summary.param_idx.==j, :]

        if nrow(curve_df) == 0 || nrow(summary_row) == 0
            panel = Plots.plot(title="$(pnames_plot[j]) | missing", legend=false, framestyle=:box)
            push!(panels, panel)
            continue
        end

        x_nat, _, y_plr = profile_likelihood_curve_data(curve_df)
        if isempty(x_nat)
            panel = Plots.plot(title="$(pnames_plot[j]) | missing", legend=false, framestyle=:box)
            push!(panels, panel)
            continue
        end

        class_label = String(summary_row.class_label[1])
        theta_hat_exp = Float64(summary_row.theta_hat_exp[1])
        threshold = Float64(summary_row.threshold[1])

        panel = Plots.plot(
            x_nat,
            y_plr;
            xscale=:log10,
            legend=false,
            title="$(pnames_plot[j]) | $(class_label)",
            lw=2,
        )

        Plots.scatter!(
            panel,
            x_nat,
            y_plr;
            ms=3,
            markercolor=:orange,
            markerstrokecolor=:black,
        )

        Plots.vline!(panel, [theta_hat_exp], color=:black, ls=:dot, lw=1.5)
        Plots.hline!(panel, [threshold], color=:green, ls=:dash, lw=2)
        Plots.scatter!(panel, [theta_hat_exp], [0.0], color=:black, ms=4)

        yvals = vcat(y_plr, [0.0, threshold])
        Plots.ylims!(panel, (minimum(yvals), 1.05 * maximum(yvals)))

        push!(panels, panel)
    end

    push!(panels, build_profile_likelihood_legend_panel())

    return Plots.plot(
        panels...;
        layout=(2, 3),
        size=(1700, 1000),
        margins=2Plots.mm,
        plot_title="PLA patient $(patient_id) | dataset $(dataset_name)",
    )
end

"""
    build_profile_likelihood_aggregate_legend_panel()

Build the aggregate PLA legend panel.
"""
# Used by: src/plotting.jl (save_profile_likelihood_aggregate_plots).
function build_profile_likelihood_aggregate_legend_panel(; style=nothing)
    title_fontsize = _style_value(style, :legend_title_fontsize, 16)
    label_fontsize = _style_value(style, :legend_label_fontsize, 12)
    note_fontsize = _style_value(style, :legend_note_fontsize, 11)

    legend_panel = Plots.plot(
        xlim=(0, 1), ylim=(0, 1),
        framestyle=:none,
        xticks=false, yticks=false,
        grid=false,
        legend=false,
    )

    Plots.annotate!(legend_panel, 0.50, 0.92, Plots.text("Legend", title_fontsize, :center))

    Plots.plot!(legend_panel, [0.08, 0.28], [0.80, 0.80], color=PROFILE_LIKELIHOOD_CLASS_COLORS["Identifiable"], lw=3)
    Plots.plot!(legend_panel, [0.08, 0.28], [0.68, 0.68], color=PROFILE_LIKELIHOOD_CLASS_COLORS["Practically identifiable"], lw=3)
    Plots.plot!(legend_panel, [0.08, 0.28], [0.56, 0.56], color=PROFILE_LIKELIHOOD_CLASS_COLORS["Unidentifiable"], lw=3)
    Plots.plot!(legend_panel, [0.08, 0.28], [0.42, 0.42], color=:green, ls=:dash, lw=2)

    Plots.annotate!(legend_panel, [
        (0.35, 0.80, Plots.text("Identifiable", label_fontsize, :left)),
        (0.35, 0.68, Plots.text("Practically identifiable", label_fontsize, :left)),
        (0.35, 0.56, Plots.text("Unidentifiable", label_fontsize, :left)),
        (0.35, 0.42, Plots.text("95% threshold", label_fontsize, :left)),
        (0.50, 0.24, Plots.text("x-axis: Δθ (log-scale parameter)", note_fontsize, :center)),
        (0.50, 0.16, Plots.text("y-axis: -2Δ profile log-likelihood", note_fontsize, :center)),
    ])

    return legend_panel
end

"""
    build_profile_likelihood_aggregate_parameter_plot(profiles, summary, pname, pname_plot; threshold)

Build one aggregate PLA plot for a single parameter.
"""
# Used by: src/plotting.jl (save_profile_likelihood_aggregate_plots).
function build_profile_likelihood_aggregate_parameter_plot(
    profiles::DataFrame,
    summary::DataFrame,
    pname::AbstractString,
    pname_plot::AbstractString;
    threshold::Real,
    style=nothing,
)
    summary_param = summary[summary.param_name.==pname, :]
    profiles_param = profiles[profiles.param_name.==pname, :]

    n_identifiable = sum(summary_param.class_label .== "Identifiable")
    n_practical = sum(summary_param.class_label .== "Practically identifiable")
    n_unidentifiable = sum(summary_param.class_label .== "Unidentifiable")

    plot_obj = Plots.plot(
        xlabel="Δ$(pname_plot)",
        ylabel="",
        legend=:best,
        gridalpha=0.15,
        title="",
        lw=1.8,
        size=(900, 650),
        legendfontsize=_style_value(style, :subplot_legend_fontsize, 8),
        tickfontsize=_style_value(style, :subplot_tickfontsize, 10),
        guidefontsize=_style_value(style, :subplot_guidefontsize, 12),
        left_margin=_plots_margin(style, :subplot_left_margin_mm, 2),
        bottom_margin=_plots_margin(style, :subplot_bottom_margin_mm, 3),
        right_margin=_plots_margin(style, :subplot_right_margin_mm, 2),
        top_margin=_plots_margin(style, :subplot_top_margin_mm, 2),
    )

    Plots.plot!(plot_obj, [NaN], [NaN]; color=:green, ls=:dash, lw=2, label="95% threshold")
    Plots.plot!(plot_obj, [NaN], [NaN]; color=PROFILE_LIKELIHOOD_CLASS_COLORS["Unidentifiable"], lw=5, label="Unidentifiable (n=$(n_unidentifiable))")
    Plots.plot!(plot_obj, [NaN], [NaN]; color=PROFILE_LIKELIHOOD_CLASS_COLORS["Practically identifiable"], lw=5, label="Practically identifiable (n=$(n_practical))")
    Plots.plot!(plot_obj, [NaN], [NaN]; color=PROFILE_LIKELIHOOD_CLASS_COLORS["Identifiable"], lw=5, label="Identifiable (n=$(n_identifiable))")
    Plots.hline!(plot_obj, [threshold], color=:green, ls=:dash, lw=2, label=nothing)

    x_focus = Float64[]
    y_focus = Float64[]

    for class_label in PROFILE_LIKELIHOOD_CLASS_ORDER
        summary_class = summary_param[summary_param.class_label.==class_label, :]
        nrow(summary_class) == 0 && continue

        for row in eachrow(summary_class)
            curve_df = profiles_param[
                (profiles_param.patient_id.==row.patient_id).&(profiles_param.param_name.==row.param_name),
                :,
            ]

            nrow(curve_df) < 2 && continue
            _, x_centered, y_plr = profile_likelihood_curve_data(curve_df)
            length(x_centered) < 2 && continue

            Plots.plot!(
                plot_obj,
                x_centered,
                y_plr;
                color=PROFILE_LIKELIHOOD_CLASS_COLORS[class_label],
                lw=1.6,
                alpha=0.75,
                label=nothing,
            )

            keep_focus = y_plr .<= 1.35 * threshold
            if any(keep_focus)
                append!(x_focus, x_centered[keep_focus])
                append!(y_focus, y_plr[keep_focus])
            else
                append!(x_focus, x_centered)
                append!(y_focus, y_plr)
            end
        end
    end

    if !isempty(x_focus)
        xhalf = profile_likelihood_empirical_quantile(abs.(x_focus), 0.98)
        xhalf = max(xhalf, 1.0)
        Plots.xlims!(plot_obj, (-1.05 * xhalf, 1.05 * xhalf))
    end

    if !isempty(y_focus)
        ytop = max(1.10 * threshold, profile_likelihood_empirical_quantile(y_focus, 0.98))
        ytop = max(ytop, 1.10 * threshold)
        Plots.ylims!(plot_obj, (0.0, 1.05 * ytop))
    else
        Plots.ylims!(plot_obj, (0.0, 1.10 * threshold))
    end

    return plot_obj
end

"""
    save_profile_likelihood_patient_plots(paths, patients, dataset_name; ...)

Save patient-level PLA composite SVG plots from per-patient CSV artifacts.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function save_profile_likelihood_patient_plots(
    paths,
    patients,
    dataset_name::AbstractString;
    param_names,
    pnames_plot,
    plotting::Bool=true,
    show_progress::Bool=true,
)
    plotting || return nothing
    mkpath(paths.composite_fig_dir)

    progress = show_progress ? Progress(length(patients); desc="Plotting PLA patients", showspeed=true) : nothing

    for i in eachindex(patients)
        patient_id = String(patients[i].id)
        patient_id_safe = safe_patient_id(patient_id)
        patient_tag = "patient_$(lpad(string(i), 4, '0'))_$(patient_id_safe)"

        try
            loaded = load_profile_likelihood_patient_csvs(paths, patient_tag)
            plot_obj = build_profile_likelihood_patient_composite_plot(
                loaded.profiles,
                loaded.summary,
                patient_id_safe,
                dataset_name;
                param_names=param_names,
                pnames_plot=pnames_plot,
            )
            Plots.savefig(plot_obj, joinpath(paths.composite_fig_dir, "$(patient_tag)_pla.svg"))
        catch err
            @warn "Skipping PLA patient plot because CSVs could not be loaded." patient=patient_id error=err
        end

        progress !== nothing && next!(progress)
    end

    progress !== nothing && finish!(progress)
    return paths.composite_fig_dir
end

"""
    save_profile_likelihood_aggregate_plots(paths, dataset_name; ...)

Save aggregate PLA parameter SVG plots from global CSV artifacts.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function save_profile_likelihood_aggregate_plots(
    paths,
    dataset_name::AbstractString;
    param_names,
    pnames_plot,
    threshold::Real,
    plotting::Bool=true,
    style=nothing,
)
    plotting || return nothing
    mkpath(paths.aggregate_fig_dir)

    loaded = load_profile_likelihood_global_csvs(paths)
    profiles = loaded.profiles_long
    summary = loaded.profiles_summary
    aggregate_panels = Plots.Plot[]

    for j in eachindex(param_names)
        pname = param_names[j]
        pname_plot = pnames_plot[j]
        plot_obj = build_profile_likelihood_aggregate_parameter_plot(
            profiles,
            summary,
            pname,
            pname_plot;
            threshold=threshold,
            style=style,
        )
        push!(aggregate_panels, plot_obj)
        Plots.savefig(plot_obj, joinpath(paths.aggregate_fig_dir, "aggregate_$(pname)_delta_theta_plr.svg"))
    end

    push!(aggregate_panels, build_profile_likelihood_aggregate_legend_panel(; style=style))

    aggregate = Plots.plot(
        aggregate_panels...;
        layout=(2, 3),
        size=(1850, 1080),
        plot_title="Aggregate profile likelihood by parameter | dataset $(dataset_name)",
        plot_titlefontsize=_style_value(style, :combined_title_fontsize, 18),
        left_margin=_plots_margin(style, :combined_left_margin_mm, 2),
        right_margin=_plots_margin(style, :combined_right_margin_mm, 2),
        bottom_margin=_plots_margin(style, :combined_bottom_margin_mm, 4),
        top_margin=_plots_margin(style, :combined_top_margin_mm, 4),
    )

    Plots.savefig(aggregate, joinpath(paths.aggregate_fig_dir, "aggregate_profile_by_parameter.svg"))
    return paths.aggregate_fig_dir
end

# =============================================================================
# Systematic Truncation Plots
# =============================================================================

"""
    save_truncation_initial_scatter(base_patient, save_path, model_cfg; display_plots=false)

Save the initial measurement scatter plot for one systematic truncation patient.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function save_truncation_initial_scatter(
    base_patient::PatientData,
    save_path::AbstractString,
    model_cfg;
    display_plots::Bool=false,
    style=nothing,
)
    mkpath(dirname(save_path))
    plot_obj = Plots.scatter(
        base_patient.timepoints,
        base_patient.ctnt_data;
        ms=6,
        alpha=0.9,
        xlabel="Time (h)",
        ylabel="cTnT [ng/mL]",
        title="Initial measurements - $(base_patient.id) [$(model_cfg.model_name)]",
        label="Initial measurements",
        size=_style_value(style, :size, (1000, 650)),
        guidefontsize=_style_value(style, :guidefontsize, 12),
        tickfontsize=_style_value(style, :tickfontsize, 10),
        titlefontsize=_style_value(style, :titlefontsize, 12),
        legendfontsize=_style_value(style, :legendfontsize, 9),
        left_margin=_plots_margin(style, :left_margin_mm, 10),
        bottom_margin=_plots_margin(style, :bottom_margin_mm, 8),
        right_margin=_plots_margin(style, :right_margin_mm, 4),
        top_margin=_plots_margin(style, :top_margin_mm, 5),
    )
    savefig(plot_obj, save_path)
    display_plots && display(plot_obj)
    return plot_obj
end

"""
    save_truncation_fit_plot(base_patient, scenario, curve_t, curve_plasma, save_path, model_cfg; display_plots=false)

Save one patient-level truncation fit plot with used and removed observations.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function save_truncation_fit_plot(
    base_patient::PatientData,
    scenario,
    curve_t::Vector{Float64},
    curve_plasma::Vector{Float64},
    save_path::AbstractString,
    model_cfg;
    display_plots::Bool=false,
    style=nothing,
)
    mkpath(dirname(save_path))
    plot_obj = Plots.plot(
        curve_t,
        curve_plasma;
        lw=2,
        label=model_cfg.curve_label,
        xlabel="Time (h)",
        ylabel="cTnT [ng/mL]",
        title="[$(model_cfg.model_name)] Base $(base_patient.id) - $(scenario.patient.id)",
        size=_style_value(style, :size, (1000, 650)),
        guidefontsize=_style_value(style, :guidefontsize, 12),
        tickfontsize=_style_value(style, :tickfontsize, 10),
        titlefontsize=_style_value(style, :titlefontsize, 12),
        legendfontsize=_style_value(style, :legendfontsize, 9),
        left_margin=_plots_margin(style, :left_margin_mm, 10),
        bottom_margin=_plots_margin(style, :bottom_margin_mm, 8),
        right_margin=_plots_margin(style, :right_margin_mm, 4),
        top_margin=_plots_margin(style, :top_margin_mm, 5),
    )

    Plots.scatter!(
        plot_obj,
        base_patient.timepoints[scenario.removed_idx],
        base_patient.ctnt_data[scenario.removed_idx];
        markershape=:x,
        markerstrokewidth=2,
        ms=7,
        color=:crimson,
        label="Removed measurements",
    )

    Plots.scatter!(
        plot_obj,
        base_patient.timepoints[scenario.kept_idx],
        base_patient.ctnt_data[scenario.kept_idx];
        markershape=:circle,
        ms=5,
        color=:dodgerblue,
        label="Used measurements",
    )

    savefig(plot_obj, save_path)
    display_plots && display(plot_obj)
    return plot_obj
end

"""
    save_truncation_parameter_boxplot(patients, validation_params, save_path, model_cfg; ...)

Save a five-panel natural-scale parameter boxplot for one truncation patient.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function save_truncation_parameter_boxplot(
    patients::Vector{PatientData},
    validation_params::Vector{Vector{Float64}},
    save_path::AbstractString,
    model_cfg;
    dataset::AbstractString="",
    data_label::AbstractString="",
    show_outliers::Bool=true,
)
    mkpath(dirname(save_path))
    param_labels = model_cfg.param_labels
    values_by_param = [Float64[] for _ in param_labels]

    for θ in validation_params
        pars = exp.(θ)
        for idx in eachindex(param_labels)
            push!(values_by_param[idx], pars[idx])
        end
    end

    for (idx, label) in enumerate(param_labels)
        vals = values_by_param[idx]
        @info "Truncation parameter summary" dataset=dataset data_label=data_label parameter=label mean=mean(vals) std=std(vals) median=median(vals) q1=quantile(vals, 0.25) q3=quantile(vals, 0.75)
    end

    fig = CairoMakie.Figure(size=(1400, 700))
    CairoMakie.Label(
        fig[0, 1:length(param_labels)],
        "Parameter distributions - $(dataset) $(data_label)";
        fontsize=22,
        tellwidth=false,
    )

    colors = [:skyblue, :orange, :lightgreen, :pink, :violet]
    x = fill(1, length(patients))
    for (idx, label) in enumerate(param_labels)
        axis = CairoMakie.Axis(
            fig[1, idx],
            title=label,
            xticklabelsvisible=false,
            xticksvisible=false,
        )
        CairoMakie.boxplot!(
            axis,
            x,
            values_by_param[idx];
            color=colors[mod1(idx, length(colors))],
            whiskerwidth=0.3,
            strokewidth=0.3,
            show_outliers=show_outliers,
        )
    end

    CairoMakie.save(save_path, fig)
    return (values=values_by_param, figure=fig)
end

"""
    save_truncation_overlay_plot(record, output_dir; plot_legend=false, axis_labels=true)

Save one ODE-vs-cUDE systematic truncation overlay plot.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function save_truncation_overlay_plot(
    record,
    output_dir::AbstractString;
    plot_legend::Bool=false,
    axis_labels::Bool=true,
    style=nothing,
)
    mkpath(output_dir)
    section_upper = uppercase(record.section)
    budget_str = lpad(string(length(record.removed_idx)), 2, "0")

    plot_obj = Plots.plot(
        record.ode_t,
        record.ode_plasma;
        lw=2,
        color=:royalblue,
        linestyle=:solid,
        label="ODE  (sMAPE=$(record.ode_smape)%, RMSLE=$(record.ode_rmsle))",
        xlabel=axis_labels ? "Time (h)" : "",
        ylabel=axis_labels ? "cTnT [ng/mL]" : "",
        legend=plot_legend ? :best : false,
        legendfontsize=_style_value(style, :legendfontsize, 9),
        guidefontsize=_style_value(style, :guidefontsize, 12),
        tickfontsize=_style_value(style, :tickfontsize, 10),
        titlefontsize=_style_value(style, :titlefontsize, 12),
        grid=true,
        size=_style_value(style, :size, (1000, 650)),
        left_margin=_plots_margin(style, :left_margin_mm, 10),
        bottom_margin=_plots_margin(style, :bottom_margin_mm, 8),
        right_margin=_plots_margin(style, :right_margin_mm, 4),
        top_margin=_plots_margin(style, :top_margin_mm, 5),
    )

    Plots.plot!(
        plot_obj,
        record.cude_t,
        record.cude_plasma;
        lw=2,
        color=:darkorange,
        linestyle=:dash,
        label="cUDE (sMAPE=$(record.cude_smape)%, RMSLE=$(record.cude_rmsle))",
    )

    if !isempty(record.removed_idx)
        Plots.scatter!(
            plot_obj,
            record.base_times[record.removed_idx],
            record.base_troponin[record.removed_idx];
            markershape=:x,
            markerstrokewidth=2,
            ms=7,
            color=:crimson,
            label="Removed (n=$(length(record.removed_idx)))",
        )
    end

    if !isempty(record.kept_idx)
        Plots.scatter!(
            plot_obj,
            record.base_times[record.kept_idx],
            record.base_troponin[record.kept_idx];
            markershape=:circle,
            ms=5,
            color=:dodgerblue,
            label="Kept (n=$(length(record.kept_idx)))",
        )
    end

    save_path = joinpath(
        output_dir,
        "overlay_$(record.patient_id)_$(section_upper)_S$(record.set_id)_B$(budget_str).svg",
    )
    savefig(plot_obj, save_path)
    return save_path
end

# =============================================================================
# Symbolic Regression Plots
# =============================================================================

"""
    save_symbolic_teacher_plot(t_grid, beta_grid, y_teacher, save_path; display_plots=false)

Save the synthetic NN teacher target curves shown to symbolic regression.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function save_symbolic_teacher_plot(
    t_grid,
    beta_grid,
    y_teacher,
    save_path::AbstractString;
    display_plots::Bool=false,
)
    mkpath(dirname(save_path))
    t_values = Float64.(collect(t_grid))
    beta_values = Float64.(collect(beta_grid))
    y_mat = reshape(y_teacher, length(t_values), length(beta_values))

    plot_obj = Plots.plot(
        xlabel="Time (h)",
        ylabel="rupture f(t_norm, β)",
        title="Synthetic NN target shown to SR",
        linewidth=2,
    )

    for (beta_idx, beta_value) in enumerate(beta_values)
        Plots.plot!(
            plot_obj,
            t_values,
            y_mat[:, beta_idx];
            label="β = $(round(beta_value, digits=2))",
        )
    end

    savefig(plot_obj, save_path)
    display_plots && display(plot_obj)
    return plot_obj
end

"""
    save_symbolic_nn_vs_sr_plot(records, save_path; display_plots=false)

Save the NN teacher versus symbolic surrogate comparison curves.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function save_symbolic_nn_vs_sr_plot(
    records,
    save_path::AbstractString;
    display_plots::Bool=false,
)
    mkpath(dirname(save_path))
    plot_obj = Plots.plot(
        xlabel="Time (h)",
        ylabel="rupture f(t_norm, β)",
        title="NN vs SR surrogate (direct fit on y_NN)",
        linewidth=2,
        legend=:best,
    )

    for (idx, record) in enumerate(records)
        nn_label = idx == 1 ? "NN (solid lines)" : false
        sr_label = idx == 1 ? "SR (dashed lines)" : false
        Plots.plot!(plot_obj, record.t_h, record.y_nn; linestyle=:solid, label=nn_label)
        Plots.plot!(plot_obj, record.t_h, record.y_sr; linestyle=:dash, label=sr_label)
    end

    savefig(plot_obj, save_path)
    display_plots && display(plot_obj)
    return plot_obj
end

"""
    save_symbolic_sr_plot(records, save_path; display_plots=false)

Save symbolic surrogate curves without the NN teacher overlay.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function save_symbolic_sr_plot(
    records,
    save_path::AbstractString;
    display_plots::Bool=false,
)
    mkpath(dirname(save_path))
    plot_obj = Plots.plot(
        xlabel="Time (h)",
        ylabel="rupture f(t_norm, β)",
        title="SR surrogate",
        linewidth=2,
        legend=:best,
    )

    for (idx, record) in enumerate(records)
        sr_label = idx == 1 ? "SR (dashed lines)" : false
        Plots.plot!(plot_obj, record.t_h, record.y_sr; linestyle=:dash, label=sr_label)
    end

    savefig(plot_obj, save_path)
    display_plots && display(plot_obj)
    return plot_obj
end

# =============================================================================
# Training Loss Plots
# =============================================================================

"""
    save_cude_training_loss_plots(paths, losses_per_model; adam_maxiters, plotting, show_progress)

Save ADAM/LBFGS loss curves for each trained cUDE initialization.
"""
# Used by: scripts/02a_run_cude_training.jl.
function save_cude_training_loss_plots(
    paths,
    losses_per_model;
    adam_maxiters::Integer,
    plotting::Bool=true,
    show_progress::Bool=true,
)
    plotting || return nothing
    mkpath(paths.fig_dir)

    progress = show_progress ? Progress(length(losses_per_model); desc="Plotting cUDE losses", showspeed=true) : nothing

    for (k, loss_vec) in enumerate(losses_per_model)
        if isempty(loss_vec)
            @warn "Skipping empty loss curve for cUDE model $(k)."
            progress !== nothing && next!(progress)
            continue
        end

        adam_end = min(adam_maxiters, length(loss_vec))
        plt = Plots.plot(
            1:adam_end,
            loss_vec[1:adam_end];
            yaxis=:log10,
            xaxis=:log10,
            label="Adam",
            color=:blue,
            xlabel="Iteration",
            ylabel="Training loss",
            title="cUDE training loss model $(k)",
        )

        if adam_end < length(loss_vec)
            Plots.plot!(
                plt,
                adam_end + 1:length(loss_vec),
                loss_vec[adam_end + 1:end];
                label="LBFGS",
                color=:red,
            )
        end

        savefig(plt, joinpath(paths.fig_dir, "loss_model_$(k).svg"))
        progress !== nothing && next!(progress)
    end

    progress !== nothing && finish!(progress)
    return nothing
end

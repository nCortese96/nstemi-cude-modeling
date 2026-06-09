"""
03a_run_model_diagnostics.jl

Compare ODE and selected cUDE patient-level outputs across MIMIC-IV and UMG.

Pipeline:
1. Load shared helpers and workflow configuration.
2. Resolve step 00 cohorts, step 01 ODE fits, step 02b/02d cUDE fits, and the
   step 02c selected cUDE model.
3. Recompute residual diagnostics from patient trajectories and fitted
   parameters.
4. Save diagnostic tables, residual plots, parameter boxplots, metric-comparison
   plots, and selected patient profile plots.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/03a_run_model_diagnostics.jl
    julia --project=. scripts/03a_run_model_diagnostics.jl plots
    julia --project=. scripts/03a_run_model_diagnostics.jl plots_metrics
    julia --project=. scripts/03a_run_model_diagnostics.jl plots_profiles

Use `config/workflow_config.jl` to switch between `results/` and
`results_test/`, choose the selected-model source, or disable plots/progress
bars. Plot-only modes read existing CSV artifacts and regenerate figures without
rewriting diagnostic tables or `delta_smape_report.txt`. `plots` regenerates all
available diagnostic plots, while `plots_metrics` and `plots_profiles` restrict
the regeneration to those plot families. The legacy `residual_plots` alias maps
to `plots`.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using CSV
using Dates
using DataFrames: DataFrame, nrow
using Logging
using Base.Threads: nthreads

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "diagnostics.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings live in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.model_diagnostics
diagnostic_mode = isempty(ARGS) ? "all" : lowercase(strip(ARGS[1]))
diagnostic_mode = diagnostic_mode == "residual_plots" ? "plots" : diagnostic_mode
valid_diagnostic_modes = ("all", "plots", "plots_metrics", "plots_profiles")
length(ARGS) <= 1 ||
    error("Model diagnostics accepts at most one mode argument. Valid modes: $(join(valid_diagnostic_modes, ", ")).")
diagnostic_mode in valid_diagnostic_modes ||
    error("Invalid model diagnostics mode '$(diagnostic_mode)'. Valid modes: $(join(valid_diagnostic_modes, ", ")).")
plot_only_mode = diagnostic_mode != "all"
run_residual_plots = diagnostic_mode in ("all", "plots")
run_metric_plots = diagnostic_mode in ("all", "plots", "plots_metrics")
run_profile_plots = settings.profile_comparison && diagnostic_mode in ("all", "plots", "plots_profiles")
mimic_dataset_config = config.datasets[settings.mimic_dataset_key]
external_dataset_config = config.datasets[settings.external_dataset_key]
selection_dataset_config = config.datasets[settings.model_selection_dataset_key]

# =============================================================================
# INPUT PATHS
# Step 03a consumes step 00 cohorts and fitted model artifacts from steps 01,
# 02b, 02c, and 02d.
# =============================================================================

cohort_dir = settings.cohort_dir
selection_paths = cude_model_selection_output_paths(settings.model_selection_dir, selection_dataset_config.dataset_name)
selected_model_path = settings.selected_model_path === nothing ?
                      selection_paths.selected_model :
                      settings.selected_model_path

# =============================================================================
# OUTPUT PATHS
# Diagnostic tables live at the step root; plots are grouped under `figs/`.
# =============================================================================

output_root = settings.output_dir
output_paths = model_diagnostics_output_paths(output_root)

# =============================================================================
# DERIVED SETTINGS
# Values derived from config and selected artifacts. No heavy side effects here.
# =============================================================================

mimic_dataset_name = mimic_dataset_config.dataset_name
external_dataset_name = external_dataset_config.dataset_name
selection_dataset_name = selection_dataset_config.dataset_name
time_edges = collect(settings.time_edges)

config.model.t_scale == T_SCALE ||
    error("Model diagnostics require config.model.t_scale=$(config.model.t_scale) to match models.jl T_SCALE=$(T_SCALE).")

# =============================================================================
# PIPELINE
# =============================================================================

@info "Model diagnostics workflow started at $(now())."
log_workflow_context(
    config;
    script_name=basename(@__FILE__),
    output_paths=(
        cohort_dir=cohort_dir,
        selected_model=selected_model_path,
        output_root=output_root,
    ),
)

@info "MIMIC dataset: $(mimic_dataset_name)"
@info "External dataset: $(external_dataset_name)"
@info "Selection dataset: $(selection_dataset_name)"
@info "Requested model diagnostics mode: $(diagnostic_mode)"
@info "Julia threads: $(nthreads())"

ensure_output_dirs!(
    (
        output_root=output_paths.output_dir,
        residuals=output_paths.residuals_fig_dir,
        boxplots=output_paths.boxplots_fig_dir,
        metrics_comparison=output_paths.metrics_comparison_fig_dir,
        profiles_comparison=output_paths.profiles_comparison_dir,
    );
    header="Ensured model diagnostics output directories",
)

@info "Loading selected cUDE model from step 02c."
selected_model = load_selected_cude_model(selected_model_path)
@info "Selected cUDE model: $(selected_model.model_id) | width=$(selected_model.nn_width) | model_idx=$(selected_model.model_idx)"

input_paths = model_diagnostics_input_paths(config, settings, selected_model)
validate_existing_paths(
    merge((selected_model=selected_model_path,), input_paths);
    header="Required model diagnostics input files",
)

@info "Loading step 00 cohorts."
mimic_cohort = load_preprocessed_cohort(mimic_dataset_name, cohort_dir)
external_cohort = load_preprocessed_cohort(external_dataset_name, cohort_dir)
mimic_patients = mimic_cohort.test
external_patients = external_cohort.test
mimic_patient_lookup = Dict(patient.id => patient for patient in mimic_patients)
external_patient_lookup = Dict(patient.id => patient for patient in external_patients)

@info "Loaded $(length(mimic_patients)) MIMIC-IV validation/test patients."
@info "Loaded $(length(external_patients)) UMG external-test patients."

@info "Loading fitted parameter and metric tables."
df_ode_mimic = CSV.read(input_paths.ode_mimic_params, DataFrame)
df_ode_umg = CSV.read(input_paths.ode_external_params, DataFrame)
df_cude_mimic_params = CSV.read(input_paths.cude_mimic_params, DataFrame)
df_cude_umg_params = CSV.read(input_paths.cude_external_params, DataFrame)
df_cude_mimic_metrics = CSV.read(input_paths.cude_mimic_metrics, DataFrame)
df_cude_umg_metrics = CSV.read(input_paths.cude_external_metrics, DataFrame)

@info "Loaded ODE MIMIC rows: $(nrow(df_ode_mimic))"
@info "Loaded ODE UMG rows: $(nrow(df_ode_umg))"
@info "Loaded cUDE MIMIC rows: $(nrow(df_cude_mimic_params))"
@info "Loaded cUDE UMG rows: $(nrow(df_cude_umg_params))"

@info "Loading selected cUDE neural-network parameters."
artifacts = load_cude_training_artifacts(settings.cude_training_input_dir, selected_model.nn_width)
neural_params = artifacts.neural_network_parameters[selected_model.model_idx]
chain = neural_network_model(selected_model.nn_depth, selected_model.nn_width; input_dims=settings.input_dim)

if plot_only_mode
    if run_residual_plots
        @info "Regenerating residual diagnostic plots from existing CSV artifacts."
        validate_existing_paths(
            (
                residuals_ode_mimic=output_paths.residuals_ode_mimic,
                residuals_ode_umg=output_paths.residuals_ode_umg,
                residuals_cude_mimic=output_paths.residuals_cude_mimic,
                residuals_cude_umg=output_paths.residuals_cude_umg,
            );
            header="Required residual CSV files for plot-only mode",
        )

        save_residual_diagnostic_plots(
            output_paths;
            residuals_ode_mimic=CSV.read(output_paths.residuals_ode_mimic, DataFrame),
            residuals_ode_umg=CSV.read(output_paths.residuals_ode_umg, DataFrame),
            residuals_cude_mimic=CSV.read(output_paths.residuals_cude_mimic, DataFrame),
            residuals_cude_umg=CSV.read(output_paths.residuals_cude_umg, DataFrame),
            edges=time_edges,
            tmax=settings.residual_tmax,
            plotting=settings.plotting,
            style=settings.residual_plot_style,
        )
        @info "Residual diagnostic plots regenerated." output_dir=output_paths.residuals_fig_dir
    end

    if run_metric_plots
        @info "Regenerating parameter and metric diagnostic plots from existing CSV artifacts."
        validate_existing_paths(
            (
                metrics_ode_mimic=output_paths.metrics_ode_mimic,
                metrics_ode_umg=output_paths.metrics_ode_umg,
                metrics_cude_mimic=output_paths.metrics_cude_mimic,
                metrics_cude_umg=output_paths.metrics_cude_umg,
            );
            header="Required metric CSV files for plot-only mode",
        )

        met_ode_mimic = CSV.read(output_paths.metrics_ode_mimic, DataFrame)
        met_ode_umg = CSV.read(output_paths.metrics_ode_umg, DataFrame)
        met_cude_mimic = CSV.read(output_paths.metrics_cude_mimic, DataFrame)
        met_cude_umg = CSV.read(output_paths.metrics_cude_umg, DataFrame)

        par_ode_mimic = diagnostic_parameter_table_from_fit_params(df_ode_mimic; model_type=:ode)
        par_ode_umg = diagnostic_parameter_table_from_fit_params(df_ode_umg; model_type=:ode)
        par_cude_mimic = diagnostic_parameter_table_from_fit_params(df_cude_mimic_params; model_type=:cude)
        par_cude_umg = diagnostic_parameter_table_from_fit_params(df_cude_umg_params; model_type=:cude)

        comparison_mimic = comparison_metrics_dataframe(met_ode_mimic, met_cude_mimic)
        comparison_umg = comparison_metrics_dataframe(met_ode_umg, met_cude_umg)

        save_parameter_diagnostic_boxplots(
            output_paths;
            par_ode_mimic=par_ode_mimic,
            par_ode_umg=par_ode_umg,
            par_cude_mimic=par_cude_mimic,
            par_cude_umg=par_cude_umg,
            plotting=settings.plotting,
        )

        save_metric_comparison_paper_plots(
            output_paths;
            comparison_mimic=comparison_mimic,
            comparison_umg=comparison_umg,
            metrics_ode_mimic=met_ode_mimic,
            metrics_ode_umg=met_ode_umg,
            metrics_cude_mimic=met_cude_mimic,
            metrics_cude_umg=met_cude_umg,
            plotting=settings.metrics_paper_plots,
        )
        @info "Metric and parameter diagnostic plots regenerated." output_dir=output_paths.metrics_comparison_fig_dir
    end

    if diagnostic_mode == "plots_profiles" && !settings.profile_comparison
        @warn "plots_profiles requested, but WORKFLOW_CONFIG.model_diagnostics.profile_comparison is false."
    end

    if run_profile_plots
        @info "Regenerating selected patient profile comparisons from existing CSV artifacts."

        ode_mimic_quartiles = select_metric_quartile_rows(
            df_ode_mimic,
            :smape;
            n_per_quartile=settings.profile_rows_per_group,
            seed=settings.profile_selection_seed,
        )
        ode_umg_quartiles = select_metric_quartile_rows(
            df_ode_umg,
            :smape;
            n_per_quartile=settings.profile_rows_per_group,
            seed=settings.profile_selection_seed,
        )

        cude_mimic_selection = cude_profile_selection_table(df_cude_mimic_metrics, df_cude_mimic_params)
        cude_umg_selection = cude_profile_selection_table(df_cude_umg_metrics, df_cude_umg_params)
        cude_mimic_quartiles = select_metric_quartile_rows(
            cude_mimic_selection,
            :smape;
            n_per_quartile=settings.profile_rows_per_group,
            seed=settings.profile_selection_seed,
        )
        cude_umg_quartiles = select_metric_quartile_rows(
            cude_umg_selection,
            :smape;
            n_per_quartile=settings.profile_rows_per_group,
            seed=settings.profile_selection_seed,
        )

        overlap_mimic = overlap_comparison_dataframe(df_ode_mimic, df_cude_mimic_metrics, df_cude_mimic_params)
        overlap_umg = overlap_comparison_dataframe(df_ode_umg, df_cude_umg_metrics, df_cude_umg_params)
        overlap_mimic_groups = select_overlap_profile_rows(overlap_mimic; n_per_group=settings.profile_rows_per_group)
        overlap_umg_groups = select_overlap_profile_rows(overlap_umg; n_per_group=settings.profile_rows_per_group)

        profile_root = output_paths.profiles_comparison_dir
        save_ode_quartile_profile_plots(ode_mimic_quartiles, mimic_patient_lookup, joinpath(profile_root, "ODE_Q_MIMIC"); plotting=settings.plotting)
        save_ode_quartile_profile_plots(ode_umg_quartiles, external_patient_lookup, joinpath(profile_root, "ODE_Q_UMG"); plotting=settings.plotting)
        save_cude_quartile_profile_plots(cude_mimic_quartiles, mimic_patient_lookup, neural_params, chain, joinpath(profile_root, "cUDE_Q_MIMIC"); plotting=settings.plotting)
        save_cude_quartile_profile_plots(cude_umg_quartiles, external_patient_lookup, neural_params, chain, joinpath(profile_root, "cUDE_Q_UMG"); plotting=settings.plotting)
        save_overlap_profile_plots(overlap_mimic_groups, mimic_patient_lookup, neural_params, chain, joinpath(profile_root, "Overlap_MIMIC"); plotting=settings.plotting)
        save_overlap_profile_plots(overlap_umg_groups, external_patient_lookup, neural_params, chain, joinpath(profile_root, "Overlap_UMG"); plotting=settings.plotting)
        @info "Selected patient profile comparisons regenerated." output_dir=profile_root
    end

    @info "Model diagnostics plot-only workflow completed at $(now())."
    exit(0)
end

@info "Computing residuals for ODE and cUDE across both datasets."
res_ode_mimic, met_ode_mimic, par_ode_mimic = compute_residuals_long_unified(
    mimic_patient_lookup,
    df_ode_mimic;
    model_type=:ode,
)
@info "Completed ODE x MIMIC-IV residuals: $(nrow(res_ode_mimic)) points."

res_ode_umg, met_ode_umg, par_ode_umg = compute_residuals_long_unified(
    external_patient_lookup,
    df_ode_umg;
    model_type=:ode,
)
@info "Completed ODE x UMG residuals: $(nrow(res_ode_umg)) points."

res_cude_mimic, met_cude_mimic, par_cude_mimic = compute_residuals_long_unified(
    mimic_patient_lookup,
    df_cude_mimic_params;
    model_type=:cude,
    chain=chain,
    nn_params=neural_params,
)
@info "Completed cUDE x MIMIC-IV residuals: $(nrow(res_cude_mimic)) points."

res_cude_umg, met_cude_umg, par_cude_umg = compute_residuals_long_unified(
    external_patient_lookup,
    df_cude_umg_params;
    model_type=:cude,
    chain=chain,
    nn_params=neural_params,
)
@info "Completed cUDE x UMG residuals: $(nrow(res_cude_umg)) points."

parameter_summary = build_parameter_summary([
    (model="ODE", dataset="MIMIC-IV", df=par_ode_mimic, param_names=["a", "b", "Cs0", "Cc0", "Td"]),
    (model="ODE", dataset="UMG", df=par_ode_umg, param_names=["a", "b", "Cs0", "Cc0", "Td"]),
    (model="cUDE", dataset="MIMIC-IV", df=par_cude_mimic, param_names=["a", "b", "Cs0", "Cc0", "beta"]),
    (model="cUDE", dataset="UMG", df=par_cude_umg, param_names=["a", "b", "Cs0", "Cc0", "beta"]),
])

metrics_summary = build_metrics_summary([
    (model="ODE", dataset="MIMIC-IV", df=met_ode_mimic),
    (model="ODE", dataset="UMG", df=met_ode_umg),
    (model="cUDE", dataset="MIMIC-IV", df=met_cude_mimic),
    (model="cUDE", dataset="UMG", df=met_cude_umg),
])

save_model_diagnostics_tables(
    output_paths;
    residuals_ode_mimic=res_ode_mimic,
    residuals_ode_umg=res_ode_umg,
    residuals_cude_mimic=res_cude_mimic,
    residuals_cude_umg=res_cude_umg,
    metrics_ode_mimic=met_ode_mimic,
    metrics_ode_umg=met_ode_umg,
    metrics_cude_mimic=met_cude_mimic,
    metrics_cude_umg=met_cude_umg,
    metrics_summary=metrics_summary,
    parameter_summary=parameter_summary,
)
@info "Saved model diagnostics tables."

comparison_mimic = comparison_metrics_dataframe(met_ode_mimic, met_cude_mimic)
comparison_umg = comparison_metrics_dataframe(met_ode_umg, met_cude_umg)
overlap_mimic = overlap_comparison_dataframe(df_ode_mimic, df_cude_mimic_metrics, df_cude_mimic_params)
overlap_umg = overlap_comparison_dataframe(df_ode_umg, df_cude_umg_metrics, df_cude_umg_params)

write_delta_smape_report(
    output_paths.delta_smape_report,
    [
        (dataset_label="MIMIC", df=overlap_mimic),
        (dataset_label="UMG", df=overlap_umg),
    ];
    threshold=settings.delta_smape_threshold,
)
@info "Saved delta-sMAPE report."

save_residual_diagnostic_plots(
    output_paths;
    residuals_ode_mimic=res_ode_mimic,
    residuals_ode_umg=res_ode_umg,
    residuals_cude_mimic=res_cude_mimic,
    residuals_cude_umg=res_cude_umg,
    edges=time_edges,
    tmax=settings.residual_tmax,
    plotting=settings.plotting,
    style=settings.residual_plot_style,
)

save_parameter_diagnostic_boxplots(
    output_paths;
    par_ode_mimic=par_ode_mimic,
    par_ode_umg=par_ode_umg,
    par_cude_mimic=par_cude_mimic,
    par_cude_umg=par_cude_umg,
    plotting=settings.plotting,
)

save_metric_comparison_paper_plots(
    output_paths;
    comparison_mimic=comparison_mimic,
    comparison_umg=comparison_umg,
    metrics_ode_mimic=met_ode_mimic,
    metrics_ode_umg=met_ode_umg,
    metrics_cude_mimic=met_cude_mimic,
    metrics_cude_umg=met_cude_umg,
    plotting=settings.metrics_paper_plots,
)

if settings.profile_comparison
    @info "Generating selected patient profile comparisons."

    ode_mimic_quartiles = select_metric_quartile_rows(
        df_ode_mimic,
        :smape;
        n_per_quartile=settings.profile_rows_per_group,
        seed=settings.profile_selection_seed,
    )
    ode_umg_quartiles = select_metric_quartile_rows(
        df_ode_umg,
        :smape;
        n_per_quartile=settings.profile_rows_per_group,
        seed=settings.profile_selection_seed,
    )

    cude_mimic_selection = cude_profile_selection_table(df_cude_mimic_metrics, df_cude_mimic_params)
    cude_umg_selection = cude_profile_selection_table(df_cude_umg_metrics, df_cude_umg_params)
    cude_mimic_quartiles = select_metric_quartile_rows(
        cude_mimic_selection,
        :smape;
        n_per_quartile=settings.profile_rows_per_group,
        seed=settings.profile_selection_seed,
    )
    cude_umg_quartiles = select_metric_quartile_rows(
        cude_umg_selection,
        :smape;
        n_per_quartile=settings.profile_rows_per_group,
        seed=settings.profile_selection_seed,
    )

    overlap_mimic_groups = select_overlap_profile_rows(overlap_mimic; n_per_group=settings.profile_rows_per_group)
    overlap_umg_groups = select_overlap_profile_rows(overlap_umg; n_per_group=settings.profile_rows_per_group)

    profile_root = output_paths.profiles_comparison_dir
    save_ode_quartile_profile_plots(ode_mimic_quartiles, mimic_patient_lookup, joinpath(profile_root, "ODE_Q_MIMIC"); plotting=settings.plotting)
    save_ode_quartile_profile_plots(ode_umg_quartiles, external_patient_lookup, joinpath(profile_root, "ODE_Q_UMG"); plotting=settings.plotting)
    save_cude_quartile_profile_plots(cude_mimic_quartiles, mimic_patient_lookup, neural_params, chain, joinpath(profile_root, "cUDE_Q_MIMIC"); plotting=settings.plotting)
    save_cude_quartile_profile_plots(cude_umg_quartiles, external_patient_lookup, neural_params, chain, joinpath(profile_root, "cUDE_Q_UMG"); plotting=settings.plotting)
    save_overlap_profile_plots(overlap_mimic_groups, mimic_patient_lookup, neural_params, chain, joinpath(profile_root, "Overlap_MIMIC"); plotting=settings.plotting)
    save_overlap_profile_plots(overlap_umg_groups, external_patient_lookup, neural_params, chain, joinpath(profile_root, "Overlap_UMG"); plotting=settings.plotting)
end

@info "Model diagnostics workflow completed at $(now())."

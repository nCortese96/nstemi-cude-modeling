"""
03c_run_systematic_truncation.jl

Run the systematic truncation stress workflow for the ODE baseline and the
selected cUDE model.

Pipeline:
1. Load shared helpers and workflow configuration.
2. Resolve the requested target and canonical step 03c output paths.
3. Load the step 00 gold-standard cohort and selected cUDE artifacts.
4. Run ODE/cUDE truncation fits when requested.
5. Save summary tables and ODE-vs-cUDE overlay plots when requested.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/03c_run_systematic_truncation.jl [target]

Valid targets:
    all
    ode
    cude
    summary
    overlay
    plots

No target means full `all`: rerun requested calculations and overwrite plots.
The `plots` target regenerates patient-level truncation figures and overlays
from existing CSV artifacts without rerunning fitting. Use
`config/workflow_config.jl` to switch between `results/` and `results_test/`,
change optimizer settings, or disable plots and progress bars.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using Base.Threads: nthreads
using CSV
using Dates
using Logging

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "fitting.jl"))
include(joinpath(@__DIR__, "..", "src", "systematic_truncation.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings live in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.systematic_truncation
cli = parse_systematic_truncation_cli(ARGS, settings)
target_keys = cli.targets

mimic_dataset_config = config.datasets[settings.mimic_dataset_key]
external_dataset_config = config.datasets[settings.external_dataset_key]
selection_dataset_config = config.datasets[settings.model_selection_dataset_key]

# =============================================================================
# INPUT PATHS
# Step 03c consumes step 00 gold-standard artifacts and, for cUDE/overlay, the
# selected model and NN weights from steps 02a/02c.
# =============================================================================

cohort_dir = settings.cohort_dir
selection_paths = cude_model_selection_output_paths(settings.model_selection_dir, selection_dataset_config.dataset_name)
selected_model_path = settings.selected_model_path === nothing ?
                      selection_paths.selected_model :
                      settings.selected_model_path

# =============================================================================
# OUTPUT PATHS
# Step 03c writes under `03_comparison_analyses/03c_truncation_stress`.
# =============================================================================

output_root = settings.output_dir
paths = systematic_truncation_output_paths(output_root)
ode_paths = systematic_truncation_model_output_paths(output_root, :ode)
cude_paths = systematic_truncation_model_output_paths(output_root, :cude)

# =============================================================================
# DERIVED SETTINGS
# Values derived from config and CLI arguments. No heavy side effects here.
# =============================================================================

config.model.t_scale == T_SCALE ||
    error("Systematic truncation requires config.model.t_scale=$(config.model.t_scale) to match models.jl T_SCALE=$(T_SCALE).")

needs_cude_artifacts = (:cude in target_keys) || (:overlay in target_keys) || (:plots in target_keys)
needs_gold_cohort = (:ode in target_keys) || (:cude in target_keys) || (:overlay in target_keys) || (:plots in target_keys)

# =============================================================================
# PIPELINE
# =============================================================================

@info "Systematic truncation workflow started at $(now())."
log_workflow_context(
    config;
    script_name=basename(@__FILE__),
    output_paths=(
        cohort_dir=cohort_dir,
        selected_model=selected_model_path,
        output_root=output_root,
    ),
)

@info "Requested systematic truncation target: $(cli.requested)"
@info "Resolved targets: $(target_keys)"
@info "Gold-standard source: $(cohort_dir)"
@info "Julia threads: $(nthreads())"

ensure_output_dirs!(
    (
        output_root=paths.output_dir,
        ode=paths.ode_dir,
        cude=paths.cude_dir,
        overlay=paths.overlay_dir,
        overlay_no_labels=paths.overlay_no_labels_dir,
        overlay_legend_on=paths.overlay_legend_on_dir,
    );
    header="Ensured systematic truncation output directories",
)

log_output_paths(
    (
        ode_metrics=ode_paths.metrics_all,
        cude_metrics=cude_paths.metrics_all,
        metrics_summary=paths.metrics_summary,
        params_summary=paths.params_summary,
        overlay_dir=paths.overlay_dir,
        overlay_no_labels_dir=paths.overlay_no_labels_dir,
        overlay_legend_on_dir=paths.overlay_legend_on_dir,
    );
    header="Systematic truncation output paths",
)

gold = nothing
if needs_gold_cohort
    @info "Loading step 00 gold-standard cohort."
    gold = load_gold_standard_cohort(
        cohort_dir,
        settings.run_dataset_name;
        mimic_dataset_name=mimic_dataset_config.dataset_name,
        external_dataset_name=external_dataset_config.dataset_name,
    )
    @info "Loaded $(length(gold.patients)) gold-standard patients." ids=gold.ids
end

selected_model = nothing
cude_artifacts = nothing

if needs_cude_artifacts
    @info "Loading selected cUDE model from step 02c."
    selected_model = load_selected_cude_model(selected_model_path)
    @info "Selected cUDE model: $(selected_model.model_id) | width=$(selected_model.nn_width) | model_idx=$(selected_model.model_idx)"

    @info "Loading cUDE training artifacts for selected width=$(selected_model.nn_width)."
    cude_artifacts = load_cude_training_artifacts(settings.cude_training_input_dir, selected_model.nn_width)
end

if :ode in target_keys
    @info "Running ODE systematic truncation target."
    ode_model_cfg = build_systematic_truncation_model_runtime(:ode, settings)
    run_systematic_truncation_target(
        gold.patients,
        gold.patient_dataset,
        ode_model_cfg,
        settings,
        ode_paths;
        initial_plot_callback=settings.plotting ?
                              (patient, save_path, model_cfg) -> save_truncation_initial_scatter(
                                  patient,
                                  save_path,
                                  model_cfg;
                                  display_plots=settings.display_plots,
                                  style=settings.plot_style,
                              ) :
                              nothing,
        scenario_plot_callback=settings.plotting ?
                               (base_patient, scenario, curve_t, curve_plasma, save_path, model_cfg) -> save_truncation_fit_plot(
                                   base_patient,
                                   scenario,
                                   curve_t,
                                   curve_plasma,
                                   save_path,
                                   model_cfg;
                                   display_plots=settings.display_plots,
                                   style=settings.plot_style,
                               ) :
                               nothing,
        parameter_boxplot_callback=settings.plotting ?
                                  (patients, validation_params, save_path, model_cfg, dataset, patient_id) -> save_truncation_parameter_boxplot(
                                      patients,
                                      validation_params,
                                      save_path,
                                      model_cfg;
                                      dataset=dataset,
                                      data_label="truncated_$(patient_id)",
                                      show_outliers=true,
                                  ) :
                                  nothing,
    )
    @info "Completed ODE systematic truncation target."
end

if :cude in target_keys
    @info "Running cUDE systematic truncation target."
    cude_model_cfg = build_systematic_truncation_model_runtime(
        :cude,
        settings;
        selected_model=selected_model,
        cude_artifacts=cude_artifacts,
    )
    run_systematic_truncation_target(
        gold.patients,
        gold.patient_dataset,
        cude_model_cfg,
        settings,
        cude_paths;
        initial_plot_callback=settings.plotting ?
                              (patient, save_path, model_cfg) -> save_truncation_initial_scatter(
                                  patient,
                                  save_path,
                                  model_cfg;
                                  display_plots=settings.display_plots,
                                  style=settings.plot_style,
                              ) :
                              nothing,
        scenario_plot_callback=settings.plotting ?
                               (base_patient, scenario, curve_t, curve_plasma, save_path, model_cfg) -> save_truncation_fit_plot(
                                   base_patient,
                                   scenario,
                                   curve_t,
                                   curve_plasma,
                                   save_path,
                                   model_cfg;
                                   display_plots=settings.display_plots,
                                   style=settings.plot_style,
                               ) :
                               nothing,
        parameter_boxplot_callback=settings.plotting ?
                                  (patients, validation_params, save_path, model_cfg, dataset, patient_id) -> save_truncation_parameter_boxplot(
                                      patients,
                                      validation_params,
                                      save_path,
                                      model_cfg;
                                      dataset=dataset,
                                      data_label="truncated_$(patient_id)",
                                      show_outliers=true,
                                  ) :
                                  nothing,
    )
    @info "Completed cUDE systematic truncation target."
end

if :plots in target_keys
    @info "Regenerating systematic truncation patient-level plots from existing CSV artifacts."
    settings.plotting || @warn "Plot regeneration requested, but settings.plotting=false; no patient-level plots will be written."

    if settings.plotting
        ode_model_cfg = build_systematic_truncation_model_runtime(:ode, settings)
        cude_model_cfg = build_systematic_truncation_model_runtime(
            :cude,
            settings;
            selected_model=selected_model,
            cude_artifacts=cude_artifacts,
        )

        regenerate_systematic_truncation_model_plots(
            gold.patients,
            gold.patient_dataset,
            ode_model_cfg,
            settings,
            ode_paths;
            initial_plot_callback=(patient, save_path, model_cfg) -> save_truncation_initial_scatter(
                patient,
                save_path,
                model_cfg;
                display_plots=settings.display_plots,
                style=settings.plot_style,
            ),
            scenario_plot_callback=(base_patient, scenario, curve_t, curve_plasma, save_path, model_cfg) -> save_truncation_fit_plot(
                base_patient,
                scenario,
                curve_t,
                curve_plasma,
                save_path,
                model_cfg;
                display_plots=settings.display_plots,
                style=settings.plot_style,
            ),
            parameter_boxplot_callback=(patients, validation_params, save_path, model_cfg, dataset, patient_id) -> save_truncation_parameter_boxplot(
                patients,
                validation_params,
                save_path,
                model_cfg;
                dataset=dataset,
                data_label="truncated_$(patient_id)",
                show_outliers=true,
            ),
        )

        regenerate_systematic_truncation_model_plots(
            gold.patients,
            gold.patient_dataset,
            cude_model_cfg,
            settings,
            cude_paths;
            initial_plot_callback=(patient, save_path, model_cfg) -> save_truncation_initial_scatter(
                patient,
                save_path,
                model_cfg;
                display_plots=settings.display_plots,
                style=settings.plot_style,
            ),
            scenario_plot_callback=(base_patient, scenario, curve_t, curve_plasma, save_path, model_cfg) -> save_truncation_fit_plot(
                base_patient,
                scenario,
                curve_t,
                curve_plasma,
                save_path,
                model_cfg;
                display_plots=settings.display_plots,
                style=settings.plot_style,
            ),
            parameter_boxplot_callback=(patients, validation_params, save_path, model_cfg, dataset, patient_id) -> save_truncation_parameter_boxplot(
                patients,
                validation_params,
                save_path,
                model_cfg;
                dataset=dataset,
                data_label="truncated_$(patient_id)",
                show_outliers=true,
            ),
        )
    end
end

if :summary in target_keys
    @info "Building integrated truncation summary tables."
    ode_tables = load_systematic_truncation_model_tables(paths.ode_dir)
    cude_tables = load_systematic_truncation_model_tables(paths.cude_dir)

    metrics_summary = build_truncation_metrics_summary(ode_tables.metrics_all, cude_tables.metrics_all)
    params_summary = build_truncation_params_summary(ode_tables.params_all, cude_tables.params_all)

    CSV.write(paths.metrics_summary, metrics_summary)
    CSV.write(paths.params_summary, params_summary)
    @info "Saved integrated truncation summaries." metrics=paths.metrics_summary params=paths.params_summary
end

if (:overlay in target_keys) || (:plots in target_keys)
    @info "Building ODE-vs-cUDE truncation overlay plots."
    selected_model === nothing && error("Selected cUDE model is required for overlay plots.")
    cude_artifacts === nothing && error("cUDE artifacts are required for overlay plots.")

    chain = neural_network_model(selected_model.nn_depth, selected_model.nn_width; input_dims=settings.input_dim)
    neural_params = Vector{Float64}(cude_artifacts.neural_network_parameters[selected_model.model_idx])

    saved_overlay_paths = String[]
    saved_overlay_no_labels_paths = String[]
    saved_overlay_legend_on_paths = String[]
    for patient in gold.patients
        ode_patient_dir = joinpath(paths.ode_dir, patient.id)
        cude_patient_dir = joinpath(paths.cude_dir, patient.id)
        isdir(ode_patient_dir) || error("Missing ODE truncation patient directory: $(ode_patient_dir)")
        isdir(cude_patient_dir) || error("Missing cUDE truncation patient directory: $(cude_patient_dir)")

        patient_overlay_dir = joinpath(paths.overlay_dir, patient.id)
        patient_overlay_no_labels_dir = joinpath(paths.overlay_no_labels_dir, patient.id)
        patient_overlay_legend_on_dir = joinpath(paths.overlay_legend_on_dir, patient.id)
        records = build_truncation_overlay_records(ode_patient_dir, cude_patient_dir, chain, neural_params)

        for record in records
            saved_path = save_truncation_overlay_plot(
                record,
                patient_overlay_dir;
                plot_legend=settings.overlay_legend,
                legend_position=settings.overlay_legend_position,
                show_count_labels=settings.overlay_count_labels,
                style=settings.plot_style,
            )
            push!(saved_overlay_paths, saved_path)

            legend_on_path = save_truncation_overlay_legend_png(
                record,
                patient_overlay_legend_on_dir;
                legend_position=settings.overlay_legend_position,
                show_count_labels=settings.overlay_count_labels,
                axis_labels=true,
                style=settings.plot_style,
            )
            push!(saved_overlay_legend_on_paths, legend_on_path)

            if settings.overlay_no_labels
                no_labels_path = save_truncation_overlay_plot(
                    record,
                    patient_overlay_no_labels_dir;
                    plot_legend=settings.overlay_legend,
                    legend_position=settings.overlay_legend_position,
                    show_count_labels=settings.overlay_count_labels,
                    axis_labels=false,
                    style=settings.plot_style,
                )
                push!(saved_overlay_no_labels_paths, no_labels_path)
            end
        end

        @info "Saved overlay plots for $(patient.id)." n=length(records) no_labels=settings.overlay_no_labels legend_position=settings.overlay_legend_position count_labels=settings.overlay_count_labels
    end

    @info "Completed overlay plot generation." total=length(saved_overlay_paths) no_labels_total=length(saved_overlay_no_labels_paths) legend_on_total=length(saved_overlay_legend_on_paths)
end

@info "Systematic truncation workflow completed at $(now())."

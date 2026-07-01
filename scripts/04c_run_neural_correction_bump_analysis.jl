"""
04c_run_neural_correction_bump_analysis.jl

Run a descriptive analysis of early non-monotonicity in the selected cUDE
neural correction.

This step characterizes a local neural-network feature,
`N_phi(t_norm, beta)`, using only existing artifacts from steps 02a, 02b, 02c,
and 02d. It does not run training, fitting, or optimization.

The observed-domain analysis over `0 <= t_norm <= 1` is the primary output.
The extended-domain analysis is a separate descriptive control.

Command line:
    JULIA_NUM_THREADS=auto julia --project=. scripts/04c_run_neural_correction_bump_analysis.jl
    julia --project=. scripts/04c_run_neural_correction_bump_analysis.jl plots

The optional `plots` mode regenerates only figures from existing step 04c CSVs.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# =============================================================================

using CSV
using DataFrames: DataFrame, eachrow, leftjoin, nrow
using Logging

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "symbolic_regression.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.neural_correction_bump_analysis
execution_mode = isempty(ARGS) ? :run :
                 length(ARGS) == 1 && lowercase(strip(ARGS[1])) == "plots" ? :plots :
                 error("Usage: julia --project=. scripts/04c_run_neural_correction_bump_analysis.jl [plots]")

mimic_config = getproperty(config.datasets, settings.mimic_dataset_key)
external_config = getproperty(config.datasets, settings.external_dataset_key)
model_selection_config = getproperty(config.datasets, settings.model_selection_dataset_key)

output_paths = neural_correction_bump_analysis_output_paths(settings.output_dir)

selection_paths = cude_model_selection_output_paths(settings.model_selection_dir, model_selection_config.dataset_name)
selected_model_path = settings.selected_model_path === nothing ? selection_paths.selected_model : settings.selected_model_path

observed_tnorm_grid = collect(range(
    Float64(settings.observed_tnorm_min),
    Float64(settings.observed_tnorm_max);
    length=Int(settings.observed_tnorm_points),
))

extended_tnorm_grid = collect(range(
    Float64(settings.extended_tnorm_min),
    Float64(settings.extended_tnorm_max);
    length=Int(settings.extended_tnorm_points),
))

feature_window_tau = Tuple(Float64.(settings.feature_window_tau))
length(feature_window_tau) == 2 && feature_window_tau[1] < feature_window_tau[2] ||
    error("feature_window_tau must be a two-value increasing tuple.")
feature_window_tnorm = (
    feature_window_tau[1] / Float64(settings.t_scale),
    feature_window_tau[2] / Float64(settings.t_scale),
)
feature_settings = merge(settings, (feature_window_tnorm=feature_window_tnorm,))
feature_tnorm_grid = collect(range(
    feature_window_tnorm[1],
    feature_window_tnorm[2];
    length=Int(settings.feature_tnorm_points),
))

# =============================================================================
# OUTPUT PATHS
# =============================================================================

log_workflow_context(
    config;
    script_name="04c_run_neural_correction_bump_analysis.jl",
    output_paths=(neural_correction_bump_analysis=output_paths.output_dir,),
)
ensure_output_dirs!(
    (
        output_root=output_paths.output_dir,
        figures=output_paths.fig_dir,
    );
    header="Ensured step 04c output directories",
)
log_output_paths(
    (
        patient_feature=output_paths.patient_feature,
        cohort_summary_feature=output_paths.cohort_summary_feature,
        beta_grid_feature=output_paths.beta_grid_feature,
        patient_curve_feature=output_paths.patient_curve_feature,
        patient_observed=output_paths.patient_observed,
        cohort_summary_observed=output_paths.cohort_summary_observed,
        beta_grid_observed=output_paths.beta_grid_observed,
        patient_curve_observed=output_paths.patient_curve_observed,
        patient_extended=output_paths.patient_extended,
        cohort_summary_extended=output_paths.cohort_summary_extended,
        beta_grid_extended=output_paths.beta_grid_extended,
        patient_curve_extended=output_paths.patient_curve_extended,
        anchor_note=output_paths.anchor_note,
        figure_feature=output_paths.fig_feature_svg,
        figure_observed=output_paths.fig_observed_svg,
        figure_extended=output_paths.fig_extended_svg,
    );
    header="Step 04c output files",
)

# =============================================================================
# PLOTTING ONLY
# =============================================================================

if execution_mode === :plots
    validate_existing_paths(
        (
            patient_feature=output_paths.patient_feature,
            beta_grid_feature=output_paths.beta_grid_feature,
            patient_curve_feature=output_paths.patient_curve_feature,
            patient_observed=output_paths.patient_observed,
            beta_grid_observed=output_paths.beta_grid_observed,
            patient_curve_observed=output_paths.patient_curve_observed,
        );
        header="Required step 04c plot artifacts",
    )
    anchor_plot_df = isfile(output_paths.anchor_source_mimic) ? CSV.read(output_paths.anchor_source_mimic, DataFrame) : nothing

    feature_patient_df = CSV.read(output_paths.patient_feature, DataFrame)
    feature_grid_df = CSV.read(output_paths.beta_grid_feature, DataFrame)
    feature_patient_curve_df = CSV.read(output_paths.patient_curve_feature, DataFrame)
    save_neural_correction_bump_analysis_plot(
        feature_grid_df,
        feature_patient_curve_df,
        feature_patient_df,
        output_paths.fig_feature_svg,
        output_paths.fig_feature_png;
        anchor_df=anchor_plot_df,
        plotting=settings.plotting,
        display_plots=settings.display_plots,
        png_px_per_unit=settings.png_px_per_unit,
        time_scale=settings.t_scale,
        feature_window_tau=settings.feature_window_tau,
        bump_beta_split=settings.bump_beta_split,
        low_beta_bump_color=settings.low_beta_bump_color,
        high_beta_bump_color=settings.high_beta_bump_color,
        no_bump_point_color=settings.no_bump_point_color,
        grid_curve_color=settings.grid_curve_color,
        grid_curve_alpha=settings.grid_curve_alpha,
        highlight_curve_linewidth=settings.highlight_curve_linewidth,
    )

    observed_patient_df = CSV.read(output_paths.patient_observed, DataFrame)
    observed_grid_df = CSV.read(output_paths.beta_grid_observed, DataFrame)
    observed_patient_curve_df = CSV.read(output_paths.patient_curve_observed, DataFrame)
    save_neural_correction_bump_analysis_plot(
        observed_grid_df,
        observed_patient_curve_df,
        observed_patient_df,
        output_paths.fig_observed_svg,
        output_paths.fig_observed_png;
        anchor_df=anchor_plot_df,
        plotting=settings.plotting,
        display_plots=settings.display_plots,
        png_px_per_unit=settings.png_px_per_unit,
        time_scale=settings.t_scale,
        bump_beta_split=settings.bump_beta_split,
        low_beta_bump_color=settings.low_beta_bump_color,
        high_beta_bump_color=settings.high_beta_bump_color,
        no_bump_point_color=settings.no_bump_point_color,
        grid_curve_color=settings.grid_curve_color,
        grid_curve_alpha=settings.grid_curve_alpha,
        highlight_curve_linewidth=settings.highlight_curve_linewidth,
    )

    if isfile(output_paths.patient_extended) && isfile(output_paths.beta_grid_extended)
        extended_patient_df = CSV.read(output_paths.patient_extended, DataFrame)
        isfile(output_paths.patient_curve_extended) ||
            error("Missing patient-specific extended curve artifact: $(output_paths.patient_curve_extended). Rerun step 04c before using plots.")
        extended_grid_df = CSV.read(output_paths.beta_grid_extended, DataFrame)
        extended_patient_curve_df = CSV.read(output_paths.patient_curve_extended, DataFrame)
        save_neural_correction_bump_analysis_plot(
            extended_grid_df,
            extended_patient_curve_df,
            extended_patient_df,
            output_paths.fig_extended_svg,
            output_paths.fig_extended_png;
            anchor_df=anchor_plot_df,
            plotting=settings.plotting,
            display_plots=settings.display_plots,
            png_px_per_unit=settings.png_px_per_unit,
            time_scale=settings.t_scale,
            bump_beta_split=settings.bump_beta_split,
            low_beta_bump_color=settings.low_beta_bump_color,
            high_beta_bump_color=settings.high_beta_bump_color,
            no_bump_point_color=settings.no_bump_point_color,
            grid_curve_color=settings.grid_curve_color,
            grid_curve_alpha=settings.grid_curve_alpha,
            highlight_curve_linewidth=settings.highlight_curve_linewidth,
        )
    else
        @info "Extended-domain 04c CSVs not found; skipping extended plot regeneration."
    end

    @info "Completed step 04c plot-only regeneration."
    exit()
end

# =============================================================================
# INPUTS
# =============================================================================

@info "Loading selected cUDE model." path=selected_model_path
selected_model = load_selected_cude_model(selected_model_path)
artifacts = load_cude_training_artifacts(settings.cude_training_input_dir, selected_model.nn_width)
1 <= selected_model.model_idx <= length(artifacts.neural_network_parameters) ||
    error("Selected cUDE model_idx=$(selected_model.model_idx) is outside available candidates.")

chain = neural_network_model(selected_model.nn_depth, selected_model.nn_width; input_dims=settings.input_dim)
neural_params = Vector{Float64}(artifacts.neural_network_parameters[selected_model.model_idx])

mimic_paths = cude_evaluation_model_output_paths(
    settings.cude_evaluation_input_dir,
    selected_model.nn_width,
    selected_model.model_idx,
    mimic_config.dataset_name,
)
external_paths = cude_external_test_output_paths(
    settings.cude_external_test_input_dir,
    external_config.dataset_name,
)

validate_existing_paths(
    (
        mimic_params=mimic_paths.patients_params_val,
        external_params=external_paths.patients_params_val,
    );
    header="Required cUDE parameter artifacts for step 04c",
)

mimic_beta_df = cude_patient_beta_dataframe(
    CSV.read(mimic_paths.patients_params_val, DataFrame);
    cohort=mimic_config.dataset_name,
)
external_beta_df = cude_patient_beta_dataframe(
    CSV.read(external_paths.patients_params_val, DataFrame);
    cohort=external_config.dataset_name,
)
patient_beta_df = vcat(mimic_beta_df, external_beta_df)
beta_grid = neural_correction_beta_grid(patient_beta_df; n_points=settings.beta_grid_points)
patient_curve_beta_values = sort(unique(Float64.(patient_beta_df.beta)))

# =============================================================================
# ANALYSIS
# =============================================================================

@info "Running feature-window neural-correction local non-monotonicity analysis." feature_window_tau=settings.feature_window_tau feature_window_tnorm=feature_window_tnorm
feature_patient_df = neural_correction_bump_dataframe(
    chain,
    neural_params,
    patient_beta_df,
    feature_tnorm_grid,
    feature_settings;
    domain="feature",
    classification=:feature,
)
feature_grid_df = neural_correction_beta_grid_dataframe(
    chain,
    neural_params,
    beta_grid,
    feature_tnorm_grid,
    feature_settings;
    domain="feature",
    classification=:feature,
)
feature_patient_curve_df = neural_correction_beta_grid_dataframe(
    chain,
    neural_params,
    patient_curve_beta_values,
    feature_tnorm_grid,
    feature_settings;
    domain="feature",
    classification=:feature,
)
feature_summary_df = neural_correction_bump_summary(feature_patient_df)

CSV.write(output_paths.patient_feature, feature_patient_df)
CSV.write(output_paths.beta_grid_feature, feature_grid_df)
CSV.write(output_paths.patient_curve_feature, feature_patient_curve_df)
CSV.write(output_paths.cohort_summary_feature, feature_summary_df)

@info "Running observed-domain neural-correction feature analysis."
observed_patient_df = neural_correction_bump_dataframe(
    chain,
    neural_params,
    patient_beta_df,
    observed_tnorm_grid,
    feature_settings;
    domain="observed",
    classification=:feature,
)
observed_grid_df = neural_correction_beta_grid_dataframe(
    chain,
    neural_params,
    beta_grid,
    observed_tnorm_grid,
    feature_settings;
    domain="observed",
    classification=:feature,
)
observed_patient_curve_df = neural_correction_beta_grid_dataframe(
    chain,
    neural_params,
    patient_curve_beta_values,
    observed_tnorm_grid,
    feature_settings;
    domain="observed",
    classification=:feature,
)
observed_summary_df = neural_correction_bump_summary(observed_patient_df)

CSV.write(output_paths.patient_observed, observed_patient_df)
CSV.write(output_paths.beta_grid_observed, observed_grid_df)
CSV.write(output_paths.patient_curve_observed, observed_patient_curve_df)
CSV.write(output_paths.cohort_summary_observed, observed_summary_df)

if settings.extended_analysis
    @info "Running extended-domain neural-correction feature analysis."
    extended_patient_df = neural_correction_bump_dataframe(
        chain,
        neural_params,
        patient_beta_df,
        extended_tnorm_grid,
        feature_settings;
        domain="extended",
        classification=:feature,
    )
    extended_grid_df = neural_correction_beta_grid_dataframe(
        chain,
        neural_params,
        beta_grid,
        extended_tnorm_grid,
        feature_settings;
        domain="extended",
        classification=:feature,
    )
    extended_patient_curve_df = neural_correction_beta_grid_dataframe(
        chain,
        neural_params,
        patient_curve_beta_values,
        extended_tnorm_grid,
        feature_settings;
        domain="extended",
        classification=:feature,
    )
    extended_summary_df = neural_correction_bump_summary(extended_patient_df)

    CSV.write(output_paths.patient_extended, extended_patient_df)
    CSV.write(output_paths.beta_grid_extended, extended_grid_df)
    CSV.write(output_paths.patient_curve_extended, extended_patient_curve_df)
    CSV.write(output_paths.cohort_summary_extended, extended_summary_df)
else
    extended_patient_df = DataFrame()
    extended_grid_df = DataFrame()
    extended_patient_curve_df = DataFrame()
    extended_summary_df = DataFrame()
end

# =============================================================================
# ANCHORING METADATA
# =============================================================================

anchor_plot_df = nothing
if settings.anchoring_analysis
    anchor_required = (
        settings.mimic_anchor_patient_file,
        settings.mimic_anchor_troponin_csv,
        settings.mimic_anchor_admission_csv,
    )

    if isfile(output_paths.anchor_source_mimic)
        @info "Loading existing MIMIC-IV anchoring metadata." path=output_paths.anchor_source_mimic
        anchor_table = CSV.read(output_paths.anchor_source_mimic, DataFrame)
    elseif all(isfile, anchor_required)
        @info "Reconstructing MIMIC-IV anchoring metadata from configured raw inputs."
        anchor_table = reconstruct_mimic_anchor_source_table(
            patient_file=settings.mimic_anchor_patient_file,
            troponin_csv=settings.mimic_anchor_troponin_csv,
            admission_csv=settings.mimic_anchor_admission_csv,
        )
    else
        missing_inputs = [path for path in anchor_required if !isfile(path)]
        open(output_paths.anchor_note, "w") do io
            println(io, "anchor-source metadata not available for this diagnostic")
            println(io, "Missing configured raw input paths:")
            foreach(path -> println(io, path), missing_inputs)
        end
        @warn "Anchoring analysis requested but raw inputs are missing; proceeding without anchoring." missing_inputs=missing_inputs
        anchor_table = nothing
    end

    if anchor_table !== nothing
        anchor_columns = [
            column for column in
            [:patient_id, :source_id, :subject_id, :hadm_id, :anchor_source, :anchor_match_status, :anchor_match_error, :anchor_match_n_points]
            if column in propertynames(anchor_table)
        ]
        :patient_id in anchor_columns ||
            error("Existing anchoring table must contain a patient_id column: $(output_paths.anchor_source_mimic)")

        mimic_feature = feature_patient_df[feature_patient_df.cohort .== mimic_config.dataset_name, :]
        anchor_join = leftjoin(mimic_feature, anchor_table[:, anchor_columns], on=:patient_id)
        CSV.write(output_paths.anchor_source_mimic, anchor_join)
        anchor_plot_df = anchor_join
        open(output_paths.anchor_note, "w") do io
            println(io, "anchor-source metadata available for MIMIC-IV")
            println(io, "source: $(isfile(output_paths.anchor_source_mimic) ? output_paths.anchor_source_mimic : "raw inputs")")
            println(io, "UMG anchor-source metadata not available for this diagnostic")
        end
    end
else
    open(output_paths.anchor_note, "w") do io
        println(io, "anchor-source metadata analysis disabled by config")
    end
end

# =============================================================================
# FIGURES AND CONSOLE SUMMARY
# =============================================================================

save_neural_correction_bump_analysis_plot(
    feature_grid_df,
    feature_patient_curve_df,
    feature_patient_df,
    output_paths.fig_feature_svg,
    output_paths.fig_feature_png;
    anchor_df=anchor_plot_df,
    plotting=settings.plotting,
    display_plots=settings.display_plots,
    png_px_per_unit=settings.png_px_per_unit,
    time_scale=settings.t_scale,
    feature_window_tau=settings.feature_window_tau,
    bump_beta_split=settings.bump_beta_split,
    low_beta_bump_color=settings.low_beta_bump_color,
    high_beta_bump_color=settings.high_beta_bump_color,
    no_bump_point_color=settings.no_bump_point_color,
    grid_curve_color=settings.grid_curve_color,
    grid_curve_alpha=settings.grid_curve_alpha,
    highlight_curve_linewidth=settings.highlight_curve_linewidth,
)

save_neural_correction_bump_analysis_plot(
    observed_grid_df,
    observed_patient_curve_df,
    observed_patient_df,
    output_paths.fig_observed_svg,
    output_paths.fig_observed_png;
    anchor_df=anchor_plot_df,
    plotting=settings.plotting,
    display_plots=settings.display_plots,
    png_px_per_unit=settings.png_px_per_unit,
    time_scale=settings.t_scale,
    bump_beta_split=settings.bump_beta_split,
    low_beta_bump_color=settings.low_beta_bump_color,
    high_beta_bump_color=settings.high_beta_bump_color,
    no_bump_point_color=settings.no_bump_point_color,
    grid_curve_color=settings.grid_curve_color,
    grid_curve_alpha=settings.grid_curve_alpha,
    highlight_curve_linewidth=settings.highlight_curve_linewidth,
)

if settings.extended_analysis
    save_neural_correction_bump_analysis_plot(
        extended_grid_df,
        extended_patient_curve_df,
        extended_patient_df,
        output_paths.fig_extended_svg,
        output_paths.fig_extended_png;
        anchor_df=anchor_plot_df,
        plotting=settings.plotting,
        display_plots=settings.display_plots,
        png_px_per_unit=settings.png_px_per_unit,
        time_scale=settings.t_scale,
        bump_beta_split=settings.bump_beta_split,
        low_beta_bump_color=settings.low_beta_bump_color,
        high_beta_bump_color=settings.high_beta_bump_color,
        no_bump_point_color=settings.no_bump_point_color,
        grid_curve_color=settings.grid_curve_color,
        grid_curve_alpha=settings.grid_curve_alpha,
        highlight_curve_linewidth=settings.highlight_curve_linewidth,
    )
end

for row in eachrow(feature_summary_df)
    @info "Feature-window bump summary." cohort=row.cohort n_patients=row.n_patients n_bump=row.n_bump n_no_bump=row.n_no_bump pct_bump=row.pct_bump
end

feature_bump_grid = feature_grid_df[feature_grid_df.bump_flag .== true, :]
if nrow(feature_bump_grid) == 0
    @info "Feature-window beta grid: no bump-classified curves."
else
    @info "Feature-window beta grid bump range." beta_min=minimum(feature_bump_grid.beta) beta_max=maximum(feature_bump_grid.beta)
end

for row in eachrow(observed_summary_df)
    @info "Observed-domain bump summary." cohort=row.cohort n_patients=row.n_patients n_bump=row.n_bump n_no_bump=row.n_no_bump pct_bump=row.pct_bump
end

observed_bump_grid = observed_grid_df[observed_grid_df.bump_flag .== true, :]
if nrow(observed_bump_grid) == 0
    @info "Observed-domain beta grid: no bump-classified curves."
else
    @info "Observed-domain beta grid bump range." beta_min=minimum(observed_bump_grid.beta) beta_max=maximum(observed_bump_grid.beta)
end

if settings.extended_analysis
    extended_bump_grid = extended_grid_df[extended_grid_df.bump_flag .== true, :]
    if nrow(extended_bump_grid) == 0
        @info "Extended-domain beta grid: no bump-classified curves."
    else
        @info "Extended-domain beta grid bump range." beta_min=minimum(extended_bump_grid.beta) beta_max=maximum(extended_bump_grid.beta)
    end
end

@info "Completed step 04c neural-correction early bump analysis."

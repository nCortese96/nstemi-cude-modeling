"""
data_io.jl

Workflow IO and patient-level input/output helpers.

Sections:
- Workflow IO: logging and output-directory helpers.
- Patient Data IO: `PatientData`, spreadsheet readers, and DataFrame converters.
- Preprocessed Cohort IO: JLD2/CSV cohort artifact readers.
- Workflow Output Artifacts: canonical path builders and artifact writers.
- Systematic Truncation IO: step 03c target, cohort, and table helpers.
- Symbolic Regression IO: step 04a output paths and stable artifact writers.
- Profile Likelihood IO: step 03b target, input, output, and CSV helpers.
"""

using CSV, XLSX
using DataFrames: DataFrame, DataFrameRow, groupby, nrow
using JLD2
using Logging

# =============================================================================
# Workflow IO
# =============================================================================

"""
    ensure_output_dirs!(output_dirs; header="Ensured output directories")

Input: a named tuple or dictionary mapping labels to output directories.
Output: created directories and console messages with absolute paths.
"""
function ensure_output_dirs!(output_dirs; header::AbstractString="Ensured output directories")
    has_dirs = false

    for (label, dir) in pairs(output_dirs)
        mkpath(string(dir))

        if !has_dirs
            @info header
            has_dirs = true
        end

        @info "  $(label): $(abspath(string(dir)))"
    end

    return output_dirs
end

ensure_output_dirs!(output_dir::AbstractString; header::AbstractString="Ensured output directories") =
    ensure_output_dirs!((output_dir=output_dir,); header=header)

"""
    log_output_paths(output_paths; header="Output paths")

Input: a named tuple or dictionary mapping labels to output paths.
Output: console messages listing absolute save paths.
"""
function log_output_paths(output_paths; header::AbstractString="Output paths")
    has_paths = false

    for (label, path) in pairs(output_paths)
        if !has_paths
            @info header
            has_paths = true
        end

        @info "  $(label): $(abspath(string(path)))"
    end

    return nothing
end

"""
    log_workflow_context(config; script_name, output_paths=NamedTuple())

Input: `WORKFLOW_CONFIG`, the running script name, and optional output paths.
Output: console messages describing run mode and active result location.
"""
function log_workflow_context(
    config;
    script_name::AbstractString,
    output_paths=NamedTuple(),
)
    @info "Workflow script: $(script_name)"

    if config.run.test_mode
        @warn "TEST MODE ACTIVE: outputs are redirected to $(config.paths.active_results_root)"
    else
        @info "Workflow mode: standard"
    end

    @info "Active results root: $(abspath(config.paths.active_results_root))"
    log_output_paths(output_paths; header="Configured output paths")

    return nothing
end

"""
    resolve_dataset_configs(config, dataset_keys)

Return dataset configuration entries selected from `config.datasets` by the
ordered `dataset_keys` collection.
"""
# Used by: scripts/00_run_preprocessing.jl, scripts/01_run_ode_tdsigmoid_fit.jl. Planned use: future workflow scripts.
function resolve_dataset_configs(config, dataset_keys)
    return [getproperty(config.datasets, key) for key in dataset_keys]
end

# =============================================================================
# Patient Data IO
# =============================================================================

"""
    PatientData

Input: patient identifier, acquisition times, and cTnT observations.
Output: mutable patient trajectory container used across preprocessing and models.
"""
struct PatientData
    id::String
    timepoints::Vector{Float64}
    ctnt_data::Vector{Float64}
end

function row2Patient(id::String, timepoints_df::AbstractVector, troponin_df::AbstractVector)
    tp_row = [x for x in collect(values(timepoints_df)) if !ismissing(x)]
    ctnt_row = [x for x in collect(values(troponin_df)) if !ismissing(x)]
    return PatientData(id, tp_row, ctnt_row)
end

function row2Patient(ids::DataFrameRow, timepoints_df::DataFrameRow, troponin_df::DataFrameRow)
    id_val = ids[1]
    tp_row = [x for x in collect(values(timepoints_df)) if !ismissing(x)]
    ctnt_row = [x for x in collect(values(troponin_df)) if !ismissing(x)]
    return PatientData(id_val, tp_row, ctnt_row)
end

function fromPatientData2DataFrame(patients::Vector{PatientData}; save::Bool=false, save_path::String="pazienti_long.csv")
    ids = String[]
    times = Float64[]
    values = Float64[]

    for p in patients
        n = length(p.timepoints)
        append!(ids, fill(p.id, n))
        append!(times, p.timepoints)
        append!(values, p.ctnt_data)
    end

    df = DataFrame(patient_id=ids, time=times, troponin=values)

    if save
        CSV.write(save_path, df)
    end
    return df
end

function fromDataFrame2PatientData(df::DataFrame)
    patients_reloaded = PatientData[]

    for gdf in groupby(df, :patient_id)
        push!(patients_reloaded, PatientData(
            gdf.patient_id[1],
            gdf.time,
            gdf.troponin
        ))
    end

    return patients_reloaded
end

function load_excel_patients(dataset_path::AbstractString, column_letter::AbstractString; data_root::AbstractString="data")
    file_path = joinpath(data_root, dataset_path)
    sheet_ids = "IDs"
    sheet_times = "times"
    sheet_values = "values"
    ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true))
    timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true))
    troponin_df = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true))
    patients = [row2Patient(ids[i, :], timepoints_df[i, :], troponin_df[i, :]) for i in 1:nrow(ids)]
    return patients, (file_path=file_path, ids=ids, timepoints_df=timepoints_df, troponin_df=troponin_df)
end

# =============================================================================
# Preprocessed Cohort IO
# =============================================================================

"""
    load_patient_ids(ids_csv_path)

Read an ID CSV and return patient identifiers as strings. The reader accepts
`patient`, `patient_id`, `id`, or the first column as the identifier column.
"""
# Used by: src/data_io.jl (load_preprocessed_cohort).
function load_patient_ids(ids_csv_path::AbstractString)
    ids_df = CSV.read(ids_csv_path, DataFrame)
    names = propertynames(ids_df)
    id_col = if :patient in names
        :patient
    elseif :patient_id in names
        :patient_id
    elseif :id in names
        :id
    else
        names[1]
    end

    return string.(ids_df[!, id_col])
end

"""
    order_patients_by_ids(patients, ids; label="cohort")

Return patients ordered by a source-of-truth ID list, erroring if any requested
ID is missing from the loaded cohort.
"""
# Used by: src/data_io.jl (load_preprocessed_cohort).
function order_patients_by_ids(patients::AbstractVector{PatientData}, ids::AbstractVector{<:AbstractString}; label::AbstractString="cohort")
    patient_map = Dict(p.id => p for p in patients)

    if length(patient_map) != length(patients)
        @warn "Duplicate patient IDs detected while ordering $(label)."
    end

    missing_ids = [id for id in ids if !haskey(patient_map, id)]
    isempty(missing_ids) || error("Missing $(length(missing_ids)) patients while ordering $(label): $(first(missing_ids))")

    extras = setdiff(collect(keys(patient_map)), collect(ids))
    if !isempty(extras)
        @warn "Ignoring $(length(extras)) patients not present in the ordered ID list for $(label)."
    end

    return [patient_map[id] for id in ids]
end

"""
    load_preprocessed_cohort(dataset_name, cohort_dir)

Load step 00 JLD2/CSV artifacts for one dataset and return ordered patients,
source IDs, split datasets, validation IDs, and source paths.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl, scripts/02a_run_cude_training.jl, scripts/02d_evaluate_cude_nn_external_test.jl.
function load_preprocessed_cohort(dataset_name::AbstractString, cohort_dir::AbstractString)
    ids_csv_path = joinpath(cohort_dir, "ids_all_eligible_$(dataset_name).csv")
    isfile(ids_csv_path) || error("Missing ordered ID file: $(ids_csv_path)")

    ordered_ids = load_patient_ids(ids_csv_path)

    if dataset_name == "MIMIC-IV"
        training_path = joinpath(cohort_dir, "MIMIC-IV_trainingset.jld2")
        test_path = joinpath(cohort_dir, "MIMIC-IV_testset.jld2")
        isfile(training_path) || error("Missing MIMIC-IV training cohort: $(training_path)")
        isfile(test_path) || error("Missing MIMIC-IV test cohort: $(test_path)")

        training = JLD2.load(training_path, "training_dataset")
        test = JLD2.load(test_path, "test_dataset")
        patients = order_patients_by_ids(vcat(training, test), ordered_ids; label=dataset_name)
        validation_ids = [p.id for p in test]

        return (
            patients=patients,
            ids=ordered_ids,
            training=training,
            test=test,
            validation_ids=validation_ids,
            source_paths=(ids=ids_csv_path, training=training_path, test=test_path),
        )
    end

    test_path = joinpath(cohort_dir, "$(dataset_name)_testset.jld2")
    isfile(test_path) || error("Missing $(dataset_name) cohort: $(test_path)")

    test = JLD2.load(test_path, "test_dataset")
    patients = order_patients_by_ids(test, ordered_ids; label=dataset_name)

    return (
        patients=patients,
        ids=ordered_ids,
        test=test,
        validation_ids=String[],
        source_paths=(ids=ids_csv_path, test=test_path),
    )
end

# =============================================================================
# Workflow Output Artifacts
# =============================================================================

"""
    ode_dataset_output_paths(output_root, dataset_name)

Return canonical step 01 output paths for one dataset.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function ode_dataset_output_paths(output_root::AbstractString, dataset_name::AbstractString)
    dataset_dir = joinpath(output_root, "$(dataset_name)_opt")
    return (
        dataset_dir=dataset_dir,
        fig_dir=joinpath(dataset_dir, "figs"),
        params_csv=joinpath(dataset_dir, "params_out.csv"),
        params_val_csv=joinpath(dataset_dir, "params_out_val.csv"),
    )
end

"""
    cude_training_output_paths(output_root, width)

Return canonical step 02a output paths for one cUDE neural-network width.
"""
# Used by: scripts/02a_run_cude_training.jl.
function cude_training_output_paths(output_root::AbstractString, width::Integer)
    width_dir = joinpath(output_root, "width_$(width)")
    return (
        width_dir=width_dir,
        fig_dir=joinpath(width_dir, "figs"),
        init_params=joinpath(width_dir, "init_params.jld2"),
        losses=joinpath(width_dir, "losses.jld2"),
        nn_weights=joinpath(width_dir, "nn_weights.jld2"),
        train_params=joinpath(width_dir, "train_params.jld2"),
        report=joinpath(width_dir, "training_report.txt"),
    )
end

"""
    cude_evaluation_width_output_paths(output_root, width, dataset_name)

Return canonical step 02b width-level output paths.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl.
function cude_evaluation_width_output_paths(output_root::AbstractString, width::Integer, dataset_name::AbstractString)
    width_dir = joinpath(output_root, "width_$(width)")
    return (
        width_dir=width_dir,
        summary_csv=joinpath(width_dir, "models_summary_$(dataset_name).csv"),
    )
end

"""
    cude_evaluation_model_output_paths(output_root, width, model_idx, dataset_name)

Return canonical step 02b model-evaluation output paths.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl.
function cude_evaluation_model_output_paths(
    output_root::AbstractString,
    width::Integer,
    model_idx::Integer,
    dataset_name::AbstractString,
)
    width_paths = cude_evaluation_width_output_paths(output_root, width, dataset_name)
    eval_dir = joinpath(width_paths.width_dir, "eval_$(model_idx)")
    profiles_dir = joinpath(eval_dir, "profiles")

    return merge(width_paths, (
        eval_dir=eval_dir,
        profiles_dir=profiles_dir,
        best_params=joinpath(eval_dir, "best_params_val_$(dataset_name).jld2"),
        patients_params_train=joinpath(eval_dir, "patients_params_train.csv"),
        patients_params_val=joinpath(eval_dir, "patients_params_val.csv"),
        patients_metrics_val=joinpath(eval_dir, "patients_metrics_val.csv"),
        correction_function=joinpath(eval_dir, "correction_function.png"),
        training_params_distribution=joinpath(eval_dir, "training_params_distribution_$(dataset_name).svg"),
        validation_params_distribution=joinpath(eval_dir, "validation_params_distribution_$(dataset_name).svg"),
    ))
end

"""
    cude_external_test_output_paths(output_root, dataset_name)

Return canonical step 02d external-test output paths. The official 02d tree is
flat at the step root, with patient plots collected under `profiles/`.
"""
# Used by: scripts/02d_evaluate_cude_nn_external_test.jl.
function cude_external_test_output_paths(output_root::AbstractString, dataset_name::AbstractString)
    return (
        output_dir=output_root,
        eval_dir=output_root,
        profiles_dir=joinpath(output_root, "profiles"),
        best_params=joinpath(output_root, "best_params_val_$(dataset_name).jld2"),
        patients_params_train=joinpath(output_root, "patients_params_train.csv"),
        patients_params_val=joinpath(output_root, "patients_params_val.csv"),
        patients_metrics_val=joinpath(output_root, "patients_metrics_val.csv"),
    )
end

"""
    model_diagnostics_output_paths(output_root)

Return canonical step 03a diagnostic output paths.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function model_diagnostics_output_paths(output_root::AbstractString)
    fig_dir = joinpath(output_root, "figs")
    residuals_fig_dir = joinpath(fig_dir, "residuals")
    boxplots_fig_dir = joinpath(fig_dir, "boxplots")
    metrics_comparison_fig_dir = joinpath(fig_dir, "metrics_comparison_paper")
    profiles_comparison_dir = joinpath(fig_dir, "profiles_comparison")

    return (
        output_dir=output_root,
        fig_dir=fig_dir,
        residuals_fig_dir=residuals_fig_dir,
        boxplots_fig_dir=boxplots_fig_dir,
        metrics_comparison_fig_dir=metrics_comparison_fig_dir,
        profiles_comparison_dir=profiles_comparison_dir,
        delta_smape_report=joinpath(output_root, "delta_smape_report.txt"),
        residuals_ode_mimic=joinpath(output_root, "residuals_ODE_MIMIC.csv"),
        residuals_ode_umg=joinpath(output_root, "residuals_ODE_UMG.csv"),
        residuals_cude_mimic=joinpath(output_root, "residuals_cUDE_MIMIC.csv"),
        residuals_cude_umg=joinpath(output_root, "residuals_cUDE_UMG.csv"),
        metrics_ode_mimic=joinpath(output_root, "metrics_ODE_MIMIC.csv"),
        metrics_ode_umg=joinpath(output_root, "metrics_ODE_UMG.csv"),
        metrics_cude_mimic=joinpath(output_root, "metrics_cUDE_MIMIC.csv"),
        metrics_cude_umg=joinpath(output_root, "metrics_cUDE_UMG.csv"),
        metrics_summary=joinpath(output_root, "metrics_summary.csv"),
        parameter_summary=joinpath(output_root, "parameter_summary.csv"),
    )
end

"""
    model_diagnostics_input_paths(config, settings, selected_model)

Return step 03a input paths derived from workflow configuration and the
selected cUDE model row produced by step 02c.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function model_diagnostics_input_paths(config, settings, selected_model)
    mimic_name = config.datasets[settings.mimic_dataset_key].dataset_name
    external_name = config.datasets[settings.external_dataset_key].dataset_name

    ode_mimic_paths = ode_dataset_output_paths(settings.ode_input_dir, mimic_name)
    ode_external_paths = ode_dataset_output_paths(settings.ode_input_dir, external_name)
    cude_mimic_paths = cude_evaluation_model_output_paths(
        settings.cude_evaluation_input_dir,
        selected_model.nn_width,
        selected_model.model_idx,
        mimic_name,
    )
    cude_external_paths = cude_external_test_output_paths(settings.cude_external_test_input_dir, external_name)

    return (
        ode_mimic_params=ode_mimic_paths.params_val_csv,
        ode_external_params=ode_external_paths.params_csv,
        cude_mimic_params=cude_mimic_paths.patients_params_val,
        cude_mimic_metrics=cude_mimic_paths.patients_metrics_val,
        cude_external_params=cude_external_paths.patients_params_val,
        cude_external_metrics=cude_external_paths.patients_metrics_val,
    )
end

"""
    validate_existing_paths(paths; header="Required input paths")

Error if any path in a named tuple or dictionary does not exist.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function validate_existing_paths(paths; header::AbstractString="Required input paths")
    missing_paths = String[]

    @info header
    for (label, path) in pairs(paths)
        @info "  $(label): $(abspath(string(path)))"
        isfile(string(path)) || push!(missing_paths, string(path))
    end

    isempty(missing_paths) ||
        error("Missing required input paths:\n" * join(missing_paths, "\n"))

    return paths
end

"""
    save_model_diagnostics_tables(paths; ...)

Write the canonical step 03a residual, metric, and summary tables.
"""
# Used by: scripts/03a_run_model_diagnostics.jl.
function save_model_diagnostics_tables(
    paths;
    residuals_ode_mimic,
    residuals_ode_umg,
    residuals_cude_mimic,
    residuals_cude_umg,
    metrics_ode_mimic,
    metrics_ode_umg,
    metrics_cude_mimic,
    metrics_cude_umg,
    metrics_summary,
    parameter_summary,
)
    mkpath(paths.output_dir)

    CSV.write(paths.residuals_ode_mimic, residuals_ode_mimic)
    CSV.write(paths.residuals_ode_umg, residuals_ode_umg)
    CSV.write(paths.residuals_cude_mimic, residuals_cude_mimic)
    CSV.write(paths.residuals_cude_umg, residuals_cude_umg)

    CSV.write(paths.metrics_ode_mimic, metrics_ode_mimic)
    CSV.write(paths.metrics_ode_umg, metrics_ode_umg)
    CSV.write(paths.metrics_cude_mimic, metrics_cude_mimic)
    CSV.write(paths.metrics_cude_umg, metrics_cude_umg)

    CSV.write(paths.metrics_summary, metrics_summary)
    CSV.write(paths.parameter_summary, parameter_summary)

    return paths
end

"""
    cude_model_selection_output_paths(output_root, dataset_name)

Return canonical step 02c output paths for cUDE model selection.
"""
# Used by: scripts/02c_grid_search.jl.
function cude_model_selection_output_paths(output_root::AbstractString, dataset_name::AbstractString)
    fig_dir = joinpath(output_root, "figs")
    plot_paths = (
        smape_median_vs_iqr=joinpath(fig_dir, "plot_smape_median_vs_iqr.png"),
        tiebreak_rmsle_vs_loss=joinpath(fig_dir, "plot_tiebreak_rmsle_vs_loss.png"),
        smape_mean_std_ranked=joinpath(fig_dir, "plot_smape_mean_std_ranked.png"),
        rmsle_mean_std_ranked=joinpath(fig_dir, "plot_rmsle_mean_std_ranked.png"),
        loss_mean_std_ranked=joinpath(fig_dir, "plot_loss_mean_std_ranked.png"),
        smape_interval_ranked=joinpath(fig_dir, "plot_smape_interval_ranked.png"),
        rmsle_interval_ranked=joinpath(fig_dir, "plot_rmsle_interval_ranked.png"),
        loss_interval_ranked=joinpath(fig_dir, "plot_loss_interval_ranked.png"),
        mean_vs_median_dumbbell=joinpath(fig_dir, "plot_mean_vs_median_dumbbell.png"),
    )

    return (
        output_dir=output_root,
        fig_dir=fig_dir,
        general_summary=joinpath(output_root, "general_summary_$(dataset_name).csv"),
        selected_model=joinpath(output_root, "robust_selected_model_$(dataset_name).csv"),
        best_by_width=joinpath(output_root, "robust_best_by_width_$(dataset_name).csv"),
        report=joinpath(output_root, "robust_selection_report.txt"),
        plots=plot_paths,
    )
end

"""
    load_cude_model_summaries(input_dir, widths, dataset_name)

Load all step 02b `models_summary` CSV files required by step 02c.
"""
# Used by: scripts/02c_grid_search.jl.
function load_cude_model_summaries(input_dir::AbstractString, widths, dataset_name::AbstractString)
    chunks = DataFrame[]
    missing_paths = String[]

    for width in widths
        summary_path = cude_evaluation_width_output_paths(input_dir, width, dataset_name).summary_csv
        if !isfile(summary_path)
            push!(missing_paths, summary_path)
            continue
        end

        push!(chunks, CSV.read(summary_path, DataFrame))
    end

    isempty(missing_paths) ||
        error("Missing required cUDE model-summary files:\n" * join(missing_paths, "\n"))

    isempty(chunks) && error("No cUDE model-summary files were loaded from $(input_dir).")
    return vcat(chunks...; cols=:union)
end

"""
    save_cude_model_selection_outputs(paths, selection)

Write canonical step 02c CSV outputs.
"""
# Used by: scripts/02c_grid_search.jl.
function save_cude_model_selection_outputs(paths, selection)
    mkpath(paths.output_dir)
    CSV.write(paths.general_summary, selection.general_summary)
    CSV.write(paths.selected_model, selection.selected_model)
    CSV.write(paths.best_by_width, selection.best_by_width)
    return paths
end

"""
    write_cude_model_selection_report(path, selection; ...)

Write a compact human-readable report for step 02c model selection.
"""
# Used by: scripts/02c_grid_search.jl.
function write_cude_model_selection_report(
    path::AbstractString,
    selection;
    dataset_name::AbstractString,
    widths,
    input_dir::AbstractString,
    output_dir::AbstractString,
)
    mkpath(dirname(path))
    selected_for_report = selection.selected_model
    best_by_width_for_report = selection.best_by_width

    open(path, "w") do io
        println(io, "Robust selection from models_summary")
        println(io, "====================================")
        println(io, "Output directory: $(abspath(output_dir))")
        println(io, "Dataset: $(dataset_name)")
        println(io, "Widths considered: $(collect(widths))")
        println(io)
        println(io, "Rule:")
        println(io, "1) choose minimum $(selection.selection_columns[1])")
        println(io, "2) tie-break on $(selection.selection_columns[2])")
        println(io, "3) tie-break on $(selection.selection_columns[3]), then $(selection.selection_columns[4])")
        println(io, "4) tie-break on $(selection.selection_columns[5]), then $(selection.selection_columns[6])")
        println(io)
        println(io, "Selected model:")
        show(io, MIME"text/plain"(), selected_for_report)
        println(io)
        println(io)
        println(io, "Best model by width:")
        show(io, MIME"text/plain"(), best_by_width_for_report)
        println(io)
    end

    return path
end

"""
    load_selected_cude_model(path)

Load the single selected cUDE model row written by step 02c.
"""
# Used by: scripts/02d_evaluate_cude_nn_external_test.jl.
function load_selected_cude_model(path::AbstractString)
    isfile(path) || error("Missing selected cUDE model file: $(path)")

    df = CSV.read(path, DataFrame)
    nrow(df) == 1 || error("Expected exactly one selected cUDE model row in $(path), found $(nrow(df)).")

    row = df[1, :]
    return (
        model_id=String(row.model_id),
        model_idx=Int(row.model_idx),
        nn_depth=Int(row.nn_depth),
        nn_width=Int(row.nn_width),
        row=DataFrame(df),
        source_path=path,
    )
end

"""
    load_cude_training_artifacts(training_input_dir, width)

Load the stable step 02a JLD2 artifacts for one cUDE width.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl, scripts/02d_evaluate_cude_nn_external_test.jl.
function load_cude_training_artifacts(training_input_dir::AbstractString, width::Integer)
    paths = cude_training_output_paths(training_input_dir, width)

    for path in (paths.init_params, paths.losses, paths.nn_weights, paths.train_params)
        isfile(path) || error("Missing cUDE training artifact: $(path)")
    end

    return (
        paths=paths,
        out_params=JLD2.load(paths.init_params, "out_params"),
        losses_per_model=JLD2.load(paths.losses, "losses_per_model"),
        neural_network_parameters=JLD2.load(paths.nn_weights, "neural_network_parameters"),
        ode_params=JLD2.load(paths.train_params, "ode_params"),
    )
end

"""
    load_existing_cude_initial_parameters(paths; enabled)

Load previously selected cUDE training initial parameters from the current
width output directory when reuse mode is enabled. Returns `nothing` when the
mode is disabled or when `init_params.jld2` is not available.
"""
# Used by: scripts/02a_run_cude_training.jl.
function load_existing_cude_initial_parameters(paths; enabled::Bool)
    if !enabled
        return nothing
    end

    if !isfile(paths.init_params)
        @warn "cUDE initial-parameter reuse is enabled, but init_params.jld2 was not found. New initial parameters will be generated." path=paths.init_params
        return nothing
    end

    @info "Loading existing cUDE initial parameters." path=abspath(paths.init_params)
    return JLD2.load(paths.init_params, "out_params")
end

"""
    ode_fit_results_dataframe(results)

Build the step 01 parameter and metric DataFrame from patient fit results.
"""
# Used by: src/data_io.jl (save_ode_fit_results).
function ode_fit_results_dataframe(results)
    isempty(results) && return DataFrame(patient=String[], smape=Float64[], rmsle=Float64[], loss=Float64[])

    n_params = maximum(length(result.params) for result in results)
    df = DataFrame(
        patient=[result.patient for result in results],
        smape=[result.smape for result in results],
        rmsle=[result.rmsle for result in results],
        loss=[result.loss for result in results],
    )

    for k in 1:n_params
        df[!, Symbol("p$(k)")] = [k <= length(result.params) ? result.params[k] : NaN for result in results]
    end

    return df
end

"""
    subset_results_by_patient_ids(df, ids)

Return the subset of fit results whose patients are listed in `ids`, preserving
the original order of `df`.
"""
# Used by: src/data_io.jl (save_ode_fit_results).
function subset_results_by_patient_ids(df::DataFrame, ids::AbstractVector{<:AbstractString})
    requested_ids = Set(ids)
    fitted_ids = Set(string.(df.patient))
    missing_ids = setdiff(requested_ids, fitted_ids)
    isempty(missing_ids) || error("Cannot build validation subset. Missing fitted patient ID: $(first(missing_ids))")

    subset_idx = findall(patient -> string(patient) in requested_ids, df.patient)
    return df[subset_idx, :]
end

"""
    save_ode_fit_results(paths, results; validation_ids=String[])

Write step 01 full fit results and optional MIMIC-IV validation subset.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function save_ode_fit_results(paths, results; validation_ids::AbstractVector{<:AbstractString}=String[])
    df = ode_fit_results_dataframe(results)
    CSV.write(paths.params_csv, df)

    validation_df = nothing
    if !isempty(validation_ids)
        validation_df = subset_results_by_patient_ids(df, validation_ids)
        CSV.write(paths.params_val_csv, validation_df)
    end

    return (all=df, validation=validation_df)
end

"""
    save_patient_metrics(path, ids, smapes, rmsles, losses)

Write patient-level metrics to CSV and return the saved DataFrame.
"""
# Planned use: scripts/02b_evaluate_cude_nn.jl and downstream evaluation scripts.
function save_patient_metrics(path::AbstractString, ids, smapes, rmsles, losses)
    mkpath(dirname(path))
    df = DataFrame(patient_id=ids, smape=smapes, rmsle=rmsles, loss=losses)
    CSV.write(path, df)
    return df
end

"""
    cude_patient_metrics_dataframe(ids, smapes, rmsles, losses)

Build the canonical patient-level metric table for cUDE evaluation.
"""
# Used by: src/data_io.jl (save_cude_evaluation_artifacts).
function cude_patient_metrics_dataframe(ids, smapes, rmsles, losses)
    return DataFrame(patient_id=ids, smape=smapes, rmsle=rmsles, loss=losses)
end

"""
    save_cude_training_artifacts(paths, training_result)

Write the canonical step 02a JLD2 artifacts for one cUDE width.
"""
# Used by: scripts/02a_run_cude_training.jl.
function save_cude_training_artifacts(paths, training_result)
    mkpath(paths.width_dir)

    JLD2.jldopen(paths.init_params, "w") do file
        file["out_params"] = training_result.out_params
    end

    JLD2.jldopen(paths.losses, "w") do file
        file["losses_per_model"] = training_result.losses_per_model
    end

    JLD2.jldopen(paths.nn_weights, "w") do file
        file["neural_network_parameters"] = training_result.neural_network_parameters
    end

    JLD2.jldopen(paths.train_params, "w") do file
        file["ode_params"] = training_result.ode_params
    end

    return paths
end

"""
    save_cude_evaluation_artifacts(paths; ...)

Write the canonical step 02b JLD2 and CSV artifacts for one evaluated cUDE
candidate model.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl, scripts/02d_evaluate_cude_nn_external_test.jl.
function save_cude_evaluation_artifacts(
    paths;
    training_ids,
    training_log_params,
    validation_ids,
    validation_log_params,
    smapes,
    rmsles,
    losses,
    n_params::Integer,
)
    mkpath(paths.eval_dir)

    JLD2.jldopen(paths.best_params, "w") do file
        file["ode_params_val"] = validation_log_params
    end

    training_df = natural_parameters_dataframe(training_ids, training_log_params; n_params=n_params)
    validation_df = natural_parameters_dataframe(validation_ids, validation_log_params; n_params=n_params)
    metrics_df = cude_patient_metrics_dataframe(validation_ids, smapes, rmsles, losses)

    CSV.write(paths.patients_params_train, training_df)
    CSV.write(paths.patients_params_val, validation_df)
    CSV.write(paths.patients_metrics_val, metrics_df)

    return (
        training_params=training_df,
        validation_params=validation_df,
        metrics=metrics_df,
    )
end

"""
    write_cude_model_summary(path, rows)

Write the canonical cUDE model-summary CSV sorted by `smape_median`.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl. Planned use: scripts/02c_grid_search.jl.
function write_cude_model_summary(path::AbstractString, rows)
    mkpath(dirname(path))

    df = isempty(rows) ? DataFrame(
        model_id=String[],
        model_idx=Int[],
        nn_depth=Int[],
        nn_width=Int[],
        n_patients=Int[],
        loss_mean=Float64[],
        loss_std=Float64[],
        loss_median=Float64[],
        loss_q1=Float64[],
        loss_q3=Float64[],
        loss_iqr=Float64[],
        smape_mean=Float64[],
        smape_std=Float64[],
        smape_median=Float64[],
        smape_q1=Float64[],
        smape_q3=Float64[],
        smape_iqr=Float64[],
        rmsle_mean=Float64[],
        rmsle_std=Float64[],
        rmsle_median=Float64[],
        rmsle_q1=Float64[],
        rmsle_q3=Float64[],
        rmsle_iqr=Float64[],
    ) : DataFrame(rows)

    sort!(df, :smape_median)
    CSV.write(path, df)
    return df
end

"""
    write_cude_training_report(path, training_result, settings; dataset_name, width, n_patients, n_threads, t_scale, paths)

Write a compact human-readable report for one step 02a cUDE training run.
"""
# Used by: scripts/02a_run_cude_training.jl.
function write_cude_training_report(
    path::AbstractString,
    training_result,
    settings;
    dataset_name::AbstractString,
    width::Integer,
    n_patients::Integer,
    n_threads::Integer,
    t_scale::Real,
    paths,
)
    mkpath(dirname(path))

    open(path, "w") do io
        println(io, "cUDE training report")
        println(io, "====================")
        println(io, "Dataset: $(dataset_name)")
        println(io, "Width: $(width)")
        println(io, "Neural depth: $(settings.nn_depth)")
        println(io, "Input dimension: $(settings.input_dim)")
        println(io, "Training patients: $(n_patients)")
        println(io, "Julia threads: $(n_threads)")
        println(io, "Time scale: $(t_scale)")
        println(io)
        println(io, "Initial search")
        println(io, "--------------")
        initial_source = hasproperty(training_result, :initial_source) ? training_result.initial_source : :generated
        initial_parameters_source = hasproperty(training_result, :initial_parameters_source) ? training_result.initial_parameters_source : nothing
        println(io, "Initial parameter source: $(initial_source)")
        if initial_parameters_source !== nothing
            println(io, "Initial parameter file: $(abspath(initial_parameters_source))")
        end
        println(io, "Initial guesses: $(settings.initial_guesses)")
        println(io, "Selected initials: $(length(training_result.out_params))")
        if training_result.selected_indices === nothing
            println(io, "Selected indices: unavailable for reused init_params.jld2")
        else
            println(io, "Selected indices: $(collect(training_result.selected_indices))")
        end
        println(io, "Selected losses: $(collect(training_result.selected_losses))")
        if training_result.selected_indices === nothing
            println(io, "Selected losses note: recomputed from reused init_params.jld2 under the current settings.")
        end
        println(io)
        println(io, "Optimization")
        println(io, "------------")
        println(io, "lambda_back: $(settings.lambda_back)")
        println(io, "kappa_bounds: $(settings.kappa_bounds)")
        println(io, "ADAM maxiters: $(settings.adam_maxiters)")
        println(io, "ADAM eta: $(settings.adam_eta)")
        println(io, "LBFGS maxiters: $(settings.lbfgs_maxiters)")
        println(io, "LBFGS tolerances: $(settings.lbfgs_tolerances)")
        println(io)
        println(io, "Training outcomes")
        println(io, "-----------------")
        for i in eachindex(training_result.final_losses)
            println(
                io,
                "model $(i): adam_retcode=$(training_result.adam_retcodes[i]), " *
                "lbfgs_retcode=$(training_result.lbfgs_retcodes[i]), " *
                "final_loss=$(training_result.final_losses[i])",
            )
        end
        println(io)
        println(io, "Output files")
        println(io, "------------")
        println(io, "init_params: $(abspath(paths.init_params))")
        println(io, "losses: $(abspath(paths.losses))")
        println(io, "nn_weights: $(abspath(paths.nn_weights))")
        println(io, "train_params: $(abspath(paths.train_params))")
        println(io, "fig_dir: $(abspath(paths.fig_dir))")
    end

    return path
end

# =============================================================================
# Systematic Truncation IO
# =============================================================================

"""
    parse_systematic_truncation_cli(args, settings)

Parse step 03c CLI arguments into an ordered list of execution targets.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function parse_systematic_truncation_cli(args, settings)
    requested = isempty(args) ? settings.default_target : lowercase(strip(args[1]))
    valid = Set(settings.valid_targets)
    requested in valid ||
        error("Invalid systematic truncation target '$(requested)'. Valid targets: $(join(settings.valid_targets, ", ")).")

    if requested == "all"
        return (requested=requested, targets=(:ode, :cude, :summary, :overlay))
    elseif requested == "ode"
        return (requested=requested, targets=(:ode,))
    elseif requested == "cude"
        return (requested=requested, targets=(:cude,))
    elseif requested == "summary"
        return (requested=requested, targets=(:summary,))
    elseif requested == "overlay"
        return (requested=requested, targets=(:overlay,))
    elseif requested == "plots"
        return (requested=requested, targets=(:plots,))
    end

    error("Unsupported systematic truncation target: $(requested)")
end

"""
    systematic_truncation_output_paths(output_root)

Return canonical step 03c output paths.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function systematic_truncation_output_paths(output_root::AbstractString)
    return (
        output_dir=output_root,
        ode_dir=joinpath(output_root, "ode"),
        cude_dir=joinpath(output_root, "cude"),
        overlay_dir=joinpath(output_root, "truncation_overlay_comparison"),
        overlay_no_labels_dir=joinpath(output_root, "truncation_overlay_comparison", "no_labels"),
        metrics_summary=joinpath(output_root, "trunc_metrics_summary.csv"),
        params_summary=joinpath(output_root, "trunc_params_summary.csv"),
    )
end

"""
    systematic_truncation_model_output_paths(output_root, model_key)

Return canonical step 03c output paths for one model target.
"""
# Used by: scripts/03c_run_systematic_truncation.jl and src/systematic_truncation.jl.
function systematic_truncation_model_output_paths(output_root::AbstractString, model_key::Symbol)
    model_key in (:ode, :cude) || error("Invalid truncation model key: $(model_key)")
    model_dir = joinpath(output_root, String(model_key))
    return (
        model_dir=model_dir,
        meta_all=joinpath(model_dir, "truncation_meta_all.csv"),
        metrics_all=joinpath(model_dir, "trunc_metrics_all.csv"),
        params_all=joinpath(model_dir, "trunc_params_all.csv"),
        patient_summary=joinpath(model_dir, "truncation_patient_summary.csv"),
        section_summary=joinpath(model_dir, "truncation_section_summary.csv"),
        param_summary=joinpath(model_dir, "truncation_param_summary.csv"),
    )
end

"""
    systematic_truncation_patient_output_paths(model_dir, patient_id)

Return canonical step 03c per-patient output paths.
"""
# Used by: src/systematic_truncation.jl.
function systematic_truncation_patient_output_paths(model_dir::AbstractString, patient_id::AbstractString)
    patient_dir = joinpath(model_dir, patient_id)
    return (
        patient_dir=patient_dir,
        data=joinpath(patient_dir, "df_$(patient_id).csv"),
        meta=joinpath(patient_dir, "truncation_meta.csv"),
        metrics=joinpath(patient_dir, "trunc_metrics.csv"),
        params=joinpath(patient_dir, "trunc_params.csv"),
        initial_scatter=joinpath(patient_dir, "patient_$(patient_id)_initial_scatter.svg"),
        parameter_boxplot=joinpath(patient_dir, "boxplots_truncated_$(patient_id).png"),
    )
end

"""
    save_systematic_truncation_patient_outputs(paths; ...)

Write canonical per-patient CSV artifacts for step 03c.
"""
# Used by: src/systematic_truncation.jl.
function save_systematic_truncation_patient_outputs(paths; patient_dataframe, meta, metrics, params)
    mkpath(paths.patient_dir)
    CSV.write(paths.data, patient_dataframe)
    CSV.write(paths.meta, meta)
    CSV.write(paths.metrics, metrics)
    CSV.write(paths.params, params)
    return paths
end

"""
    save_systematic_truncation_model_outputs(paths; ...)

Write canonical aggregate CSV artifacts for one step 03c model target.
"""
# Used by: src/systematic_truncation.jl.
function save_systematic_truncation_model_outputs(paths; meta_all, metrics_all, params_all, patient_summary, section_summary, param_summary)
    mkpath(paths.model_dir)
    CSV.write(paths.meta_all, meta_all)
    CSV.write(paths.metrics_all, metrics_all)
    CSV.write(paths.params_all, params_all)
    CSV.write(paths.patient_summary, patient_summary)
    CSV.write(paths.section_summary, section_summary)
    CSV.write(paths.param_summary, param_summary)
    return paths
end

"""
    load_gold_standard_cohort(cohort_dir, run_dataset_name; ...)

Load tagged gold-standard patients produced by step 00 while preserving the
ordered ID file as the source of truth.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function load_gold_standard_cohort(
    cohort_dir::AbstractString,
    run_dataset_name::AbstractString;
    mimic_dataset_name::AbstractString="MIMIC-IV",
    external_dataset_name::AbstractString="UMG",
)
    ids_path = joinpath(cohort_dir, "ids_gold_std_patients_$(run_dataset_name).csv")
    report_path = joinpath(cohort_dir, "gold_std_filter_report_$(run_dataset_name).csv")
    isfile(ids_path) || error("Missing gold-standard ID file: $(ids_path)")
    isfile(report_path) || error("Missing gold-standard filter report: $(report_path)")

    ids_df = CSV.read(ids_path, DataFrame)
    required = Set([:patient, :dataset])
    missing = setdiff(required, Set(Symbol.(names(ids_df))))
    isempty(missing) || error("Gold-standard ID file is missing columns: $(missing)")

    mimic_cohort = load_preprocessed_cohort(mimic_dataset_name, cohort_dir)
    external_cohort = load_preprocessed_cohort(external_dataset_name, cohort_dir)

    patient_sources = Dict{String,Dict{String,PatientData}}(
        mimic_dataset_name => Dict(patient.id => patient for patient in mimic_cohort.test),
        external_dataset_name => Dict(patient.id => patient for patient in external_cohort.test),
    )

    patients = PatientData[]
    patient_dataset = Dict{String,String}()

    for row in eachrow(ids_df)
        tagged_id = string(row.patient)
        dataset_name = string(row.dataset)
        haskey(patient_sources, dataset_name) || error("Unsupported gold-standard dataset: $(dataset_name)")

        parts = split(tagged_id, "_"; limit=2)
        length(parts) == 2 || error("Gold-standard patient ID must be tagged as <dataset>_<id>: $(tagged_id)")
        raw_id = parts[2]

        source_map = patient_sources[dataset_name]
        haskey(source_map, raw_id) || error("Missing raw patient $(raw_id) for tagged gold-standard ID $(tagged_id)")

        source_patient = source_map[raw_id]
        push!(patients, PatientData(tagged_id, copy(source_patient.timepoints), copy(source_patient.ctnt_data)))
        patient_dataset[tagged_id] = dataset_name
    end

    return (
        patients=patients,
        patient_dataset=patient_dataset,
        ids=string.(ids_df.patient),
        ids_path=ids_path,
        report_path=report_path,
        report=CSV.read(report_path, DataFrame),
    )
end

"""
    load_systematic_truncation_model_tables(model_dir)

Load aggregate and per-patient output tables for one step 03c model target.
"""
# Used by: scripts/03c_run_systematic_truncation.jl.
function load_systematic_truncation_model_tables(model_dir::AbstractString)
    paths = (
        meta_all=joinpath(model_dir, "truncation_meta_all.csv"),
        metrics_all=joinpath(model_dir, "trunc_metrics_all.csv"),
        params_all=joinpath(model_dir, "trunc_params_all.csv"),
    )
    validate_existing_paths(paths; header="Required systematic truncation aggregate files")
    return (
        meta_all=CSV.read(paths.meta_all, DataFrame),
        metrics_all=CSV.read(paths.metrics_all, DataFrame),
        params_all=CSV.read(paths.params_all, DataFrame),
        paths=paths,
    )
end

# =============================================================================
# Symbolic Regression IO
# =============================================================================

"""
    symbolic_regression_output_paths(output_root)

Return canonical step 04a symbolic-regression output paths.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_regression_output_paths(output_root::AbstractString)
    fig_dir = joinpath(output_root, "figs")
    sr_outputs_dir = joinpath(output_root, "sr_outputs")

    return (
        output_dir=output_root,
        fig_dir=fig_dir,
        sr_outputs_dir=sr_outputs_dir,
        teacher_dataset=joinpath(output_root, "sr_teacher_dataset_direct.csv"),
        pareto_frontier=joinpath(output_root, "sr_pareto_frontier_direct.csv"),
        selected_model=joinpath(output_root, "selected_symbolic_model.txt"),
        teacher_plot=joinpath(fig_dir, "sr_teacher_direct.png"),
        nn_vs_sr_plot=joinpath(fig_dir, "nn_vs_sr_direct.png"),
        sr_plot=joinpath(fig_dir, "sr_direct.png"),
    )
end

"""
    save_symbolic_regression_tables(paths; teacher, frontier)

Write the stable step 04a CSV artifacts.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function save_symbolic_regression_tables(paths; teacher::DataFrame, frontier::DataFrame)
    mkpath(paths.output_dir)
    CSV.write(paths.teacher_dataset, teacher)
    CSV.write(paths.pareto_frontier, frontier)
    return (teacher_dataset=paths.teacher_dataset, pareto_frontier=paths.pareto_frontier)
end

"""
    write_symbolic_regression_report(path, selection, metrics; ...)

Write a compact report describing the selected symbolic surrogate.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function write_symbolic_regression_report(
    path::AbstractString,
    selection,
    metrics;
    selected_model,
    settings,
    output_paths,
)
    mkpath(dirname(path))

    open(path, "w") do io
        println(io, "Symbolic regression selected model")
        println(io, "==================================")
        println(io, "Selected cUDE model: $(selected_model.model_id)")
        println(io, "cUDE width: $(selected_model.nn_width)")
        println(io, "cUDE model index: $(selected_model.model_idx)")
        println(io, "Input dimension: $(settings.input_dim)")
        println(io, "Time scale: $(settings.t_scale)")
        println(io)
        println(io, "Teacher grid")
        println(io, "------------")
        println(io, "Time points: $(length(settings.t_grid))")
        println(io, "Beta points: $(length(settings.beta_grid))")
        println(io, "Training points: $(length(settings.t_grid) * length(settings.beta_grid))")
        println(io, "Validation enabled: $(settings.use_validation)")
        println(io)
        println(io, "Symbolic regression")
        println(io, "-------------------")
        println(io, "Warm-up iterations: $(settings.niterations_warmup)")
        println(io, "Main iterations: $(settings.niterations_main)")
        println(io, "Seed: $(settings.seed)")
        println(io, "Max size: $(settings.maxsize)")
        println(io, "Populations: $(settings.populations)")
        println(io, "Parsimony: $(settings.parsimony)")
        println(io)
        println(io, "Selected frontier member")
        println(io, "------------------------")
        println(io, "Frontier index: $(selection.best_idx)")
        println(io, "Complexity: $(selection.complexity)")
        println(io, "Validation loss: $(selection.validation_loss)")
        println(io, "Equation:")
        println(io, selection.equation)
        println(io)
        println(io, "Synthetic grid metrics")
        println(io, "----------------------")
        println(io, "MSE: $(metrics.mse)")
        println(io, "MAE: $(metrics.mae)")
        println(io, "R2: $(metrics.r2)")
        println(io)
        println(io, "Output files")
        println(io, "------------")
        println(io, "teacher_dataset: $(abspath(output_paths.teacher_dataset))")
        println(io, "pareto_frontier: $(abspath(output_paths.pareto_frontier))")
        println(io, "sr_outputs_dir: $(abspath(output_paths.sr_outputs_dir))")
        println(io, "fig_dir: $(abspath(output_paths.fig_dir))")
    end

    return path
end

# =============================================================================
# Symbolic Formula Evaluation IO
# =============================================================================

"""
    symbolic_formula_output_paths(output_root, dataset_name)

Return canonical step 04b output paths for one evaluated dataset.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function symbolic_formula_output_paths(output_root::AbstractString, dataset_name::AbstractString)
    dataset_dir = joinpath(output_root, "$(dataset_name)_test")
    residuals_dir = joinpath(dataset_dir, "residuals")
    profiles_dir = joinpath(dataset_dir, "profiles")
    dataset_label = "$(dataset_name)_FORMULA"

    return (
        dataset_dir=dataset_dir,
        residuals_dir=residuals_dir,
        profiles_dir=profiles_dir,
        best_params=joinpath(dataset_dir, "best_params_val_formula_$(dataset_name).jld2"),
        patients_metrics=joinpath(dataset_dir, "patients_metrics_val_formula.csv"),
        patients_params=joinpath(dataset_dir, "patients_params_val_formula.csv"),
        residuals_csv=joinpath(dataset_dir, "residuals_$(dataset_label).csv"),
        parameter_boxplot=joinpath(dataset_dir, "boxplots_$(dataset_label).png"),
        correction_surrogate=joinpath(dataset_dir, "correction_surrogate.svg"),
        correction_surrogate_with_title=joinpath(dataset_dir, "correction_surrogate_with_title.svg"),
        residuals_vs_time=joinpath(residuals_dir, "residuals_vs_time_$(dataset_label).png"),
        residuals_vs_fitted=joinpath(residuals_dir, "residuals_vs_fitted_$(dataset_label).png"),
    )
end

"""
    save_symbolic_formula_artifacts(paths; patient_ids, flat_log_params, smapes, rmsles, losses, n_params)

Write the canonical step 04b optimized-parameter and patient-metric artifacts.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function save_symbolic_formula_artifacts(
    paths;
    patient_ids,
    flat_log_params::AbstractVector,
    smapes,
    rmsles,
    losses,
    n_params::Integer,
)
    mkpath(paths.dataset_dir)

    JLD2.jldopen(paths.best_params, "w") do file
        file["params_list_flat"] = collect(flat_log_params)
    end

    metrics_df = cude_patient_metrics_dataframe(patient_ids, smapes, rmsles, losses)
    params_df = natural_parameters_dataframe(patient_ids, flat_log_params; n_params=n_params)

    CSV.write(paths.patients_metrics, metrics_df)
    CSV.write(paths.patients_params, params_df)

    return (
        metrics=metrics_df,
        params=params_df,
    )
end

# =============================================================================
# Profile Likelihood IO
# =============================================================================

"""
    safe_patient_id(pid)

Return a filesystem-safe patient identifier while preserving the readable ID.
"""
# Used by: src/profile_likelihood.jl and src/plotting.jl profile likelihood helpers.
safe_patient_id(pid) = replace(string(pid), r"[^\w\-]+" => "_")

"""
    parse_workflow_bool(value)

Parse CLI boolean values accepted by workflow scripts.
"""
# Used by: src/data_io.jl (parse_profile_likelihood_cli).
function parse_workflow_bool(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized in ("true", "t", "1", "yes", "y") && return true
    normalized in ("false", "f", "0", "no", "n") && return false
    error("Cannot parse boolean value: $(value)")
end

"""
    parse_profile_likelihood_cli(args, settings)

Parse step 03b CLI arguments into an optional target and execution mode.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function parse_profile_likelihood_cli(args, settings)
    target = settings.default_target
    mode = "run"
    run_compute = settings.run_compute
    run_plot_patients = settings.run_plot_patients
    run_plot_aggregate = settings.run_plot_aggregate

    valid_targets = Set(vcat(["all"], collect(string.(settings.targets))))
    plot_modes = Dict(
        "plots" => (run_compute=false, run_plot_patients=true, run_plot_aggregate=true),
        "plots_patients" => (run_compute=false, run_plot_patients=true, run_plot_aggregate=false),
        "plots_aggregate" => (run_compute=false, run_plot_patients=false, run_plot_aggregate=true),
    )

    seen_target = false
    seen_mode = false

    for arg in args
        token = lowercase(strip(arg))

        if haskey(plot_modes, token)
            seen_mode && error("Only one PLA plot mode can be provided. Got duplicate mode: $(arg)")
            selected = plot_modes[token]
            mode = token
            run_compute = selected.run_compute
            run_plot_patients = selected.run_plot_patients
            run_plot_aggregate = selected.run_plot_aggregate
            seen_mode = true
            continue
        end

        if token in valid_targets
            seen_target && error("Only one PLA target can be provided. Got duplicate target: $(arg)")
            target = token
            seen_target = true
            continue
        end

        occursin("=", arg) || error(
            "Invalid PLA argument '$(arg)'. Valid targets: $(join(sort(collect(valid_targets)), ", ")). " *
            "Valid plot modes: plots, plots_patients, plots_aggregate."
        )

        # Compatibility path for older key=value invocations.
        key, value = split(arg, "="; limit=2)
        key = lowercase(strip(key))
        parsed = parse_workflow_bool(value)
        if key == "run_compute"
            run_compute = parsed
        elseif key == "run_plot_patients"
            run_plot_patients = parsed
        elseif key == "run_plot_aggregate"
            run_plot_aggregate = parsed
        else
            error("Invalid PLA override '$(key)'. Valid keys: run_compute, run_plot_patients, run_plot_aggregate.")
        end
    end

    return (
        target=target,
        mode=mode,
        run_compute=run_compute,
        run_plot_patients=run_plot_patients,
        run_plot_aggregate=run_plot_aggregate,
    )
end

"""
    resolve_profile_likelihood_targets(settings, requested_target)

Resolve the step 03b CLI target into ordered internal target keys.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function resolve_profile_likelihood_targets(settings, requested_target)
    requested = lowercase(string(requested_target))
    requested == "all" && return collect(settings.targets)

    valid = Dict(string(key) => key for key in settings.targets)
    haskey(valid, requested) ||
        error("Invalid PLA target '$(requested_target)'. Valid targets: all, $(join(sort(collect(keys(valid))), ", ")).")

    return [valid[requested]]
end

"""
    profile_likelihood_target_specs(config, settings)

Return canonical target metadata for step 03b.
"""
# Used by: scripts/03b_run_profile_likelihood.jl and src/data_io.jl.
function profile_likelihood_target_specs(config, settings)
    mimic_name = config.datasets[settings.mimic_dataset_key].dataset_name
    external_name = config.datasets[settings.external_dataset_key].dataset_name

    cude_params = ["a", "b", "Cs0", "Cc0", "beta"]
    cude_plot = ["a", "b", "Cs0", "Cc0", "β"]
    ode_params = ["a", "b", "Cs0", "Cc0", "Td"]
    ode_plot = ["a", "b", "Cs0", "Cc0", "Td"]

    return Dict(
        :cude_mimic => (
            key=:cude_mimic,
            model_kind=:cude,
            dataset_key=settings.mimic_dataset_key,
            dataset_name=mimic_name,
            target_name="cude_$(mimic_name)",
            param_names=cude_params,
            pnames_plot=cude_plot,
            lower=Float64.(settings.cude_lower),
            upper=Float64.(settings.cude_upper),
        ),
        :cude_umg => (
            key=:cude_umg,
            model_kind=:cude,
            dataset_key=settings.external_dataset_key,
            dataset_name=external_name,
            target_name="cude_$(external_name)",
            param_names=cude_params,
            pnames_plot=cude_plot,
            lower=Float64.(settings.cude_lower),
            upper=Float64.(settings.cude_upper),
        ),
        :ode_mimic => (
            key=:ode_mimic,
            model_kind=:ode,
            dataset_key=settings.mimic_dataset_key,
            dataset_name=mimic_name,
            target_name="ode_$(mimic_name)",
            param_names=ode_params,
            pnames_plot=ode_plot,
            lower=Float64.(settings.ode_lower),
            upper=Float64.(settings.ode_upper),
        ),
        :ode_umg => (
            key=:ode_umg,
            model_kind=:ode,
            dataset_key=settings.external_dataset_key,
            dataset_name=external_name,
            target_name="ode_$(external_name)",
            param_names=ode_params,
            pnames_plot=ode_plot,
            lower=Float64.(settings.ode_lower),
            upper=Float64.(settings.ode_upper),
        ),
    )
end

"""
    profile_likelihood_output_paths(output_root, target_name)

Return canonical output paths for one step 03b target.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function profile_likelihood_output_paths(output_root::AbstractString, target_name::AbstractString)
    target_dir = joinpath(output_root, target_name)
    per_patient_dir = joinpath(target_dir, "per_patient")
    fig_dir = joinpath(target_dir, "figs")
    composite_dir = joinpath(fig_dir, "composite")
    aggregate_dir = joinpath(fig_dir, "aggregate")

    return (
        target_dir=target_dir,
        per_patient_dir=per_patient_dir,
        fig_dir=fig_dir,
        composite_fig_dir=composite_dir,
        aggregate_fig_dir=aggregate_dir,
        profiles_long=joinpath(target_dir, "pla_profiles_long.csv"),
        profiles_summary=joinpath(target_dir, "pla_profiles_summary.csv"),
    )
end

"""
    profile_likelihood_input_paths(settings, spec, selected_model)

Return canonical input paths for one step 03b target.
"""
# Used by: src/data_io.jl (load_profile_likelihood_target_inputs).
function profile_likelihood_input_paths(settings, spec, selected_model)
    if spec.model_kind == :ode
        ode_paths = ode_dataset_output_paths(settings.ode_input_dir, spec.dataset_name)
        params_csv = spec.dataset_name == "MIMIC-IV" ? ode_paths.params_val_csv : ode_paths.params_csv
        return (params_csv=params_csv,)
    elseif spec.model_kind == :cude
        cude_paths = spec.dataset_name == "MIMIC-IV" ?
                     cude_evaluation_model_output_paths(
                         settings.cude_evaluation_input_dir,
                         selected_model.nn_width,
                         selected_model.model_idx,
                         spec.dataset_name,
                     ) :
                     cude_external_test_output_paths(settings.cude_external_test_input_dir, spec.dataset_name)
        training_paths = cude_training_output_paths(settings.cude_training_input_dir, selected_model.nn_width)

        return (
            params_csv=cude_paths.patients_params_val,
            best_params=cude_paths.best_params,
            nn_weights=training_paths.nn_weights,
            train_params=training_paths.train_params,
        )
    end

    error("Unsupported PLA model kind: $(spec.model_kind)")
end

function _profile_likelihood_ode_parameter_matrix(path::AbstractString, patients)
    df = CSV.read(path, DataFrame)
    found_cols = Symbol.(names(df))
    required = [:p1, :p2, :p3, :p4, :p5]
    missing = setdiff(required, found_cols)
    isempty(missing) || error("Missing ODE parameter columns in $(path): $(missing)")

    id_col = :patient in found_cols ? :patient :
             (:patient_id in found_cols ? :patient_id : error("Missing patient identifier column in $(path)."))

    rows_by_id = Dict{String,NTuple{5,Float64}}()
    for row in eachrow(df)
        pid = string(row[id_col])
        haskey(rows_by_id, pid) && error("Duplicate patient ID in ODE parameter file: $(pid)")
        rows_by_id[pid] = (Float64(row.p1), Float64(row.p2), Float64(row.p3), Float64(row.p4), Float64(row.p5))
    end

    patient_ids = [patient.id for patient in patients]
    missing_ids = filter(id -> !haskey(rows_by_id, id), patient_ids)
    isempty(missing_ids) || error("Missing ODE parameters for patient IDs: $(missing_ids)")

    matrix = Matrix{Float64}(undef, length(patient_ids), 5)
    for (i, pid) in enumerate(patient_ids)
        params = rows_by_id[pid]
        for j in 1:5
            matrix[i, j] = params[j]
        end
    end

    all(isfinite, matrix) || error("ODE parameter matrix contains non-finite values: $(path)")
    return matrix, df
end

function _profile_likelihood_cude_parameter_matrix(paths, patients, n_params::Integer)
    params_df = CSV.read(paths.params_csv, DataFrame)
    found_cols = Symbol.(names(params_df))
    required = [:patient_id, :a, :b, :Cs0, :Cc0, :beta]
    missing = setdiff(required, found_cols)
    isempty(missing) || error("Missing cUDE parameter columns in $(paths.params_csv): $(missing)")

    patient_ids = [patient.id for patient in patients]
    csv_ids = string.(params_df.patient_id)
    patient_ids == csv_ids ||
        error("cUDE patients_params_val.csv is not aligned with the step 00 cohort ordering for $(paths.params_csv).")

    flat_log_params = Vector{Float64}(JLD2.load(paths.best_params, "ode_params_val"))
    length(flat_log_params) == length(patients) * n_params ||
        error("Expected $(length(patients) * n_params) cUDE log parameters in $(paths.best_params), got $(length(flat_log_params)).")

    reshaped = permutedims(reshape(flat_log_params, n_params, :))
    theta_from_csv = hcat(
        log.(Float64.(params_df.a)),
        log.(Float64.(params_df.b)),
        log.(Float64.(params_df.Cs0)),
        log.(Float64.(params_df.Cc0)),
        log.(Float64.(params_df.beta)),
    )

    max_abs_diff = maximum(abs.(theta_from_csv .- reshaped))
    max_abs_diff <= 1e-8 ||
        error("Mismatch between cUDE JLD2 and CSV log-scale parameters. max_abs_diff=$(max_abs_diff)")

    all(isfinite, reshaped) || error("cUDE parameter matrix contains non-finite values: $(paths.best_params)")
    return reshaped, params_df
end

"""
    load_profile_likelihood_target_inputs(config, settings, spec, selected_model)

Load the step 03b cohort, parameter starts, and selected cUDE NN artifacts for
one target while validating patient/parameter alignment.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function load_profile_likelihood_target_inputs(config, settings, spec, selected_model)
    cohort = load_preprocessed_cohort(spec.dataset_name, settings.cohort_dir)
    patients = cohort.test
    input_paths = profile_likelihood_input_paths(settings, spec, selected_model)

    validate_existing_paths(input_paths; header="Required PLA input files for $(spec.target_name)")

    if spec.model_kind == :ode
        reshaped_params, params_df = _profile_likelihood_ode_parameter_matrix(input_paths.params_csv, patients)
        return (
            patients=patients,
            reshaped_params=reshaped_params,
            params_df=params_df,
            chain=nothing,
            neural_params=nothing,
            input_paths=input_paths,
            cohort=cohort,
        )
    end

    reshaped_params, params_df = _profile_likelihood_cude_parameter_matrix(input_paths, patients, settings.n_params)
    artifacts = load_cude_training_artifacts(settings.cude_training_input_dir, selected_model.nn_width)
    1 <= selected_model.model_idx <= length(artifacts.neural_network_parameters) ||
        error("Selected model_idx=$(selected_model.model_idx) is outside available cUDE candidates.")

    chain = neural_network_model(selected_model.nn_depth, selected_model.nn_width; input_dims=settings.input_dim)
    neural_params = artifacts.neural_network_parameters[selected_model.model_idx]

    return (
        patients=patients,
        reshaped_params=reshaped_params,
        params_df=params_df,
        chain=chain,
        neural_params=neural_params,
        input_paths=input_paths,
        cohort=cohort,
    )
end

function _assert_profile_likelihood_target_under_root(root::AbstractString, target_dir::AbstractString)
    root_abs = abspath(root)
    target_abs = abspath(target_dir)
    startswith(target_abs, root_abs * string(Base.Filesystem.path_separator)) ||
        target_abs == root_abs ||
        error("Refusing to clean PLA output outside configured output root: $(target_abs)")
    return nothing
end

"""
    reset_profile_likelihood_output!(paths; root, compute, plot_patients, plot_aggregate)

Clean only the requested step 03b target output folders before writing new
artifacts.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function reset_profile_likelihood_output!(
    paths;
    root::AbstractString,
    compute::Bool,
    plot_patients::Bool,
    plot_aggregate::Bool,
)
    _assert_profile_likelihood_target_under_root(root, paths.target_dir)

    if compute
        isdir(paths.target_dir) && rm(paths.target_dir; recursive=true, force=true)
    else
        if plot_patients
            isdir(paths.composite_fig_dir) && rm(paths.composite_fig_dir; recursive=true, force=true)
        end
        if plot_aggregate
            isdir(paths.aggregate_fig_dir) && rm(paths.aggregate_fig_dir; recursive=true, force=true)
        end
    end

    ensure_output_dirs!(
        (
            target=paths.target_dir,
            per_patient=paths.per_patient_dir,
            composite=paths.composite_fig_dir,
            aggregate=paths.aggregate_fig_dir,
        );
        header="Ensured PLA output directories",
    )

    return paths
end

"""
    save_profile_likelihood_patient_csvs(paths, patient_tag, profiles, summary)

Write one patient's step 03b profile and summary CSV artifacts.
"""
# Used by: src/profile_likelihood.jl.
function save_profile_likelihood_patient_csvs(paths, patient_tag::AbstractString, profiles::DataFrame, summary::DataFrame)
    mkpath(paths.per_patient_dir)
    profile_path = joinpath(paths.per_patient_dir, "$(patient_tag)_profiles.csv")
    summary_path = joinpath(paths.per_patient_dir, "$(patient_tag)_summary.csv")
    CSV.write(profile_path, profiles)
    CSV.write(summary_path, summary)
    return (profiles=profile_path, summary=summary_path)
end

"""
    save_profile_likelihood_global_csvs(paths, profiles_long, summary)

Write global step 03b profile and summary CSV artifacts.
"""
# Used by: src/profile_likelihood.jl.
function save_profile_likelihood_global_csvs(paths, profiles_long::DataFrame, summary::DataFrame)
    mkpath(paths.target_dir)
    CSV.write(paths.profiles_long, profiles_long)
    CSV.write(paths.profiles_summary, summary)
    return (profiles_long=paths.profiles_long, profiles_summary=paths.profiles_summary)
end

"""
    load_profile_likelihood_global_csvs(paths)

Read global step 03b profile and summary CSV artifacts.
"""
# Used by: src/plotting.jl PLA helpers.
function load_profile_likelihood_global_csvs(paths)
    isfile(paths.profiles_long) || error("Missing PLA profiles CSV: $(paths.profiles_long)")
    isfile(paths.profiles_summary) || error("Missing PLA summary CSV: $(paths.profiles_summary)")
    return (
        profiles_long=CSV.read(paths.profiles_long, DataFrame),
        profiles_summary=CSV.read(paths.profiles_summary, DataFrame),
    )
end

"""
    load_profile_likelihood_patient_csvs(paths, patient_tag)

Read one patient's step 03b profile and summary CSV artifacts.
"""
# Used by: src/plotting.jl PLA helpers.
function load_profile_likelihood_patient_csvs(paths, patient_tag::AbstractString)
    profile_path = joinpath(paths.per_patient_dir, "$(patient_tag)_profiles.csv")
    summary_path = joinpath(paths.per_patient_dir, "$(patient_tag)_summary.csv")
    isfile(profile_path) || error("Missing PLA patient profile CSV: $(profile_path)")
    isfile(summary_path) || error("Missing PLA patient summary CSV: $(summary_path)")
    return (
        profiles=CSV.read(profile_path, DataFrame),
        summary=CSV.read(summary_path, DataFrame),
        paths=(profiles=profile_path, summary=summary_path),
    )
end

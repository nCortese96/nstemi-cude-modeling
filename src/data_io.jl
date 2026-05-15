"""
data_io.jl

Workflow IO and patient-level input/output helpers.

Sections:
- Workflow IO: logging and output-directory helpers.
- Patient Data IO: `PatientData`, spreadsheet readers, and DataFrame converters.
- Preprocessed Cohort IO: JLD2/CSV cohort artifact readers.
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
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
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

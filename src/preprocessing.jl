"""
preprocessing.jl

Patient preprocessing, reporting, and cohort export utilities.

Sections:
- Patient Preprocessing: duplicate handling, trimming, and eligibility filters.
- Preprocessing Reports: formal reports and cohort artifact export.
"""

using CSV
using DataFrames: DataFrame
using Dates
using JLD2
using Logging
using Random
using Statistics: mean, median, quantile, std

# =============================================================================
# Patient Preprocessing
# =============================================================================

"""
    patient_duplicate_time_counts(patients::Vector{PatientData}; tol=0.0)

Return `Dict(id => n_duplicates)` after sorting each patient's timepoints.
"""
function patient_duplicate_time_counts(patients::Vector{PatientData}; tol::Real=0.0)
    counts = Dict{String,Int}()
    for p in patients
        t = sort(p.timepoints)
        length(t) < 2 && continue
        ndup = count(abs.(diff(t)) .<= tol)
        if ndup > 0
            counts[p.id] = ndup
        end
    end
    return counts
end

"""
    collapse_duplicate_times(tp, ct; agg=mean, tol=0.0)

Input: one patient's time and troponin vectors.
Output: sorted vectors with duplicate timepoints collapsed by `agg`.
"""
function collapse_duplicate_times(tp::Vector{Float64}, ct::Vector{Float64};
    agg=mean,
    tol::Float64=0.0
)
    @assert length(tp) == length(ct) "Input time and cTnT vectors must have the same length"

    n = length(tp)
    n <= 1 && return (copy(tp), copy(ct))

    idx = sortperm(tp)
    t = tp[idx]
    y = ct[idx]

    t2 = Float64[]
    y2 = Float64[]

    i = 1
    while i <= n
        j = i
        while j < n && abs(t[j+1] - t[i]) <= tol
            j += 1
        end
        push!(t2, t[i])
        push!(y2, agg(@view y[i:j]))
        i = j + 1
    end

    return t2, y2
end

"""
    collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

Input: patient vector and duplicate counts from `patient_duplicate_time_counts`.
Output: `(n_patients_modified, n_points_removed_total)` after in-place collapse.
"""
function collapse_duplicates!(
    patients::Vector{PatientData},
    dup_counts::AbstractDict{String,Int};
    agg=mean,
    tol::Float64=0.0
)
    isempty(dup_counts) && return (0, 0)

    n_modified = 0
    n_removed_total = 0

    for p in patients
        haskey(dup_counts, p.id) || continue

        n_before = length(p.timepoints)
        tp2, ct2 = collapse_duplicate_times(p.timepoints, p.ctnt_data; agg=agg, tol=tol)
        n_after = length(tp2)

        empty!(p.timepoints)
        append!(p.timepoints, tp2)
        empty!(p.ctnt_data)
        append!(p.ctnt_data, ct2)

        n_modified += 1
        n_removed_total += (n_before - n_after)
    end

    return (n_modified, n_removed_total)
end

function trim_time(patients::AbstractVector{PatientData}, time_val)
    filtered_patients = PatientData[]

    for p in patients
        mask = p.timepoints .<= time_val

        if any(mask)
            tp = p.timepoints[mask]
            ct = p.ctnt_data[mask]
            push!(filtered_patients, PatientData(p.id, tp, ct))
        else
            @warn "Patient $(p.id) has no acquisitions <= $(time_val) h and will be excluded"
        end
    end

    @info "Kept $(length(filtered_patients)) patients out of $(length(patients))"

    return filtered_patients
end

"""
    count_acq_in_window_sorted(timepoints_sorted, h)

Input: sorted timepoint vector and time threshold.
Output: number of acquisitions with `t <= h`.
"""
@inline function count_acq_in_window_sorted(timepoints_sorted::AbstractVector{<:Real}, h::Real)
    return searchsortedlast(timepoints_sorted, h)
end

"""
    max_gap_sorted(timepoints_sorted)::Float64

Input: sorted timepoint vector.
Output: maximum consecutive time gap.
"""
@inline function max_gap_sorted(timepoints_sorted::AbstractVector{<:Real})
    n = length(timepoints_sorted)
    if n < 2
        return 0.0
    end
    gmax = 0.0
    @inbounds for i in 2:n
        g = timepoints_sorted[i] - timepoints_sorted[i-1]
        if g > gmax
            gmax = g
        end
    end
    return gmax
end

function find_anomalies(
    patients::Vector{PatientData},
    meas_min_number::Int=1,
    min_acq_time_before::Real=300.0,
    min_acq_n_before::Int=1,
    min_acq_time_after::Real=0.0,
    min_acq_n_after::Int=0,
    min_time::Real=0.0;
    max_gap_h::Union{Nothing,Real}=nothing,
    verbose::Bool=true
)
    anomalies = Dict{String,Vector{String}}()

    for p in patients
        issues = String[]

        tp = p.timepoints
        ct = p.ctnt_data
        n_tp = length(tp)
        n_ct = length(ct)

        if n_ct == 0
            push!(issues, "empty cTnT data")
        end
        if n_tp == 0
            push!(issues, "empty acquisition timepoints")
        end
        if n_tp != n_ct
            push!(issues, "timepoint/cTnT length mismatch")
        end

        if n_tp > 0
            tmax = maximum(tp)
            if tmax < min_time
                push!(issues, "maximum acquisition time < $(min_time)h")
            end

            if any(<(0), tp)
                push!(issues, "negative acquisition time")
            end
            if n_ct > 0 && any(<(0), ct)
                push!(issues, "negative cTnT value")
            end

            if n_tp < meas_min_number
                push!(issues, "fewer than $meas_min_number acquisition timepoints")
            end

            if issorted(tp; lt=(a, b) -> a <= b)
                n_before = count_acq_in_window_sorted(tp, min_acq_time_before)
                if n_before < min_acq_n_before
                    push!(issues, "fewer than $min_acq_n_before observation(s) within the first $(min_acq_time_before)h")
                end

                n_after = length(tp) - count_acq_in_window_sorted(tp, min_acq_time_after)
                if n_after < min_acq_n_after
                    push!(issues, "fewer than $min_acq_n_after observation(s) after $(min_acq_time_after)h")
                end

                if max_gap_h !== nothing
                    gmax = max_gap_sorted(tp)
                    if gmax > max_gap_h
                        push!(issues, "maximum consecutive gap > $(max_gap_h)h (observed=$(round(gmax, digits=3))h)")
                    end
                end
            else
                push!(issues, "acquisition times are not sorted")
            end
        end

        if !isempty(issues)
            anomalies[p.id] = issues
        end
    end
    if verbose
        if isempty(anomalies)
            println("No anomalies found")
        else
            for (id, issues) in anomalies
                @warn "Patient " * id * ": " * join(issues, ", ")
            end
        end
    end

    return anomalies
end


# =============================================================================
# Preprocessing Reports
# =============================================================================

"""
    DatasetReporter

Input: a dataset name and output directory through the constructor.
Output: mutable report state used by `report_step!` and `finalize_report`.
"""
mutable struct DatasetReporter
    dataset_name::String
    report_path::String
    io::IOStream
    step_count::Int
end

"""
    DatasetReporter(dataset_name; report_dir="res")

Input: dataset name and report directory.
Output: initialized report writer with a text header already written.
"""
function DatasetReporter(dataset_name::String; report_dir::String="res")
    mkpath(report_dir)
    ts = Dates.format(now(), "yyyymmdd_HHMMss")
    fname = "dataset_report_$(dataset_name)_$(ts).txt"
    report_path = joinpath(report_dir, fname)
    io = open(report_path, "w")

    sep = "="^78
    header = """
    $(sep)
    cTnT DATASET PREPROCESSING REPORT
    $(sep)
    Dataset:      $(dataset_name)
    Generated at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    Report file:  $(report_path)

    Summary format
      Distributional fields are reported as: median [Q1-Q3] (min-max).
      Missing values (NaN) are excluded from counts and summary statistics.

    Metric definitions
      Patients:
        Number of patient trajectories present at the current pipeline step.
      Total cTnT observations:
        Count of non-missing cTnT concentration values.
      Total acquisition timepoints:
        Count of non-missing acquisition times.
      cTnT observations per patient:
        Distribution of non-missing cTnT observations per patient.
      First acquisition time (h):
        Distribution of the first valid acquisition time per patient.
      Last acquisition time (h):
        Distribution of the last valid acquisition time per patient.
      Observation span (h):
        Last acquisition time minus first acquisition time per patient.
      Max consecutive acquisition gap (h):
        Largest gap between consecutive valid acquisition times per patient.
    $(sep)
    """

    _dual_print(io, header)
    return DatasetReporter(dataset_name, report_path, io, 0)
end

"""
    _summary_stats(v)

Input: numeric vector.
Output: NamedTuple with n, mean, std, median, quartiles, min, and max after dropping NaN.
"""
function _summary_stats(v::AbstractVector{<:Real})
    clean = filter(!isnan, v)
    n = length(clean)
    n == 0 && return (n=0, mean=NaN, std=NaN, median=NaN,
        q1=NaN, q3=NaN, min=NaN, max=NaN)
    return (
        n=n,
        mean=mean(clean),
        std=length(clean) > 1 ? std(clean) : 0.0,
        median=median(clean),
        q1=quantile(clean, 0.25),
        q3=quantile(clean, 0.75),
        min=minimum(clean),
        max=maximum(clean),
    )
end

"""
    _dual_print(io, text)

Input: open IO stream and text.
Output: writes the same text to the report file and stdout.
"""
function _dual_print(io::IOStream, text::String)
    print(io, text)
    print(stdout, text)
    flush(io)
end

"""
    _fmt_stat(s; digits=2)

Input: summary-stat NamedTuple.
Output: compact median/IQR/min-max string for reports.
"""
function _fmt_stat(s::NamedTuple; digits::Int=2)
    if isnan(s.median)
        return "N/A"
    end
    return string(
        round(s.median; digits),
        " [", round(s.q1; digits), "-", round(s.q3; digits), "]",
        " (", round(s.min; digits), "-", round(s.max; digits), ")"
    )
end

"""
    report_step!(reporter, step_name, patients; extra_info="")

Input: report state, step label, patient collection, and optional text.
Output: appends preprocessing statistics to both stdout and the report file.
"""
function report_step!(reporter::DatasetReporter,
    step_name::String,
    patients::AbstractVector{PatientData};
    extra_info::String="")

    reporter.step_count += 1
    step_n = reporter.step_count
    n_patients = length(patients)

    ctnt_counts = [count(!isnan, p.ctnt_data) for p in patients]
    ctnt_counts_stats = _summary_stats(Float64.(ctnt_counts))
    total_ctnt = sum(ctnt_counts)

    tp_counts = [count(!isnan, p.timepoints) for p in patients]
    total_tp = sum(tp_counts)

    first_times = Float64[]
    last_times = Float64[]
    obs_spans = Float64[]
    max_gaps = Float64[]

    for p in patients
        valid_tp = filter(!isnan, p.timepoints)
        isempty(valid_tp) && continue

        ft = first(valid_tp)
        lt = last(valid_tp)
        push!(first_times, ft)
        push!(last_times, lt)
        push!(obs_spans, lt - ft)

        if length(valid_tp) >= 2
            push!(max_gaps, maximum(diff(valid_tp)))
        end
    end

    first_stats = _summary_stats(first_times)
    last_stats = _summary_stats(last_times)
    span_stats = _summary_stats(obs_spans)
    gap_stats = _summary_stats(max_gaps)

    sep = "-"^78
    block = """

    $(sep)
    STEP $(lpad(string(step_n), 2, '0')): $(step_name)
    $(sep)
    Patients:                              $(n_patients)
    Total cTnT observations:               $(total_ctnt)
    Total acquisition timepoints:          $(total_tp)
    $(extra_info == "" ? "" : "Additional information:                 $(extra_info)\n")cTnT observations per patient:         $(_fmt_stat(ctnt_counts_stats))
    First acquisition time (h):            $(_fmt_stat(first_stats))
    Last acquisition time (h):             $(_fmt_stat(last_stats))
    Observation span (h):                  $(_fmt_stat(span_stats))
    Max consecutive acquisition gap (h):   $(_fmt_stat(gap_stats))
    $(sep)

    """

    _dual_print(reporter.io, block)
    return nothing
end

"""
    finalize_report(reporter)

Input: active dataset reporter.
Output: closes the report file and returns its path.
"""
function finalize_report(reporter::DatasetReporter)
    sep = "="^78
    footer = """
    $(sep)
    REPORT COMPLETE
    Recorded preprocessing steps: $(reporter.step_count)
    Report saved to: $(reporter.report_path)
    $(sep)
    """
    _dual_print(reporter.io, footer)
    close(reporter.io)
    @info "Report closed: $(reporter.report_path)"
    return reporter.report_path
end

"""
    preprocessing_filter_description(...)

Input: filtering thresholds and removed-patient count.
Output: one-line description used in preprocessing reports.
"""
function preprocessing_filter_description(;
    meas_min_number::Int,
    min_acq_time_before::Real,
    min_acq_n_before::Int,
    min_acq_time_after::Real,
    min_acq_n_after::Int,
    min_time::Real,
    max_gap::Real,
    removed_count::Int)

    return string(
        "Eligibility filters: minimum observations=$(meas_min_number), ",
        "early window=at least $(min_acq_n_before) observation(s) within the first $(min_acq_time_before)h, ",
        "late window=at least $(min_acq_n_after) observation(s) after $(min_acq_time_after)h, ",
        "minimum observation span=$(min_time)h, maximum consecutive gap=$(max_gap)h. ",
        "Removed patients: $(removed_count)."
    )
end

"""
    split_patients(patients; train_fraction=0.8, seed=1234)

Return a reproducible shuffled copy of `patients` plus training and test splits.
The input vector is not mutated.
"""
# Used by: src/preprocessing.jl (save_preprocessed_dataset!). Planned use: scripts/02a_run_cude_training.jl.
function split_patients(patients::Vector{PatientData}; train_fraction::Real=0.8, seed::Integer=1234)
    shuffled = copy(patients)
    Random.seed!(seed)
    shuffle!(shuffled)
    n_train = Int(round(length(shuffled) * train_fraction))
    return (
        patients=shuffled,
        training=shuffled[1:n_train],
        test=shuffled[n_train+1:end],
    )
end

"""
    save_preprocessed_dataset!(dataset_name, cleaned_patients, report_dir; train_fraction=0.8, seed=1234, reporter=nothing)

Input: cleaned patient collection and output settings.
Output: writes JLD2 train/test artifacts and returns the saved patient sets.
"""
# Used by: src/preprocessing.jl (run_dataset_report).
function save_preprocessed_dataset!(
    dataset_name::String,
    cleaned_patients::Vector{PatientData},
    report_dir::String;
    train_fraction::Real=0.8,
    seed::Integer=1234,
    reporter::Union{Nothing,DatasetReporter}=nothing
)
    if dataset_name == "MIMIC-IV"
        split = split_patients(cleaned_patients; train_fraction=train_fraction, seed=seed)
        training_dataset = split.training
        test_dataset = split.test

        if reporter !== nothing
            report_step!(reporter, "Training split ($(round(Int, train_fraction * 100))%)", training_dataset;
                extra_info="Split: $(length(training_dataset)) training, $(length(test_dataset)) test")
            report_step!(reporter, "Test split ($(round(Int, (1 - train_fraction) * 100))%)", test_dataset)
        end

        @save "$(report_dir)/$(dataset_name)_trainingset.jld2" training_dataset
        @save "$(report_dir)/$(dataset_name)_testset.jld2" test_dataset

        return (cleaned_patients=split.patients, training=training_dataset, test=test_dataset)
    end

    test_dataset = cleaned_patients
    @save "$(report_dir)/$(dataset_name)_testset.jld2" test_dataset
    return (cleaned_patients=cleaned_patients, test=test_dataset)
end

"""
    run_dataset_report(; dataset_name, dataset_path, column_letter, ...)

Input: dataset identity, Excel location, preprocessing thresholds, and report/output directory.
Output: cleaned patients, saved split artifacts, and report path.
"""
function run_dataset_report(;
    dataset_name::String,
    dataset_path::String,
    column_letter::String,
    T_SCALE::Float64,
    meas_min_number::Int,
    min_acq_time_before::Float64,
    min_acq_n_before::Int,
    min_acq_time_after::Float64,
    min_acq_n_after::Int,
    min_time::Float64,
    max_gap::Float64,
    report_dir::String="res",
    train_fraction::Real=0.8,
    split_seed::Integer=1234,
    data_root::String="data"
)
    reporter = DatasetReporter(dataset_name; report_dir=report_dir)

    patients, load_info = load_excel_patients(dataset_path, column_letter; data_root=data_root)

    report_step!(reporter, "Raw dataset loaded", patients;
        extra_info="File: $(load_info.file_path)")

    dup_counts = patient_duplicate_time_counts(patients; tol=0.0)
    nmod, nrm = collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

    report_step!(reporter, "Duplicate timepoints collapsed", patients;
        extra_info="Modified patients: $(nmod), removed duplicate points: $(nrm)")

    trimmed_patients = trim_time(patients, T_SCALE)

    report_step!(reporter, "Trimmed to $(T_SCALE)h", trimmed_patients)

    anoms = find_anomalies(trimmed_patients,
        meas_min_number,
        min_acq_time_before, min_acq_n_before,
        min_acq_time_after, min_acq_n_after,
        min_time;
        max_gap_h=max_gap,
        verbose=false
    )

    cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_patients)

    filter_info = preprocessing_filter_description(
        meas_min_number=meas_min_number,
        min_acq_time_before=min_acq_time_before,
        min_acq_n_before=min_acq_n_before,
        min_acq_time_after=min_acq_time_after,
        min_acq_n_after=min_acq_n_after,
        min_time=min_time,
        max_gap=max_gap,
        removed_count=length(anoms),
    )

    report_step!(reporter, "Anomaly filtering (All Eligible)", cleaned_patients;
        extra_info=filter_info)

    df_ae_ids = DataFrame(patient=[p.id for p in cleaned_patients])
    CSV.write(joinpath(report_dir, "ids_all_eligible_$(dataset_name).csv"), df_ae_ids)

    saved_sets = save_preprocessed_dataset!(
        dataset_name,
        cleaned_patients,
        report_dir;
        train_fraction=train_fraction,
        seed=split_seed,
        reporter=reporter,
    )

    report_path = finalize_report(reporter)
    return merge(saved_sets, (report_path=report_path,))
end

"""
    preprocessing_output_paths(dataset_name, output_dir; train_fraction=0.8, report_path=nothing)

Input: preprocessing dataset name, output directory, split fraction, and optional report path.
Output: named tuple of artifacts expected from `run_dataset_report`.
"""
function preprocessing_output_paths(
    dataset_name::AbstractString,
    output_dir::AbstractString;
    train_fraction::Real=0.8,
    report_path=nothing,
)
    paths = report_path === nothing ? NamedTuple() : (report=report_path,)
    paths = merge(paths, (
        all_eligible_ids=joinpath(output_dir, "ids_all_eligible_$(dataset_name).csv"),
    ))

    if 0 < train_fraction < 1
        return merge(paths, (
            training_set=joinpath(output_dir, "$(dataset_name)_trainingset.jld2"),
            test_set=joinpath(output_dir, "$(dataset_name)_testset.jld2"),
        ))
    end

    return merge(paths, (
        test_set=joinpath(output_dir, "$(dataset_name)_testset.jld2"),
    ))
end

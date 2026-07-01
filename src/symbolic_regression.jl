"""
symbolic_regression.jl

Reusable non-plotting helpers for workflow steps 04a and 04c.

Sections:
- Teacher Dataset: deterministic synthetic grids and NN teacher evaluation.
- Neural-Correction Early Bump Analysis: descriptive local-feature summaries.
- Symbolic Regression: bounded loss, options, search, and Pareto selection.
- Evaluation Tables: stable teacher, frontier, and selected-model summaries.
"""

using CSV, XLSX
using DataFrames: DataFrame, eachrow, groupby, leftjoin, nrow
using Dates: DateTime
import Dates
using Statistics: mean, median, quantile
using SymbolicRegression
import SymbolicRegression: string_tree, compute_complexity

# =============================================================================
# Teacher Dataset
# =============================================================================

"""
    build_symbolic_teacher_grid(t_grid, beta_grid; t_scale)

Build the deterministic cUDE symbolic-regression teacher grid.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function build_symbolic_teacher_grid(t_grid, beta_grid; t_scale::Real)
    patient_id = String[]
    t_h = Float64[]
    t_norm = Float64[]
    beta = Float64[]

    for (beta_idx, beta_value) in enumerate(beta_grid)
        for time_h in t_grid
            push!(patient_id, "synth$(beta_idx)")
            push!(t_h, Float64(time_h))
            push!(t_norm, Float64(time_h / t_scale))
            push!(beta, Float64(beta_value))
        end
    end

    X = [t_norm'; beta']
    size(X) == (2, length(t_norm)) ||
        error("Invalid symbolic teacher matrix size: $(size(X))")

    return (
        patient_id=patient_id,
        t_h=t_h,
        t_norm=t_norm,
        beta=beta,
        X=X,
        t_grid=Float64.(collect(t_grid)),
        beta_grid=Float64.(collect(beta_grid)),
    )
end

"""
    evaluate_symbolic_nn_teacher(chain, neural_params, X)

Evaluate the selected cUDE neural correction on a `2 x N` teacher grid.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function evaluate_symbolic_nn_teacher(chain, neural_params, X)
    y = Vector{Float64}(undef, size(X, 2))

    for idx in axes(X, 2)
        y[idx] = chain((@view X[:, idx]), neural_params)[1]
    end

    all(isfinite, y) || error("Symbolic teacher target contains non-finite values.")
    return y
end

# =============================================================================
# Neural-Correction Early Bump Analysis
# =============================================================================

"""
    cude_patient_beta_dataframe(params_df; cohort)

Build a validated patient-level beta table for the step 04c descriptive
analysis. The `beta` column is expected on the natural scale.
"""
# Used by: scripts/04c_run_neural_correction_bump_analysis.jl.
function cude_patient_beta_dataframe(params_df::DataFrame; cohort::AbstractString)
    required = (:patient_id, :beta)
    missing_columns = setdiff(required, propertynames(params_df))
    isempty(missing_columns) ||
        error("cUDE parameter table for $(cohort) is missing columns: $(join(missing_columns, ", ")).")

    patient_id = String[]
    beta = Float64[]

    for row in eachrow(params_df)
        value = Float64(row.beta)
        if isfinite(value) && value > 0
            push!(patient_id, String(row.patient_id))
            push!(beta, value)
        end
    end

    isempty(beta) && error("No finite positive beta values found for $(cohort).")

    return DataFrame(
        cohort=fill(String(cohort), length(beta)),
        patient_id=patient_id,
        beta=beta,
    )
end

"""
    neural_correction_beta_grid(patient_beta_df; n_points)

Return a regular beta grid spanning the observed patient-specific beta range.
"""
# Used by: scripts/04c_run_neural_correction_bump_analysis.jl.
function neural_correction_beta_grid(patient_beta_df::DataFrame; n_points::Integer)
    n_points > 0 || error("beta grid requires a positive number of points.")
    beta_values = Float64.(patient_beta_df.beta)
    beta_min = minimum(beta_values)
    beta_max = maximum(beta_values)
    beta_min == beta_max && return fill(beta_min, n_points)
    return collect(range(beta_min, beta_max; length=n_points))
end

"""
    classify_neural_correction_curve(tnorm, values; ...)

Classify early non-monotonicity in a neural-correction curve. This is a
descriptive thresholded summary of a local neural-network feature, not a
biological claim.
"""
# Used by: src/symbolic_regression.jl (patient and beta-grid analyses).
function classify_neural_correction_curve(
    tnorm::AbstractVector,
    values::AbstractVector;
    early_window_tnorm::Real,
    min_curve_range::Real,
    min_abs_drop::Real,
    min_rel_drop::Real,
)
    length(tnorm) == length(values) ||
        error("Neural-correction curve time/value vectors must have the same length.")

    y = Float64.(values)
    t = Float64.(tnorm)

    if isempty(y) || any(!isfinite, y) || any(!isfinite, t)
        return (
            bump_flag=false,
            tnorm_peak=missing,
            peak_value=missing,
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=missing,
            relative_drop=missing,
            nonmonotonicity_score=0.0,
            no_bump_reason="invalid_curve",
            curve_range=missing,
        )
    end

    curve_range = maximum(y) - minimum(y)
    if curve_range < min_curve_range
        return (
            bump_flag=false,
            tnorm_peak=missing,
            peak_value=missing,
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=0.0,
            relative_drop=0.0,
            nonmonotonicity_score=0.0,
            no_bump_reason="near_flat_curve",
            curve_range=curve_range,
        )
    end

    early_idxs = findall(<=(Float64(early_window_tnorm)), t)
    isempty(early_idxs) && return (
        bump_flag=false,
        tnorm_peak=missing,
        peak_value=missing,
        tnorm_valley=missing,
        valley_value=missing,
        absolute_drop=0.0,
        relative_drop=0.0,
        nonmonotonicity_score=0.0,
        no_bump_reason="no_early_window_points",
        curve_range=curve_range,
    )

    local_peak_pos = argmax(@view y[early_idxs])
    peak_idx = early_idxs[local_peak_pos]
    if peak_idx == last(early_idxs)
        return (
            bump_flag=false,
            tnorm_peak=t[peak_idx],
            peak_value=y[peak_idx],
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=0.0,
            relative_drop=0.0,
            nonmonotonicity_score=0.0,
            no_bump_reason="early_peak_at_window_boundary",
            curve_range=curve_range,
        )
    end

    if peak_idx >= length(y)
        return (
            bump_flag=false,
            tnorm_peak=t[peak_idx],
            peak_value=y[peak_idx],
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=0.0,
            relative_drop=0.0,
            nonmonotonicity_score=0.0,
            no_bump_reason="no_post_peak_points",
            curve_range=curve_range,
        )
    end

    post_peak_minimum = minimum(@view y[(peak_idx + 1):end])
    post_peak_minimum_idx = peak_idx + argmin(@view y[(peak_idx + 1):end])
    absolute_drop = y[peak_idx] - post_peak_minimum
    relative_drop = absolute_drop / curve_range
    score = max(absolute_drop, 0.0)
    bump_flag = absolute_drop >= min_abs_drop && relative_drop >= min_rel_drop

    return (
        bump_flag=bump_flag,
        tnorm_peak=t[peak_idx],
        peak_value=y[peak_idx],
        tnorm_valley=t[post_peak_minimum_idx],
        valley_value=y[post_peak_minimum_idx],
        absolute_drop=absolute_drop,
        relative_drop=relative_drop,
        nonmonotonicity_score=score,
        no_bump_reason=bump_flag ? "" : "insufficient_drop",
        curve_range=curve_range,
    )
end

"""
    classify_neural_correction_feature_curve(tnorm, values; ...)

Classify the extended-domain local neural-network feature visible in 04a by
searching for a local peak followed by a local valley inside a configurable
feature window. This is a descriptive diagnostic only.
"""
# Used by: src/symbolic_regression.jl (step 04c feature analysis).
function classify_neural_correction_feature_curve(
    tnorm::AbstractVector,
    values::AbstractVector;
    feature_window_tnorm,
    min_curve_range::Real,
    min_abs_drop::Real,
    min_rel_drop::Real,
)
    length(tnorm) == length(values) ||
        error("Neural-correction curve time/value vectors must have the same length.")

    y = Float64.(values)
    t = Float64.(tnorm)

    if isempty(y) || any(!isfinite, y) || any(!isfinite, t)
        return (
            bump_flag=false,
            tnorm_peak=missing,
            peak_value=missing,
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=missing,
            relative_drop=missing,
            nonmonotonicity_score=0.0,
            no_bump_reason="invalid_curve",
            curve_range=missing,
        )
    end

    window = Tuple(Float64.(feature_window_tnorm))
    length(window) == 2 && window[1] < window[2] ||
        error("feature_window_tnorm must be a two-value increasing tuple.")

    feature_idxs = findall(t_value -> window[1] <= t_value <= window[2], t)
    if length(feature_idxs) < 3
        return (
            bump_flag=false,
            tnorm_peak=missing,
            peak_value=missing,
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=0.0,
            relative_drop=0.0,
            nonmonotonicity_score=0.0,
            no_bump_reason="no_local_peak_in_feature_window",
            curve_range=missing,
        )
    end

    local_range = maximum(@view y[feature_idxs]) - minimum(@view y[feature_idxs])
    if local_range < min_curve_range
        return (
            bump_flag=false,
            tnorm_peak=missing,
            peak_value=missing,
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=0.0,
            relative_drop=0.0,
            nonmonotonicity_score=0.0,
            no_bump_reason="near_flat_curve",
            curve_range=local_range,
        )
    end

    best = nothing
    for peak_idx in feature_idxs[2:(end - 1)]
        is_local_peak = y[peak_idx] > y[peak_idx - 1] && y[peak_idx] >= y[peak_idx + 1]
        is_local_peak || continue

        valley_candidates = feature_idxs[feature_idxs .> peak_idx]
        isempty(valley_candidates) && continue
        valley_idx = valley_candidates[argmin(@view y[valley_candidates])]
        absolute_drop = y[peak_idx] - y[valley_idx]
        relative_drop = absolute_drop / local_range
        candidate = (
            peak_idx=peak_idx,
            valley_idx=valley_idx,
            absolute_drop=absolute_drop,
            relative_drop=relative_drop,
        )
        if best === nothing || candidate.absolute_drop > best.absolute_drop
            best = candidate
        end
    end

    if best === nothing
        return (
            bump_flag=false,
            tnorm_peak=missing,
            peak_value=missing,
            tnorm_valley=missing,
            valley_value=missing,
            absolute_drop=0.0,
            relative_drop=0.0,
            nonmonotonicity_score=0.0,
            no_bump_reason="no_local_peak_in_feature_window",
            curve_range=local_range,
        )
    end

    score = max(best.absolute_drop, 0.0)
    bump_flag = best.absolute_drop >= min_abs_drop && best.relative_drop >= min_rel_drop

    return (
        bump_flag=bump_flag,
        tnorm_peak=t[best.peak_idx],
        peak_value=y[best.peak_idx],
        tnorm_valley=t[best.valley_idx],
        valley_value=y[best.valley_idx],
        absolute_drop=best.absolute_drop,
        relative_drop=best.relative_drop,
        nonmonotonicity_score=score,
        no_bump_reason=bump_flag ? "" : "insufficient_feature_drop",
        curve_range=local_range,
    )
end

function _classify_neural_correction_curve(tnorm_grid, curve, settings; classification::Symbol)
    if classification === :feature
        return classify_neural_correction_feature_curve(
            tnorm_grid,
            curve;
            feature_window_tnorm=settings.feature_window_tnorm,
            min_curve_range=settings.feature_min_curve_range,
            min_abs_drop=settings.feature_min_abs_drop,
            min_rel_drop=settings.feature_min_rel_drop,
        )
    elseif classification === :early
        return classify_neural_correction_curve(
            tnorm_grid,
            curve;
            early_window_tnorm=settings.early_window_tnorm,
            min_curve_range=settings.min_curve_range,
            min_abs_drop=settings.min_abs_drop,
            min_rel_drop=settings.min_rel_drop,
        )
    else
        error("Unsupported neural-correction classification mode: $(classification)")
    end
end

"""
    neural_correction_curve(chain, neural_params, tnorm_grid, beta)

Evaluate `N_phi(tnorm, beta)` for one natural-scale beta value.
"""
# Used by: scripts/04c_run_neural_correction_bump_analysis.jl.
function neural_correction_curve(chain, neural_params, tnorm_grid, beta::Real)
    beta_value = Float64(beta)
    return [Float64(chain([Float64(tnorm), beta_value], neural_params)[1]) for tnorm in tnorm_grid]
end

"""
    neural_correction_bump_dataframe(chain, neural_params, beta_df, tnorm_grid, settings; domain)

Evaluate and classify patient-specific neural-correction curves.
"""
# Used by: scripts/04c_run_neural_correction_bump_analysis.jl.
function neural_correction_bump_dataframe(
    chain,
    neural_params,
    beta_df::DataFrame,
    tnorm_grid,
    settings;
    domain::AbstractString,
    classification::Symbol=:early,
)
    rows = NamedTuple[]

    for row in eachrow(beta_df)
        curve = neural_correction_curve(chain, neural_params, tnorm_grid, row.beta)
        curve_classification = _classify_neural_correction_curve(
            tnorm_grid,
            curve,
            settings;
            classification=classification,
        )

        push!(rows, (
            domain=String(domain),
            cohort=String(row.cohort),
            patient_id=String(row.patient_id),
            beta=Float64(row.beta),
            bump_flag=curve_classification.bump_flag,
            tnorm_peak=curve_classification.tnorm_peak,
            peak_value=curve_classification.peak_value,
            tnorm_valley=curve_classification.tnorm_valley,
            valley_value=curve_classification.valley_value,
            absolute_drop=curve_classification.absolute_drop,
            relative_drop=curve_classification.relative_drop,
            nonmonotonicity_score=curve_classification.nonmonotonicity_score,
            no_bump_reason=curve_classification.no_bump_reason,
            curve_range=curve_classification.curve_range,
        ))
    end

    return DataFrame(rows)
end

"""
    neural_correction_beta_grid_dataframe(chain, neural_params, beta_grid, tnorm_grid, settings; domain)

Evaluate and classify regular-grid beta curves.
"""
# Used by: scripts/04c_run_neural_correction_bump_analysis.jl.
function neural_correction_beta_grid_dataframe(
    chain,
    neural_params,
    beta_grid,
    tnorm_grid,
    settings;
    domain::AbstractString,
    classification::Symbol=:early,
)
    rows = NamedTuple[]

    for beta_value in Float64.(collect(beta_grid))
        curve = neural_correction_curve(chain, neural_params, tnorm_grid, beta_value)
        curve_classification = _classify_neural_correction_curve(
            tnorm_grid,
            curve,
            settings;
            classification=classification,
        )

        for (tnorm, y_value) in zip(tnorm_grid, curve)
            push!(rows, (
                domain=String(domain),
                beta=beta_value,
                tnorm=Float64(tnorm),
                y_nn=Float64(y_value),
                bump_flag=curve_classification.bump_flag,
                tnorm_peak=curve_classification.tnorm_peak,
                peak_value=curve_classification.peak_value,
                tnorm_valley=curve_classification.tnorm_valley,
                valley_value=curve_classification.valley_value,
                absolute_drop=curve_classification.absolute_drop,
                relative_drop=curve_classification.relative_drop,
                nonmonotonicity_score=curve_classification.nonmonotonicity_score,
                no_bump_reason=curve_classification.no_bump_reason,
                curve_range=curve_classification.curve_range,
            ))
        end
    end

    return DataFrame(rows)
end

_median_iqr(values) = begin
    finite_values = [Float64(v) for v in values if !ismissing(v) && isfinite(Float64(v))]
    isempty(finite_values) && return (median=missing, q1=missing, q3=missing)
    (median=median(finite_values), q1=quantile(finite_values, 0.25), q3=quantile(finite_values, 0.75))
end

"""
    neural_correction_bump_summary(patient_df)

Build cohort-level summaries for patient-specific early non-monotonicity.
"""
# Used by: scripts/04c_run_neural_correction_bump_analysis.jl.
function neural_correction_bump_summary(patient_df::DataFrame)
    rows = NamedTuple[]

    for sub in groupby(patient_df, [:domain, :cohort])
        total = nrow(sub)
        n_bump = count(Bool.(sub.bump_flag))
        n_no_bump = total - n_bump
        beta_all = _median_iqr(sub.beta)
        beta_bump = _median_iqr(sub[sub.bump_flag .== true, :beta])
        beta_no_bump = _median_iqr(sub[sub.bump_flag .== false, :beta])
        score = _median_iqr(sub.nonmonotonicity_score)

        push!(rows, (
            domain=String(sub.domain[1]),
            cohort=String(sub.cohort[1]),
            n_patients=total,
            n_bump=n_bump,
            pct_bump=100 * n_bump / total,
            n_no_bump=n_no_bump,
            pct_no_bump=100 * n_no_bump / total,
            beta_median=beta_all.median,
            beta_q1=beta_all.q1,
            beta_q3=beta_all.q3,
            beta_bump_median=beta_bump.median,
            beta_bump_q1=beta_bump.q1,
            beta_bump_q3=beta_bump.q3,
            beta_no_bump_median=beta_no_bump.median,
            beta_no_bump_q1=beta_no_bump.q1,
            beta_no_bump_q3=beta_no_bump.q3,
            score_median=score.median,
            score_q1=score.q1,
            score_q3=score.q3,
        ))
    end

    return DataFrame(rows)
end

function _parse_mimic_datetime(value)
    if ismissing(value) || value === nothing || value == ""
        return missing
    elseif value isa DateTime
        return value
    else
        return DateTime(replace(String(value), " " => "T"))
    end
end

function _read_workflow_anchor_patients(path::AbstractString)
    ids = DataFrame(XLSX.readtable(path, "IDs", "A:B", header=false, infer_eltypes=true))
    times = DataFrame(XLSX.readtable(path, "times", "A:Z", header=false, infer_eltypes=true))
    nrow(ids) == nrow(times) ||
        error("MIMIC-IV anchor ID/time sheets have different row counts in $(path).")

    records = NamedTuple[]
    for idx in 1:nrow(ids)
        source_id = String(ids[idx, 1])
        patient_id = String(ids[idx, 2])
        parts = split(source_id, "_")
        length(parts) == 2 || continue
        workflow_times = Float64[
            Float64(value) for value in collect(values(times[idx, :]))
            if !ismissing(value) && value !== nothing
        ]
        push!(records, (
            patient_id=patient_id,
            source_id=source_id,
            subject_id=parse(Int, parts[1]),
            hadm_id=parse(Int, parts[2]),
            workflow_times=workflow_times,
        ))
    end
    return records
end

function _read_mimic_charttime_map(path::AbstractString)
    df = CSV.read(path, DataFrame)
    required = (:subject_id, :hadm_id, :charttime)
    missing_columns = setdiff(required, propertynames(df))
    isempty(missing_columns) ||
        error("MIMIC-IV troponin table is missing columns: $(join(missing_columns, ", ")).")

    chart_map = Dict{Tuple{Int,Int},Vector{DateTime}}()
    for row in eachrow(df)
        key = (Int(row.subject_id), Int(row.hadm_id))
        charttime = _parse_mimic_datetime(row.charttime)
        ismissing(charttime) && continue
        push!(get!(chart_map, key, DateTime[]), charttime)
    end
    for values in values(chart_map)
        sort!(values)
    end
    return chart_map
end

function _read_mimic_admission_map(path::AbstractString)
    df = CSV.read(path, DataFrame)
    required = (:subject_id, :hadm_id, :admittime, :edregtime)
    missing_columns = setdiff(required, propertynames(df))
    isempty(missing_columns) ||
        error("MIMIC-IV admission table is missing columns: $(join(missing_columns, ", ")).")

    admission_map = Dict{Tuple{Int,Int},NamedTuple}()
    for row in eachrow(df)
        key = (Int(row.subject_id), Int(row.hadm_id))
        admission_map[key] = (
            admittime=_parse_mimic_datetime(row.admittime),
            edregtime=_parse_mimic_datetime(row.edregtime),
        )
    end
    return admission_map
end

function _hours_from_anchor(charttimes::Vector{DateTime}, anchor)
    ismissing(anchor) && return Float64[]
    return [Dates.value(charttime - anchor) / (1000 * 60 * 60) for charttime in charttimes]
end

function _subsequence_match(workflow_times::Vector{Float64}, candidate_times::Vector{Float64}; tol::Real=1e-3)
    isempty(workflow_times) && return (matched=false, n_points=0, max_error=missing)
    isempty(candidate_times) && return (matched=false, n_points=0, max_error=missing)

    wf_idx = 1
    max_error = 0.0
    for candidate in candidate_times
        wf_idx > length(workflow_times) && break
        err = abs(workflow_times[wf_idx] - candidate)
        if err <= tol
            max_error = max(max_error, err)
            wf_idx += 1
        end
    end

    matched = wf_idx > length(workflow_times)
    return (
        matched=matched,
        n_points=wf_idx - 1,
        max_error=matched ? max_error : missing,
    )
end

"""
    reconstruct_mimic_anchor_source_table(; patient_file, troponin_csv, admission_csv)

Reconstruct MIMIC-IV anchor-source metadata from trusted local raw artifacts.
Only unique full-sequence matches are assigned as `edregtime` or `admittime`.
"""
# Used by: scripts/04c_run_neural_correction_bump_analysis.jl when enabled by config.
function reconstruct_mimic_anchor_source_table(;
    patient_file::AbstractString,
    troponin_csv::AbstractString,
    admission_csv::AbstractString,
)
    patients = _read_workflow_anchor_patients(patient_file)
    chart_map = _read_mimic_charttime_map(troponin_csv)
    admission_map = _read_mimic_admission_map(admission_csv)
    rows = NamedTuple[]

    for patient in patients
        key = (patient.subject_id, patient.hadm_id)
        if !haskey(chart_map, key) || !haskey(admission_map, key)
            push!(rows, (
                patient_id=patient.patient_id,
                source_id=patient.source_id,
                subject_id=patient.subject_id,
                hadm_id=patient.hadm_id,
                anchor_source="not_available",
                anchor_match_status="not_available",
                anchor_match_error=missing,
                anchor_match_n_points=0,
            ))
            continue
        end

        charttimes = chart_map[key]
        anchors = admission_map[key]
        admit_match = _subsequence_match(patient.workflow_times, _hours_from_anchor(charttimes, anchors.admittime))
        ed_match = _subsequence_match(patient.workflow_times, _hours_from_anchor(charttimes, anchors.edregtime))
        matches = String[]
        admit_match.matched && push!(matches, "admittime")
        ed_match.matched && push!(matches, "edregtime")

        anchor_source = length(matches) == 1 ? matches[1] :
                        length(matches) > 1 ? "ambiguous" :
                        "unmatched"
        selected_match = anchor_source == "admittime" ? admit_match :
                         anchor_source == "edregtime" ? ed_match :
                         (matched=false, n_points=max(admit_match.n_points, ed_match.n_points), max_error=missing)

        push!(rows, (
            patient_id=patient.patient_id,
            source_id=patient.source_id,
            subject_id=patient.subject_id,
            hadm_id=patient.hadm_id,
            anchor_source=anchor_source,
            anchor_match_status=anchor_source,
            anchor_match_error=selected_match.max_error,
            anchor_match_n_points=selected_match.n_points,
        ))
    end

    return DataFrame(rows)
end

"""
    symbolic_teacher_dataframe(grid, y)

Build the canonical symbolic-regression teacher dataset table.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_teacher_dataframe(grid, y)
    length(y) == length(grid.t_norm) ||
        error("Teacher target length $(length(y)) does not match grid length $(length(grid.t_norm)).")

    return DataFrame(
        patient_id=grid.patient_id,
        t_h=grid.t_h,
        t_norm=grid.t_norm,
        beta=grid.beta,
        y_nn=y,
    )
end

# =============================================================================
# Symbolic Regression
# =============================================================================

"""
    smooth_relu_fast(x; eps_value=1e-5)

Smooth positive-part approximation used by the bounded symbolic-regression loss.
"""
# Used by: src/symbolic_regression.jl (build_symbolic_regression_loss).
smooth_relu_fast(x; eps_value::Real=1e-5) = 0.5 * (x + sqrt(x * x + eps_value * eps_value))

"""
    build_symbolic_regression_loss(settings)

Return the bounded elementwise loss used during symbolic regression.
"""
# Used by: src/symbolic_regression.jl (build_symbolic_regression_options).
function build_symbolic_regression_loss(settings)
    lambda_negative = settings.lambda_negative
    lambda_high = settings.lambda_high
    smooth_eps = settings.smooth_eps

    return (y_pred, y_true) ->
        (y_pred - y_true)^2 +
        lambda_negative * smooth_relu_fast(-y_pred; eps_value=smooth_eps)^2 +
        lambda_high * smooth_relu_fast(y_pred - 1.0; eps_value=smooth_eps)^2
end

"""
    build_symbolic_regression_options(settings, output_directory)

Create `SymbolicRegression.Options` from workflow config.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function build_symbolic_regression_options(settings, output_directory::AbstractString)
    return Options(
        binary_operators=settings.binary_operators,
        unary_operators=settings.unary_operators,
        maxsize=settings.maxsize,
        populations=settings.populations,
        parsimony=settings.parsimony,
        complexity_of_constants=settings.complexity_of_constants,
        batching=settings.batching,
        batch_size=settings.batch_size,
        should_optimize_constants=settings.should_optimize_constants,
        elementwise_loss=build_symbolic_regression_loss(settings),
        output_directory=output_directory,
        save_to_file=true,
        seed=settings.seed,
    )
end

"""
    run_symbolic_regression_search(X, y, settings, options)

Run the warm-up and main symbolic-regression searches, returning the main hall
of fame object.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function run_symbolic_regression_search(X, y, settings, options)
    variable_names = collect(settings.variable_names)

    if settings.niterations_warmup > 0
        @info "Running symbolic-regression warm-up." niterations=settings.niterations_warmup
        equation_search(
            X,
            y;
            niterations=settings.niterations_warmup,
            options=options,
            parallelism=:multithreading,
            progress=settings.progress_bars,
            variable_names=variable_names,
        )
    end

    @info "Running main symbolic-regression search." niterations=settings.niterations_main
    return equation_search(
        X,
        y;
        niterations=settings.niterations_main,
        options=options,
        parallelism=:multithreading,
        progress=settings.progress_bars,
        variable_names=variable_names,
    )
end

"""
    symbolic_sr_eval(tree, X)

Evaluate one SymbolicRegression tree and replace non-finite outputs with the
configured large penalty value.
"""
# Used by: scripts/04a_run_symbolic_regression.jl and src/symbolic_regression.jl.
function symbolic_sr_eval(tree, X)
    out = eval_tree_array(tree, X)
    vals = out isa Tuple ? collect(out[1]) : collect(out)
    y = Float64.(vals)

    for idx in eachindex(y)
        if !isfinite(y[idx])
            y[idx] = 1e6
        end
    end

    return y
end

"""
    select_symbolic_regression_model(hof, X_teacher, y_teacher, settings, options)

Select the simplest Pareto member whose teacher-grid MSE is within the
configured tolerance of the minimum teacher-grid MSE.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function select_symbolic_regression_model(hof, X_teacher, y_teacher, settings, options)
    frontier = calculate_pareto_frontier(hof)
    isempty(frontier) && error("Symbolic regression produced an empty Pareto frontier.")

    teacher_mses = Float64[]
    complexities = Int[]

    for member in frontier
        y_hat_teacher = symbolic_sr_eval(member.tree, X_teacher)
        push!(teacher_mses, mean((y_teacher .- y_hat_teacher) .^ 2))
        push!(complexities, compute_complexity(member, options))
    end

    best_idx = select_symbolic_regression_index(
        teacher_mses,
        complexities;
        tolerance=settings.teacher_mse_tolerance,
    )

    best = frontier[best_idx]
    equation = string_tree(best.tree; variable_names=collect(settings.variable_names))

    return (
        frontier=frontier,
        best=best,
        best_idx=best_idx,
        equation=equation,
        teacher_mse=teacher_mses[best_idx],
        complexity=complexities[best_idx],
        teacher_mses=teacher_mses,
        complexities=complexities,
    )
end

"""
    select_symbolic_regression_index(teacher_mses, complexities; tolerance)

Return the least-complex candidate whose teacher-grid MSE is within `tolerance`
of the minimum MSE.
"""
# Used by: src/symbolic_regression.jl.
function select_symbolic_regression_index(teacher_mses, complexities; tolerance::Real)
    isempty(teacher_mses) && error("Cannot select a symbolic surrogate from an empty candidate set.")
    length(teacher_mses) == length(complexities) ||
        error("Symbolic surrogate MSE and complexity vectors must have the same length.")

    minimum_mse_idx = argmin(teacher_mses)
    near_best = findall(teacher_mses .<= tolerance * teacher_mses[minimum_mse_idx])
    return near_best[argmin(complexities[near_best])]
end

"""
    symbolic_equation_eval(equation, X)

Evaluate a trusted symbolic equation on a `2 x N` teacher grid.
"""
# Used by: src/symbolic_regression.jl (report-only symbolic selection).
function symbolic_equation_eval(equation::AbstractString, X)
    body = Meta.parse(equation; raise=true)
    correction = Core.eval(@__MODULE__, :((t_norm, β) -> $body))
    # Report mode evaluates a newly compiled callable immediately. invokelatest
    # is intentionally confined here: candidate equations are never injected
    # into the step 04b ODE solve.
    return [Float64(Base.invokelatest(correction, X[1, idx], X[2, idx])) for idx in axes(X, 2)]
end

"""
    symbolic_teacher_arrays(teacher_table; t_scale)

Reconstruct the teacher matrix and target vector from the stable step 04a CSV.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_teacher_arrays(teacher_table::DataFrame; t_scale::Real)
    required = (:t_h, :t_norm, :beta, :y_nn)
    missing_columns = setdiff(required, propertynames(teacher_table))
    isempty(missing_columns) ||
        error("Symbolic teacher table is missing columns: $(join(missing_columns, ", ")).")

    t_h = Float64.(teacher_table.t_h)
    t_norm = Float64.(teacher_table.t_norm)
    beta = Float64.(teacher_table.beta)
    y = Float64.(teacher_table.y_nn)

    all(isapprox.(t_norm, t_h ./ t_scale)) ||
        error("Symbolic teacher table is inconsistent with t_scale=$(t_scale).")

    return (
        X=[t_norm'; beta'],
        y=y,
        t_grid=unique(t_h),
        beta_grid=unique(beta),
        training_points=length(y),
    )
end

"""
    select_symbolic_regression_model(frontier_table, X_teacher, y_teacher, settings)

Select a trusted equation from the stable Pareto-frontier CSV without rerunning
symbolic regression.
"""
# Used by: scripts/04a_run_symbolic_regression.jl (`report` mode).
function select_symbolic_regression_model(frontier_table::DataFrame, X_teacher, y_teacher, settings)
    required = (:idx, :complexity, :equation)
    missing_columns = setdiff(required, propertynames(frontier_table))
    isempty(missing_columns) ||
        error("Symbolic frontier table is missing columns: $(join(missing_columns, ", ")).")
    isempty(frontier_table.idx) && error("Symbolic frontier table is empty.")

    teacher_mses = [
        mean((y_teacher .- symbolic_equation_eval(String(equation), X_teacher)) .^ 2)
        for equation in frontier_table.equation
    ]
    complexities = Int.(frontier_table.complexity)
    position = select_symbolic_regression_index(
        teacher_mses,
        complexities;
        tolerance=settings.teacher_mse_tolerance,
    )
    symbolic_target = symbolic_equation_eval(String(frontier_table.equation[position]), X_teacher)

    return (
        best_idx=Int(frontier_table.idx[position]),
        equation=String(frontier_table.equation[position]),
        teacher_mse=teacher_mses[position],
        complexity=complexities[position],
        teacher_mses=teacher_mses,
        complexities=complexities,
        symbolic_target=symbolic_target,
    )
end

# =============================================================================
# Evaluation Tables
# =============================================================================

"""
    symbolic_frontier_dataframe(frontier, options, variable_names)

Build the canonical symbolic-regression Pareto frontier table.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_frontier_dataframe(frontier, options, variable_names)
    return DataFrame(
        idx=collect(1:length(frontier)),
        complexity=[compute_complexity(member, options) for member in frontier],
        loss=[member.loss for member in frontier],
        equation=[string_tree(member.tree; variable_names=collect(variable_names)) for member in frontier],
    )
end

"""
    symbolic_grid_metrics(y_true, y_pred)

Return MSE, MAE, and R2 for symbolic surrogate predictions on a grid.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function symbolic_grid_metrics(y_true, y_pred)
    mse = mean((y_true .- y_pred) .^ 2)
    mae = mean(abs.(y_true .- y_pred))
    r2 = 1 - sum((y_true .- y_pred) .^ 2) / sum((y_true .- mean(y_true)) .^ 2)
    return (mse=mse, mae=mae, r2=r2)
end

"""
    build_symbolic_plot_curves(chain, neural_params, tree, t_grid, beta_values; t_scale)

Build NN and symbolic surrogate curves used by step 04a comparison plots.
"""
# Used by: scripts/04a_run_symbolic_regression.jl.
function build_symbolic_plot_curves(chain, neural_params, tree, t_grid, beta_values; t_scale::Real)
    records = Vector{Any}()
    t_values = Float64.(collect(t_grid))

    for beta_value in beta_values
        beta_float = Float64(beta_value)
        y_nn = [chain([time_h / t_scale, beta_float], neural_params)[1] for time_h in t_values]
        X_tmp = hcat([[time_h / t_scale, beta_float] for time_h in t_values]...)
        y_sr = symbolic_sr_eval(tree, X_tmp)

        push!(records, (
            beta=beta_float,
            t_h=t_values,
            y_nn=Float64.(y_nn),
            y_sr=y_sr,
        ))
    end

    return records
end

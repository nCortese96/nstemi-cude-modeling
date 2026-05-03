using StableRNGs, DataFrames, StatsBase, XLSX, CSV, Random
using Statistics
using Dates
using Logging

using Revise
includet("ctnt-ude-model.jl")
includet("dataset_report.jl")

@info "═══ Dataset Report Script started $(now()) ═══"

T_SCALE = 240.0;

# ── Parametri di filtering (identici a main_cude_min_max.jl) ───────────────
meas_min_number = 5;
min_acq_time_before = 12.0;
min_acq_n_before = 1;
min_acq_time_after = 48.0;
min_acq_n_after = 1;
min_time = 72.0;
max_gap = 72.0;

# ══════════════════════════════════════════════════════════════════════════════
# Funzione che esegue la pipeline completa su un dataset e ne fa il report
# ══════════════════════════════════════════════════════════════════════════════
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
    report_dir::String="res"
)

    reporter = DatasetReporter(dataset_name; report_dir=report_dir)

    # ─── STEP 1: Caricamento raw ─────────────────────────────────────────
    file_path = "data/$(dataset_path)"
    sheet_ids = "IDs"
    sheet_times = "times"
    sheet_values = "values"

    xf = XLSX.readxlsx(file_path)
    ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true))
    timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true))
    troponin_df = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true))

    patients = [row2Patient(ids[i, :], timepoints_df[i, :], troponin_df[i, :]) for i in 1:nrow(ids)]

    report_step!(reporter, "Raw dataset loaded", patients;
        extra_info="File: $(file_path)")

    # ─── STEP 2: Collapse duplicati temporali ────────────────────────────
    dup_counts = patient_duplicate_time_counts(patients; tol=0.0)
    (nmod, nrm) = collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

    report_step!(reporter, "Duplicate timepoints collapsed", patients;
        extra_info="Pazienti modificati: $(nmod), punti rimossi: $(nrm)")

    # ─── STEP 3: Trimming a T_SCALE ──────────────────────────────────────
    trimmed_p = trim_time(patients, T_SCALE)

    report_step!(reporter, "Trimmed to $(T_SCALE)h", trimmed_p)

    # ─── STEP 4: Anomaly filtering (All Eligible) ────────────────────────
    anoms = find_anomalies(trimmed_p,
        meas_min_number,
        min_acq_time_before, min_acq_n_before,
        min_acq_time_after, min_acq_n_after,
        min_time;
        max_gap_h=max_gap,
        verbose=false
    )

    cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p)

    filter_info = string(
        "Filtro: min_meas=$(meas_min_number), ",
        "before=$(min_acq_n_before)×$(min_acq_time_before)h, ",
        "after=$(min_acq_n_after)×$(min_acq_time_after)h, ",
        "min_span=$(min_time)h, max_gap=$(max_gap)h. ",
        "Rimossi: $(length(anoms))"
    )
    report_step!(reporter, "Anomaly filtering (All Eligible)", cleaned_patients;
        extra_info=filter_info)

    # ─── STEP 5: Salvataggio IDs all eligible ────────────────────────────
    df_ae_ids = DataFrame(patient=[p.id for p in cleaned_patients])
    # CSV.write(joinpath(report_dir, "ids_all_eligible_$(dataset_name).csv"), df_ae_ids)

    # ─── STEP 6: Train / Test split (80/20) ──────────────────────────────
    Random.seed!(1234)
    rng = StableRNG(42)
    shuffle!(cleaned_patients)
    n_train = Int(round(length(cleaned_patients) * 0.8))
    training_dataset = cleaned_patients[1:n_train]
    test_dataset = cleaned_patients[n_train+1:end]

    report_step!(reporter, "Training split (80%)", training_dataset;
        extra_info="Split: $(n_train) training, $(length(test_dataset)) test")

    report_step!(reporter, "Test split (20%)", test_dataset)

    # ─── Chiusura ────────────────────────────────────────────────────────
    report_path = finalize_report(reporter)

    return (cleaned_patients=cleaned_patients,
        training=training_dataset,
        test=test_dataset,
        report_path=report_path)
end

# ══════════════════════════════════════════════════════════════════════════════
#  Esecuzione per entrambi i dataset
# ══════════════════════════════════════════════════════════════════════════════

# ─── MIMIC-IV ────────────────────────────────────────────────────────────────
@info "▶ Processing MIMIC-IV..."
mimic = run_dataset_report(
    dataset_name="MIMIC-IV",
    dataset_path="MIMIC-IV/NSTEMI_reorganized_skipped.xlsx",
    column_letter="B",
    T_SCALE=T_SCALE,
    meas_min_number=meas_min_number,
    min_acq_time_before=min_acq_time_before,
    min_acq_n_before=min_acq_n_before,
    min_acq_time_after=min_acq_time_after,
    min_acq_n_after=min_acq_n_after,
    min_time=min_time,
    max_gap=max_gap,
)

# ─── UMG ─────────────────────────────────────────────────────────────────────
@info "▶ Processing UMG..."
umg = run_dataset_report(
    dataset_name="UMG",
    dataset_path="UMG_NSTEMI_Dataset.xlsx",
    column_letter="A",
    T_SCALE=T_SCALE,
    meas_min_number=meas_min_number,
    min_acq_time_before=min_acq_time_before,
    min_acq_n_before=min_acq_n_before,
    min_acq_time_after=min_acq_time_after,
    min_acq_n_after=min_acq_n_after,
    min_time=min_time,
    max_gap=max_gap,
)

@info "═══ Report completati ═══"
@info "MIMIC-IV report: $(mimic.report_path)"
@info "UMG report:      $(umg.report_path)"

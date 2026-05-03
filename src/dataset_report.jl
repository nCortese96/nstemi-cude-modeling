using Statistics
using Dates

# ─────────────────────────────────────────────────────────────────────────────
# DatasetReporter — accumula step di preprocessing e riporta statistiche
# ─────────────────────────────────────────────────────────────────────────────

"""
    DatasetReporter

Accumula gli step di preprocessing del dataset e li riporta sia a terminale
sia su un file TXT.

# Uso tipico
```julia
reporter = DatasetReporter("UMG")
report_step!(reporter, "Raw dataset loaded", patients)
# … uno o più step di modifica …
report_step!(reporter, "After trimming", trimmed_patients)
finalize_report(reporter)
```
"""
mutable struct DatasetReporter
    dataset_name::String
    report_path::String       # percorso del file TXT
    io::IOStream              # file handle aperto
    step_count::Int
end

"""
    DatasetReporter(dataset_name; report_dir="res")

Crea un reporter e apre (sovrascrive) il file TXT in `report_dir`.
"""
function DatasetReporter(dataset_name::String; report_dir::String="res")
    mkpath(report_dir)
    ts = Dates.format(now(), "yyyymmdd_HHMMss")
    fname = "dataset_report_$(dataset_name)_$(ts).txt"
    report_path = joinpath(report_dir, fname)
    io = open(report_path, "w")

    header = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           DATASET PREPROCESSING REPORT                     ║
    ║  Dataset:  $(rpad(dataset_name, 46))║
    ║  Date:     $(rpad(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), 46))║
    ╚══════════════════════════════════════════════════════════════╝
    """

    _dual_print(io, header)
    return DatasetReporter(dataset_name, report_path, io, 0)
end

# ── Funzioni statistiche interne ────────────────────────────────────────────

"""
Calcola statistiche sintetiche su un vettore numerico, ignorando NaN.
Ritorna una NamedTuple (n, mean, std, median, q1, q3, min, max).
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

"""Stampa simultaneamente su IOStream e stdout."""
function _dual_print(io::IOStream, text::String)
    print(io, text)
    print(stdout, text)
    flush(io)
end

"""Formatta stat come: median [Q1–Q3] (min–max)."""
function _fmt_stat(s::NamedTuple; digits::Int=2)
    if isnan(s.median)
        return "N/A"
    end
    return string(
        round(s.median; digits),
        " [", round(s.q1; digits), "–", round(s.q3; digits), "]",
        " (", round(s.min; digits), "–", round(s.max; digits), ")"
    )
end

# ── report_step! ────────────────────────────────────────────────────────────

"""
    report_step!(reporter, step_name, patients; extra_info="")

Registra uno step della pipeline.  Calcola e stampa le statistiche chiave
del vettore `patients::AbstractVector{PatientData}` sia a terminale sia
sul file TXT.
"""
function report_step!(reporter::DatasetReporter,
    step_name::String,
    patients::AbstractVector{PatientData};
    extra_info::String="")

    reporter.step_count += 1
    step_n = reporter.step_count
    n_patients = length(patients)

    # ── metriche ────────────────────────────────────────────────────────────
    # Acquisizioni cTnT per paziente (solo non-NaN)
    ctnt_counts = [count(!isnan, p.ctnt_data) for p in patients]
    ctnt_counts_stats = _summary_stats(Float64.(ctnt_counts))
    total_ctnt = sum(ctnt_counts)

    # Timepoints totali (solo non-NaN)
    tp_counts = [count(!isnan, p.timepoints) for p in patients]
    total_tp = sum(tp_counts)

    # Primo timepoint (ore) – solo pazienti con almeno 1 non-NaN
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

        # max gap tra consecutivi
        if length(valid_tp) >= 2
            gaps = diff(valid_tp)
            push!(max_gaps, maximum(gaps))
        end
    end

    first_stats = _summary_stats(first_times)
    last_stats = _summary_stats(last_times)
    span_stats = _summary_stats(obs_spans)
    gap_stats = _summary_stats(max_gaps)

    # ── formattazione ───────────────────────────────────────────────────────
    sep = "─"^62
    block = """

    $(sep)
      STEP $(step_n): $(step_name)
    $(sep)
      Numero pazienti:                     $(n_patients)
      Acquisizioni cTnT totali:            $(total_ctnt)
      Timepoints totali:                   $(total_tp)
    $(extra_info == "" ? "" : "  Info aggiuntive:                     $(extra_info)\n")
      Acquisizioni per paziente:           $(_fmt_stat(ctnt_counts_stats))
      Primo timepoint (h):                 $(_fmt_stat(first_stats))
      Ultimo timepoint (h):                $(_fmt_stat(last_stats))
      Observation span (h):                $(_fmt_stat(span_stats))
      Max gap consecutivo (h):             $(_fmt_stat(gap_stats))
    $(sep)

    """

    _dual_print(reporter.io, block)
    return nothing
end

# ── finalize_report ─────────────────────────────────────────────────────────

"""
    finalize_report(reporter)

Stampa un sommario finale e chiude il file TXT.
"""
function finalize_report(reporter::DatasetReporter)
    footer = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  REPORT COMPLETATO                                         ║
    ║  Step totali registrati: $(lpad(reporter.step_count, 3))                                ║
    ║  File salvato in: $(rpad(reporter.report_path, 39))║
    ╚══════════════════════════════════════════════════════════════╝
    """
    _dual_print(reporter.io, footer)
    close(reporter.io)
    @info "Report chiuso: $(reporter.report_path)"
    return reporter.report_path
end

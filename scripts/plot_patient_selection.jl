# =============================================================================
# Patient Selection and Plotting (ODE and cUDE Overlay)
# =============================================================================

using DataFrames, CSV, JLD2
using Plots
using OrdinaryDiffEq: Tsit5
using SciMLBase: successful_retcode, ODEProblem, remake
using ComponentArrays: ComponentArray
using Random
using Statistics

# Configura Plots con Plotly/CairoMakie o vanilla GR (che e' il default)
# plots_theme(:dao)

include("ctnt-ude-model.jl")

# =========================== IO Directories =================================

const OUT_DIR = "res/patient_paper_selections"
mkpath(OUT_DIR)
rm(joinpath(OUT_DIR, "delta_smape_report.txt"), force=true)

# ODE Paths
const ODE_MIMIC_CSV = "res/NSTEMI_ODE_TdSigmoid/MIMIC-IV_opt_lambda1/models/params_out.csv"
const ODE_UMG_CSV = "res/NSTEMI_ODE_TdSigmoid/UMG_opt_lambda1/models/params_out.csv"

# cUDE Paths
const CUDE_BASE = "res/NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback/models"

const CUDE_MIMIC_METRICS = joinpath(CUDE_BASE, "MIMIC-IV_test_NN_3_ms_test", "patients_metrics_val.csv")
const CUDE_MIMIC_PARAMS = joinpath(CUDE_BASE, "MIMIC-IV_test_NN_3_ms_test", "patients_params_val.csv")

const CUDE_UMG_METRICS = joinpath(CUDE_BASE, "UMG_test_NN_ab10_3_ms_test", "patients_metrics_val.csv")
const CUDE_UMG_PARAMS = joinpath(CUDE_BASE, "UMG_test_NN_ab10_3_ms_test", "patients_params_val.csv")

const NN_PARAMS_JLD2 = joinpath(CUDE_BASE, "nnNSTEMI_NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback.jld2")
const BEST_NN_IDX = 3
const NN_DEPTH = 2
const NN_WIDTH = 8
const NN_IN_DIM = 2

# Dataset Paths
const JLD2_MIMIC_TRAIN = "res/MIMIC-IV_trainingset.jld2"
const JLD2_MIMIC_TEST = "res/MIMIC-IV_testset.jld2"
const JLD2_UMG_TEST = "res/UMG_testset.jld2"

# ======================== Load Global NN and Data ============================

@info "Loading Neural Network Params..."
@load NN_PARAMS_JLD2 neural_network_parameters
const BEST_NN = Vector{Float64}(neural_network_parameters[BEST_NN_IDX])
const CHAIN = neural_network_model(NN_DEPTH, NN_WIDTH; input_dims=NN_IN_DIM)

@info "Loading Patient Sequences..."
PATIENT_DICT = Dict{String,PatientData}()

function load_ds(path, key)
    if isfile(path)
        ds = JLD2.load(path, key)
        for p in ds
            PATIENT_DICT[p.id] = p
        end
    else
        @warn "File not found: $(path)"
    end
end

load_ds(JLD2_MIMIC_TRAIN, "training_dataset")
load_ds(JLD2_MIMIC_TEST, "test_dataset")
load_ds(JLD2_UMG_TEST, "test_dataset")

@info "Loaded $(length(PATIENT_DICT)) distinct patient profiles."

# =========================== Solving Functions ===============================

function solve_ode(params_log::Vector{Float64}, tmax::Float64)
    # ODE in param_log is [p1, p2, p3, p4, p5] -> a, b, Cs0, Cc0, Td (all log-scale)
    u0 = [exp(params_log[3]), exp(params_log[4]), 0.0]
    prob = ODEProblem(troponin_ode!, u0, (0.0, tmax + 10.0), params_log)
    sol = solve(prob, Tsit5(); saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || @warn "ODE solve failed"
    return sol.t, sol[3, :]
end

function solve_cude(params_natural::Vector{Float64}, nn_p::Vector{Float64}, tmax::Float64)
    # params_natural is [a, b, Cs0, Cc0, beta]
    θ_log = log.(params_natural .+ 1e-15) # safe log
    u0 = [params_natural[3], params_natural[4], 0.0]
    cude_f!(du, u, p, t) = ctnt_cude!(du, u, p, t, CHAIN)
    prob = ODEProblem(cude_f!, u0, (0.0, tmax + 10.0))
    p_full = ComponentArray(ode=θ_log, neural=nn_p)
    sol = solve(prob, Tsit5(); p=p_full, saveat=1.0, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || @warn "cUDE solve failed"
    return sol.t, sol[3, :]
end

# ======================== Plot Annotazioni e Stili ===========================

function plot_base(p_data::PatientData, patient_id::String, tmax::Float64)
    plt = Plots.plot(
        size=(800, 500),
        xlabel="Time (h)", ylabel="cTnT [ng/mL]",
        # title="Patient ID: $(patient_id)",
        legend=false,
        margin=5Plots.mm,
        grid=true
    )
    # Punti misurati
    Plots.scatter!(
        plt,
        p_data.timepoints, p_data.ctnt_data,
        markershape=:circle, color=:red,
        ms=6, markerstrokewidth=1.5,
        label="Observation"
    )
    return plt
end

function get_tmax(p_data::PatientData)
    return maximum(p_data.timepoints) + 10.0
end

# ========================== Parte 1: ODE Quartili ============================

function process_ode_quartiles(df_ode::DataFrame, dataset_name::String, dirname::String)
    df_ode = dropmissing(df_ode, :smape)
    sort!(df_ode, :smape)

    n = nrow(df_ode)
    q_len = div(n, 4)

    Random.seed!(42) # Riproducibilita'

    out_folder = joinpath(OUT_DIR, dirname)
    mkpath(out_folder)

    @info "ODE sMAPE Quartiles per $(dataset_name) (Total: $(n))"

    for (q_idx, start_idx) in enumerate([1, q_len + 1, 2 * q_len + 1, 3 * q_len + 1])
        end_idx = (q_idx == 4) ? n : start_idx + q_len - 1

        chunk = df_ode[start_idx:end_idx, :]
        if nrow(chunk) >= 10
            selected_rows = chunk[shuffle(1:nrow(chunk))[1:10], :]
        else
            selected_rows = chunk
        end

        for row in eachrow(selected_rows)
            pid = String(row.patient)
            haskey(PATIENT_DICT, pid) || continue
            p_data = PATIENT_DICT[pid]

            tmax = get_tmax(p_data)
            p_log = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]

            t_ode, y_ode = solve_ode(p_log, tmax)
            plt = plot_base(p_data, pid, tmax)

            # Annotazione spostata dentro il plot poichè la legenda è nascosta
            # ann_str = "ODE Model\nsMAPE: $(round(row.smape, digits=2))%\nRMSLE: $(round(row.rmsle, digits=4))"
            Plots.plot!(plt, t_ode, y_ode, lw=2, color=:blue)
            # Plots.annotate!(plt, tmax * 0.95, maximum(p_data.ctnt_data) * 0.95,
            #     Plots.text(ann_str, 9, :right, :top))

            Plots.savefig(plt, joinpath(out_folder, "ODE_Q$(q_idx)_$(pid).svg"))
        end
    end
end

@info "1) ODE Quartili MIMIC"
df_ode_mimic = CSV.read(ODE_MIMIC_CSV, DataFrame)
process_ode_quartiles(df_ode_mimic, "MIMIC-val", "ODE_Q_MIMIC")

@info "2) ODE Quartili UMG"
df_ode_umg = CSV.read(ODE_UMG_CSV, DataFrame)
process_ode_quartiles(df_ode_umg, "UMG-test", "ODE_Q_UMG")


# ========================== Parte 1B: cUDE Quartili ==========================

function process_cude_quartiles(cude_metrics_csv::String, cude_params_csv::String, dataset_name::String, dirname::String)
    df_cude_metrics = CSV.read(cude_metrics_csv, DataFrame)
    df_cude_params = CSV.read(cude_params_csv, DataFrame)

    merged = innerjoin(df_cude_metrics, df_cude_params, on=:patient_id, makeunique=true)
    merged = dropmissing(merged, :smape)
    sort!(merged, :smape)

    n = nrow(merged)
    q_len = div(n, 4)

    Random.seed!(42) # Riproducibilita'

    out_folder = joinpath(OUT_DIR, dirname)
    mkpath(out_folder)

    @info "cUDE sMAPE Quartiles per $(dataset_name) (Total: $(n))"

    for (q_idx, start_idx) in enumerate([1, q_len + 1, 2 * q_len + 1, 3 * q_len + 1])
        end_idx = (q_idx == 4) ? n : start_idx + q_len - 1

        chunk = merged[start_idx:end_idx, :]
        if nrow(chunk) >= 10
            selected_rows = chunk[shuffle(1:nrow(chunk))[1:10], :]
        else
            selected_rows = chunk
        end

        for row in eachrow(selected_rows)
            pid = String(row.patient_id)
            haskey(PATIENT_DICT, pid) || continue
            p_data = PATIENT_DICT[pid]

            tmax = get_tmax(p_data)
            p_nat_cude = Float64[row.a, row.b, row.Cs0, row.Cc0, row.beta]

            t_cude, y_cude = solve_cude(p_nat_cude, BEST_NN, tmax)
            plt = plot_base(p_data, pid, tmax)

            # ann_str = "cUDE Model\nsMAPE: $(round(row.smape, digits=2))%\nRMSLE: $(round(row.rmsle, digits=4))"
            Plots.plot!(plt, t_cude, y_cude, lw=2, color=:darkorange, linestyle=:dash)
            # Plots.annotate!(plt, tmax * 0.95, maximum(p_data.ctnt_data) * 0.95,
            #     Plots.text(ann_str, 9, :right, :top))

            Plots.savefig(plt, joinpath(out_folder, "cUDE_Q$(q_idx)_$(pid).svg"))
        end
    end
end

@info "3) cUDE Quartili MIMIC"
process_cude_quartiles(CUDE_MIMIC_METRICS, CUDE_MIMIC_PARAMS, "MIMIC-val", "cUDE_Q_MIMIC")

@info "4) cUDE Quartili UMG"
process_cude_quartiles(CUDE_UMG_METRICS, CUDE_UMG_PARAMS, "UMG-test", "cUDE_Q_UMG")


# ======================== Parte 2: cUDE vs ODE Overlap =======================

function process_overlap(df_ode::DataFrame, cude_metrics_csv::String, cude_params_csv::String, dataset_name::String, dirname::String)
    df_cude_metrics = CSV.read(cude_metrics_csv, DataFrame)
    df_cude_params = CSV.read(cude_params_csv, DataFrame)

    # Merge on patient id
    # df_ode has :patient, df_cude_metrics has :patient_id
    df_ode_renamed = rename(df_ode, :patient => :patient_id)

    merged = innerjoin(df_ode_renamed, df_cude_metrics, on=:patient_id, makeunique=true)
    merged = innerjoin(merged, df_cude_params, on=:patient_id, makeunique=true)

    # Rinomino per comodita'
    rename!(merged, :smape_1 => :smape_cude, :smape => :smape_ode, :rmsle_1 => :rmsle_cude, :rmsle => :rmsle_ode)

    merged.delta_smape = merged.smape_cude .- merged.smape_ode

    # --- REPORT PERCENTUALI DELTA sMAPE ---
    n_tot = nrow(merged)
    cude_wins = count(x -> x < -1.0, merged.delta_smape)
    ties = count(x -> abs(x) <= 1.0, merged.delta_smape)
    ode_wins = count(x -> x > 1.0, merged.delta_smape)

    report_path = joinpath(OUT_DIR, "delta_smape_report.txt")
    open(report_path, "a") do io
        println(io, "=== DELTA sMAPE REPORT ($(dataset_name)) ===")
        println(io, "Total Patients: ", n_tot)
        println(io, "cUDE Wins (ΔsMAPE < -1.0%): ", round(100 * cude_wins / n_tot, digits=2), "% (", cude_wins, " patients)")
        println(io, "Ties (|ΔsMAPE| ≤ 1.0%):     ", round(100 * ties / n_tot, digits=2), "% (", ties, " patients)")
        println(io, "ODE Wins (ΔsMAPE > 1.0%):   ", round(100 * ode_wins / n_tot, digits=2), "% (", ode_wins, " patients)")
        println(io, "-----------------------------------------")
        println(io, "Patient Breakdown:")
        for row in eachrow(sort(merged, :delta_smape))
            esito = if row.delta_smape < -1.0
                "cUDE Wins"
            elseif row.delta_smape > 1.0
                "ODE Wins"
            else
                "Tie"
            end
            println(io, "$(row.patient_id) | ODE: $(round(row.smape_ode, digits=2))% | cUDE: $(round(row.smape_cude, digits=2))% | Delta: $(round(row.delta_smape, digits=2))% | Outcome: $(esito)")
        end
        println(io, "=========================================\n")
    end

    # Voglio 3 categorie (cUDE advantage, Neutral, ODE advantage)
    sort!(merged, :delta_smape)

    # = delta_smape = cUDE - ODE.  
    # cUDE advantage means delta is VERY NEGATIVE.
    # ODE advantage means delta is VERY POSITIVE.
    # Neutral means delta is close to 0.

    # Category 1: cUDE Advantage (top 10 negative)
    cude_adv = length(merged.delta_smape) >= 10 ? merged[1:10, :] : merged

    # Category 2: Neutral (closest to 0 absolute value)
    merged.abs_delta = abs.(merged.delta_smape)
    neutral_sorted = sort(merged, :abs_delta)
    neutral = length(neutral_sorted.abs_delta) >= 10 ? neutral_sorted[1:10, :] : neutral_sorted

    # Category 3: ODE Advantage (top 10 positive)
    # Note: merged is sorted by delta_smape ascending!
    ode_adv = length(merged.delta_smape) >= 10 ? merged[end-9:end, :] : merged

    out_folder = joinpath(OUT_DIR, dirname)
    mkpath(out_folder)

    function make_overlap_plot(group_df, group_prefix)
        for row in eachrow(group_df)
            pid = String(row.patient_id)
            haskey(PATIENT_DICT, pid) || continue
            p_data = PATIENT_DICT[pid]

            tmax = get_tmax(p_data)

            p_log_ode = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]
            p_nat_cude = Float64[row.a, row.b, row.Cs0, row.Cc0, row.beta]

            t_ode, y_ode = solve_ode(p_log_ode, tmax)
            t_cude, y_cude = solve_cude(p_nat_cude, BEST_NN, tmax)

            plt = plot_base(p_data, pid, tmax)
            # Plots.title!(plt, "Patient: $(pid) | ΔsMAPE: $(round(row.delta_smape, digits=2))")

            # Annotazioni esplicite 
            # ann_str = "ODE (Blue): sMAPE $(round(row.smape_ode, digits=2))%, RMSLE $(round(row.rmsle_ode, digits=4))\ncUDE (Orange): sMAPE $(round(row.smape_cude, digits=2))%, RMSLE $(round(row.rmsle_cude, digits=4))"

            # ODE Plot (Blue)
            Plots.plot!(plt, t_ode, y_ode, lw=2, color=:blue, label="ODE")

            # cUDE Plot (Orange dashed)
            Plots.plot!(plt, t_cude, y_cude, lw=2, color=:darkorange, linestyle=:dash, label="cUDE")

            # Aggiungiamo il delta sMAPE come titolo della legenda per fare in modo che l'engine
            # di posizionamento `:best` lo sposti in automatico per non coprire i punti/linee.
            delta_str = "ΔsMAPE: $(round(row.delta_smape, digits=2))%"
            Plots.plot!(plt, legend=:best, legendtitle=delta_str, legendtitlefontsize=9)

            Plots.savefig(plt, joinpath(out_folder, "Overlap_$(group_prefix)_$(pid).svg"))
        end
    end

    @info "  >> Overlaps per $(dataset_name)"
    make_overlap_plot(cude_adv, "CUDE_Advantage")
    make_overlap_plot(neutral, "Neutral_Advantage")
    make_overlap_plot(ode_adv, "ODE_Advantage")
end

@info "5) Overlap cUDE vs ODE: MIMIC"
process_overlap(df_ode_mimic, CUDE_MIMIC_METRICS, CUDE_MIMIC_PARAMS, "MIMIC", "Overlap_MIMIC")

@info "6) Overlap cUDE vs ODE: UMG"
process_overlap(df_ode_umg, CUDE_UMG_METRICS, CUDE_UMG_PARAMS, "UMG", "Overlap_UMG")

@info "Execution Completed Successfully!"

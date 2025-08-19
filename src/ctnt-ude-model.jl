using SimpleChains: SimpleChain, TurboDense, static, init_params
using SciMLBase: successful_retcode, ODEProblem, OptimizationSolution, OptimizationFunction, OptimizationProblem
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using DataFrames: DataFrame
using StableRNGs, StatsBase
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches
using OrdinaryDiffEq: AutoTsit5, Rosenbrock23, Tsit5
using DiffEqCallbacks

using ProgressMeter: Progress, next!
using DifferentialEquations
using DiffEqBase
using Base.Threads
using DiffEqGPU
using CUDA

softplus(x) = log(1 + exp(x))

sigmoid(x) = 1 / (1 + exp(-x))

const DELTA = 1e-12; # 0.007 # con cutoff 0.014 ng/mL # 1e-3
T_SCALE = 350.0;
const dt = 0.1
COMMON_TIME = 0.0:dt:T_SCALE;

# const POS_CB   = PositiveDomain()
# const NEG_TEST = (u, p, t) -> minimum(u) < 0

smape(pred, obs) = 200 * mean(abs.(pred .- obs) ./ (abs.(pred) .+ abs.(obs)))

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    β = exp(p.ode[5]) # Positive conditional parameter

    a = exp(p.ode[1])
    b = exp(p.ode[2])
    # Cs0 = exp(p.ode[3])
    # Cc0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    # correction = chain([u[1], t, a, b, Cs0, Cc0, β], p.neural)[1]

    # correction = chain([u[1], t, β], p.neural)[1]

    t_norm   = t / T_SCALE

    correction = chain([t_norm, β], p.neural)[1]

    du[1] = - (Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a*(Cc - Cp)
    du[3] = a*(Cc - Cp) - b*Cp

end

struct ctntCUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

function ctntCUDEModel(
    # ctnt_timepoints::AbstractVector{T},
    θ,
    chain::SimpleChain,
    tspan::Tuple{T,T}
    ) where T <: Real
    
    # println("In model: ", θ)
    # construct the ude function
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)

    # tspan = (ctnt_timepoints[1], ctnt_timepoints[end])
    
    Cs0 = exp(θ[3]) # exp both if params in log
    Cc0 = exp(θ[4])

    u0 = [Cs0, Cc0, 0];

    # ode = ODEProblem(cude!, u0, tspan, θ)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntCUDEModel(ode, chain)
end

# Definizione della struttura per i dati del paziente
struct PatientData
    id::String
    timepoints::Vector{Float64}   # vettore dei timepoints per il paziente
    ctnt_data::Vector{Float64}    # vettore dei valori di troponina
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

function neural_network_model(depth::Int, width::Int; input_dims::Int = 7)

    layers = []

    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))
    # push!(layers, TurboDense{true}(sigmoid, 1))

    SimpleChain(static(input_dims), layers...)
end

function sample_initial_neural_parameters(n_initials::Int, chain::SimpleChain, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_parameters(n_patients::Int, n_initials::Int, lhs_lb::AbstractVector{T}, lhs_ub::AbstractVector{T}, rng::AbstractRNG) where T <: Real
    # return sample(n_initials, lhs_lb, lhs_ub, LatinHypercubeSample(rng))
    return sample(n_initials, repeat(lhs_lb, n_patients), repeat(lhs_ub, n_patients), LatinHypercubeSample(rng))
end

########################## LOSS FUNCTIONS ##########################################

function compute_loss(θ, (model, timepoints, ctnt_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: ctntCUDEModel
    # solve the ODE problem
        sol = solve(model.problem, AutoTsit5(Rosenbrock23()); p=θ, saveat=timepoints,
                    # callback = POS_CB, isoutofdomain = NEG_TEST
                    )

        if !successful_retcode(sol)
            # If the solver fails, return infinity
            return Inf
        end

        sol = max.(Array(sol), DELTA)

        plasm = sol[3,:];

        if length(plasm) != length(ctnt_data)
            @error "Error on data"
            println(timepoints)
            println(plasm)
            println(ctnt_data)
        end
        # return sum(abs2, plasm - ctnt_data)
        return sum(abs2, log.(plasm) .- log.(ctnt_data))
        # return sum(((plasm - ctnt_data).^2).*ctnt_data)
        # return smape(plasm, ctnt_data)   # % su base 0–100
        # return 100 * mean(abs, (plasm .- ctnt_data) ./ (ctnt_data .+ DELTA))
        # return sqrt(mean((log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)).^2))
end

## Finito il train si estraggono i parametri della rete 
# patient_loss: Quando sono noti i parametri della rete
function patient_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode = θ, neural = fixed_nn_params)

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]

    # ODEProblem aggiornato
    prob = remake(model.problem; u0 = u0, p = p)

    sol = solve(prob, AutoTsit5(Rosenbrock23()); p=p, saveat=timepoints,
    #callback = POS_CB, isoutofdomain = NEG_TEST
    ) 

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    sol = max.(Array(sol), DELTA)

    plasm = sol[3,:];

    # return sum(abs2, plasm - ctnt_data)
    return sum(abs2, log.(plasm) .- log.(ctnt_data))
    # return sum(((plasm - ctnt_data).^2).*ctnt_data)
    # return smape(plasm, ctnt_data)
end
# La differenza sta nel dove si crea il component array:
# Se lo dai in pasto alla loss lo ottimizza tutto,
# se lo costruisci dentro ottimizza solo i parametri del modello

function smape_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode = θ, neural = fixed_nn_params)

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]

    # ODEProblem aggiornato
    prob = remake(model.problem; u0 = u0, p = p)

    sol = solve(prob, AutoTsit5(Rosenbrock23()); p=p, saveat=timepoints,
    #callback = POS_CB, isoutofdomain = NEG_TEST
    ) 

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    sol = max.(Array(sol), DELTA)

    plasm = sol[3,:];

    return smape(plasm, ctnt_data)
end

function training_loss(p, (models, training_dataset))
    loss_tot = 0.0
    for (i, model) in enumerate(models)
        patient = training_dataset[i];
        idx_start = 5*(i-1) + 1
        idx_end   = 5*i
        θ = ComponentArray(ode = p.ode[idx_start:idx_end], neural = p.neural)
        u0_new = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]
        prob = remake(model.problem; u0 = u0_new, p = θ)
        new_model = ctntCUDEModel(prob, model.chain) 
        loss_tot += compute_loss(θ, (new_model, patient.timepoints, patient.ctnt_data))
    end
    return loss_tot / length(training_dataset)
end


function deduplicate_times!(p::PatientData)
    tp, ct = p.timepoints, p.ctnt_data
    @assert length(tp) == length(ct) "Lunghezze diverse per ID=$(p.id)"

    seen = Set{Float64}()       # tempi già incontrati
    removed = 0
    # scorri al contrario così gli indici restanti non cambiano
    for i in reverse(eachindex(tp))
        t = tp[i]
        if t in seen            # duplicato → rimuovi
            println("ID=$(p.id)  t=$(t)  ctnt=$(ct[i])  [rimosso]")
            splice!(tp, i)
            splice!(ct, i)
            removed += 1
        else
            push!(seen, t)
        end
    end
    return removed
end

function trim_time(patients::AbstractVector{PatientData}, time_val)
    filtered_patients = PatientData[]

    for p in patients
        # mask dei timepoints validi
        mask = p.timepoints .<= time_val

        if any(mask)
            # conserva solo i punti ≤ time_val
            tp = p.timepoints[mask]
            ct = p.ctnt_data[mask]
            push!(filtered_patients, PatientData(p.id, tp, ct))
        else
            @warn "Patient $(p.id) has no acquisitions ≤ $(time_val) h and will be excluded"
        end
    end

    @info "Kept $(length(filtered_patients)) patients out of $(length(patients))"

    return filtered_patients
end

"""
    find_anomalies(patients::Vector{PatientData})

Restituisce un dizionario id → Lista di messaggi di errore:
  • “negative time”      → sono presenti timepoints < 0
  • “negative ctnt”      → sono presenti valori di troponina < 0
  • “times not sorted”   → i timepoints non sono in ordine non decrescente
"""
function find_anomalies(patients::Vector{PatientData}, n::Int64)
    anomalies = Dict{String, Vector{String}}()
    for p in patients
        issues = String[]

        if isempty(p.ctnt_data) || isempty(p.timepoints)
            push!(issues, "empty ctnt data")
        end

        if isempty(p.timepoints)
            push!(issues, "empty timepoints data")
        end

        if length(p.timepoints) != length(p.ctnt_data)
            push!(issues, "time ctnt mismatch")
        end

        # tempi negativi
        if any(t -> t < 0, p.timepoints)
            push!(issues, "negative time")
        end

        # ctnt negativi
        if any(c -> c < 0, p.ctnt_data)
            push!(issues, "negative ctnt")
        end

        if length(p.timepoints) <= n
            push!(issues, "n acquisizion < $n")
        end

        # ordinamento dei tempi
        if !issorted(p.timepoints; lt = ≤)  # lt=≤ permette anche tempo duplicato
            push!(issues, "times not sorted")
        end

        if !isempty(issues)
            anomalies[p.id] = issues
        end

    end

    if isempty(anomalies)
        println("No anomalies found")
    else
        for (id, issues) in anomalies
            @warn "Patient ", id, ": ", join(issues, ", ")
        end
    end
    return anomalies
end

function patient_dims(patients::AbstractVector{PatientData})
    # 1. Calcola il numero di acquisizioni per ciascun paziente
    counts = [length(p.ctnt_data) for p in patients]

    # 2. Trova gli indici del max e del min
    i_max = argmax(counts)
    i_min = argmin(counts)

    # 3. Stampa id, numero di acquisizioni e (opzionale) il vettore dei valori
    p_max = patients[i_max]
    p_min = patients[i_min]

    println("Patient with MAX acquisitions: ", p_max.id,
            " -> ", counts[i_max], " samples; ctnt_data = ", length(p_max.ctnt_data))

    println("Patient with MIN acquisitions: ", p_min.id,
            " -> ", counts[i_min], " samples; ctnt_data = ", length(p_min.ctnt_data))
    
    return (length(p_min.ctnt_data), length(p_max.ctnt_data))
end

function plot_distribution(patients::AbstractVector{PatientData})
    all_times = vcat([p.timepoints for p in patients]...)                 # Vector{Float64}
    all_ctnt  = vcat([p.ctnt_data for p in patients]...)                # Vector{Float64}

    t_min = minimum(all_times)
    t_max = maximum(all_times)

    @info "Tempo  min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h"

    c_min = minimum(all_ctnt)
    c_max = maximum(all_ctnt)

    @info "CTnT   min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL"

    all_ctnt_log = log.(all_ctnt .+ DELTA);

    # ------------------------------------------------------------------
    # 3) GRAFICO DELLE DISTRIBUZIONI  (tempo & troponina)  --------------
    # ------------------------------------------------------------------

    plt1 = histogram(all_times;
                    bins = 40,
                    xlabel = "Time (h)",
                    ylabel = "#",
                    title = "Time-points distribution",
                    legend = false)

    plt2 = histogram(all_ctnt_log;
                    bins = 40, # log-scale consigliata
                    xlabel = "CTnT (ng/mL)",
                    ylabel = "#",
                    title = "Troponin log distribution",
                    legend = false)

    dist = plot(plt1, plt2; layout = (2,1), size = (900,600))

    return all_times, all_ctnt, t_min, t_max, c_min, c_max, dist 
end

function scutter_patients(patients::AbstractVector{PatientData})
    plt = plot(; xlabel="Time (h)", ylabel="cTnT (ng/mL)",
           title="All patients: troponin vs time", legend=false)

    # Sovrapponi ogni paziente come una linea sottile
    for p in patients
        scatter!(plt, p.timepoints, p.ctnt_data; lw=1, alpha=0.5)
    end
    return plt
end
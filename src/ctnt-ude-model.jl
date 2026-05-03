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
using Plots, CairoMakie
# using DiffEqCallbacks

using ProgressMeter
using DifferentialEquations
using DiffEqBase
using Base.Threads

using StaticArrays: SVector

softplus(x) = log(1 + exp(x))
softplus_stable(x) = log1p(exp(-abs(x))) + max(x, 0)
relu_smooth(x; κ=0.05) = κ * softplus_stable(x / κ)
# pospart(x; κ=20.0) = softplus_stable(κ*x)/κ

sigmoid(x) = 1 / (1 + exp(-x))

relu(x) = ifelse(x > 0, x, zero(x))
# to_bounds(x, lb, ub) = lb + (ub - lb) * sigmoid(x)

to_bounds(x, lb, ub; κ=0.05) = x +
                               κ * softplus_stable((lb - x) / κ) -
                               κ * softplus_stable((x - ub) / κ)

# Sensitivity algorithms
# SENSE = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
# SENSE = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing=true) # No
# SENSE = InterpolatingAdjoint(autojacvec = EnzymeVJP())
# SENSE = GaussAdjoint(autojacvec = ZygoteVJP()) # No

const DELTA = 1e-6; # 0.007 # con cutoff 0.014 ng/mL # 1e-3
T_SCALE = 240.0; #350.0
const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, T_SCALE];
# const dt = 0.1
# COMMON_TIME = 0.0:dt:T_SCALE;

# const POS_CB   = PositiveDomain()
# const NEG_TEST = (u, p, t) -> minimum(u) < 0

smape(pred, obs) = 200 * mean(abs.(pred .- obs) ./ (abs.(pred) .+ abs.(obs)))
rmsle(y_true, y_pred) = sqrt(mean((log.(y_pred .+ 1) .- log.(y_true .+ 1)) .^ 2))

function troponin_ode!(du, u, p, τ)
    Cs, Cc, Cp = u

    a = exp(p[1])
    b = exp(p[2])
    Td = exp(p[5])

    n = 3.0
    τn = τ^n
    Td_n = Td^n
    fτ = τn / (τn + Td_n)

    du[1] = -(Cs - Cc) * fτ
    du[2] = (Cs - Cc) * fτ - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp
end

function ctnt_ude!(du, u, p, t, chain::SimpleChain)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    a = exp(p.ode[1])
    b = exp(p.ode[2])

    t_norm = t / T_SCALE #NORMALIZZAZIONE

    correction = chain([t_norm], p.neural)[1]

    du[1] = -(Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp

end

struct ctntUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

function ctntUDEModel(θ, chain::SimpleChain, pat_times::Vector{Float64}=Float64[])

    cude!(du, u, p, t) = ctnt_ude!(du, u, p, t, chain)

    Cs0 = exp(θ[3]) # exp both if params in log
    Cc0 = exp(θ[4])

    u0 = [Cs0, Cc0, 0.0]
    # u0 = SVector(Cs0, Cc0, 0.0);

    tspan = (0.0, pat_times[end] + 10.0)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntUDEModel(ode, chain)
end

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    β = exp(p.ode[end]) # Positive conditional parameter

    a = exp(p.ode[1])
    b = exp(p.ode[2])

    t_norm = t / T_SCALE #NORMALIZZAZIONE

    correction = chain([t_norm, β], p.neural)[1]

    du[1] = -(Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp

end

function ctntCUDEModel(θ, chain::SimpleChain, pat_times::Vector{Float64}=Float64[]) # where T <: Real

    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)

    Cs0 = exp(θ[3]) # exp both if params in log
    Cc0 = exp(θ[4])

    u0 = [Cs0, Cc0, 0.0]
    # u0 = SVector(Cs0, Cc0, 0.0);

    tspan = (0.0, pat_times[end] + 10.0)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntUDEModel(ode, chain)
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

function fromPatientData2DataFrame(patients::Vector{PatientData}; save::Bool=false, save_path::String="pazienti_long.csv")
    ids = String[]
    times = Float64[]
    values = Float64[]

    for p in patients
        n = length(p.timepoints)
        # Ripetiamo l'ID per ogni misurazione del paziente
        append!(ids, fill(p.id, n))
        # Accodiamo i tempi e i valori
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
        # gdf è un sotto-dataframe per un singolo paziente
        push!(patients_reloaded, PatientData(
            gdf.patient_id[1],     # ID (è uguale per tutte le righe del gruppo)
            gdf.time,              # Vettore tempi
            gdf.troponin           # Vettore valori
        ))
    end

    return patients_reloaded
end

function fromCSV2PateintData(csv_path::String)
    df_loaded = CSV.read(csv_path, DataFrame)
    return fromDataFrame2PatientData(df_loaded)
end

function neural_network_model(depth::Int, width::Int; input_dims::Int=7)

    layers = []

    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    # push!(layers, TurboDense{true}(softplus, 1))
    push!(layers, TurboDense{true}(sigmoid, 1))
    # push!(layers, TurboDense{true}(softplus_stable, 1))

    SimpleChain(static(input_dims), layers...)
end

function sample_initial_neural_parameters(n_initials::Int, chain::SimpleChain, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_parameters(n_patients::Int, n_initials::Int, lhs_lb::AbstractVector{T}, lhs_ub::AbstractVector{T}, rng::AbstractRNG) where T<:Real
    # return sample(n_initials, lhs_lb, lhs_ub, LatinHypercubeSample(rng))
    return sample(n_initials, repeat(lhs_lb, n_patients), repeat(lhs_ub, n_patients), LatinHypercubeSample(rng))
end

########################## LOSS FUNCTIONS ##########################################

function compute_loss(θ, (model, timepoints, ctnt_data)) # ::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real # where M <: ctntUDEModel
    # solve the ODE problem
    sol = solve(
        model.problem, Tsit5(); p=θ, saveat=timepoints,
        # sensealg = SENSE,
        abstol=1e-8, reltol=1e-6
        # callback = POS_CB, isoutofdomain = NEG_TEST
    )

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    # plasm = max.(Array(sol[3,:]), DELTA);
    # sol = max.(Array(sol), DELTA)
    plasm = sol[3, :]
    # plasm = max.(sol[3, :], DELTA);

    if length(plasm) != length(ctnt_data)
        @error "Error on data"
        # println(timepoints)
        # println(plasm)
        # println(ctnt_data)
    end
    # return sum(abs2, plasm - ctnt_data)
    # return sum(abs2, log.(plasm) .- log.(ctnt_data))
    return mean(abs2, log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA))
    # return sum(((plasm - ctnt_data).^2).*ctnt_data)
    # return smape(plasm, ctnt_data)   # % su base 0–100
    # return 100 * mean(abs, (plasm .- ctnt_data) ./ (ctnt_data .+ DELTA))
    # return sqrt(mean((log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)).^2))
    # return 0.1 * sum(abs2, (plasm - ctnt_data) / maximum(ctnt_data) ) + 0.8 * sum(abs2, log.(plasm) .- log.(ctnt_data))
end

# patient_loss: Quando sono noti i parametri della rete
function patient_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params); λ_back=0.0)
    p = ComponentArray(ode=θ, neural=fixed_nn_params)

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]
    # u0 = SVector(exp(θ[3]), exp(θ[4]), 0.0)

    # ODEProblem aggiornato
    prob = remake(model.problem; u0=u0, p=p)

    # sol = solve(prob, AutoTsit5(Rosenbrock23()); p=p, saveat=timepoints,
    #callback = POS_CB, isoutofdomain = NEG_TEST
    # ) 

    sol = solve(
        prob, Tsit5(); p=p, saveat=timepoints,
        # sensealg = SENSE,
        abstol=1e-8, reltol=1e-6
    )
    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    # plasm = max.(Array(sol[3,:]), DELTA);

    # sol = max.(Array(sol), DELTA)
    # plasm = max.(sol[3, :], DELTA);
    # plasm = sol[3, :];

    # sarcomere = sol[1, :]
    # cytosol = sol[2, :]
    plasm = sol[3, :]

    # back_sc = mean(abs2, relu.(cytosol .- sarcomere))  # >0 se Cc > Cs
    # back_cp = mean(abs2, relu.(plasm .- cytosol))  # >0 se Cp > Cc
    # ic_pen = relu(θ[4] - θ[3])^2
    ic_pen = relu_smooth(θ[4] - θ[3])^2

    return mean(abs2, log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)) + λ_back * ic_pen
end

function patient_loss_formula(θ, (problem, timepoints, ctnt_data); λ_back=0.0)
    # println("loss")
    # println(θ)
    u0 = [exp(θ[3]), exp(θ[4]), 0.0]

    # ODEProblem aggiornato
    prob = remake(problem; u0=u0, p=θ)

    sol = solve(
        prob,
        Tsit5();
        p=θ,
        saveat=timepoints,
        #callback = POS_CB, isoutofdomain = NEG_TEST
        abstol=1e-8, reltol=1e-6
    )

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    # sol = max.(Array(sol), DELTA)

    plasm = sol[3, :]

    ic_pen = relu_smooth(θ[4] - θ[3])^2

    # return sum(abs2, plasm - ctnt_data)
    # return sum(abs2, log.(plasm) .- log.(ctnt_data))
    return mean(abs2, log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)) + λ_back * ic_pen
    # return sum(((plasm - ctnt_data).^2).*ctnt_data)
    # return smape(plasm, ctnt_data)
end

# function smape_loss_formula(θ, (problem, timepoints, ctnt_data))

#     u0 = [exp(θ[3]), exp(θ[4]), 0.0]

#     # ODEProblem aggiornato
#     prob = remake(problem; u0=u0, p=θ)

#     sol = solve(
#         prob,
#         Tsit5();
#         p=θ,
#         saveat=timepoints,
#         #callback = POS_CB, isoutofdomain = NEG_TEST
#     )

#     if !successful_retcode(sol)
#         # If the solver fails, return infinity
#         return Inf
#     end

#     # sol = max.(Array(sol), DELTA)

#     plasm = sol[3, :]

#     return smape(plasm, ctnt_data)
# end

# function training_loss(p, (models, training_dataset); n_params::Int=5)
#     loss_tot = 0.0
#     for (i, model) in enumerate(models)
#         patient = training_dataset[i]
#         idx_start = n_params * (i - 1) + 1
#         idx_end = n_params * i
#         θ = ComponentArray(ode=p.ode[idx_start:idx_end], neural=p.neural)
#         u0_new = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]
#         prob = remake(model.problem; u0=u0_new, p=θ)
#         new_model = ctntUDEModel(prob, model.chain)
#         loss_tot += compute_loss(θ, (new_model, patient.timepoints, patient.ctnt_data))
#     end
#     return loss_tot / length(training_dataset)
# end

# function serial_training_loss(p, (models, training_dataset); n_params::Int=5)
#     n = length(training_dataset)
#     loss_tot = zero(eltype(p.ode))  # ok anche con Dual

#     @inbounds @views for i in 1:n
#         patient = training_dataset[i]
#         model = models[i]

#         idx1 = n_params * (i - 1) + 1
#         idx2 = n_params * i

#         # idx1 = length(p.ode)*(i-1) + 1
#         # idx2 = length(p.ode)*i

#         ode_i = p.ode[idx1:idx2]  # no copy
#         θ = ComponentArray(ode=ode_i, neural=p.neural)

#         u0_new = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]
#         # u0_new = SVector(exp(θ.ode[3]), exp(θ.ode[4]), 0.0)
#         prob = remake(model.problem; u0=u0_new, p=θ)

#         # evita new_model: risolvi direttamente
#         sol = solve(
#             prob, Tsit5();
#             p=θ, saveat=patient.timepoints,
#             # sensealg = SENSE,
#             abstol=1e-8, reltol=1e-6
#         )

#         if !successful_retcode(sol)
#             loss_tot += oftype(loss_tot, Inf)
#             continue
#         end

#         # evita Array(sol) 3xN: prendi solo Cp
#         # plasm = max.(sol[3, :], DELTA);
#         plasm = sol[3, :]
#         # data-term identico al tuo (ma senza allocazioni globali)
#         loss_tot += mean(abs2, log.(plasm .+ DELTA) .- log.(patient.ctnt_data .+ DELTA))
#     end

#     return loss_tot / n
# end

function par_training_loss(p, (models, training_dataset);
    n_params::Int=5,
    lb_param=nothing, ub_param=nothing,
    κ_bounds::Real=0.05,
    λ_back::Real=0.0
)
    n = length(training_dataset)
    T = eltype(p.ode)
    partial = fill(zero(T), Threads.maxthreadid())

    # Se 1 thread, evita proprio l'overhead e i task
    if Threads.nthreads() == 1
        loss_tot = zero(eltype(p.ode))

        @inbounds @views for i in 1:n
            patient = training_dataset[i]
            model = models[i]

            idx1 = n_params * (i - 1) + 1
            idx2 = n_params * i

            if κ_bounds != 0.0
                @views ode_raw = p.ode[idx1:idx2]

                if lb_param === nothing
                    ode_i = ode_raw
                else
                    ode_i = similar(ode_raw)
                    @inbounds for k in 1:n_params
                        ode_i[k] = to_bounds(ode_raw[k], lb_param[k], ub_param[k]; κ=κ_bounds)
                    end
                end
            else
                ode_i = p.ode[idx1:idx2]
            end

            θ = ComponentArray(ode=ode_i, neural=p.neural)

            u0_new = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]   # Cs0, Cc0
            # u0_new = SVector(exp(θ.ode[3]), exp(θ.ode[4]), 0.0)
            prob = remake(model.problem; u0=u0_new, p=θ)

            sol = solve(
                prob, Tsit5();
                saveat=patient.timepoints,
                # sensealg = SENSE,
                abstol=1e-8, reltol=1e-6
            )

            if !successful_retcode(sol)
                return oftype(loss_tot, Inf)
            end
            # plasm = max.(sol[3, :], DELTA);
            # sarcomere = sol[1, :]
            # cytosol = sol[2, :]
            plasm = sol[3, :]

            # back_sc = mean(abs2, relu.(cytosol .- sarcomere))  # >0 se Cc > Cs
            # back_cp = mean(abs2, relu.(plasm .- cytosol))  # >0 se Cp > Cc
            # ic_pen = relu(θ.ode[4] - θ.ode[3])^2
            ic_pen = relu_smooth(θ.ode[4] - θ.ode[3])^2

            loss_tot += mean(abs2, log.(plasm .+ DELTA) .- log.(patient.ctnt_data .+ DELTA)) + λ_back * ic_pen # λ_back*(back_sc + back_cp)
            # loss_tot += mean(abs2, log.(plasm .+ DELTA) .- log.(patient.ctnt_data .+ DELTA))
        end

        return loss_tot / n
    end

    Threads.@threads for i in 1:n
        patient = training_dataset[i]
        model = models[i]

        idx1 = n_params * (i - 1) + 1
        idx2 = n_params * i

        # idx1 = length(p.ode)*(i-1) + 1
        # idx2 = length(p.ode)*i

        if κ_bounds != 0.0
            @views ode_raw = p.ode[idx1:idx2]

            if lb_param === nothing
                ode_i = ode_raw
            else
                ode_i = similar(ode_raw)
                @inbounds for k in 1:n_params
                    ode_i[k] = to_bounds(ode_raw[k], lb_param[k], ub_param[k]; κ=κ_bounds)
                end
            end
        else
            ode_i = p.ode[idx1:idx2]
        end

        θ = ComponentArray(ode=ode_i, neural=p.neural)

        u0_new = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]
        # u0_new = SVector(exp(θ.ode[3]), exp(θ.ode[4]), 0.0)
        prob = remake(model.problem; u0=u0_new, p=θ)

        sol = solve(
            prob, Tsit5();
            p=θ, saveat=patient.timepoints,
            # sensealg = SENSE,
            abstol=1e-8, reltol=1e-6
        )
        loss_i = if !successful_retcode(sol)
            oftype(zero(T), Inf)
        else
            # plasm = max.(sol[3, :], DELTA);
            # sarcomere = sol[1, :]
            # cytosol = sol[2, :]
            plasm = sol[3, :]

            # back_sc = mean(abs2, relu.(cytosol .- sarcomere))  # >0 se Cc > Cs
            # back_cp = mean(abs2, relu.(plasm .- cytosol))  # >0 se Cp > Cc
            # ic_pen = relu(θ.ode[4] - θ.ode[3])^2
            ic_pen = relu_smooth(θ.ode[4] - θ.ode[3])^2

            mean(abs2, log.(plasm .+ DELTA) .- log.(patient.ctnt_data .+ DELTA)) + λ_back * ic_pen # λ_back*(back_sc + back_cp)
            # mean(abs2, log.(plasm .+ DELTA) .- log.(patient.ctnt_data .+ DELTA))
        end

        partial[Threads.threadid()] += loss_i
    end

    return sum(partial) / n
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

"""
    patient_duplicate_time_counts(patients::Vector{PatientData}; tol=0.0)

Ritorna Dict(id => ndup) dove ndup è il numero di duplicati (gap consecutivi <= tol)
dopo ordinamento dei timepoints.
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

Ritorna (tp2, ct2) ordinati e con duplicati collassati.
- agg: mean/median (o qualunque funzione aggregante)
- tol: tolleranza per considerare tempi "uguali" (0.0 = uguali esatti)
"""
function collapse_duplicate_times(tp::Vector{Float64}, ct::Vector{Float64};
    agg=mean,
    tol::Float64=0.0
)
    @assert length(tp) == length(ct) "Lunghezze diverse"

    n = length(tp)
    n ≤ 1 && return (copy(tp), copy(ct))

    idx = sortperm(tp)
    t = tp[idx]
    y = ct[idx]

    t2 = Float64[]
    y2 = Float64[]

    i = 1
    while i ≤ n
        j = i
        while j < n && abs(t[j+1] - t[i]) ≤ tol
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

Collassa IN-PLACE i duplicati temporali dei soli pazienti presenti come chiavi in `dup_counts`,
dove `dup_counts` è il Dict prodotto da `patient_duplicate_time_counts`.

Modifica `timepoints` e `ctnt_data` del paziente (ordina e collassa).
Ritorna (n_patients_modified, n_points_removed_total).
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

        # in-place update dei vettori del paziente
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

# """
#     count_acq_in_window(timepoints, h)::Int
# Conteggia le acquisizioni con t ≤ h.
# """
# @inline function count_acq_in_window(timepoints::AbstractVector{<:Real}, h::Real)
#     sum(t -> isfinite(t) && t ≤ h, timepoints)
# end

# """
#     has_min_acq_in_window(p::PatientData; max_hour=12.0, min_acq=1)::Bool
# True se il paziente ha almeno `min_acq` acquisizioni entro `max_hour` ore.
# """
# @inline function has_min_acq_in_window(p::PatientData; max_hour::Real=12.0, min_acq::Int=1)
#     count_acq_in_window(p.timepoints, max_hour) ≥ min_acq
# end

# """
#     filter_patients_by_window(patients; max_hour=12.0, min_acq=1)
# Ritorna (keep, dropped_ids) dopo il filtro per finestra/numero minimo.
# """
# function filter_patients_by_window(patients::Vector{PatientData}; max_hour::Real=12.0, min_acq::Int=1)
#     keep = Vector{PatientData}()
#     dropped_ids = String[]
#     for p in patients
#         if has_min_acq_in_window(p; max_hour=max_hour, min_acq=min_acq)
#             push!(keep, p)
#         else
#             push!(dropped_ids, p.id)
#         end
#     end
#     return keep, dropped_ids
# end

# Conteggio efficiente se timepoints ordinati
@inline function count_acq_in_window_sorted(timepoints_sorted::AbstractVector{<:Real}, h::Real)
    # assume sorted non-decrescente
    return searchsortedlast(timepoints_sorted, h)
end

"""
    max_gap_sorted(timepoints_sorted)::Float64
Massimo gap tra timepoints consecutivi (assume sorted).
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
    max_gap_h::Union{Nothing,Real}=nothing,   # <-- nuovo, opzionale
    verbose::Bool=true
)
    anomalies = Dict{String,Vector{String}}()

    for p in patients
        issues = String[]

        tp = p.timepoints
        ct = p.ctnt_data
        n_tp = length(tp)
        n_ct = length(ct)

        # empties / mismatch
        if n_ct == 0
            push!(issues, "empty ctnt data")
        end
        if n_tp == 0
            push!(issues, "empty timepoints data")
        end
        if n_tp != n_ct
            push!(issues, "time ctnt mismatch")
        end

        # se non ho timepoints non posso fare molti check (evita maximum/issorted inutili)
        if n_tp > 0
            # min_time check (usa maximum una sola volta)
            tmax = maximum(tp)
            if tmax < min_time
                push!(issues, "less then $(min_time)h max time")
            end

            # negativi (una scansione ciascuno)
            if any(<(0), tp)
                push!(issues, "negative time")
            end
            if n_ct > 0 && any(<(0), ct)
                push!(issues, "negative ctnt")
            end

            # numero acquisizioni: tipicamente "almeno meas_min_number"
            # nel tuo codice era <=; se intendi "minimo 5" allora deve essere < 5
            if n_tp < meas_min_number
                push!(issues, "n acquisizion < $meas_min_number")
            end

            # # ordinamento: se non sorted lo segnalo, ma posso comunque usare una copia ordinata per i check successivi
            # sorted_ok = issorted(tp; lt=≤)
            # if !sorted_ok
            #     push!(issues, "times not sorted")
            # end

            if issorted(tp; lt=≤)
                # conteggi finestre: O(log n) se sorted (qui lo è per costruzione)
                n_before = count_acq_in_window_sorted(tp, min_acq_time_before)
                if n_before < min_acq_n_before
                    push!(issues, "less then $min_acq_n_before measurements in the first $(min_acq_time_before)h")
                end

                n_after = length(tp) - count_acq_in_window_sorted(tp, min_acq_time_after)
                if n_after < min_acq_n_after
                    push!(issues, "less then $min_acq_n_after measurements in the last $(min_acq_time_after)h")
                end

                # gap massimo opzionale
                if max_gap_h !== nothing
                    gmax = max_gap_sorted(tp)
                    if gmax > max_gap_h
                        push!(issues, "max gap > $(max_gap_h)h (max=$(round(gmax, digits=3))h)")
                    end
                end
            else
                push!(issues, "times not sorted")
            end

            # t_sorted = sorted_ok ? tp : sort(copy(tp))
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

function param_stats(v::AbstractVector)
    # v_clean = collect(skipmissing(v))              # rimuove eventuali missing
    μ = mean(v)
    σ = std(v)
    q1, med, q3 = quantile(v, (0.25, 0.5, 0.75))
    iqr = q3 - q1
    return μ, σ, q1, med, q3, iqr
end

function plot_distribution(patients::AbstractVector{PatientData})
    all_times = vcat([p.timepoints for p in patients]...)                 # Vector{Float64}
    all_ctnt = vcat([p.ctnt_data for p in patients]...)                # Vector{Float64}

    t_min = minimum(all_times)
    t_max = maximum(all_times)

    @info "Tempo  min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h"

    c_min = minimum(all_ctnt)
    c_max = maximum(all_ctnt)

    @info "CTnT   min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL"

    all_ctnt_log = log.(all_ctnt .+ DELTA)

    # ------------------------------------------------------------------
    # 3) GRAFICO DELLE DISTRIBUZIONI  (tempo & troponina)  --------------
    # ------------------------------------------------------------------

    plt1 = Plots.histogram(all_times;
        bins=40,
        xlabel="Time (h)",
        ylabel="#",
        title="Time-points distribution",
        legend=false)

    plt2 = Plots.histogram(all_ctnt_log;
        bins=40, # log-scale consigliata
        xlabel="CTnT (ng/mL)",
        ylabel="#",
        title="Troponin log distribution",
        legend=false)

    dist = Plots.plot(plt1, plt2; layout=(2, 1), size=(900, 600))

    return all_times, all_ctnt, t_min, t_max, c_min, c_max, dist
end

function scutter_patients(patients::AbstractVector{PatientData})
    plt = Plots.plot(; xlabel="Time (h)", ylabel="cTnT (ng/mL)",
        title="All patients: troponin vs time", legend=false)

    # Sovrapponi ogni paziente come una linea sottile
    for p in patients
        Plots.scatter!(plt, p.timepoints, p.ctnt_data; lw=1, alpha=0.5)
    end
    return plt
end

function log_residuals(y, yhat; ϵ=DELTA)
    return log.(y .+ ϵ) .- log.(yhat .+ ϵ)
end

function compute_residuals_patient(model::ctntUDEModel, patient::PatientData, p::ComponentArray;
    plotting::Bool=false,
    abstol=1e-8,
    reltol=1e-6)

    pred = solve(model.problem, Tsit5(); p=p, saveat=patient.timepoints, abstol=abstol, reltol=reltol)

    # if any(pred .< 0)
    #     println("Warning: negative prediction for patient $(patient.id)")
    #     println(pred[3,:])
    # end

    if plotting

        sol = solve(model.problem, Tsit5(); p=p, abstol=abstol, reltol=reltol)

        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1, 1],
            xlabel="Time (h)",
            ylabel="cTnT",
            title="cTnT simulation patient $(patient.id)")

        # curva continua del modello (variabile 3)
        CairoMakie.lines!(ax, sol.t, sol[3, :], color=:blue, label="cTnT simulation")

        # dati sperimentali
        CairoMakie.scatter!(ax, patient.timepoints, patient.ctnt_data,
            color=:red, label="Data", markersize=8)

        axislegend(ax, position=:rt)

        display(fig)
    end

    yhat = vec(pred[3, :])
    y = patient.ctnt_data
    res = log_residuals(y, yhat)

    return y, yhat, res
end

function compute_residuals_patient(problem::ODEProblem, patient::PatientData, p::Vector{Float64};
    plotting::Bool=false,
    abstol=1e-8,
    reltol=1e-6)

    pred = solve(problem, Tsit5(); p=p, saveat=patient.timepoints, abstol=abstol, reltol=reltol)

    # if any(pred .< 0)
    #     println("Warning: negative prediction for patient $(patient.id)")
    #     println(pred[3,:])
    # end

    if plotting

        sol = solve(problem, Tsit5(); p=p, abstol=abstol, reltol=reltol)

        fig = CairoMakie.Figure(size=(800, 500))
        ax = CairoMakie.Axis(fig[1, 1],
            xlabel="Time (h)",
            ylabel="cTnT",
            title="cTnT simulation patient $(patient.id)")

        # curva continua del modello (variabile 3)
        CairoMakie.lines!(ax, sol.t, sol[3, :], color=:blue, label="cTnT simulation")

        # dati sperimentali
        CairoMakie.scatter!(ax, patient.timepoints, patient.ctnt_data,
            color=:red, label="Data", markersize=8)

        axislegend(ax, position=:rt)

        display(fig)
    end

    yhat = vec(pred[3, :])
    y = patient.ctnt_data
    res = log_residuals(y, yhat)

    return y, yhat, res
end

function add_time_bins!(df::DataFrame, edges::Vector{Float64})
    # bin index: 1..(length(edges)-1)
    nb = length(edges) - 1
    b = similar(df.t, Int)
    for i in eachindex(df.t)
        # searchsortedlast dà indice dell'edge <= t
        k = searchsortedlast(edges, df.t[i])
        # clamp a [1, nb]
        b[i] = clamp(k, 1, nb)
    end
    df.bin = b
    df.bin_center = [0.5 * (edges[k] + edges[k+1]) for k in b]
    return df
end

function bin_summary(df::DataFrame)
    g = groupby(df, :bin)
    centers = Float64[]
    q1 = Float64[]
    med = Float64[]
    q3 = Float64[]
    n = Int[]

    for sub in g
        push!(centers, first(sub.bin_center))
        push!(q1, quantile(sub.res, 0.25))
        push!(med, quantile(sub.res, 0.50))
        push!(q3, quantile(sub.res, 0.75))
        push!(n, nrow(sub))
    end

    # ordina per centro bin
    ord = sortperm(centers)
    return (centers=centers[ord], q1=q1[ord], med=med[ord], q3=q3[ord], n=n[ord])
end

function plot_residuals_vs_time(df::DataFrame, edges::Vector{Float64}; title="Residuals vs time", TMAX=T_SCALE, nmin::Int=1)
    s = bin_summary(df)
    for i in eachindex(s.centers)
        @info "bin_center=$(s.centers[i])  n=$(s.n[i])"
    end

    # maschera: metti NaN dove n è troppo basso
    med_mask = Float64[]
    q1_mask = Float64[]
    q3_mask = Float64[]
    for i in eachindex(s.n)
        if s.n[i] ≥ nmin
            push!(med_mask, s.med[i])
            push!(q1_mask, s.q1[i])
            push!(q3_mask, s.q3[i])
        else
            push!(med_mask, NaN)
            push!(q1_mask, NaN)
            push!(q3_mask, NaN)
        end
    end

    fig = CairoMakie.Figure(size=(950, 450))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="Time (h)",
        ylabel="log residual log(y) - log(ŷ)",
        title=title)

    # scatter di tutti i punti (leggero)
    CairoMakie.scatter!(ax, df.t, df.res; markersize=4, color=(:black, 0.25))

    # linea mediana per bin s.med
    CairoMakie.lines!(ax, s.centers, med_mask; linewidth=2, label="Median (per bin)", color=:blue)

    # banda IQR s.q1, s.q3
    CairoMakie.band!(ax, s.centers, q1_mask, q3_mask; color=(:gray, 0.2), label="IQR (Q1-Q3)")

    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6), label="Zero line (horizontal)")

    CairoMakie.xlims!(ax, 0, TMAX)

    CairoMakie.vlines!(ax, edges[2:end-1];
        color=(:black, 0.35),
        linewidth=1.5,
        linestyle=:dash,
        label="Bins (vertical)")

    for i in eachindex(s.centers)
        x_rel = clamp(s.centers[i] / TMAX, 0.0, 1.0)
        CairoMakie.text!(ax, x_rel, 0.96;  # 0.98 = in alto
            text="n=$(s.n[i])",
            space=:relative,
            align=(:center, :top),
            rotation=pi / 4,
            fontsize=12,
            color=(:black, 0.8))

        # xedge_rel = clamp(s.centers[i] / TMAX, 0.0, 1.0)
        # lines!(ax, [xedge_rel, xedge_rel], [0.0, 0.90]; space=:relative,
        #     color=(:black, 0.35), linewidth=1.5, linestyle = :dash)
    end

    CairoMakie.text!(ax, 0.99, 0.02;
        #   text="Median/IQR shown only if n ≥ $nmin",
        text="n = number of points in bin",
        space=:relative, align=(:right, :bottom),
        fontsize=12, color=(:black, 0.7))

    CairoMakie.Legend(fig[1, 2], ax;)

    fig
end

function plot_residuals_vs_fitted(df::DataFrame; title="Residuals vs fitted", ϵ=1e-10)
    fig = CairoMakie.Figure(size=(550, 450))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel="log predicted ŷ ",
        ylabel="log residual log(y) - log(ŷ)",
        title=title)

    CairoMakie.scatter!(ax, log.(df.yhat .+ ϵ), df.res; markersize=5, color=(:black, 0.25), label="Residuals")
    CairoMakie.hlines!(ax, [0.0]; linestyle=:dash, color=(:black, 0.6), label="Zero line")

    # CairoMakie.Legend(fig[1, 2], ax;)

    fig
end

# Only for original model at the moment
function compute_residuals(odefun::Function,
    params::DataFrame,
    patients::Vector{PatientData};
    plotting::Bool=false,
    tpad::Real=10.0,
    abstol=1e-12,
    reltol=1e-10)
    # residuals = []
    ids = []
    residuals_list = []
    for patient in patients
        patient_params = filter(row -> row.patient == patient.id, params)
        if nrow(patient_params) > 0
            p = [patient_params.p1[1], patient_params.p2[1], patient_params.p3[1],
                patient_params.p4[1], patient_params.p5[1]]
            u0 = [exp(p[3]), exp(p[4]), 0.0]
            tspan = (0.0, patient.timepoints[end] + tpad)
            prob = ODEProblem(odefun, u0, tspan, p)
            pred = solve(prob, Tsit5(); saveat=patient.timepoints, abstol=abstol, reltol=reltol)
            # pred = [sol(t)[3] for t in patient.timepoints]
            # Check comupte_residuals_patient
            if plotting
                # sol = solve(prob, Tsit5(); abstol=abstol, reltol=reltol)
                # plt = Plots.plot(sol; idxs=3, label = "cTnT simulation patient $(patient.id)")
                # Plots.scatter!(plt, patient.timepoints, patient.ctnt_data, label = "Data")
                # display(plt)
                sol = solve(prob, Tsit5(); abstol=abstol, reltol=reltol)

                fig = CairoMakie.Figure(size=(800, 500))
                ax = CairoMakie.Axis(fig[1, 1],
                    xlabel="Time (h)",
                    ylabel="cTnT",
                    title="cTnT simulation patient $(patient.id)")

                # curva continua del modello (variabile 3)
                CairoMakie.lines!(ax, sol.t, sol[3, :], color=:blue, label="cTnT simulation")

                # dati sperimentali
                CairoMakie.scatter!(ax, patient.timepoints, patient.ctnt_data,
                    color=:red, label="Data", markersize=8)

                CairoMakie.axislegend(ax, position=:rt)

                display(fig)
            end
            # predicted = [sol(t)[3] for t in patient.timepoints]
            # residual = patient.ctnt_data .- sol[3,:]
            residual = log_residuals(patient.ctnt_data, pred[3, :])
            # if any(sol[3,:] .< 0)
            #     println(sol[3,:])
            #     println(patient_params.patient)
            # end
            # push!(residuals, (id=patient.id, residuals=residual))
            push!(ids, patient.id)
            push!(residuals_list, residual)
        end
    end
    # return residuals
    return DataFrame(id=ids, residuals=residuals_list)
end

# Also for this - imported from TroponinRelease
function compute_residuals_long(odefun::Function,
    params::DataFrame,
    patients::Vector{PatientData};
    plotting::Bool=false,
    tpad::Real=20.0,
    abstol=1e-12,
    reltol=1e-10)

    out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])

    for patient in patients
        patient_params = filter(row -> row.patient == patient.id, params)
        nrow(patient_params) == 0 && continue

        p = [patient_params.p1[1], patient_params.p2[1], patient_params.p3[1],
            patient_params.p4[1], patient_params.p5[1]] # params a b Td Cc0 Cs0

        u0 = [exp(p[3]), exp(p[4]), 0.0] # initial conditions Cs0 Cc0 Cp0
        tspan = (0.0, patient.timepoints[end] + tpad)

        prob = ODEProblem(odefun, u0, tspan, p)

        # predizione ESATTAMENTE ai timepoints (coerente per residual)
        pred = solve(prob, Tsit5(); saveat=patient.timepoints, abstol=abstol, reltol=reltol)

        # if any(pred .< 0)
        #     println("Warning: negative prediction for patient $(patient.id)")
        #     println(pred[3,:])
        # end

        if plotting

            sol = solve(prob, Tsit5(); abstol=abstol, reltol=reltol)

            fig = CairoMakie.Figure(size=(800, 500))
            ax = CairoMakie.Axis(fig[1, 1],
                xlabel="Time (h)",
                ylabel="cTnT",
                title="cTnT simulation patient $(patient.id)")

            # curva continua del modello (variabile 3)
            CairoMakie.lines!(ax, sol.t, sol[3, :], color=:blue, label="cTnT simulation")

            # dati sperimentali
            CairoMakie.scatter!(ax, patient.timepoints, patient.ctnt_data,
                color=:red, label="Data", markersize=8)

            axislegend(ax, position=:rt)

            display(fig)
        end

        yhat = vec(pred[3, :])
        y = patient.ctnt_data
        res = log_residuals(y, yhat)

        append!(out, DataFrame(id=fill(patient.id, length(y)),
            t=patient.timepoints,
            y=y,
            yhat=yhat,
            res=res))
    end

    return out
end

function compute_plot_residuals(patients::Vector{PatientData}, ode_params_val::Vector{Float64}, best_nn::Vector{Float64},
    chain::SimpleChain; EDGES::Vector{Float64}=EDGES, N_params::Int=5, UDE::Bool=false,
    hi::Bool=false, show_plots::Bool=false, figsave_path::String="./", modelssave_path::String="./",
    dataset_label::String="")

    out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])
    smape_out = DataFrame(id=String[], smape=Float64[])

    a = []
    b = []
    Cs0 = []
    Cc0 = []
    β = []

    @showprogress desc = "Computing residuals..." for (i, patient) in enumerate(patients)

        idx1 = N_params * (i - 1) + 1
        idx2 = N_params * i
        ode_p = ode_params_val[idx1:idx2]
        p = ComponentArray(ode=ode_p, neural=best_nn)

        push!(a, exp(p.ode[1]))
        push!(b, exp(p.ode[2]))
        push!(Cs0, exp(p.ode[3]))
        push!(Cc0, exp(p.ode[4]))
        if N_params == 5
            push!(β, exp(p.ode[5]))
        end

        # u0 = [p.ode[end], p.ode[end-1], 0.0]
        # tspan = (0.0, patient.timepoints[end] + 10.0);
        # model = UDE ? ctntUDEModel(p, chain, tspan) : ctntCUDEModel(p, chain, tspan);

        model = UDE ? ctntUDEModel(p, chain, patient.timepoints) : ctntCUDEModel(p, chain, patient.timepoints)

        y, yhat, res = compute_residuals_patient(model, patient, p; plotting=show_plots)

        append!(out, DataFrame(id=fill(patient.id, length(y)),
            t=patient.timepoints,
            y=y,
            yhat=yhat,
            res=res))

        smape_val = smape(yhat, y)
        append!(smape_out, DataFrame(id=patient.id, smape=smape_val))
    end

    params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β]

    @info "Saving residuals data to CSV and plotting"

    add_time_bins!(out, EDGES)

    fig_vs_time = plot_residuals_vs_time(
        out,
        EDGES;
        title="Residuals vs time - $dataset_label", TMAX=T_SCALE, nmin=1)

    fig_vs_fitted = plot_residuals_vs_fitted(
        out;
        title="Residuals vs fitted - $dataset_label")

    @info "Boxplotting params"

    par_names = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"]

    x = vcat([fill(1, length(a))]...)

    f = Figure(
        size=(1400, 700), # input
    )

    Label(
        f[0, 1:length(par_names)],
        "Parameter distributions — $dataset_label";
        fontsize=22,
        tellwidth=false
    )

    axes = []

    # max_y_values = [15, 5, 200.0, 200.0, 3.0];

    @showprogress desc = "Generating axes..." for (i, p) in enumerate(par_names)
        ax = Axis(f[1, i],
            title=p,
            xticklabelsvisible=false, # Nasconde i numeri (1, 2...)
            xticksvisible=false,      # Nasconde le tacchette
            # limits=(nothing, nothing, nothing, max_y_values[i])
        )
        push!(axes, ax)
        # push!(axes, (Axis(f[1, i], title = p)))
    end

    my_colors = [:skyblue, :orange, :lightgreen, :pink, :violet]
    @showprogress desc = "Generating boxplots..." for (i, (ax, p)) in enumerate(zip(axes, params))
        current_color = my_colors[mod1(i, length(my_colors))]
        # i = (i-1)+1;
        CairoMakie.boxplot!(
            ax, x, p;
            # color = x, 
            # width = 0.5,
            # mediancolor = :red,
            # whiskercolor = :gray,
            # outliercolor = :green,
            # show_notch = true
            # color = :skyblue,
            color=current_color,
            whiskerwidth=0.3,
            strokewidth=0.3,
            show_outliers=true
        )
        # ax.xticks = (1:length(exps), exps_names);
        # ax.xticklabelrotation = pi/3;
        # autolimits!(ax)
    end

    if hi
        CSV.write("$(modelssave_path)/residuals_$(dataset_label)_hi.csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label)_hi.png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label)_hi.png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label)_hi.png", f)
    else
        CSV.write("$(modelssave_path)/residuals_$(dataset_label).csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label).png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label).png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label).png", f)
    end

    if show_plots
        display(fig_vs_time)
        display(fig_vs_fitted)
        display(f)
    end

    return out, smape_out
end

function compute_plot_residuals(patients::Vector{PatientData}, ode_params_val, ode_func::Function;
    EDGES::Vector{Float64}=EDGES, N_params::Int=5, UDE::Bool=false,
    hi::Bool=false, show_plots::Bool=false, figsave_path::String="./",
    modelssave_path::String="./", dataset_label::String="")

    out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])
    smape_out = DataFrame(id=String[], smape=Float64[])

    a = []
    b = []
    Cs0 = []
    Cc0 = []
    β = []

    @showprogress desc = "Computing residuals..." for (i, patient) in enumerate(patients)

        p = vcat(ode_params_val[i]...)

        push!(a, exp(p[1]))
        push!(b, exp(p[2]))
        push!(Cs0, exp(p[3]))
        push!(Cc0, exp(p[4]))
        if N_params == 5
            push!(β, exp(p[5]))
        end

        u0_init = [exp(p[3]), exp(p[4]), 0.0]

        tspan = (0.0, patient.timepoints[end] + 10.0)

        problem = ODEProblem(ode_func, u0_init, tspan)

        y, yhat, res = compute_residuals_patient(problem, patient, p; plotting=show_plots)

        append!(out, DataFrame(id=fill(patient.id, length(y)),
            t=patient.timepoints,
            y=y,
            yhat=yhat,
            res=res))

        smape_val = smape(yhat, y)
        append!(smape_out, DataFrame(id=patient.id, smape=smape_val))
    end

    params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β]

    @info "Saving residuals data to CSV and plotting"

    add_time_bins!(out, EDGES)

    fig_vs_time = plot_residuals_vs_time(
        out,
        EDGES;
        title="Residuals vs time - $dataset_label", TMAX=T_SCALE, nmin=1)

    fig_vs_fitted = plot_residuals_vs_fitted(out; title="Residuals vs fitted - $dataset_label")

    @info "Boxplotting params"

    par_names = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"]

    x = vcat([fill(1, length(a))]...)

    f = Figure(
        size=(1400, 700), # input
    )

    Label(
        f[0, 1:length(par_names)],
        "Parameter distributions — $dataset_label";
        fontsize=22,
        tellwidth=false
    )

    axes = []

    # max_y_values = [15, 5, 200.0, 200.0, 3.0];

    @showprogress desc = "Generating axes..." for (i, p) in enumerate(par_names)
        ax = Axis(f[1, i],
            title=p,
            xticklabelsvisible=false, # Nasconde i numeri (1, 2...)
            xticksvisible=false,      # Nasconde le tacchette
            # limits=(nothing, nothing, nothing, max_y_values[i])
        )
        push!(axes, ax)
        # push!(axes, (Axis(f[1, i], title = p)))
    end

    my_colors = [:skyblue, :orange, :lightgreen, :pink, :violet]
    @showprogress desc = "Generating boxplots..." for (i, (ax, p)) in enumerate(zip(axes, params))
        current_color = my_colors[mod1(i, length(my_colors))]
        # i = (i-1)+1;
        CairoMakie.boxplot!(
            ax, x, p;
            # color = x, 
            # width = 0.5,
            # mediancolor = :red,
            # whiskercolor = :gray,
            # outliercolor = :green,
            # show_notch = true
            # color = :skyblue,
            color=current_color,
            whiskerwidth=0.3,
            strokewidth=0.3,
            show_outliers=true
        )
        # ax.xticks = (1:length(exps), exps_names);
        # ax.xticklabelrotation = pi/3;
        # autolimits!(ax)
    end

    if hi
        CSV.write("$(modelssave_path)/residuals_$(dataset_label)_hi.csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label)_hi.png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label)_hi.png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label)_hi.png", f)
    else
        CSV.write("$(modelssave_path)/residuals_$(dataset_label).csv", out)
        CairoMakie.save("$(figsave_path)/residuals_vs_time_$(dataset_label).png", fig_vs_time)
        CairoMakie.save("$(figsave_path)/residuals_vs_fitted_$(dataset_label).png", fig_vs_fitted)
        save("$(figsave_path)/boxplots_$(dataset_label).png", f)
    end

    if show_plots
        display(fig_vs_time)
        display(fig_vs_fitted)
        display(f)
    end

    return out, smape_out
end

function params_extraction(
    patients::Vector{PatientData},
    ode_params_val::Vector{Float64};
    # best_nn::Union{Vector{Float64}, Nothing} = nothing,
    UDE::Bool=false,
    N_params::Int=5,
    data_label::String="", # "training" / "validation"
    dataset::String="",
    figsave_path::String="",
    show_outliers::Bool=false,
    savefigure::Bool=false
)

    a = []
    b = []
    Cs0 = []
    Cc0 = []
    β = []

    @showprogress desc = "$(data_label) prams extraction..." for (i, patient) in enumerate(patients)

        idx1 = N_params * (i - 1) + 1
        idx2 = N_params * i
        local ode_p = ode_params_val[idx1:idx2]
        # local p = ComponentArray(ode=ode_p, neural=best_nn)

        # push!(a, exp(p.ode[1]))
        # push!(b, exp(p.ode[2]))
        # push!(Cs0, exp(p.ode[3]))
        # push!(Cc0, exp(p.ode[4]))
        # if N_params == 5
        #     push!(β, exp(p.ode[5]))
        # end
        push!(a, exp(ode_p[1]))
        push!(b, exp(ode_p[2]))
        push!(Cs0, exp(ode_p[3]))
        push!(Cc0, exp(ode_p[4]))
        if N_params == 5
            push!(β, exp(ode_p[5]))
        end

    end

    @info "Average, STD in $data_label param a: $(mean(a)) std: $(std(a))"
    @info "Average, STD in $data_label param b: $(mean(b)) std: $(std(b))"
    @info "Average, STD in $data_label param Cs0: $(mean(Cs0)) std: $(std(Cs0))"
    @info "Average, STD in $data_label param Cc0: $(mean(Cc0)) std: $(std(Cc0))"
    if N_params == 5
        @info "Average, STD in $data_label param β: $(mean(β)) std: $(std(β))"
    end

    @info "Median [Q1-Q3] in $data_label param a: $(median(a)) [$(quantile(a, 0.25)) - $(quantile(a, 0.75))]"
    @info "Median [Q1-Q3] in $data_label param b: $(median(b)) [$(quantile(b, 0.25)) - $(quantile(b, 0.75))]"
    @info "Median [Q1-Q3] in $data_label param Cs0: $(median(Cs0)) [$(quantile(Cs0, 0.25)) - $(quantile(Cs0, 0.75))]"
    @info "Median [Q1-Q3] in $data_label param Cc0: $(median(Cc0)) [$(quantile(Cc0, 0.25)) - $(quantile(Cc0, 0.75))]"
    if N_params == 5
        @info "Median [Q1-Q3] in $data_label param β: $(median(β)) [$(quantile(β, 0.25)) - $(quantile(β, 0.75))]"
    end

    params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β]

    @info "Boxplotting params"

    par_names = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"]

    x = vcat([fill(1, length(a))]...)

    f = Figure(
        size=(1400, 700), # input
    )

    Label(
        f[0, 1:length(par_names)],
        "Parameter distributions $data_label — $dataset dataset", ;
        fontsize=22,
        tellwidth=false
    )

    axes = []

    # max_y_values = [15, 5, 200.0, 200.0, 3.0];

    @showprogress desc = "Generating axes..." for (i, p) in enumerate(par_names)
        ax = Axis(f[1, i],
            title=p,
            xticklabelsvisible=false, # Nasconde i numeri (1, 2...)
            xticksvisible=false,      # Nasconde le tacchette
            # limits=(nothing, nothing, nothing, max_y_values[i])
        )
        push!(axes, ax)
        # push!(axes, (Axis(f[1, i], title = p)))
    end

    my_colors = [:skyblue, :orange, :lightgreen, :pink, :violet]
    @showprogress desc = "Generating boxplots..." for (i, (ax, p)) in enumerate(zip(axes, params))
        current_color = my_colors[mod1(i, length(my_colors))]
        # i = (i-1)+1;
        CairoMakie.boxplot!(
            ax, x, p;
            # color = x, 
            # width = 0.5,
            # mediancolor = :red,
            # whiskercolor = :gray,
            # outliercolor = :green,
            # show_notch = true
            # color = :skyblue,
            color=current_color,
            whiskerwidth=0.3,
            strokewidth=0.3,
            show_outliers=show_outliers
        )
        # ax.xticks = (1:length(exps), exps_names);
        # ax.xticklabelrotation = pi/3;
        # autolimits!(ax)
    end

    if savefigure
        save("$(figsave_path)/$(data_label)_params_distribution_$(dataset).svg", f)
        @info "Figure saved at: $(figsave_path)/$(data_label)_params_distribution_$(dataset).svg"
    end
    return a, b, Cs0, Cc0, β, f
end

########################## DATA-ONLY OBJECTIVES FOR PLA ##########################################

"""
Solve the patient-specific cUDE/UDE model with fixed NN parameters and return plasma predictions.
Returns a Vector{Float64} on success, or `nothing` on failure.
"""
function patient_plasma_prediction(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode=θ, neural=fixed_nn_params)

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]
    prob = remake(model.problem; u0=u0, p=p)

    sol = try
        solve(
            prob,
            Tsit5();
            p=p,
            saveat=timepoints,
            abstol=1e-8,
            reltol=1e-6
        )
    catch
        return nothing
    end

    if !successful_retcode(sol)
        return nothing
    end

    plasma = vec(sol[3, :])

    if length(plasma) != length(ctnt_data) || any(!isfinite, plasma)
        return nothing
    end

    return plasma
end

"""
Log-residual vector:
r_k(θ) = log(pred_k + DELTA) - log(obs_k + DELTA)
"""
function patient_log_residuals(θ, data)
    model, timepoints, ctnt_data, fixed_nn_params = data

    plasma = patient_plasma_prediction(θ, data)
    plasma === nothing && return nothing

    resid = log.(plasma .+ DELTA) .- log.(ctnt_data .+ DELTA)

    if any(!isfinite, resid)
        return nothing
    end

    return resid
end

"""
Residual sum of squares on the log scale.
This is the data-only part, with no physics-informed penalty.
"""
function patient_rss_log(θ, data)
    resid = patient_log_residuals(θ, data)
    resid === nothing && return Inf
    return sum(abs2, resid)
end

# """
# Mean squared log error on the log scale.
# Useful if you want continuity with your current training metric,
# but not recommended as the main inferential objective for PLA.
# """
# function patient_mse_log(θ, data)
#     resid = patient_log_residuals(θ, data)
#     resid === nothing && return Inf
#     return mean(abs2, resid)
# end

"""
Negative log-likelihood under a Gaussian error model on the log-transformed data.

If sigma2 === nothing:
    returns the profiled objective n * log(RSS/n)
    which is equivalent to -2 log L up to an additive constant.

If sigma2 is provided:
    returns the full NLL up to the constant n/2 * log(2π).
"""
function patient_nll_log_gaussian(θ, data; sigma2::Union{Nothing,Float64}=nothing)
    resid = patient_log_residuals(θ, data)
    resid === nothing && return Inf

    n = length(resid)
    n == 0 && return Inf

    rss = max(sum(abs2, resid), eps(Float64))

    if !isfinite(rss)
        return Inf
    end

    if sigma2 === nothing
        # Profiled Gaussian likelihood on log-data:
        # -2 log L_prof = n * log(RSS/n) + const
        return n * log(rss / n)
    else
        if !(isfinite(sigma2) && sigma2 > 0.0)
            return Inf
        end
        # Full Gaussian NLL on log-data, up to additive constant n/2*log(2π)
        return 0.5 * n * log(sigma2) + rss / (2 * sigma2)
    end
end

### ODE Only ###

"""
    parse_log_to_csv(inpath::AbstractString;
                     out_csv::AbstractString = "params.csv",
                     meta_csv::Union{Nothing,AbstractString} = "meta.csv")

Parsa un log tipo:
  "dataset: ...\nUB: [..]\nPatient: n1 | smape: ... | params: [a,b,c,...]"
e scrive:
  - `out_csv`: tabella con patient, smape, p1..pK
  - `meta_csv` (se non `nothing`): riga con dataset e UB

Ritorna il DataFrame dei parametri.
"""
function parse_log_to_csv(inpath::AbstractString; out_csv::AbstractString="params.csv",
    meta_csv::Union{Nothing,AbstractString}="meta.csv")

    txt = read(inpath, String)

    dataset = match(r"(?m)^\s*dataset:\s*(.+)$", txt)
    dataset = dataset === nothing ? missing : strip(dataset.captures[1])

    ub_m = match(r"(?m)^\s*UB:\s*\[([^\]]+)\]", txt)
    UB = ub_m === nothing ? missing :
         [parse(Float64, strip(x)) for x in split(ub_m.captures[1], ',')]

    num_re = raw"([+-]?(?:Inf|NaN|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))"
    pat_re = Regex(
        raw"(?m)^\s*Patient:\s*([^\|]+?)\s*\|\s*smape:\s*" * num_re *
        raw"(?:\s*\|\s*rmsle:\s*" * num_re *
        raw"\s*\|\s*loss:\s*" * num_re * raw")?" *
        raw"\s*\|\s*params:\s*\[([^\]]*)\]"
    )
    ms = collect(eachmatch(pat_re, txt))

    params_lists = Vector{Vector{Float64}}(undef, length(ms))
    patients = Vector{String}(undef, length(ms))
    smapes = Vector{Float64}(undef, length(ms))
    rmsles = Vector{Union{Missing,Float64}}(undef, length(ms))
    losses = Vector{Union{Missing,Float64}}(undef, length(ms))
    maxK = 0

    for (i, m) in enumerate(ms)
        patients[i] = strip(m.captures[1])
        smapes[i] = parse(Float64, m.captures[2])
        rmsles[i] = m.captures[3] === nothing ? missing : parse(Float64, m.captures[3])
        losses[i] = m.captures[4] === nothing ? missing : parse(Float64, m.captures[4])

        params_str = strip(m.captures[5])
        plist = isempty(params_str) ? Float64[] :
                [parse(Float64, strip(x)) for x in split(params_str, ',')]

        params_lists[i] = plist
        maxK = max(maxK, length(plist))
    end

    df = DataFrame(patient=patients, smape=smapes, rmsle=rmsles, loss=losses)
    for k in 1:maxK
        col = Vector{Union{Missing,Float64}}(undef, length(params_lists))
        @inbounds for i in eachindex(params_lists)
            col[i] = k <= length(params_lists[i]) ? params_lists[i][k] : missing
        end
        df[!, Symbol("p$k")] = col
    end

    CSV.write(out_csv, df)

    if meta_csv !== nothing
        meta = DataFrame(dataset=[dataset],
            UB=[UB === missing ? missing : join(UB, ",")])
        CSV.write(meta_csv, meta)
    end

    return df
end

# function compute_residuals_long(params::Vector{Float64},
#                                 patients::Vector{PatientData},
#                                 UDE::Bool=false,
#                                 n_params::Int=5;
#                                 chain::SimpleChain,
#                                 fixed_nn_params = Vector{Float64},
#                                 plotting::Bool=false,
#                                 tpad::Real=10.0,
#                                 abstol=1e-12,
#                                 reltol=1e-10)

#     out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])

#     for (i, patient) in enumerate(patients)
#         # patient_params = filter(row -> row.patient == patient.id, params)
#         # nrow(patient_params) == 0 && continue
#         idx1 = n_params*(i-1) + 1;
#         idx2 = n_params*i;
#         p = ComponentArray(ode = params[idx1:idx2], neural = fixed_nn_params);

#         # u0 = [p.ode[end], p.ode[end-1], 0.0]
#         tspan = (0.0, patient.timepoints[end] + tpad);

#         model = UDE ? ctntUDEModel(p, chain, tspan) : ctntCUDEModel(p, chain, tspan);

#         # predizione ESATTAMENTE ai timepoints (coerente per residual)
#         pred = solve(model.problem, Tsit5(); saveat=patient.timepoints, abstol=abstol, reltol=reltol)

#         # if any(pred .< 0)
#         #     println("Warning: negative prediction for patient $(patient.id)")
#         #     println(pred[3,:])
#         # end

#         if plotting

#             sol = solve(model.problem, Tsit5(); abstol=abstol, reltol=reltol)

#             fig = CairoMakie.Figure(size = (800, 500))
#             ax  = CairoMakie.Axis(fig[1, 1],
#                     xlabel = "Time (h)",
#                     ylabel = "cTnT",
#                     title = "cTnT simulation patient $(patient.id)")

#             # curva continua del modello (variabile 3)
#             CairoMakie.lines!(ax, sol.t, sol[3, :], color = :blue,label = "cTnT simulation")

#             # dati sperimentali
#             CairoMakie.scatter!(ax, patient.timepoints, patient.ctnt_data,
#                     color = :red, label = "Data", markersize = 8)

#             axislegend(ax, position = :rt)

#             display(fig)
#         end

#         yhat = vec(pred[3, :])
#         y    = patient.ctnt_data
#         res  = log_residuals(y, yhat)

#         append!(out, DataFrame(id = fill(patient.id, length(y)),
#                                t  = patient.timepoints,
#                                y  = y,
#                                yhat = yhat,
#                                res  = res))
#     end

#     return out
# end

# =============================================================================
# UNIFIED RESIDUAL COMPUTATION (both ODE and cUDE from CSV DataFrames)
# =============================================================================
#
# This helper can replace:
#   - compute_residuals_long(odefun, params::DataFrame, patients; ...)
#       (lines ~1210-1274 above, ODE-only, reads from DataFrame with :patient, :p1..p5)
#   - The per-patient loop inside compute_plot_residuals(patients, ode_params_val::Vector, best_nn, chain; ...)
#       (lines ~1276-1412 above, cUDE-only, reads from flat Vector with N_params stride)
#   - extract_p_from_df in boxplotting_cude.jl
#   - compute_residuals(odefun, params::DataFrame, patients; ...)
#       (lines ~1148-1207 above, ODE-only)
#
# Migration example (OLD → NEW):
#
#   # OLD (ODE):
#   #   res_long = compute_residuals_long(troponin_ode!, params_df, patients)
#   #
#   # NEW (ODE):
#   #   res_long, metrics, extracted = compute_residuals_long_unified(
#   #       patients, params_df; model_type=:ode)
#
#   # OLD (cUDE):
#   #   out, smape_out = compute_plot_residuals(patients, ode_params_val, best_nn, chain;
#   #       N_params=5, UDE=false, ...)
#   #
#   # NEW (cUDE):
#   #   res_long, metrics, extracted = compute_residuals_long_unified(
#   #       patients, cude_params_df; model_type=:cude,
#   #       chain=chain, nn_params=best_nn)
#
# =============================================================================

"""
    compute_residuals_long_unified(patients, params_df;
        model_type=:ode, chain=nothing, nn_params=nothing,
        tpad=10.0, abstol=1e-8, reltol=1e-6)

Unified residual computation for both ODE and cUDE models.

Returns `(residuals_df, metrics_df, params_extracted)`:
  - `residuals_df`: long-format DataFrame(id, t, y, yhat, res)
  - `metrics_df`: DataFrame(id, smape_val, rmsle_val)
  - `params_extracted`: DataFrame(patient_id, a, b, Cs0, Cc0, p5) where p5 is Td (ODE) or beta (cUDE)

Arguments:
  - `patients`: Dict{String, PatientData} or Vector{PatientData}
  - `params_df`: DataFrame with patient-specific parameters
      * For `:ode`: columns :patient, :p1…:p5 (log-scale)
      * For `:cude`: columns :patient_id, :a, :b, :Cs0, :Cc0, :beta (natural-scale)
  - `model_type`: `:ode` or `:cude`
  - `chain`: SimpleChain (required for :cude)
  - `nn_params`: Vector{Float64} of NN weights (required for :cude)
"""
function compute_residuals_long_unified(
    patients,  # Dict{String,PatientData} or Vector{PatientData}
    params_df::DataFrame;
    model_type::Symbol=:ode,
    chain=nothing,
    nn_params=nothing,
    tpad::Real=10.0,
    abstol=1e-12,
    reltol=1e-10
)
    @assert model_type in (:ode, :cude) "model_type must be :ode or :cude"
    if model_type == :cude
        @assert chain !== nothing "chain required for cUDE"
        @assert nn_params !== nothing "nn_params required for cUDE"
    end

    # Build lookup if Vector
    patient_lookup = if patients isa Dict
        patients
    else
        Dict(p.id => p for p in patients)
    end

    residuals_out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[])
    metrics_out = DataFrame(id=String[], smape_val=Float64[], rmsle_val=Float64[])
    params_out = DataFrame(patient_id=String[], a=Float64[], b=Float64[],
        Cs0=Float64[], Cc0=Float64[], p5=Float64[])

    id_col = model_type == :ode ? :patient : :patient_id

    for row in eachrow(params_df)
        pid = String(row[id_col])
        haskey(patient_lookup, pid) || continue
        p_data = patient_lookup[pid]

        tmax = maximum(p_data.timepoints) + tpad

        if model_type == :ode
            p_log = Float64[row.p1, row.p2, row.p3, row.p4, row.p5]
            u0 = [exp(p_log[3]), exp(p_log[4]), 0.0]
            prob = ODEProblem(troponin_ode!, u0, (0.0, tmax), p_log)
            pred = solve(prob, Tsit5(); saveat=p_data.timepoints, abstol=abstol, reltol=reltol)

            # Extract natural-scale params
            push!(params_out, (pid, exp(p_log[1]), exp(p_log[2]),
                exp(p_log[3]), exp(p_log[4]), exp(p_log[5])))
        else  # :cude
            p_nat = Float64[row.a, row.b, row.Cs0, row.Cc0, row.beta]
            # θ_log = log.(p_nat .+ 1e-15)
            θ_log = log.(p_nat)
            u0 = [p_nat[3], p_nat[4], 0.0]

            cude_f!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)
            prob = ODEProblem(cude_f!, u0, (0.0, tmax))
            p_full = ComponentArray(ode=θ_log, neural=nn_params)
            pred = solve(prob, Tsit5(); p=p_full, saveat=p_data.timepoints, abstol=abstol, reltol=reltol)

            push!(params_out, (pid, p_nat[1], p_nat[2], p_nat[3], p_nat[4], p_nat[5]))
        end

        if !successful_retcode(pred)
            @warn "Solve failed for patient $pid (model=$model_type)"
            continue
        end

        yhat = vec(pred[3, :])
        y = p_data.ctnt_data

        if length(yhat) != length(y)
            @warn "Length mismatch for patient $pid: yhat=$(length(yhat)), y=$(length(y))"
            continue
        end

        res = log_residuals(y, yhat)

        append!(residuals_out, DataFrame(
            id=fill(pid, length(y)),
            t=p_data.timepoints,
            y=y,
            yhat=yhat,
            res=res
        ))

        push!(metrics_out, (pid, smape(yhat, y), rmsle(y, yhat)))
    end

    return residuals_out, metrics_out, params_out
end
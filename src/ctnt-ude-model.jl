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
# using DiffEqCallbacks

using ProgressMeter: Progress, next!
using DifferentialEquations
using DiffEqBase
using Base.Threads

using StaticArrays: SVector

softplus(x) = log(1 + exp(x))

sigmoid(x) = 1 / (1 + exp(-x))

const DELTA = 1e-12; # 0.007 # con cutoff 0.014 ng/mL # 1e-3
T_SCALE = 350.0;
# const dt = 0.1
# COMMON_TIME = 0.0:dt:T_SCALE;

# const POS_CB   = PositiveDomain()
# const NEG_TEST = (u, p, t) -> minimum(u) < 0

smape(pred, obs) = 200 * mean(abs.(pred .- obs) ./ (abs.(pred) .+ abs.(obs)))

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    β = exp(p.ode[end]) # Positive conditional parameter

    a = exp(p.ode[1])
    b = exp(p.ode[2])
    # Cs0 = exp(p.ode[3])
    # Cc0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    # correction = chain([u[1], t, a, b, Cs0, Cc0, β], p.neural)[1]

    # correction = chain([u[1], t, β], p.neural)[1]

    t_norm = t / T_SCALE

    # correction = chain([t_norm, β], p.neural)[1]
    # t_in, β_in = promote(t_norm, β)
    correction = chain(SVector(t_norm, promote(t_norm, β)), p.neural)[1]

    du[1] = - (Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a*(Cc - Cp)
    du[3] = a*(Cc - Cp) - b*Cp

end

function ctnt_sigmoid_cude!(du, u, p, t, chain::SimpleChain)
    # States
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]

    # ODE parameters (log-space)
    a  = exp(p.ode[1])
    b  = exp(p.ode[2])
    Td = exp(p.ode[3])     # Procopio sigmoid parameter (3rd)

    # Patient-specific conditional parameter
    β = exp(p.ode[end])

    # Procopio sigmoid (n=3)
    n   = 3.0
    τn  = t^n
    TdN = Td^n
    fτ  = τn / (τn + TdN)

    # --- NN residual correction on the sigmoid ---
    # Work in logit space so that δNN = 0 => f_corr == fτ exactly.
    # Center softplus output near 0 by subtracting softplus(0)=log(2) so init ≈ no-correction.
    t_norm = t / T_SCALE
    correction = chain(SVector{2,Float64}(t_norm, β), p.neural)[1]

    κ      = 1.0  # scaling of correction (tunable)

    ϵ          = eps(typeof(fτ))
    # fτc        = clamp(fτ, ϵ, one(fτ) - ϵ)
    # logit_fτ   = log(fτc / (one(fτc) - fτc))
    f_corr     = sigmoid(logit_fτ + κ * correction)

    # ODE system
    du[1] = - (Cs - Cc) * f_corr
    du[2] =   (Cs - Cc) * f_corr - a*(Cc - Cp)
    du[3] =   a*(Cc - Cp) - b*Cp
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
    
    Cs0 = exp(θ[end-2]) # exp both if params in log
    Cc0 = exp(θ[end-1])

    u0 = [Cs0, Cc0, 0];

    # ode = ODEProblem(cude!, u0, tspan, θ)
    # isoutofdomain = (u, p, t) -> any(<(0), u)
    ode = ODEProblem(cude!, u0, tspan;
        # isoutofdomain = isoutofdomain
        )

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
    sol = solve(
        model.problem, Tsit5(); p=θ, saveat=timepoints,
        # abstol=1e-10, reltol=1e-8,
        # callback = POS_CB, isoutofdomain = NEG_TEST
        )

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    # plasm = max.(Array(sol[3,:]), DELTA);
    # sol = max.(Array(sol), DELTA)
    # plasm = sol[3,:];
    plasm = max.(sol[3, :], DELTA);

    if length(plasm) != length(ctnt_data)
        @error "Error on data"
        # println(timepoints)
        # println(plasm)
        # println(ctnt_data)
    end
    # return sum(abs2, plasm - ctnt_data)
    # return sum(abs2, log.(plasm) .- log.(ctnt_data))
    return mean(abs2, log.(plasm) .- log.(ctnt_data))
    # return sum(((plasm - ctnt_data).^2).*ctnt_data)
    # return smape(plasm, ctnt_data)   # % su base 0–100
    # return 100 * mean(abs, (plasm .- ctnt_data) ./ (ctnt_data .+ DELTA))
    # return sqrt(mean((log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)).^2))
    # return 0.1 * sum(abs2, (plasm - ctnt_data) / maximum(ctnt_data) ) + 0.8 * sum(abs2, log.(plasm) .- log.(ctnt_data))
end

## Finito il train si estraggono i parametri della rete 
# patient_loss: Quando sono noti i parametri della rete
function patient_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode = θ, neural = fixed_nn_params)

    u0 = [exp(θ[end-2]), exp(θ[end-1]), 0.0]

    # ODEProblem aggiornato
    prob = remake(model.problem; u0 = u0, p = p)

    # sol = solve(prob, AutoTsit5(Rosenbrock23()); p=p, saveat=timepoints,
    #callback = POS_CB, isoutofdomain = NEG_TEST
    # ) 

    sol = solve(
        prob, Tsit5(); p=p, saveat=timepoints,
        # abstol=1e-10, reltol=1e-8
        )
    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    # plasm = max.(Array(sol[3,:]), DELTA);

    # sol = max.(Array(sol), DELTA)
    plasm = max.(sol[3, :], DELTA);

    # return sum(abs2, plasm - ctnt_data)
    # return sum(abs2, log.(plasm) .- log.(ctnt_data))
    return mean(abs2, log.(plasm) .- log.(ctnt_data))
    # return sum(((plasm - ctnt_data).^2).*ctnt_data)
    # return smape(plasm, ctnt_data)
    # return 0.1 * sum(abs2, (plasm - ctnt_data) / maximum(ctnt_data)) + 0.8 * sum(abs2, log.(plasm) .- log.(ctnt_data))
end
# La differenza sta nel dove si crea il component array:
# Se lo dai in pasto alla loss lo ottimizza tutto,
# se lo costruisci dentro ottimizza solo i parametri del modello

function smape_loss(θ, (model, timepoints, ctnt_data, fixed_nn_params))
    p = ComponentArray(ode = θ, neural = fixed_nn_params)

    u0 = [exp(θ[end-2]), exp(θ[end-1]), 0.0]

    # ODEProblem aggiornato
    prob = remake(model.problem; u0 = u0, p = p)

    # sol = solve(prob, AutoTsit5(Rosenbrock23()); p=p, saveat=timepoints,
    #callback = POS_CB, isoutofdomain = NEG_TEST
    # )

    sol = solve(
        prob, Tsit5(); p=p, saveat=timepoints,
        # abstol=1e-10, reltol=1e-8
        );

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    # sol = max.(Array(sol), DELTA)

    plasm = max.(sol[3, :], DELTA);

    return smape(plasm, ctnt_data)
end

function training_loss(p, (models, training_dataset))
    loss_tot = 0.0
    for (i, model) in enumerate(models)
        patient = training_dataset[i];
        idx_start = 5*(i-1) + 1
        idx_end   = 5*i
        θ = ComponentArray(ode = p.ode[idx_start:idx_end], neural = p.neural)
        u0_new = [exp(θ.ode[end-2]), exp(θ.ode[end-1]), 0.0]
        prob = remake(model.problem; u0 = u0_new, p = θ)
        new_model = ctntCUDEModel(prob, model.chain)
        loss_tot += compute_loss(θ, (new_model, patient.timepoints, patient.ctnt_data))
    end
    return loss_tot / length(training_dataset)
end

function serial_training_loss(p, (models, training_dataset))
    n = length(training_dataset)
    loss_tot = zero(eltype(p.ode))  # ok anche con Dual
    # sense = GaussAdjoint(autojacvec=ZygoteVJP());

    @inbounds @views for i in 1:n
        patient = training_dataset[i]
        model   = models[i]

        idx1 = 5*(i-1) + 1;
        idx2 = 5*i;

        # idx1 = length(p.ode)*(i-1) + 1
        # idx2 = length(p.ode)*i

        ode_i = p.ode[idx1:idx2]  # no copy
        θ     = ComponentArray(ode = ode_i, neural = p.neural)

        u0_new = (exp(θ.ode[end-2]), exp(θ.ode[end-1]), 0.0)
        prob   = remake(model.problem; u0 = collect(u0_new))  # o SVector se vuoi dopo

        # evita new_model: risolvi direttamente
        sol = solve(
            prob, Tsit5();
            p=θ, saveat=patient.timepoints,
            # sensealg = sense,
            # abstol=1e-12, reltol=1e-10
            );

        if !successful_retcode(sol)
            loss_tot += oftype(loss_tot, Inf)
            continue
        end

        # evita Array(sol) 3xN: prendi solo Cp
        plasm = max.(sol[3, :], DELTA);
        # data-term identico al tuo (ma senza allocazioni globali)
        loss_tot += mean(abs2, log.(plasm) .- log.(patient.ctnt_data))
    end

    return loss_tot / n
end

function par_training_loss(p, (models, training_dataset))
    n = length(training_dataset)
    T = eltype(p.ode)
    partial = fill(zero(T), Threads.maxthreadid())
    # sense = GaussAdjoint(autojacvec=ZygoteVJP());

        # Se 1 thread, evita proprio l'overhead e i task
    if Threads.nthreads() == 1
        loss_tot = zero(eltype(p.ode))

        @inbounds @views for i in 1:n
            patient = training_dataset[i]
            model   = models[i]

            idx1 = 5*(i-1) + 1
            idx2 = 5*i

            ode_i = p.ode[idx1:idx2]   # con @views => view, no copy
            θ     = ComponentArray(ode=ode_i, neural=p.neural)

            u0_new = [exp(θ.ode[end-2]), exp(θ.ode[end-1]), 0.0]   # Cs0, Cc0
            prob   = remake(model.problem; u0=u0_new, p=θ)

            sol = solve(
                prob, Tsit5();
                saveat=patient.timepoints,
                # sensealg = sense,
                # abstol=1e-12, reltol=1e-10
                )

            if !successful_retcode(sol)
                return oftype(loss_tot, Inf)
            end
            plasm = max.(sol[3, :], DELTA);
            loss_tot += mean(abs2, log.(plasm) .- log.(patient.ctnt_data))
        end

        return loss_tot / n
    end

    Threads.@threads for i in 1:n
        patient = training_dataset[i]
        model   = models[i]

        idx1 = 5*(i-1) + 1;
        idx2 = 5*i;

        # idx1 = length(p.ode)*(i-1) + 1
        # idx2 = length(p.ode)*i

        @views ode_i = p.ode[idx1:idx2]
        θ = ComponentArray(ode = ode_i, neural = p.neural)

        u0_new = (exp(θ.ode[end-2]), exp(θ.ode[end-1]), 0.0)
        prob   = remake(model.problem; u0 = collect(u0_new))

        sol = solve(
            prob, Tsit5();
            p=θ, saveat=patient.timepoints,
            # sensealg = sense,
            # abstol=1e-12, reltol=1e-10
            )
        loss_i = if !successful_retcode(sol)
            oftype(zero(T), Inf)
        else
            plasm = max.(sol[3, :], DELTA);
            mean(abs2, log.(plasm) .- log.(patient.ctnt_data))
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
)
    anomalies = Dict{String, Vector{String}}()

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

                n_after = count_acq_in_window_sorted(tp, min_acq_time_after)
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

    if isempty(anomalies)
        println("No anomalies found")
    else
        for (id, issues) in anomalies
            @warn "Patient " * id * ": " * join(issues, ", ")
        end
    end

    return anomalies
end

# """
#     find_anomalies(patients::Vector{PatientData})

# Restituisce un dizionario id → Lista di messaggi di errore:
#   • “negative time”      → sono presenti timepoints < 0
#   • “negative ctnt”      → sono presenti valori di troponina < 0
#   • “times not sorted”   → i timepoints non sono in ordine non decrescente
# """
# function find_anomalies(patients::Vector{PatientData}, n::Int64, min_acq_time::Real=300.0, min_acq_n::Int=1)
#     anomalies = Dict{String, Vector{String}}()
#     for p in patients
#         issues = String[]

#         if isempty(p.ctnt_data) || isempty(p.timepoints)
#             push!(issues, "empty ctnt data")
#         end

#         if isempty(p.timepoints)
#             push!(issues, "empty timepoints data")
#         end

#         if length(p.timepoints) != length(p.ctnt_data)
#             push!(issues, "time ctnt mismatch")
#         end

#         # tempi negativi
#         if any(t -> t < 0, p.timepoints)
#             push!(issues, "negative time")
#         end

#         # ctnt negativi
#         if any(c -> c < 0, p.ctnt_data)
#             push!(issues, "negative ctnt")
#         end

#         if length(p.timepoints) <= n
#             push!(issues, "n acquisizion < $n")
#         end

#         # ordinamento dei tempi
#         if !issorted(p.timepoints; lt = ≤)  # lt=≤ permette anche tempo duplicato
#             push!(issues, "times not sorted")
#         end

#         if count_acq_in_window(p.timepoints, min_acq_time) < min_acq_n
#             push!(issues, "less then $min_acq_n measurements in the first $(min_acq_time)h")
#         end

#         if !isempty(issues)
#             anomalies[p.id] = issues
#         end

#     end

#     if isempty(anomalies)
#         println("No anomalies found")
#     else
#         for (id, issues) in anomalies
#             @warn "Patient ", id, ": ", join(issues, ", ")
#         end
#     end
#     return anomalies
# end

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
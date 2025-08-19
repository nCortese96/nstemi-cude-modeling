############################################################
# ensemble_training.jl
# ----------------------------------------------------------
# Drop‑in replacement per la funzione `training_loss` seriale
# (file ctnt-ude-model.jl) basata su EnsembleProblem. Si usa
# con 500 pazienti o più per sfruttare al meglio CPU multi‑core
# o una singola GPU.  
# ----------------------------------------------------------
# API:
#   l = ensemble_training_loss(θ, (patients, chain);
#                               backend = :threads,   # :gpu | :threads | :cluster
#                               saveat  = COMMON_TIME)
# ----------------------------------------------------------
# Dipendenze: OrdinaryDiffEq, DiffEqGPU, SciMLSensitivity,
#             ComponentArrays, StaticArrays (già nel tuo Proj.toml)
############################################################

module EnsembleTraining

using OrdinaryDiffEq, DiffEqGPU, SciMLSensitivity
using ComponentArrays, StaticArrays           # StaticArray → buono su CPU e GPU
using CUDA

import Main: ctnt_cude!

export ensemble_training_loss               # Funzione che userai in Optimization

const Δ = 1e-8                             # piccola costante per log‑safety
T_SCALE = 350.0;
const dt = 0.1
COMMON_TIME = 0.0:dt:T_SCALE;

# ----------------------------
# 1. Template generico dell'ODE
# ----------------------------
"""
    build_template(chain, θ, patients)

Crea un `ODEProblem` generico che copre lo `tspan` massimo
fra tutti i pazienti. Serve come base del tuo `EnsembleProblem`.
"""
function build_template(chain, θ, patients)
    tmax = maximum(p.timepoints[end] for p in patients)
    # la funzione ctnt_cude! è importata dal Main (ctnt-ude-model.jl)
    cude!(du,u,p,t) = ctnt_cude!(du,u,p,t,chain)
    u0 = @SVector [1.0, 1.0, 0.0]          # placeholder, verrà sovrascritto
    return ODEProblem(cude!, u0, (0.0,tmax))
end

# ----------------------------
# 2. Funzione di loss integrata
# ----------------------------
function ensemble_training_loss(θ::ComponentArray,
                                 (patients, chain);
                                 backend::Symbol = :threads,
                                 saveat::Union{Nothing,AbstractVector}=nothing)
    # 2.1  – Template & EnsembleProblem
    template = build_template(chain, θ, patients)
    n_pat    = length(patients)

    # prob_func genera ogni traiettoria con i parametri del rispettivo paziente
    function prob_func(prob, i, repeat)
        pat = patients[i]
        # slice dei 5 parametri ODE dedicati al paziente i
        θi = ComponentArray(ode    = θ.ode[5(i-1)+1:5*i],
                            neural = θ.neural)
        u0  = @MVector [exp(θi.ode[3]), exp(θi.ode[4]), 0.0]
        remake(prob; u0 = u0, p = θi, tspan = (0.0, pat.timepoints[end]))
    end

    ep = EnsembleProblem(template; prob_func  = prob_func, safetycopy = false)

    # 2.2  – Scelta del backend parallelo
    ensemblealg = backend == :gpu ? EnsembleGPUArray(CUDA.CUDABackend()) : EnsembleThreads()

    # sol = solve(ep, Tsit5(); ensemblealg,
    #                          trajectories = n_pat,
    #                          sensealg = InterpolatingAdjoint(checkpointing = true),   # AD‑aware
    #                          saveat   = saveat,
    #                          progress = false)               # usa la tua progress esterna

    # ---- risoluzione ----
    common_kwargs = (; ensemblealg, trajectories=n_pat,
                       sensealg = InterpolatingAdjoint(checkpointing = true),
                       progress=false, abstol = 1e-8, reltol = 1e-4)

    sol = isnothing(saveat) ?
          solve(ep, Tsit5(); common_kwargs..., save_everystep=false) :
          solve(ep, Tsit5(); common_kwargs..., saveat=saveat)

    # 2.3  – Loss aggregata: SSE sui log della 3ª variabile (plasma Ctnt)
    l = 0.0
    @inbounds for i in 1:n_pat
        pat         = patients[i]
        # interpola la soluzione alla griglia dei dati reali del paziente
        model_ctnt  = sol[i](pat.timepoints; idxs = 3)        # Vector{Float64}
        data_ctnt   = pat.ctnt_data                           # stessa lunghezza
        l          += sum(abs2, log.(max.(model_ctnt, Δ)) .- log.(data_ctnt))
    end
    return l / n_pat
end

end # module

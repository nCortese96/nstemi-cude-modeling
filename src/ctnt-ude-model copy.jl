
"""
    ensemble_training_loss(p, data)

Data:
- `p`        : ComponentArray con campi `.ode` e `.neural`
- `data`     : tuple `(training_dataset, use_gpu_flag)`

Restituisce la loss media su tutti i pazienti, integrando
in parallelo con CPU o GPU.
"""
function ensemble_training_loss(p, (training_dataset, use_gpu))

    N = length(training_dataset)

    # 1) Base problem (template sul primo set di parametri)
    base_prob = ctntCUDEModel(p.ode[1:5], chain,
                              (0.0, last(COMMON_TIME))).problem

    # 2) Come generare il problema i-esimo
    prob_func = function (prob, i, _)
        θi = ComponentArray(
            neural = p.neural,
            ode    = p.ode[5*(i-1)+1 : 5*i]
        )
        u0 = [exp(θi.ode[3]), exp(θi.ode[4]), 0.0]
        remake(prob; p = θi, u0 = u0)
    end

    # 3) Costruisci l’EnsembleProblem
    enprob = EnsembleProblem(base_prob; prob_func = prob_func)

    # 4) Scegli l’algoritmo (CPU o GPU)
    ens_alg = use_gpu ? EnsembleGPUArray(CUDA.default_device()) : EnsembleThreads()

    # 5) Risolvi tutte le traiettorie in parallelo
    sol = solve(enprob, AutoTsit5(Rosenbrock23());
                trajectories  = N,
                ensemblealg   = ens_alg,
                saveat        = COMMON_TIME,
                progress      = true)

    # 6) Calcola la loss mediata
    loss_sum = 0.0
    for i in 1:N
        cp = clamp.(sol[i][3, :], DELTA, Inf)
        # tp = training_dataset[i].timepoints
        ct = training_dataset[i].ctnt_data .+ DELTA
        loss_sum += sum(abs2, log.(cp) .- log.(ct))
    end
    return loss_sum / N
end

# function parallel_training_loss(p, (models, training_dataset))
#     loss_tot = Atomic{Float64}(0.0)
#     @threads for i in eachindex(models)
#         model = models[i];
#         patient = training_dataset[i];
#         idx_start = 5*(i-1) + 1
#         idx_end   = 5*i
#         θ = ComponentArray(ode = p.ode[idx_start:idx_end], neural = p.neural)
#         u0_new = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]
#         prob = remake(model.problem; u0 = u0_new, p = θ)
#         new_model = ctntCUDEModel(prob, model.chain) 
#         loss = compute_loss(θ, (new_model, patient.timepoints, patient.ctnt_data))
#         atomic_add!(loss_tot, loss)
#     end
#     return loss_tot[] / length(training_dataset)
# end

# function parallel_training_loss(p, (models, training_dataset))
#     nt = Threads.nthreads()

#     # ➊  scegli il tipo del valore che sommeremo (Float64 o Dual)
#     #     => calcoliamo un singolo loss "dummy" fuori dal @threads
#     first_loss = let
#         m = models[1]; pat = training_dataset[1]
#         θ = ComponentArray(neural = p.neural,
#                            ode    = p.ode[1:5])
#         u0 = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]
#         new_mod = ctntCUDEModel(remake(m.problem; p = θ, u0 = u0), m.chain)
#         compute_loss(θ, (new_mod, pat.timepoints, pat.ctnt_data))
#     end

#     # ➋  vettore "per thread" inizializzato a zero
#     partial = fill(zero(first_loss), nt)            # Vector{Float64} o Vector{Dual}

#     # ➌  loop parallelo senza atomiche
#     @threads for i in eachindex(models)
#         model   = models[i]
#         patient = training_dataset[i]

#         θ       = ComponentArray(
#                      neural = p.neural,
#                      ode    = p.ode[5*(i-1)+1:5*i])
#         u0_new  = [exp(θ.ode[3]), exp(θ.ode[4]), 0.0]

#         prob    = remake(model.problem; p = θ, u0 = u0_new)
#         new_mod = ctntCUDEModel(prob, model.chain)

#         l = compute_loss(θ, (new_mod, patient.timepoints, patient.ctnt_data))

#         partial[Threads.threadid()] += l            # ➍  scrittura esclusiva
#     end

#     # ➎  riduzione finale in seriale
#     return sum(partial) / length(training_dataset)
# end

# function ensamble_training_loss(p, training_dataset)
#     # 1.  problema “template”
#     base_prob = ctntCUDEModel(p.ode[1:5], chain, (0.0, COMMON_TIME[end])).problem

#     # 2.  funzione che produce la variante i
#     function prob_func(prob, i, _)
#         θi  = ComponentArray(neural = p.neural,
#                              ode    = p.ode[5*(i-1)+1:5*i])
#         u0  = [exp(θi.ode[3]), exp(θi.ode[4]), 0.0]
#         remake(prob; p = θi, u0 = u0,
#                       tspan = (0.0, COMMON_TIME[end]))   # ← allineato alla griglia
#     end

#     enprob = EnsembleProblem(base_prob; prob_func)

#     # 3.  integrazione parallela con griglia comune
#     sol = solve(enprob, Tsit5();
#                 trajectories = length(training_dataset),
#                 saveat       = COMMON_TIME,            
#                 ensemble     = EnsembleThreads(),
#                 progress     = false)

#     # 4.  riduzione: usa compute_loss “slice-based”
#     loss_tot = mapreduce(i -> begin
#                        slice = sol[i][3, :]              # Cp su tutta la griglia
#                        compute_loss(slice, idx_map[i],   # seleziona solo i tempi reali
#                                     training_dataset[i].ctnt_data)
#                    end,
#                    +, eachindex(training_dataset))

#     return loss_tot / length(training_dataset)
# end

# @inline function interp_slice(slice::AbstractVector, t::Float64)
#     pos   = t / dt                       # posizione “fra le celle”
#     iL    = Int(floor(pos)) + 1          # indice sinistro (1-based)
#     α     = pos - floor(pos)             # peso destro
#     return (1-α) * slice[iL] + α * slice[iL+1]
# end

# @inline function troponin_pred(slice::AbstractVector, tvec)
#     preds = similar(tvec, eltype(slice))
#     @inbounds @simd for j in eachindex(tvec)
#         preds[j] = interp_slice(slice, tvec[j])
#     end
#     return preds
# end

# @inline function compute_loss(cp_slice::AbstractVector{<:Real}, timepoints, ctnt_data)
#     preds = troponin_pred(cp_slice, timepoints)  # CPU-only interp.
#     return sum(abs2, log.(preds .+ DELTA) .- log.(ctnt_data .+ DELTA))
# end

# @inline function compute_loss(cp_slice::CUDA.CuArray{<:Real,1}, timepoints, ctnt_data)
#     preds = troponin_pred(Array(cp_slice), timepoints)  # copia bulk → CPU
#     return sum(abs2, log.(preds .+ DELTA) .- log.(ctnt_data .+ DELTA))
# end

# function ensamble_training_loss(p, training_dataset; on_gpu = true)
#     ens = on_gpu ? EnsembleGPUArray(CUDADevice()) : EnsembleThreads()
#     base_prob = ctntCUDEModel(p.ode[1:5], chain,
#                               (0.0, last(COMMON_TIME))).problem

#     function prob_func(prob, i, _)
#         θi = ComponentArray(neural = p.neural,
#                             ode    = p.ode[5*(i-1)+1:5*i])
#         u0 = [exp(θi.ode[3]), exp(θi.ode[4]), 0.0]
#         remake(prob; p = θi, u0 = u0, tspan = (0.0, last(COMMON_TIME)))
#     end

#     enprob = EnsembleProblem(base_prob; prob_func)

#     sol = solve(enprob, AutoTsit5(Rosenbrock23());
#                 trajectories  = length(training_dataset),
#                 saveat        = COMMON_TIME,      
#                 ensemblealg   = ens,
#                 callback      = POS_CB,
#                 isoutofdomain = NEG_TEST,
#                 progress      = false)

#     loss_tot = 0.0
#     # @inbounds for i in eachindex(training_dataset)
#     #     patient = training_dataset[i]
#     #     loss_tot += compute_loss(sol[i], patient.timepoints, patient.ctnt_data)
#     # end

#     loss_tot = mapreduce(i -> 
#                             begin
#                                 cp_slice = sol[i][3, :]          # CuArray o Vector
#                                 compute_loss(cp_slice,
#                                         training_dataset[i].timepoints,
#                                         training_dataset[i].ctnt_data)
#                             end,
#                         +, eachindex(training_dataset))

#     return loss_tot / length(training_dataset)
# end
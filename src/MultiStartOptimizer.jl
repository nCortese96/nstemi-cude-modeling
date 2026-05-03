module MultiStartOptimizer

using Optimization, OptimizationOptimJL
using Random, StableRNGs, Distributions, QuasiMonteCarlo
using Base.Threads
using CSV, Tables
using ProgressMeter

export run_multistart

# function run_multistart(
#     loss::Function,
#     N::Int;
#     lower::Union{Nothing, Vector{<:Real}} = nothing,
#     upper::Union{Nothing, Vector{<:Real}} = nothing,
#     optimizer = Fminbox(LBFGS()),
#     verbose::Bool = true,
#     callback::Function = (x, f) -> nothing,
#     callback_every::Int = 0,
#     save_to_csv::Union{Nothing, String} = nothing,
#     rng::AbstractRNG = StableRNG(42),
#     maxiters::Int = 1000,
#     maxtime::Float64 = 80.0
# )
#     t = @elapsed begin
#         results = Vector{Any}(undef, N)

#         # Precompute starts
#         lhs_starts = QuasiMonteCarlo.sample(N, lower, upper, LatinHypercubeSample(rng))

#         # Optimization function (build once)
#         optf = OptimizationFunction((x, _) -> loss(x), AutoForwardDiff())

#         # Progress state (single-writer)
#         done = Threads.Atomic{Int}(0)
#         best_loss_atomic = Threads.Atomic{Float64}(Inf)
#         # Progres bar
#         p = Progress(N; desc="MultiStart", dt=0.5, showspeed=true)

#         monitor = Threads.@spawn begin
#             last = 0
#             while true
#                 d = done[]
#                 if d > last
#                     ProgressMeter.update!(p, d; showvalues=[(:best_loss, best_loss_atomic[])])
#                     last = d
#                 end
#                 d >= N && break
#                 sleep(0.1)
#             end
#         end

#         # Thread-safe best tracking
#         best_lock = ReentrantLock()
#         best_loss = Inf
#         best_sol = nothing

#         @threads for i in 1:N
#             # materialize start vector
#             p0 = Vector(lhs_starts[:, i])

#             optprob = isnothing(lower) ?
#                 OptimizationProblem(optf, p0) :
#                 OptimizationProblem(optf, p0; lb=lower, ub=upper)

#             result = try
#                 solve(optprob, optimizer; maxiters=maxiters, maxtime=maxtime)
#             catch err
#                 nothing
#             end

#             # result = solve(optprob, optimizer; maxiters=maxiters, maxtime=maxtime);

#             results[i] = result

#             if result !== nothing
#                 # update best (locked, rare)
#                 lock(best_lock) do
#                     if result.minimum < best_loss
#                         best_loss = result.minimum
#                         best_sol = result
#                         best_loss_atomic[] = best_loss
#                         # callback can be expensive; keep it but note cost
#                         callback(result.u, result.minimum)
#                     end
#                 end
#             end

#             done[] += 1
#         end

#         wait(monitor)

#         # optional summary
#         if verbose
#             nfail = count(r -> r === nothing, results)
#             println("Finished MultiStart: best_loss=$(best_loss), failed=$(nfail)/$(N)")
#         end

#         # Save CSV if requested
#         if save_to_csv !== nothing
#             rows = [ (; start=i,
#                        loss = (r === nothing ? NaN : r.minimum),
#                        params = (r === nothing ? [] : r.u)) for (i, r) in enumerate(results) ]
#             CSV.write(save_to_csv, Tables.columntable(rows))
#         end

#         return best_sol, results
#     end

#     # keep your timing print if you want
#     # (cannot print best_sol here unless we store it outside; easiest: return already contains it)
# end

function run_multistart(
    loss::Function,
    N::Int;
    lower::Union{Nothing, Vector{<:Real}} = nothing,
    upper::Union{Nothing, Vector{<:Real}} = nothing,
    optimizer = Fminbox(LBFGS()),
    verbose::Bool = true,
    callback::Function = (x, f) -> nothing,
    callback_every::Int = 0,
    save_to_csv::Union{Nothing, String} = nothing,
    rng::AbstractRNG = StableRNG(42),
    maxiters::Int = 1000,
    maxtime::Float64 = 80.0,
    prescreen::Bool = false,
    topk::Int = 8
)
    t = @elapsed begin
        # Precompute starts
        lhs_starts = QuasiMonteCarlo.sample(N, lower, upper, LatinHypercubeSample(rng))
        starts = [Vector(lhs_starts[:, i]) for i in 1:N]

        # Optional prescreen on initial loss
        selected_idx = collect(1:N)
        prescreen_losses = fill(NaN, N)

        if prescreen
            # prescreen_losses = map(starts) do s
            #     try
            #         loss(s)
            #     catch
            #         Inf
            #     end
            # end
            prescreen_losses = [loss(s) for s in starts]

            finite_idx = findall(isfinite, prescreen_losses)
            isempty(finite_idx) && error("Prescreen failed: no finite starting point found.")

            # if isempty(finite_idx)
            #     verbose && println("Prescreen failed: no finite starting point found.")
            #     return nothing, Any[]
            # end

            keep_n = min(topk, length(finite_idx))
            ord = sortperm(prescreen_losses[finite_idx])
            selected_idx = finite_idx[ord[1:keep_n]]
            starts = starts[selected_idx]
        end

        results = Vector{Any}(undef, length(starts))

        # Optimization function (build once)
        optf = OptimizationFunction((x, _) -> loss(x), AutoForwardDiff())

        # Progress state (single-writer)
        done = Threads.Atomic{Int}(0)
        best_loss_atomic = Threads.Atomic{Float64}(Inf)

        # Progress bar
        p = Progress(length(starts); desc="MultiStart", dt=0.1, showspeed=true)

        monitor = Threads.@spawn begin
            last = 0
            while true
                d = done[]
                if d > last
                    ProgressMeter.update!(p, d; showvalues=[(:best_loss, best_loss_atomic[])])
                    last = d
                end
                d >= length(starts) && break
                sleep(0.1)
            end
        end

        # Thread-safe best tracking
        best_lock = ReentrantLock()
        best_loss = Inf
        best_sol = nothing

        @threads for i in eachindex(starts)
            p0 = starts[i]

            optprob = isnothing(lower) ?
                OptimizationProblem(optf, p0) :
                OptimizationProblem(optf, p0; lb=lower, ub=upper)

            result = solve(optprob, optimizer; maxiters=maxiters, maxtime=maxtime)
            # result = try
            #     solve(optprob, optimizer; maxiters=maxiters, maxtime=maxtime)
            # catch
            #     nothing
            # end

            results[i] = result

            if result !== nothing
                lock(best_lock) do
                    if result.minimum < best_loss
                        best_loss = result.minimum
                        best_sol = result
                        best_loss_atomic[] = best_loss
                        callback(result.u, result.minimum)
                    end
                end
            end

            done[] += 1
        end

        wait(monitor)
        finish!(p)

        if verbose
            nfail = count(r -> r === nothing, results)
            println("Finished MultiStart: best_loss=$(best_loss), failed=$(nfail)/$(length(starts))")
        end

        if save_to_csv !== nothing
            rows = [
                (
                    start = selected_idx[i],
                    prescreen_loss = prescreen ? prescreen_losses[selected_idx[i]] : NaN,
                    loss = (r === nothing ? NaN : r.minimum),
                    params = (r === nothing ? [] : r.u)
                )
                for (i, r) in enumerate(results)
            ]
            CSV.write(save_to_csv, Tables.columntable(rows))
        end

        best_sol === nothing && error("MultiStart failed: no valid solution found among $(length(starts)) starts.")
        return best_sol, results
    end
end

end # module
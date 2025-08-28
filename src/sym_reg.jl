using DataFrames
using JLD2
using Logging
using Random
using Printf
using SymbolicRegression
using CSV
using Dates
using ProgressMeter

println("⚠️ Algorithm SR started $(now())")

include("ctnt-ude-model.jl")

input_dim = 2;
nn_depth = 2;
nn_width = 8;
inputs_str = "t, β";
if input_dim == 3
    inputs_str = "u[1], t, β";
elseif input_dim == 7
    inputs_str = "u[1], t, a, b, Cs0, Cc0, β";
end

# USE_GPU = true;
T_SCALE = 350.0;
# dt = 0.1;

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

experiment = "NSTEMI_partrvalMIMIC_SSEf_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Symbolic Regression Started")
#     println(io, "********************************")
# end

mkpath("$(models_path)/sr_outputs")

@load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
@assert length(best_nn) == length(init_params(chain))
@load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
@load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;

    # t_norm   = t / T_SCALE

    # correction = chain([t_norm, β], p.neural)[1]

times_norm_all = Float64[]
for p in test_dataset
    append!(times_norm_all, p.timepoints/T_SCALE)
end

β_all = Float64.(exp.([sol.u[5] for sol in best_solution]))  # β è positive ⇒ exp del log-param
q = (0.05, 0.95)
t_lo, t_hi   = quantile(times_norm_all, q);
β_lo, β_hi   = quantile(β_all, q);
logβ_lo, logβ_hi = log(β_lo), log(β_hi)

@info "Range t_norm: [$t_lo, $t_hi]"
@info "Range β     : [$β_lo, $β_hi]"

M = 500              # n. pazienti sintetici
# Kmin, Kmax = 5, 15   # n. misure per paziente (intero casuale)
Random.seed!(42)

# --- Costruzione dataset coerente ---
patient_id = String[]            # per tracciabilità/validazioni
t_norm     = Float64[]
beta       = Float64[]

for i in 1:M
    βi = Float64(exp(rand()*(logβ_hi - logβ_lo) + logβ_lo))  # β fissato per il paziente i
    # Ki = rand(Kmin:Kmax)
    Ki = 15
    for k in 1:Ki
        ti = Float64(rand()*(t_hi - t_lo) + t_lo)            # tempi ~ U[t_lo,t_hi]
        push!(patient_id, "synth$i")
        push!(t_norm, ti)
        push!(beta,    βi)                                   # stesso β per tutte le righe di i
    end
end

# --- Matrice X (features×samples) coerente con SymbolicRegression: [t_norm; β] ---
X_syn = [t_norm'; beta']   # 2×N: riga1=t_norm, riga2=β
@assert size(X_syn) == (2, length(t_norm))

# --- Target dalla tua rete (positiva): y_syn = correction_nn(t_norm, β) ---
# (Assumo che 'chain', 'best_nn' siano già definiti come nel tuo setup)
y_syn = similar(t_norm)
for j in eachindex(t_norm)
    y_syn[j] = chain((@view X_syn[:, j]), best_nn)[1]
end

# patient_id :: Vector{Int}
# t_norm     :: Vector{Float32}
# beta       :: Vector{Float32}
# y_syn      :: Vector{Float32}   # output NN (correction)

# 1) Costruisci il DataFrame "campione per riga"
df = DataFrame(
    patient_id = patient_id,
    t_norm     = t_norm,
    beta       = beta,
    y          = y_syn
)

CSV.write("$(models_path)/symRegDataset.csv", df)

# opts = Options(
#     binary_operators = ( +, *),
#     unary_operators  = ( inv, ),
#     maxsize = 18,
#     populations = 24,
#     should_optimize_constants = true,
#     output_directory = "$(models_path)/sr_outputs",   # default sarebbe "./outputs"
#     save_to_file     = true,
#     seed=42
# )

opts = Options(
  binary_operators = (+, *),
  unary_operators  = (inv,),
  maxsize = 18,
  populations = 24,           # ↓ da 24
#   population_size = 300,      # <-- IMPOSTALO (riduce il picco di memoria)
  parsimony = 1e-4,
  should_optimize_constants = true,
  output_directory = "$(models_path)/sr_outputs",   # default sarebbe "./outputs"
#   optimizer_probability = 0.5,    # non ottimizzare in tutte le iterazioni
#   optimizer_nrestarts = 3,
#   optimizer_iterations = 10,
#   batching = true,                # usa mini-batch
#   batch_size = 4096,
  save_to_file = true,
  seed = 42
)

# Warm-up breve per compilare e testare che tutto funzioni
equation_search(X_syn, y_syn;
    niterations = 300,
    options = opts,
    # parallelism = :multithreading,
    # return_state = true,
    progress = true,
    variable_names = ["t_norm","β"])

println("Warm-up terminated")
# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Warm-up terminated")
#     println(io, "********************************")
# end

# Esegui a “blocchi” per tenere sotto controllo memoria/ETA
# for _ in 1:10   # totale 10*2_000 = 20k iterazioni
#     global hof, state
#     hof, state = equation_search(X_syn, y_syn;
#         niterations = 2_000,
#         options = opts,
#         parallelism = :multithreading,
#         saved_state = state,
#         return_state = true,
#         progress = true,
#         variable_names = ["t_norm","β"])
# end

hof = equation_search(
  X_syn, y_syn;
  niterations = 25_000,
  options     = opts,
#   parallelism = :multithreading,
#   return_state = true,
  progress    = true,
  variable_names = ["t_norm","β"]
)

println("Symbolic regression process terminated succesfully")
open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "********************************")
    println(io, "Symbolic regression process terminated succesfully")
end

# Pareto + migliore formula (di solito l’ultima)
frontier = calculate_pareto_frontier(hof)
best     = frontier[end]
println("correction(t_norm, β) ≈ ", string_tree(best.tree; variable_names=["t_norm","β"]))

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "********************************")
    println(io, ("correction(t_norm, β) ≈ ", string_tree(best.tree; variable_names=["t_norm","β"])))
    println(io, "********************************")
end

# y_hat = eval_tree_array(best.tree, X_syn)
# r2  = 1 - sum((y_syn .- y_hat).^2) / sum((y_syn .- mean(y_syn)).^2)
# mae = mean(abs.(y_syn .- y_hat))
# @show r2 mae string_tree(best.tree; variable_names=["t_norm","β"])
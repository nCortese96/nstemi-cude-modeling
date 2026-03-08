using DataFrames, CSV, JLD2
using ProgressMeter, Logging, Printf
using Random, Dates
using SymbolicRegression

println("⚠️ Algorithm SR started $(now())")

include("ctnt-ude-model.jl")

UMG_data = false;

best_idx = 2; # index of the best model to test

UDE = false; # false for cUDE

# norm_type = 1;

if UDE
    @info "Using UDE model"
    input_dim = 1
    nn_depth = 2
    nn_width = 8
    N_params = 4
    inputs_str = "τ"
else
    @info "Using cUDE model"
    input_dim = 2
    nn_depth = 2
    nn_width = 8
    N_params = 5
    inputs_str = "τ, β"
end

# USE_GPU = true;
T_SCALE = 200.0;
# dt = 0.1;

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

# experiment = "NSTEMI_partrvalMIMIC_SSEf_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
# experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus";
experiment = "NSTEMI_cUDE_hi_MIMIC-IV_MSE_min_max_28";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

mkpath("$(models_path)/sr_outputs")

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Symbolic Regression Started")
#     println(io, "********************************")
# end

@info "Loading dataset"
test_dataset = if UMG_data
    dataset = "UMG"

    figsave_path = "$(fig_path)/umg_test_nn_$(best_idx)"
    modelssave_path = "$(models_path)/umg_test_nn_$(best_idx)"

    @load "$(modelssave_path)/UMG_testset.jld2" test_dataset
    @info "$dataset test dataset loaded from previous save"
    test_dataset
else
    dataset = "MIMIC-IV"

    figsave_path = "res/$(experiment)/figs/test_NN_$(best_idx)"
    modelssave_path = "res/$(experiment)/models/test_NN_$(best_idx)"

    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset
    @info "Using $dataset dataset for residuals"

    test_dataset
end
@info "Dataset $dataset loaded with $(length(test_dataset)) patients"
patient_dims(test_dataset)
all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(test_dataset);
display(dist)

@info "Loading best model and parameters"
best_nn, ode_params_val = try
    @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn
    @load "$(modelssave_path)/best_params_val_$(dataset).jld2" ode_params_val
    best_nn, ode_params_val
catch e

    @warn "Error loading model: $e"

    @info "Using best idx [$(best_idx)] for load model"

    @load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters
    @load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params

    best_nn = neural_network_parameters[best_idx]
    ode_params_val = ode_params[best_idx] # log version where 1 is the index of the best model in info_output
    best_nn, ode_params_val
end

@info "Best model and parameters loaded successfully"
@info "Loaded $(length(ode_params_val)) parameters"

# @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
# @assert length(best_nn) == length(init_params(chain))
# @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
# @load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;

# t_norm   = t / T_SCALE

# correction = chain([t_norm, β], p.neural)[1]
times_all = Float64[];
times_norm_all = Float64[];
# for p in test_dataset
#     append!(times_all, p.timepoints)
#     if norm_type == 0
#         append!(times_norm_all, p.timepoints/T_SCALE)
#     elseif norm_type == 1
#         append!(times_norm_all, (p.timepoints .- minimum(p.timepoints)) ./ (maximum(p.timepoints) - minimum(p.timepoints)))
#     elseif norm_type == 2
#         append!(times_norm_all, log(p.timepoints + DELTA))
#     end
# end
for p in test_dataset
    append!(times_all, p.timepoints)
    append!(times_norm_all, p.timepoints / T_SCALE)
end

# β_all = Float64.(exp.([sol.u[5] for sol in best_solution]))  # β è positive ⇒ exp del log-param
β_all = [exp.(ode_params_val[N_params*(i-1)+1:N_params*i][end]) for i in 1:length(ode_params_val)÷N_params];
β_all = vcat(β_all...);
q = (0.05, 0.95)
t_lo_q, t_hi_q = quantile(times_all, q)
t_lo, t_hi = quantile(times_norm_all, q)
β_lo, β_hi = quantile(β_all, q)
logβ_lo, logβ_hi = log(β_lo), log(β_hi)

@info "Range t_norm: [$t_lo, $t_hi]"
@info "Range β     : [$β_lo, $β_hi]"

M = 500              # n. pazienti sintetici
# Kmin, Kmax = 5, 15   # n. misure per paziente (intero casuale)
Random.seed!(42)

# --- Costruzione dataset coerente ---
patient_id = String[]            # per tracciabilità/validazioni
t_norm = Float64[]
beta = Float64[]

for i in 1:M
    βi = Float64(exp(rand() * (logβ_hi - logβ_lo) + logβ_lo))  # β fissato per il paziente i
    # Ki = rand(Kmin:Kmax)
    Ki = 15
    for k in 1:Ki
        # ti = Float64(rand()*(t_hi - t_lo) + t_lo)            # tempi ~ U[t_lo,t_hi]
        ti = rand(times_norm_all)
        push!(patient_id, "synth$i")
        push!(t_norm, ti)
        push!(beta, βi)                                   # stesso β per tutte le righe di i
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
    patient_id=patient_id,
    t_norm=t_norm,
    beta=beta,
    y=y_syn
)

CSV.write("$(models_path)/symRegDatase_test.csv", df)

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
    binary_operators=(+, *),
    unary_operators=(inv,),
    maxsize=18,
    populations=24,           # ↓ da 24
    #   population_size = 300,      # <-- IMPOSTALO (riduce il picco di memoria)
    parsimony=1e-4,
    should_optimize_constants=true,
    output_directory="$(models_path)/sr_outputs",   # default sarebbe "./outputs"
    #   optimizer_probability = 0.5,    # non ottimizzare in tutte le iterazioni
    #   optimizer_nrestarts = 3,
    #   optimizer_iterations = 10,
    #   batching = true,                # usa mini-batch
    #   batch_size = 4096,
    save_to_file=true,
    seed=42
)

# Warm-up breve per compilare e testare che tutto funzioni
equation_search(X_syn, y_syn;
    niterations=300,
    options=opts,
    parallelism=:multithreading,
    # return_state = true,
    progress=true,
    variable_names=["t_norm", "β"])

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
    niterations=25_000,
    options=opts,
    parallelism=:multithreading,
    #   return_state = true,
    progress=true,
    variable_names=["t_norm", "β"]
)

println("Symbolic regression process terminated succesfully")
open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "********************************")
    println(io, "Symbolic regression process terminated succesfully")
end

# Pareto + migliore formula (di solito l’ultima)
frontier = calculate_pareto_frontier(hof)
best = frontier[end]
println("correction(t_norm, β) ≈ ", string_tree(best.tree; variable_names=["t_norm", "β"]))

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "********************************")
    println(io, ("TEST correction(t_norm, β) ≈ ", string_tree(best.tree; variable_names=["t_norm", "β"])))
    println(io, "********************************")
end

# y_hat = eval_tree_array(best.tree, X_syn)
# r2  = 1 - sum((y_syn .- y_hat).^2) / sum((y_syn .- mean(y_syn)).^2)
# mae = mean(abs.(y_syn .- y_hat))
# @show r2 mae string_tree(best.tree; variable_names=["t_norm","β"])
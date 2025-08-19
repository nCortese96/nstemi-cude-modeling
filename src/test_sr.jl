using SimpleChains: SimpleChain, TurboDense, static
using JLD2, Random, Statistics
using SymbolicRegression  # Pkg.add("SymbolicRegression")
using Printf

# --- Impostazioni coerenti col training ---
T_SCALE = 350.0           # come nel tuo modello
nn_depth = 2
nn_width = 8
input_dim = 2             # [t_norm, β]

# Ricostruisco la stessa architettura usata in training
softplus(x) = log(1 + exp(x))
function neural_network_model(depth::Int, width::Int; input_dims::Int = 2)
    layers = [TurboDense{true}(tanh, width) for _ in 1:depth]
    push!(layers, TurboDense{true}(softplus, 1))  # uscita positiva, come nel tuo codice
    return SimpleChain(static(input_dims), layers...)
end
chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim)

# --- Carico i file prodotti dal tuo training ---
experiment = "NSTEMI_partrvalMIMIC_SSEf_ts350.0_28_inp2_multipl_softplus"  # <-- metti il tuo
resdir = "res/$(experiment)/models"

@load "$(resdir)/best_nn_NSTEMI_$(experiment).jld2" best_nn
@load "$(resdir)/testsetNSTEMI_$(experiment).jld2"  test_dataset
@load "$(resdir)/best_solutionNSTEMI_$(experiment).jld2" best_solution  # vettore di OptimizationSolution

# Estrai i β stimati per ciascun paziente di test (in log nello stato -> fai exp)
β_vec = [exp(sol.u[5]) for sol in best_solution]   # coerente con il tuo codice
# (se l’ordine dei best_solution corrisponde a test_dataset, si allineano 1:1 come in test_code.jl)

# --- Costruisco (X, y) reali: punti tempo dei pazienti e β del paziente ---
# X deve essere 2×N (features × samples), y deve essere 1×N (o Vector) per SymbolicRegression
t_all = Float32[]          # t_norm
b_all = Float32[]          # β
y_all = Float32[]          # output rete (correction)

for (i, patient) in enumerate(test_dataset)
    βi = Float32(β_vec[i])
    for t in patient.timepoints
        tnorm = Float32(t / T_SCALE)
        yi = chain([tnorm, βi], best_nn)[1]   # valuta la tua NN coi migliori pesi
        push!(t_all, tnorm)
        push!(b_all, βi)
        push!(y_all, Float32(yi))
    end
end

# Trasformo in matrici per SR: X è 2×N (col-major), y è Vector{Float32}(N)
X = reshape(collect(Iterators.flatten((t_all, b_all))), 2, :)
y = y_all

# (scelta consigliata) Regressione su z = log(y + eps) per garantire positività poi via exp
eps = 1f-8
z = log.(y .+ eps)

# --- Split semplice train/val per controllare overfitting nella SR ---
N = length(z)
Random.seed!(123)
perm = randperm(N)
ntrain = Int(round(0.8N))
train_idx = perm[1:ntrain];  val_idx = perm[ntrain+1:end]
Xtr, ztr = X[:, train_idx], z[train_idx]
Xva, zva = X[:, val_idx],   z[val_idx]

# --- Impostazioni SymbolicRegression (API moderne) ---
# vedi: equation_search, Options, calculate_pareto_frontier, string_tree
# https://ai.damtp.cam.ac.uk/symbolicregression/dev/
opts = Options(
    binary_operators = [+, *, -, /],
    unary_operators  = [exp, log, sqrt],  # tieni funzioni "lisce" e interpretabili
    maxsize = 30,       # complessità massima dell’albero
    populations = 20,   # numero popolazioni
    parsimony = 1e-4,   # penalità complessità
)

# Cerca sull’insieme di training
hof = equation_search(
    Xtr, ztr;
    niterations = 200,
    options = opts,
    parallelism = :multithreading,
    progress = true
)dsd

# Prendo il Pareto front e la “migliore” formula (ultima è tipicamente la più accurata)
dominating = calculate_pareto_frontier(hof)
best_tree  = dominating[end].tree

# Stampo formula in termini di t_norm e β (questa è la formula di z = log(correction))
formula_z_str = string_tree(best_tree, opts; variable_names = ["t_norm","β"])
@printf("\nLog-correction(t_norm, β) ≈ %s\n", formula_z_str)

# Valutazione valida (R^2 su z); poi ricorda: correction_hat = exp(z_hat)
z_hat_val = best_tree(Xva)   # gli Expression sono "callable" su X
ss_res = sum((zva .- z_hat_val).^2)
ss_tot = sum((zva .- mean(zva)).^2)
r2 = 1 - ss_res/ss_tot
@printf("R^2 (val, su log-correction): %.4f\n", r2)

# -------------------------------------------------------------
# symbolic_extraction_simple.jl
# Estrarre una formula simbolica che approssima la correzione
# neurale f(t_norm, β) usata nel modello ctnt_cude!.
# Script lineare, senza funzioni né moduli: copia‑incolla e vai.
# -------------------------------------------------------------

using Random
using Distributions: Uniform
using SimpleChains
using JLD2
using SymbolicRegression

# 1. Ricrea la rete esattamente come nel training -----------------------------

depth  = 2            # numero di strati nascosti
width  = 8            # neuroni per strato
chain  = SimpleChain(
            static(2),                 # input = (t_norm, β)
            TurboDense(tanh,  width),
            TurboDense(softplus, 1)
         )

# 2. Carica i pesi migliori ----------------------------------------------------

θ_best = load("checkpoints/best_nn_NSTEMI.jld2", "θ")
@assert length(θ_best) == length(init_params(chain))

# 3. Costanti e dominio reale --------------------------------------------------

const T_SCALE = 350.0             # ore → identico al training
β_min = 0.05                      # ← sostituisci con il tuo minimo reale
β_max = 5.0                       # ← sostituisci con il tuo massimo reale

# 4. Costruisci il dataset per SymbolicRegression -----------------------------

n       = 8_000
X       = Matrix{Float64}(undef, n, 2)
X[:,1] .= rand(n)                          # t_norm  ~ U(0,1)
X[:,2] .= rand(Uniform(β_min, β_max), n)   # β       ~ dominio reale

y_raw  = [chain(x, θ_best)[1] for x in eachrow(X)]
y      = log.(y_raw)                      # log → uscita softplus + stabile

# 5. Lancia la regressione simbolica -----------------------------------------

ops     = [:+, :*, :/, :exp, :log, :^]
options = SymbolicRegression.Options(
             niterations     = 10_000,
             population_size = 2_000,
             maxsize         = 30,
             parsimony       = 1e-3,
             deterministic   = true
         )
hof = EquationSearch(y, X, ops, options)

# 6. Recupera la miglior equazione e creala come funzione ---------------------

best_eq = hof[1].tree
f_sym   = SymbolicRegression.lambdify(best_eq)
correction_sym(t, β) = exp(f_sym(t / T_SCALE, β))

println("Miglior formula simbolica (dominio normalizzato):")
println(best_eq)

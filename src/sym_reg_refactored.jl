using DataFrames, CSV, JLD2, Dates, Statistics
using Plots
using SymbolicRegression
import SymbolicRegression: string_tree, compute_complexity

println("⚠️ Symbolic regression started $(now())")

include("ctnt-ude-model.jl")

# ============================================================
# 1) CONFIG
# ============================================================
dataset_id = 0           # 0 = MIMIC-IV, 1 = UMG
UDE = false              # false = cUDE
best_idx = 3

nn_depth = 2
nn_width = 8
input_dim = UDE ? 1 : 2

use_multistart = true

T_SCALE = 240.0

if dataset_id == 0
    dataset_name = "MIMIC-IV"
else
    dataset_name = "UMG"
end

experiment = "NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback"

fig_path = "res/$(experiment)/figs"
models_path = "res/$(experiment)/models"

figsave_path = "$(fig_path)/$(dataset_name)_test_NN_$(best_idx)$(use_multistart ? "_ms_test" : "")/sr_outputs_extended_refactored_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))/figs"
modelssave_path = "$(models_path)/$(dataset_name)_test_NN_$(best_idx)$(use_multistart ? "_ms_test" : "")/sr_outputs_extended_refactored_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))"

mkpath(figsave_path)
mkpath(modelssave_path)

# Explicit SR domain
TMAX_SR_H = 2400.0#960.0 # 240.0
# t_hours_grid = collect(0.01:0.1:TMAX_SR_H)      # hours shown in plot

# t_hours_grid = unique(vcat(
#     collect(0.01:0.1:24.0),      # onset / early rise
#     collect(24.5:0.5:120.0),    # transition
#     collect(121.0:1.0:240.0),   # observed late phase
#     collect(241.0:10.0:TMAX_SR_H)  # plateau / extrapolated tail
# ))

# t_hours_grid = unique(vcat(
#     collect(0.01:0.15:24.0),    # 160
#     collect(24.5:0.75:120.0),   # 128
#     collect(121.0:1.5:240.0),   # 80
#     collect(241.0:15.0:TMAX_SR_H)   # 48
# ))

# t_hours_grid = unique(vcat(
#     # collect(0.01:0.5:120.0),      # baseline/early
#     # collect(121.0:1.0:400.0),     # onset
#     collect(0.01:1.0:750.0),
#     collect(752.0:2.0:1250.0),    # transizioni (zona più importante)
#     collect(1260.0:10.0:1800.0),   # plateau
#     collect(1802.0:2.0:TMAX_SR_H)
# ))

# t_hours_grid = unique(vcat(
#     collect(0.01:1.0:250.0),     # inizio (abbastanza denso)
#     # collect(202.0:5.0:800.0),   # fase di salita
#     # collect(805.0:10.0:1250.0),  # avvicinamento al plateau
#     # collect(1270.0:20.0:2000.0),
#     collect(252.5:5.0:2100.0),
#     collect(2101.0:2.0:TMAX_SR_H) # plateau/coda
# ))

# t_hours_grid = unique(vcat(
#     collect(0.01:1.0:800.0),     # inizio (abbastanza denso)
#     # collect(202.0:5.0:800.0),   # fase di salita
#     # collect(805.0:10.0:1250.0),  # avvicinamento al plateau
#     # collect(1270.0:20.0:2000.0),
#     collect(802.5:20.0:2300.0),
#     collect(2301.0:2.0:TMAX_SR_H) # plateau/coda
# ))

# t_hours_grid = unique(vcat(
#     collect(0.01:1.0:120.0), # early
#     collect(121.0:2.0:500.0), # onset/transition
#     collect(505.0:5.0:1500.0), # late transition
#     collect(1520.0:20.0:TMAX_SR_H) # plateau
# ))

# Validation grid: same domain, different sampling
# t_hours_val_grid = unique(vcat(
#     collect(0.15:0.15:24.0),
#     collect(24.75:0.75:120.0),
#     collect(121.5:1.5:240.0),
#     collect(247.5:12.5:TMAX_SR_H)
# ))

# t_hours_val_grid = unique(vcat(
#     collect(0.08:0.20:24.0),
#     collect(24.8:1.0:120.0),
#     collect(122.0:2.0:240.0),
#     collect(250.0:20.0:TMAX_SR_H)
# ))

# t_hours_val_grid = unique(vcat(
#     collect(0.26:0.5:120.26),
#     collect(121.5:1.0:400.5),
#     collect(403.0:2.0:1801.0),
#     collect(1815.0:10.0:TMAX_SR_H)
# ))

# β_grid = collect(exp.(range(log(0.01), log(1.0), length=20)))
# β_grid = collect(range(0.01, 1.0, length=20))
# β_grid = collect(exp.(range(log(0.1), log(1.0), length=12)))
# β_val_grid = collect(exp.(range(log(0.012), log(0.95), length=15)))
# β_val_grid = collect(exp.(range(log(0.012), log(0.95), length=12)))
# β_val_grid = collect(exp.(range(log(0.011), log(0.99), length=12)))

t_hours_grid = unique(vcat(
    collect(0.01:2.0:250.0),      # step 2 → 0.01, 2.01, 4.01, ...
    collect(255.0:10.0:2100.0),   # step 10
    collect(2101.0:2.0:TMAX_SR_H)    # step 2 finali
))

β_grid = collect(range(0.1, 1.0, length=18))

# ============================================================
# 2) LOAD TRAINED NETWORK
# ============================================================
@info "Loading trained neural network"
chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim)

@load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters
best_nn = neural_network_parameters[best_idx]

@info "Network loaded"

# ============================================================
# 3a) BUILD SYNTHETIC TEACHER DATASET
# ============================================================
patient_id = String[]
t_h = Float64[]
t_norm = Float64[]
beta = Float64[]

for (iβ, βi) in enumerate(β_grid)
    for th in t_hours_grid
        push!(patient_id, "synth$(iβ)")
        push!(t_h, th)
        push!(t_norm, th / T_SCALE)
        push!(beta, βi)
    end
end

X_syn = [t_norm'; beta']   # 2 x N
@assert size(X_syn) == (2, length(t_norm))

# Direct NN teacher target
y_syn = Vector{Float64}(undef, length(t_norm))
for j in eachindex(t_norm)
    y_syn[j] = chain((@view X_syn[:, j]), best_nn)[1]
end

# Save teacher dataset
df = DataFrame(
    patient_id=patient_id,
    t_h=t_h,
    t_norm=t_norm,
    beta=beta,
    y_nn=y_syn
)
CSV.write("$(modelssave_path)/sr_teacher_dataset_direct.csv", df)

# ============================================================
# 4) PLOT EXACTLY WHAT SR SEES
# ============================================================
y_mat = reshape(y_syn, length(t_hours_grid), length(β_grid))

p_teacher = Plots.plot(
    xlabel="Time (h)",
    ylabel="rupture f(t_norm, β)",
    title="Synthetic NN target shown to SR",
    linewidth=2
)

for (iβ, βi) in enumerate(β_grid)
    Plots.plot!(p_teacher, t_hours_grid, y_mat[:, iβ],
        label="β = $(round(βi, digits=2))")
end

display(p_teacher)
savefig(p_teacher, "$(figsave_path)/sr_teacher_direct.png")

# ============================================================
# 5) SYMBOLIC REGRESSION
# ============================================================
# opts_warm = Options(
#     binary_operators = (+, *),
#     unary_operators  = (inv,),
#     maxsize = 18,
#     populations = 24,
#     parsimony = 5e-4,
#     should_optimize_constants = true,
#     # complexity_of_constants = 2,
#     # complexity_of_operators = [inv => 2],
#     output_directory = "$(modelssave_path)/sr_outputs",
#     save_to_file = false,
#     seed = 42
# )

# opts = Options(
#     binary_operators = (+, *),
#     unary_operators  = (inv,),
#     maxsize = 18,
#     populations = 24,
#     parsimony = 5e-5,
#     # complexity_of_constants = 2,
#     # complexity_of_operators = [inv => 2],
#     batching = true,
#     batch_size = 512,
#     should_optimize_constants = true,
#     output_directory = "$(modelssave_path)/sr_outputs",
#     save_to_file = true,
#     seed = 42
# )

# refactored 04-17_16-43-23 dal salvataggio timeline 03-27_12_53_41 sym_reg_controlled.jl
λ_neg = 200.0   # penalità forte per y_pred < 0
λ_hi = 20.0    # opzionale: penalità leggera per y_pred > 1

smooth_relu_fast(x; ϵ=1e-5) = 0.5 * (x + sqrt(x * x + ϵ * ϵ))

loss_smooth_fast = (y_pred, y_true) ->
    (y_pred - y_true)^2 +
    λ_neg * smooth_relu_fast(-y_pred)^2 +
    λ_hi * smooth_relu_fast(y_pred - 1.0)^2

opts = Options(
    binary_operators=(+, *),
    unary_operators=(inv,),
    maxsize=20,
    populations=24,
    parsimony=5e-4,
    complexity_of_constants=2,
    # complexity_of_operators=[exp => 4, inv => 2],
    batching=true,
    batch_size=512,
    should_optimize_constants=true,
    # elementwise_loss = (y_pred, y_true) ->
    #     (y_pred - y_true)^2 +
    #     λ_neg * max(-y_pred, 0.0)^2 +
    #     λ_hi  * max(y_pred - 1.0, 0.0)^2,
    elementwise_loss=loss_smooth_fast,
    output_directory="$(modelssave_path)/sr_outputs",
    save_to_file=true,
    seed=42
)

# refactored 04-17_16_16_01 dedotto
# opts = Options(
#     binary_operators=(+, *),
#     unary_operators=(inv,),
#     maxsize=18,
#     populations=24,
#     parsimony=5e-4,
#     # complexity_of_constants=2,
#     # complexity_of_operators=[inv => 2],
#     # batching = true,
#     # batch_size = 512,
#     should_optimize_constants=true,
#     output_directory="$(modelssave_path)/sr_outputs",
#     save_to_file=true,
#     seed=42
# )

@info "Warm-up SR"
equation_search(
    X_syn, y_syn;
    # weights = weights_syn,
    niterations=300,
    options=opts,
    parallelism=:multithreading,
    progress=true,
    variable_names=["t_norm", "β"]
)

# opts = Options(
#     binary_operators = (+, *),
#     unary_operators  = (inv,),
#     maxsize = 18,
#     populations = 24,
#     parsimony = 5e-4,
#     should_optimize_constants = true,
#     # complexity_of_constants = 2,
#     # complexity_of_operators = [inv => 2],
#     output_directory = "$(modelssave_path)/sr_outputs",
#     save_to_file = true,
#     seed = 42
# )

@info "Starting main SR search"
hof = equation_search(
    X_syn, y_syn;
    # weights = weights_syn,
    niterations=25_000,
    options=opts,
    parallelism=:multithreading,
    progress=true,
    variable_names=["t_norm", "β"]
)

# ============================================================
# 6) FORMULA SELECTION FROM THE PARETO FRONTIER
# ============================================================
#=
  Symbolic Regression does not return a single formula, but a
  **Pareto frontier**: a set of non-dominated formulas representing
  the trade-off between accuracy and complexity. A formula A dominates
  B if it is both simpler AND more accurate; all non-dominated formulas
  form the frontier.

  Selection is performed in two phases:

  PHASE 1 — Selection by accuracy
  ────────────────────────────────
  For each formula on the frontier, the MSE is evaluated on the
  synthetic teacher dataset (X_syn, y_syn). The formula with the
  lowest error is identified:

      best_idx_front = argmin(val_losses)

  PHASE 2 — Tie-break by simplicity (post-hoc parsimony)
  ────────────────────────────────────────────────────────
  Among all formulas whose MSE ≤ tol × MSE_minimum (with tol = 1.02,
  i.e. within 2% of the minimum), the one with the **lowest
  complexity** is selected. This favours simpler, more physically
  interpretable expressions when accuracy differences are negligible.

  Note: complexity is computed by SymbolicRegression by counting tree
  nodes, with additional penalties for constants
  (complexity_of_constants=2) and operators (complexity_of_operators).
=#

# Evaluate each frontier member on the synthetic training dataset
function sr_eval(tree, X)
    out = eval_tree_array(tree, X)
    vals = out isa Tuple ? collect(out[1]) : collect(out)
    y = Float64.(vals)
    for i in eachindex(y)
        if !isfinite(y[i])
            y[i] = 1e6   # penalizza valori non finiti (Inf, NaN)
        end
    end
    return y
end

frontier = calculate_pareto_frontier(hof)

# Phase 1: compute MSE for each frontier member
val_losses = Float64[]
complexities = Int[]

for member in frontier
    y_hat_val = sr_eval(member.tree, X_syn)
    push!(val_losses, mean((y_syn .- y_hat_val) .^ 2))
    push!(complexities, compute_complexity(member, opts))
end

# Phase 1: select the formula with minimum error
best_idx_front = argmin(val_losses)

# Phase 2: among formulas within 2% of the minimum, pick the simplest
tol = 1.02
near_best = findall(val_losses .<= tol * val_losses[best_idx_front])
if !isempty(near_best)
    best_idx_front = near_best[argmin(complexities[near_best])]
end

best = frontier[best_idx_front]

eq_best = string_tree(best.tree; variable_names=["t_norm", "β"])
println("Selected equation:")
println(eq_best)
println("Validation loss = ", val_losses[best_idx_front])
println("Complexity = ", complexities[best_idx_front])

# Save frontier
front_df = DataFrame(
    idx=collect(1:length(frontier)),
    complexity=[compute_complexity(x, opts) for x in frontier],
    loss=[x.loss for x in frontier],
    equation=[string_tree(x.tree; variable_names=["t_norm", "β"]) for x in frontier]
)
CSV.write("$(modelssave_path)/sr_pareto_frontier_direct.csv", front_df)

# ============================================================
# 7) EVALUATE SURROGATE ON THE SAME SYNTHETIC GRID
# ============================================================
y_hat = sr_eval(best.tree, X_syn)

mse_sr = mean((y_syn .- y_hat) .^ 2)
mae_sr = mean(abs.(y_syn .- y_hat))
r2_sr = 1 - sum((y_syn .- y_hat) .^ 2) / sum((y_syn .- mean(y_syn)) .^ 2)

println("SR metrics on synthetic grid:")
println("  MSE = ", mse_sr)
println("  MAE = ", mae_sr)
println("  R²  = ", r2_sr)

# ============================================================
# 8) FINAL COMPARISON PLOT: NN vs SR
# ============================================================
p_cmp = Plots.plot(
    xlabel="Time (h)",
    ylabel="rupture f(t_norm, β)",
    title="NN vs SR surrogate (direct fit on y_NN)",
    linewidth=2,
    legend=:best
)

for (k, βi) in enumerate(0.1:0.1:1.0)
    y_nn = [chain([t / T_SCALE, βi], best_nn)[1] for t in t_hours_grid]

    X_tmp = hcat([[t / T_SCALE, βi] for t in t_hours_grid]...)  # 2 x T
    y_sr = sr_eval(best.tree, X_tmp)

    lbl_nn = (k == 1) ? "NN (solid lines)" : false
    lbl_sr = (k == 1) ? "SR (dashed lines)" : false

    Plots.plot!(p_cmp, t_hours_grid, y_nn, linestyle=:solid, label=lbl_nn)
    Plots.plot!(p_cmp, t_hours_grid, y_sr, linestyle=:dash, label=lbl_sr)
end

display(p_cmp)
savefig(p_cmp, "$(figsave_path)/nn_vs_sr_direct.png")

p_surr = Plots.plot(
    xlabel="Time (h)",
    ylabel="rupture f(t_norm, β)",
    title="SR surrogate",
    linewidth=2,
    legend=:best
)

for (k, βi) in enumerate(0.1:0.1:1.0)
    X_tmp = hcat([[t / T_SCALE, βi] for t in t_hours_grid]...)  # 2 x T
    y_sr = sr_eval(best.tree, X_tmp)

    lbl_sr = (k == 1) ? "SR (dashed lines)" : false
    Plots.plot!(p_surr, t_hours_grid, y_sr, linestyle=:dash, label=lbl_sr)
end

display(p_surr)
savefig(p_surr, "$(figsave_path)/sr_direct.png")
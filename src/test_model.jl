using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics

include("ctnt-ude-model.jl")

println("Dataset loading...")
@load "res/models/trainingsetNSTEMI_SSE_0806log.jld2" training_dataset;
@load "res/models/testsetNSTEMI_SSE_0806log.jld2" test_dataset;

@load "res/models/lossesNSTEMI_SSE_0806log.jld2" losses;

segments = []         # raccoglierà i range di indici
start = 1             # inizio del primo segmento
n = length(losses)
@assert n > 0 "Array vuoto!"

for i in 1:(n-1)
    # rottura se zero o salto ≥ threshold
    if i+1 < length(losses) && abs(losses[i+1] / losses[i]) ≥ 100
        push!(segments, start:i)
        start = i + 1
    end
end
# chiudi l’ultimo segmento
push!(segments, start:n)

n = length(segments)

for i in 1:n
    loss = losses[segments[i]]
    pl_losses = plot(1:1000, loss[1:1000], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
    plot!(1001:length(loss), loss[1001:end], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)
    display(pl_losses)
end

# idx_start = adam_maxiters*(i-1) + 1  # per il primo paziente
# idx_end   = adam_maxiters*i

# pl_losses = plot(idx_start:idx_end, losses[idx_start:idx_end], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)

# idx_start = (adam_maxiters + 1)*(i-1) + 1  # per il primo paziente
# idx_end   = (adam_maxiters + 1 + lbfgs_maxiters)*i

# plot!(idx_start:idx_end, losses[idx_start:idx_end], yaxis = :log10, xaxis = :log10,
#     xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)

@load "res/models/optsolsNSTEMI_SSE_0806log.jld2" optsols;

@load "res/models/nnNSTEMI_SSE_0806log.jld2" neural_network_parameters;
@load "res/models/odebetasNSTEMI_SSE_0806log.jld2" ode_params;

@load "res/models/objectivesNSTEMI_SSE_0806log.jld2" objectives;

@load "res/models/odebetastestNSTEMI_SSE_0806log.jld2" ode_betas_test;
@load "res/models/lossestestNSTEMI_SSE_0806log.jld2" losses_test;

@load "res/models/best_nn_NSTEMI_SSE_0806log.jld2" best_nn
@load "res/models/best_ode_beta_NSTEMI_SSE_0806log.jld2" best_ode_beta

# @load "res/models/trainingsetNSTEMI_SSE_0706log.jld2" training_dataset;
# @load "res/models/testsetNSTEMI_SSE_0706log.jld2" test_dataset;

# println("Neural network loading...")
# @load "res/models/nnNSTEMI_SSE_0706log.jld2" neural_network_parameters;

# println("ODE and Betas loading...")
# @load "res/models/odebetasNSTEMI_SSE_0706log.jld2" ode_params;

# @load "res/models/optsolsNSTEMI_SSE_0706log.jld2" optsols;

# @load "res/models/best_nn_NSTEMI_SSE_0706log.jld2" best_nn;
# @load "res/models/best_ode_beta_NSTEMI_SSE_0706log.jld2" best_ode_beta;

# @load "res/models/odebetastestNSTEMI_SSE_0706log.jld2" ode_betas_test;

opt_sol = optsols[1]
best_nn1 = opt_sol.u.neural
ode_betas_test = opt_sol.u.ode

i = 5
patient = test_dataset[i]
idx_start = 5*(i-1) + 1
idx_end = 5*i
patient_params = ode_betas_test[idx_start:idx_end]  # usa il guess iniziale, per esempio
println(exp.(patient_params))
tspan = (0.0, patient.timepoints[end]+10)

chain = neural_network_model(2, 4; input_dims=7);

# Costruisci il modello per questo paziente:
model = ctntCUDEModel(patient_params, chain, tspan)

p = ComponentArray(ode = patient_params, neural = best_nn1)

lhs_lb = log.([0.001, 0.001, 0.01, 0.001, 0.001]);
lhs_ub = log.([5.0, 5.0, 300.0, 400.0, 3]);

initial = vec(mean(reshape(opt_sol.u.ode, :, 5), dims=1))
println(exp.(initial))

optfunc = OptimizationFunction(patient_loss, AutoForwardDiff())

optprob = OptimizationProblem(optfunc, initial,
                (model, patient.timepoints, patient.ctnt_data, p.neural),
                lb = lhs_lb, ub = lhs_ub)

optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()),
    maxiters=1000)

println(exp.(optsol.u))
println(optsol.objective)

p_opt = ComponentArray(ode = optsol.u, neural = best_nn1)

# sol = Array(solve(model.problem, p=p_opt, saveat=1))
sol = solve_model(p_opt, (model, patient.timepoints, patient.ctnt_data))
sol = Array(sol)
println(patient_loss(p_opt, (model, patient.timepoints, patient.ctnt_data, p.neural)))
println(compute_loss(p_opt, (model, patient.timepoints, patient.ctnt_data)))
# sol = Array(solve_model(p, (model, patient.timepoints, patient.ctnt_data)))
pred = sol[3,:]
# pred = [u[3] for u in sol.u]

plot(patient.timepoints, pred; lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")
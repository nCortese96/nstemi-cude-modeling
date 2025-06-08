using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics

include("ctnt-ude-model.jl")

println("Dataset loading...")

@load "res/models/trainingsetNSTEMI_SSE_0706log.jld2" training_dataset;
@load "res/models/testsetNSTEMI_SSE_0706log.jld2" test_dataset;

println("Neural network loading...")
@load "res/models/nnNSTEMI_SSE_0706log.jld2" neural_network_parameters;

println("ODE and Betas loading...")
@load "res/models/odebetasNSTEMI_SSE_0706log.jld2" ode_params;

@load "res/models/optsolsNSTEMI_SSE_0706log.jld2" optsols;

@load "res/models/best_nn_NSTEMI_SSE_0706log.jld2" best_nn;
@load "res/models/best_ode_beta_NSTEMI_SSE_0706log.jld2" best_ode_beta;

@load "res/models/odebetastestNSTEMI_SSE_0706log.jld2" ode_betas_test;

opt_sol = optsols[1]
best_nn1 = opt_sol.u.neural

i = 1
patient = test_dataset[i]
patient_params = ode_betas_test[i]  # usa il guess iniziale, per esempio
println(exp.(patient_params))
tspan = (0.0, patient.timepoints[end]+10)

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
    maxiters=200)

println(exp.(optsol.u))
println(optsol.objective)

chain = neural_network_model(2, 6; input_dims=7);

# Costruisci il modello per questo paziente:
model = ctntCUDEModel(patient_params, chain, tspan)

sol = solve(model.problem, p=p, saveat=1)
println(patient_loss(p, (model, patient.timepoints, patient.ctnt_data, p.neural)))
println(compute_loss(p, (model, patient.timepoints, patient.ctnt_data)))
# sol = Array(solve_model(p, (model, patient.timepoints, patient.ctnt_data)))
# pred = sol[3,:]
pred = [u[3] for u in sol.u]

plot(pred, lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")
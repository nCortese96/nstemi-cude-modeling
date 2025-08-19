using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics
using Dates
using Logging
using Base.Threads: @threads, nthreads

println("⚠️ Test formula algorithm started $(now())")

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

T_SCALE = 350.0;

experiment = "NSTEMI_partrvalMIMIC_logSSEf_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

figsave_path = "$(fig_path)/test_formula";
modelssave_path = "$(models_path)/test_formula";

mkpath(figsave_path)
mkpath(modelssave_path)

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Test formula algorithm Started")
#     println(io, "********************************")
# end

# @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn
@load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
@load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;

lhs_lb = log.([0.001, 0.001, 0.01, 0.01, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1]); # 5.0, 5.0, 300.0, 400.0, 3

ode_p = best_solution;

const c1 = 0.13256218391741428;
const c2 = 13.260793182859695;
const c3 = 67.46423088085042;
corr(t, β) = c1 / (c2*t + c3*β + 1/(t + eps(Float64)))   # guardrail su t≈0

function ctnt_ode!(du, u, p, t)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]
    # println("ode")
    # println(p)
    β = exp(p[5]) # Positive conditional parameter

    a = exp(p[1])
    b = exp(p[2])
    # Cs0 = exp(p.ode[3])
    # Cc0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    # correction = chain([u[1], t, a, b, Cs0, Cc0, β], p.neural)[1]

    # correction = chain([u[1], t, β], p.neural)[1]

    t_norm   = t / T_SCALE

    correction = corr(t_norm, β)

    du[1] = - (Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a*(Cc - Cp)
    du[3] = a*(Cc - Cp) - b*Cp
end

function patient_loss_formula(θ, (problem, timepoints, ctnt_data))
    # println("loss")
    # println(θ)
    u0 = [exp(θ[3]), exp(θ[4]), 0.0]

    # ODEProblem aggiornato
    prob = remake(problem; u0 = u0, p = θ)

    sol = solve(prob, Tsit5(); p=θ, saveat=timepoints,
    #callback = POS_CB, isoutofdomain = NEG_TEST
    ) 

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    sol = max.(Array(sol), DELTA)

    plasm = sol[3,:];

    # return sum(abs2, plasm - ctnt_data)
    return sum(abs2, log.(plasm) .- log.(ctnt_data))
    # return sum(((plasm - ctnt_data).^2).*ctnt_data)
    # return smape(plasm, ctnt_data)
end

function smape_loss_formula(θ, (problem, timepoints, ctnt_data))

    u0 = [exp(θ[3]), exp(θ[4]), 0.0]

    # ODEProblem aggiornato
    prob = remake(problem; u0 = u0, p = θ)

    sol = solve(prob, Tsit5(); p=θ, saveat=timepoints,
    #callback = POS_CB, isoutofdomain = NEG_TEST
    ) 

    if !successful_retcode(sol)
        # If the solver fails, return infinity
        return Inf
    end

    sol = max.(Array(sol), DELTA)

    plasm = sol[3,:];

    return smape(plasm, ctnt_data)
end

pguess = mean([optsol.u for optsol in ode_p]);

optfunc = OptimizationFunction(patient_loss_formula, AutoForwardDiff());

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "*********************************")
    println(io, "Evaluating Formula from SymReg with sMAPE")
end

smape_values = [];
loss_values = [];

ev_bar = Progress(length(test_dataset); desc = "Validating", color = :cyan, showspeed = true)
for (i, patient) in enumerate(test_dataset)

    u0_init = [exp(pguess[3]), exp(pguess[4]), 0.0];

    tspan = (0.0, patient.timepoints[end] + 10.0);

    problem = ODEProblem(ctnt_ode!, u0_init, tspan);

    optprob = OptimizationProblem(optfunc, pguess, (problem, patient.timepoints, patient.ctnt_data), lb = lhs_lb, ub = lhs_ub);

    optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000);

    p_opt = optsol_lbfgs.u;

    println("For $(patient.id), params: ", p_opt)
    println("Params: ", exp.(p_opt))
    # push!(optsols_valid, optsol_lbfgs);

    u0_new = [exp(p_opt[3]), exp(p_opt[4]), 0.0]
    prob   = remake(problem; u0 = u0_new, p = p_opt)

    sol = Array(solve(prob, Tsit5(); p=p_opt, saveat=1));
    println("Patient loss: ", patient_loss_formula(p_opt, (prob, patient.timepoints, patient.ctnt_data)))
    # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
    push!(loss_values, optsol_lbfgs.objective)
    println("Objective:    ", optsol_lbfgs.objective)
    validation_metric = smape_loss_formula(p_opt, (prob, patient.timepoints, patient.ctnt_data));
    println("sMAPE: ", validation_metric)
    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Patient $(patient.id) sMAPE Formula validation: $validation_metric")
    end
    push!(smape_values, validation_metric)
    pred = sol[3,:];

    pl = plot(pred; lw=2, label="Model with Formula Prediction", xlabel="Time", ylabel="Troponin", title="Patient $(patient.id)")
    scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

    display(pl)

    save("$(figsave_path)/patient_$(patient.id).svg", pl)
    next!(ev_bar)
end

println(median(smape_values))
println(median(loss_values))

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Median sMAPE with formula: $(median(smape_values))")
    println(io, "--> Median logSSE with formula: $(median(loss_values))")
end


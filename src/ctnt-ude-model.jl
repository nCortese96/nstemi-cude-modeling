using SimpleChains: SimpleChain, TurboDense, static, init_params
using SciMLBase: ODEProblem, OptimizationSolution
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using DataFrames: DataFrame
using StableRNGs

using ProgressMeter: Progress, next!

using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches

softplus(x) = log(1 + exp(x))

# Definizione della struttura per i dati del paziente
struct PatientData
    id::String
    timepoints::Vector{Float64}   # vettore dei timepoints per il paziente
    ctnt_data::Vector{Float64}      # vettore dei valori di troponina
    init_params::Vector{Float64}    # guess iniziale, ad esempio: [a, b, Cs0, Cc0]
end

# Funzione per convertire la riga i-esima dei tre DataFrame in una struttura PatientData.
# Per ogni riga vengono rimossi i valori mancanti (missing) e si mantengono solo i valori validi.
function row_to_patient(i, ids::DataFrame, timepoints_df::DataFrame, troponin_df::DataFrame, initial_params::AbstractVector{Float64})
    # Estrai l'ID (se serve)
    id_val = ids[i, :][1]
    
    # Estrai i valori della riga come vettori.
    # timepoints_df[i, :] restituisce una NamedTuple; convertiamola in vettore.
    tp_row = [x for x in collect(values(timepoints_df[i, :])) if !ismissing(x)]
    ctnt_row = [x for x in collect(values(troponin_df[i, :])) if !ismissing(x)]
    
    # Opzionalmente, se i valori sono stringhe, li puoi convertire in Float64.
    # In questo esempio assumiamo che infer_eltypes=true li abbia già convertiti.
    
    # Definisci dei guess iniziali standard, ad esempio:
    # init_params = [0.005, 0.005, 0.1, 0.001, log(0.1)]
    
    return PatientData(id_val, tp_row, ctnt_row, initial_params)
end

function neural_network_model(depth::Int, width::Int; input_dims::Int = 7)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    # Esempio di termini dinamici (da adattare al modello specifico)
    # Termini base (senza correzione)
    # p.ode = [a, b, Cs0, Cc0]
    # Cs_ctnt = u[1]
    # Cc_ctnt = u[2]
    # Cp_ctnt = u[3]

    β = exp(p.ode[5]) # sempre positivo

    # a = 10 ^ θ[1]
    # b = 10 ^ θ[2]

    a = exp(p.ode[1])
    b = exp(p.ode[2])
    Cc0 = exp(p.ode[3])
    Cs0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    correction = chain([u[1], t, a, b, Cc0, Cs0, β], p.neural)[1]

    du[1] = - (u[1] - u[2] + correction)
    du[2] = (u[1] - u[2] + correction) - a*(u[2] - u[3])
    du[3] = a*(u[2] - u[3]) - b*u[3]

end

struct ctntCUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

function ctntCUDEModel(
    # ctnt_timepoints::AbstractVector{T},
    θ,
    chain::SimpleChain,
    tspan::Tuple{T,T}
    ) where T <: Real

    # construct the ude function
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)

    # tspan = (ctnt_timepoints[1], ctnt_timepoints[end])

    Cc0 = exp(θ[3])
    Cs0 = exp(θ[4])

    u0 = [Cc0, Cs0, 0];

    # ode = ODEProblem(cude!, u0, tspan, θ)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntCUDEModel(ode, chain)
end

################################# PREDICT ##########################################

function solve_model(θ, (model, timepoints, ctnt_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: ctntCUDEModel
    return solve(model.problem, p=θ, saveat=timepoints)
end

########################## LOSS FUNCTIONS ##########################################

function compute_loss(θ, (model, timepoints, ctnt_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: ctntCUDEModel
    # solve the ODE problem
    sol = solve_model(θ, (model, timepoints, ctnt_data))
    pred = [u[3] for u in sol.u]
    # Calculate the squared error
    return sum((pred .- ctnt_data).^2)
end

# 4. Funzione training_loss: somma la loss su tutti i pazienti del training
#
# x è un vettore contenente:
#   - i parametri della rete neurale (globali) (primi N_nn elementi)
#   - per ciascun paziente, 5 parametri specifici: [a, b, Cs0, Cc0, log(β)]
function training_loss(p, training_dataset, nn_params_init)
    N_nn = length(nn_params_init)
    loss_tot = 0.0
    nn_param_vec = p[1:N_nn]
    for (i, patient) in enumerate(training_dataset)
        idx_start = N_nn + 5*(i-1) + 1
        idx_end   = N_nn + 5*i
        patient_params = p[idx_start:idx_end]
        tspan = (patient.timepoints[1], patient.timepoints[end])

        model = ctntCUDEModel(patient_params, chain, tspan)
        θ = ComponentArray(ode = patient_params, neural = nn_param_vec)
        ### Calcolo cost function ###
        loss_tot += compute_loss(θ, (model, patient.timepoints, patient.ctnt_data))
        # loss_tot += sum(abs2, sol[3,:] - patient.ctnt_data)
    end
    return loss_tot / length(training_dataset) # MSE
end

function sample_initial_neural_parameters(n_initials::Int, chain::SimpleChain, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_ode_parameters(n_initials::Int, lhs_lb::AbstractVector{T}, lhs_ub::AbstractVector{T}, rng::AbstractRNG) where T <: Real
    return sample(n_initials, lhs_lb, lhs_ub, LatinHypercubeSample(rng))
end

function create_start_points(
    chain::SimpleChain,
    initial_guesses::Int = 25_000,
    lhs_lb::AbstractVector{T} = [0.001, 0.001, 0.01, 0.001, -Inf],
    lhs_ub::AbstractVector{T} = [5, 5, 300, 400, Inf],
    n_params_guess::Int = 1,
    rng::AbstractRNG = StableRNG(42)
    ) where T <: Real

    initial_nn = sample_initial_neural_parameters(initial_guesses, chain, rng)
    initial_ode = sample_initial_ode_parameters(initial_guesses, lhs_lb, lhs_ub, rng)

    initial_parameters = [ComponentArray(
        neural = initial_nn[i],
        ode = repeat(initial_ode[:,i], 1, n_params_guess)
    ) for i in eachindex(initial_guesses)]
    return initial_parameters
end

function otpimize(optfunc::OptimizationFunction, θ_init, adam_maxiters, lbfgs_maxiters)

    # Definisci la funzione di loss come una funzione che accetta due argomenti:
    # - θ: il vettore dei parametri
    # - data: una tupla contenente il training_dataset (in questo caso)
    # optfunc = OptimizationFunction((θ, x) -> training_loss(θ, training_dataset, θ_init.neural), AutoForwardDiff());

    # Primo step: utilizziamo Gradient Descent per una convergenza rapida
    optprob = Optimization.OptimizationProblem(optfunc, θ_init);
    opt_result1 = Optimization.solve(optprob, Optimisers.Adam(0.01), maxiters=adam_maxiters);

    optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u);
    opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=lbfgs_maxiters);

    return opt_result2
end

function train(training_dataset, initial_parameters, adam_maxiters, lbfgs_maxiters)
    optsols = OptimizationSolution[]
    optfunc = OptimizationFunction((θ, x) -> training_loss(θ, training_dataset, θ_init.neural), AutoForwardDiff())
    prog = Progress(selected_initials; dt=1.0, desc="Optimizing...", color=:blue)
    for i in initial_parameters
        opt_sol = optimize(optfunc, initial_parameters[i], adam_maxiters, lbfgs_maxiters)
        push!(optsols, opt_sol)
        next!(prog)
    end
    return optsols
end

## Finito il train si estraggono i parametri della rete 
# patient_loss: calcola la loss per un singolo paziente dato un vettore di parametri specifici
function patient_loss(patient_params, model::ctntCUDEModel, timepoints, ctnt_data, fixed_nn_params)
    p = ComponentArray(ode = patient_params, neural = fixed_nn_params)
    sol = solve(model.problem, Tsit5(); p=p, saveat=timepoints)
    pred = [u[3] for u in sol.u]
    return sum((pred .- ctnt_data).^2)
end
# La differenza sta nel dove si crea il component array:
# Se lo dai in pasto alla loss lo ottimizza tutto,
# se lo costruisci dentro ottimizza solo i parametri del modello

function otpimize(optfunc::OptimizationFunction,
    lbfgs_maxiters,
    initial_ode_params::AbstractVector{T} = [0.005, 0.005, 0.1, 0.001]) where T<:Real

    optprob2 = Optimization.OptimizationProblem(optfunc, initial_ode_params);
    opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=lbfgs_maxiters);

    return opt_result2
end

# function train(validation_dataset, initial_ode_params, fixed_nn_params, lower_bounds, upper_bounds, lbfgs_maxiters)
#     optsols = OptimizationSolution[]
#     model = ctntCUDEModel(patient_params, chain, tspan)
#     optfunc = OptimizationFunction((θ, x) -> patient_loss(θ, training_dataset, θ_init.neural), AutoForwardDiff())
#     prog = Progress(selected_initials; dt=1.0, desc="Optimizing...", color=:blue)
#     for i in initial_parameters
#         opt_sol = optimize(optfunc, initial_parameters[i], validation_dataset, adam_maxiters, lbfgs_maxiters)
#         push!(optsols, opt_sol)
#         next!(prog)
#     end
#     return optsols
# end

function select_best_starts(
    chain::SimpleChain,
    n_best::Int = 25,
    initial_guesses::Int = 25_000,
    lhs_lb::AbstractVector{T} = [0.001, 0.001, 0.01, 0.001, -Inf],
    lhs_ub::AbstractVector{T} = [5, 5, 300, 400, Inf],
    n_params_guess::Int = 1,
    rng::AbstractRNG = StableRNG(42)
    ) where T <: Real

    initial_parameters = create_start_points(initial_guesses, chain, lhs_lb, lhs_ub, n_params_guess, rng)
    losses = Float64[]
    prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
    for p in initial_parameters
        loss_value = compute_loss(p, (models, timepoints, cpeptide_data))
        push!(losses, loss_value)
        next!(prog)
    end

    println("Initial parameters evaluated. Optimizing for the best $(n_best) initial parameters.")
    best_indices = partialsortperm(losses, 1:n_best)
    return initial_parameters[best_indices] # collezione di θ_init
end



function select_model(validation_dataset, fixed_nn_params, initial_ode_params::AbstractVector{T} = [0.005, 0.005, 0.1, 0.001], 
    
) where T<:Real
end

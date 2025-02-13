
using SciMLBase: ODEProblem, OptimizationSolution
using SimpleChains: SimpleChain, TurboDense, static, init_params
using DataInterpolations: LinearInterpolation
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using ProgressMeter: Progress, next!
using StatsBase: countmap

using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches

COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
)

COLORLIST = [
    RGBf(252/255, 253/255, 191/255),
    RGBf(254/255, 191/255, 132/255),
    RGBf(250/255, 127/255, 94/255),
]

abstract type CPeptideModel end

softplus(x) = log(1 + exp(x))

"""
neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

Constructs a neural network model with a given depth and width. The input dimensions are set to 2 by default.

# Arguments
- `depth::Int`: The depth of the neural network.
- `width::Int`: The width of the neural network.
- `input_dims::Int`: The number of input dimensions. Default is 2.

# Returns
- `SimpleChain`: A neural network model.
"""
function neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

"""
c_peptide_kinetic_parameters(age::Real, t2dm::Bool)

Calculates the kinetic parameters for the c-peptide model based on the age and the presence of type 2 diabetes. The
parameters are based on the van Cauter model. [1]

# Arguments
- `age::Real`: The age of the individual.
- `t2dm::Bool`: A boolean indicating whether the individual has type 2 diabetes.

# Returns
- `Tuple`: A tuple containing the kinetic parameters k0, k1, and k2.

[1]: Van Cauter, E., Mestrez, F., Sturis, J., Polonsky, K. S. (1992). Estimation of insulin secretion rates from C-peptide levels. Comparison of individual and standard kinetic parameters for C-peptide clearance. Diabetes, 41(3), 368-377.
"""
function c_peptide_kinetic_parameters(age::Real, t2dm::Bool)

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    k1 = fraction * (log(2)/long_half_life) + (1-fraction) * (log(2)/short_half_life)
    k0 = (log(2)/short_half_life)*(log(2)/long_half_life)/k1
    k2 = (log(2)/short_half_life) + (log(2)/long_half_life) - k0 - k1

    return k0, k1, k2
end


"""
c_peptide_cude!(du, u, p, t, chain::SimpleChain, glucose::LinearInterpolation, glucose_t0::Real, Cb::T, k0::T, k1::T, k2::T) where T <: Real

The ODE function for the c-peptide model with a _conditional_ neural network for c-peptide production. 
The model consists of two compartments: plasma c-peptide and interstitial c-peptide. 

# Arguments
- `du`: The derivative vector.
- `u`: The state vector.
- `p`: The parameter vector.
- `t`: The time.
- `chain::SimpleChain`: The neural network model.
- `glucose::LinearInterpolation`: The glucose data as a linear interpolation.
- `glucose_t0::Real`: The initial timepoint for the glucose data.
- `Cb::T`: The basal c-peptide value.
- `k0::T`: The kinetic parameter k0.
- `k1::T`: The kinetic parameter k1.
- `k2::T`: The kinetic parameter k2.

# Returns
- `Nothing`: The derivative vector is updated in place.
"""
function c_peptide_cude!(du, u, p, t, chain::SimpleChain, glucose::LinearInterpolation, 
    glucose_t0::Real, Cb::T, k0::T, k1::T, k2::T) where T <: Real

    # extract vector of conditional parameters
    β = exp.(p.ode)

    # production by neural network, forced in steady-state at t0
    ΔG = glucose(t) - glucose(glucose_t0)
    production = chain([ΔG; β], p.neural)[1] - chain([0.0; β], p.neural)[1]

    # two c-peptide compartments

    # plasma c-peptide
    du[1] = -(k0 + k2) * u[1] + k1 * u[2] + Cb*k0 + production

    # interstitial c-peptide
    du[2] = -k1*u[2] + k2*u[1]

end

struct CPeptideCUDEModel<:CPeptideModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
CPeptideCUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, chain::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool)

Constructs a c-peptide model with a conditional neural network for c-peptide production.

# Arguments
- `glucose_data::AbstractVector{T}`: The glucose data.
- `glucose_timepoints::AbstractVector{T}`: The timepoints for the glucose data.
- `age::Real`: The age of the individual.
- `chain::SimpleChain`: The neural network model.
- `cpeptide_data::AbstractVector{T}`: The c-peptide data.
- `t2dm::Bool`: A boolean indicating whether the individual has type 2 diabetes.

# Returns
- `CPeptideCUDEModel`: A c-peptide model with a conditional neural network for c-peptide production.
"""
function CPeptideCUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, 
    chain::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool) where T <: Real

    # interpolate glucose data
    glucose = LinearInterpolation(glucose_data, glucose_timepoints)
    
    # basal c-peptide
    Cb = cpeptide_data[1]

    # get kinetic parameters
    k0, k1, k2 = c_peptide_kinetic_parameters(age, t2dm)

    # construct the ude function
    cude!(du, u, p, t) = c_peptide_cude!(du, u, p, t, chain, glucose, glucose_timepoints[1], Cb, k0, k1, k2)

    # initial conditions
    u0 = [Cb, (k2/k1)*Cb]

    # time span
    tspan = (glucose_timepoints[1], glucose_timepoints[end])

    # construct the ode problem
    ode = ODEProblem(cude!, u0, tspan)

    return CPeptideCUDEModel(ode, chain)
end

"""
loss(θ, (model, timepoints, cpeptide_data))

Sum of squared errors loss function for the c-peptide model.

CALCOLO LOSS DI BASE

# Arguments
- `θ`: The parameter vector.
- `model::CPeptideModel`: The c-peptide model.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractVector{T}`: The c-peptide data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, cpeptide_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: CPeptideModel

    # solve the ODE problem
    sol = Array(solve(model.problem, p=θ, saveat=timepoints))
    # Calculate the mean squared error
    return sum(abs2, sol[1,:] - cpeptide_data)
end

"""
loss(θ, (models, timepoints, cpeptide_data, neural_network_parameters))

Sum of squared errors loss function for the conditional UDE c-peptide model with known neural network parameters.

LOSS QUANTO LA RETE È DEFINITA

# Arguments
- `θ`: The parameter vector.
- `p`: The tuple containing the following elements:
    - `models::CPeptideCUDEModel`: The conditional c-peptide models.
    - `timepoints::AbstractVector{T}`: The timepoints.
    - `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
    - `neural_network_parameters::AbstractVector{T}`: The neural network parameters.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, cpeptide_data, neural_network_parameters)::Tuple{CPeptideCUDEModel, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}) where T <: Real

    # construct the parameter vector
    p = ComponentArray(ode=θ, neural=neural_network_parameters)
    return loss(p, (model, timepoints, cpeptide_data))
end

"""
loss(θ, (models, timepoints, cpeptide_data))

Sum of squared errors loss function for the conditional UDE c-peptide model with multiple models.

QUANDO DEVI OTTIMIZZARE ANCORA TUTTO

# Arguments
- `θ`: The parameter vector.
- `p`: The tuple containing the following elements:
    - `models::AbstractVector{CPeptideCUDEModel}`: The conditional c-peptide models.
    - `timepoints::AbstractVector{T}`: The timepoints.
    - `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (models, timepoints, cpeptide_data)::Tuple{AbstractVector{CPeptideCUDEModel}, AbstractVector{T}, AbstractMatrix{T}}) where T <: Real
    # calculate the loss for each model
    error = 0.0
    for (i, model) in enumerate(models)
        p_model = ComponentArray(ode = θ.ode[i,:], neural=θ.neural)
        error += loss(p_model, (model, timepoints, cpeptide_data[i,:]))
    end
    return error / length(models)
end

function sample_initial_neural_parameters(chain::SimpleChain, n_initials::Int, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_ode_parameters(n_models::Int, lhs_lb::T, lhs_ub::T, n_initials, rng::AbstractRNG) where T <: Real
    return sample(n_initials, repeat([lhs_lb], n_models), repeat([lhs_ub], n_models), LatinHypercubeSample(rng))
end

function create_progressbar_callback(its, run)
    prog = Progress(its; dt=1, desc="Optimizing run $(run) ", showspeed=true, color=:blue)
    function callback(_, _)
        next!(prog)
        false
    end

    return callback
end

function _optimize(optfunc::OptimizationFunction,
    initial_parameters,
    model::CPeptideCUDEModel, 
    timepoints::AbstractVector{T}, 
    cpeptide_data::AbstractVector{T},
    neural_network_parameters::AbstractVector{T},
    lower_bound,
    upper_bound,
    number_of_iterations_lbfgs::Int
    ) where T <: Real

    optprob = OptimizationProblem(optfunc, initial_parameters, (model, timepoints, cpeptide_data, neural_network_parameters),
    lb = lower_bound, ub = upper_bound)
    optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=number_of_iterations_lbfgs)

    return optsol
end

function _optimize(optfunc::OptimizationFunction, 
    initial_parameters,
    models::AbstractVector{CPeptideCUDEModel}, 
    timepoints::AbstractVector{T}, 
    cpeptide_data::AbstractMatrix{T},
    number_of_iterations_adam::Int,
    number_of_iterations_lbfgs::Int,
    learning_rate_adam::Real
    ) where T <: Real

    # training step 1 (Adam)
    optprob_train = OptimizationProblem(optfunc, initial_parameters, (models, timepoints, cpeptide_data))
    optsol_train = Optimization.solve(optprob_train, Optimisers.Adam(learning_rate_adam), maxiters=number_of_iterations_adam)
    
    # training step 2 (LBFGS)
    optprob_train_2 = OptimizationProblem(optfunc, optsol_train.u, (models, timepoints, cpeptide_data))
    optsol_train_2 = Optimization.solve(optprob_train_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=number_of_iterations_lbfgs)

    return optsol_train_2
end

"""
train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, neural_network_parameters::AbstractVector{T}; 
    initial_beta::Real = -2.0,
    lbfgs_lower_bound::Real = -4.0,
    lbfgs_upper_bound::Real = 1.0,
    lbfgs_iterations::Int = 1000) where T <: Real

Trains a c-peptide model with a conditional neural network for c-peptide production using the conditional UDE framework. This function is used when the neural network parameters are known
and fixed. Only the conditional parameter(s) are optimized.

OTTIMIZZAZIONE DEI SOLI PARAMETRI

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `neural_network_parameters::AbstractVector{T}`: The neural network parameters.
- `initial_beta::Real`: The initial beta value. Default is -2.0.
- `lbfgs_lower_bound::Real`: The lower bound for the L-BFGS optimizer. Default is -4.0.
- `lbfgs_upper_bound::Real`: The upper bound for the L-BFGS optimizer. Default is 1.0.
- `lbfgs_iterations::Int`: The number of iterations for the L-BFGS optimizer. Default is 1,000.

# Returns
- `AbstractVector{OptimizationSolution}`: The optimization solutions.
"""
function train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, 
    neural_network_parameters::AbstractVector{T};
    initial_beta = -2.0,
    lbfgs_lower_bound::V = -4.0,
    lbfgs_upper_bound::V = 1.0,
    lbfgs_iterations::Int = 1000
    ) where T <: Real where V <: Real

    optsols = OptimizationSolution[]
    optfunc = OptimizationFunction(loss, AutoForwardDiff())
    for (i,model) in enumerate(models)
        optsol = _optimize(optfunc, [initial_beta],  model, timepoints, cpeptide_data[i,:], neural_network_parameters, lbfgs_lower_bound, lbfgs_upper_bound, lbfgs_iterations)
        push!(optsols, optsol)
    end

    return optsols
end

"""
train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, rng::AbstractRNG; 
    initial_guesses::Int = 25_000,
    selected_initials::Int = 25,
    lhs_lower_bound::V = -2.0,
    lhs_upper_bound::V = 0.0,
    n_conditional_parameters::Int = 1,
    number_of_iterations_adam::Int = 1000,
    number_of_iterations_lbfgs::Int = 1000,
    learning_rate_adam::Real = 1e-2) where T <: Real where V <: Real

Trains a c-peptide model with a conditional neural network for c-peptide production using the conditional UDE framework. This function is used when the neural network parameters are unknown.
Both the neural network and conditional parameters are optimized.

OTTIMIZZAZIONE DI RETE E PARAMETRI

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `rng::AbstractRNG`: The random number generator.
- `initial_guesses::Int`: The number of initial guesses. Default is 25,000.
- `selected_initials::Int`: The number of selected initials. Default is 25.
- `lhs_lower_bound::V`: The lower bound for the LHS sampling. Default is -2.0.
- `lhs_upper_bound::V`: The upper bound for the LHS sampling. Default is 0.0.
- `n_conditional_parameters::Int`: The number of conditional parameters. Default is 1.
- `number_of_iterations_adam::Int`: The number of iterations for the Adam optimizer. Default is 1,000.
- `number_of_iterations_lbfgs::Int`: The number of iterations for the L-BFGS optimizer. Default is 1,000.
- `learning_rate_adam::Real`: The learning rate for the Adam optimizer. Default is 1e-2.

# Returns
- `AbstractVector{OptimizationSolution}`: The optimization solutions.
"""
function train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractVecOrMat{T}, rng::AbstractRNG; 
    initial_guesses::Int = 25_000,
    selected_initials::Int = 25,
    lhs_lower_bound::V = -2.0,
    lhs_upper_bound::V = 0.0,
    n_conditional_parameters::Int = 1,
    number_of_iterations_adam::Int = 1000,
    number_of_iterations_lbfgs::Int = 1000,
    learning_rate_adam::Real = 1e-2) where T <: Real where V <: Real

    # sample initial parameters
    initial_neural_params = sample_initial_neural_parameters(models[1].chain, initial_guesses, rng)
    initial_ode_params = sample_initial_ode_parameters(length(models), lhs_lower_bound, lhs_upper_bound, initial_guesses, rng)

    initial_parameters = [ComponentArray(
        neural = initial_neural_params[i],
        ode = repeat(initial_ode_params[:,i],1, n_conditional_parameters)
    ) for i in eachindex(initial_neural_params)]

    # preselect initial parameters
    losses_initial = Float64[]
    prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
    for p in initial_parameters
        loss_value = loss(p, (models, timepoints, cpeptide_data))
        push!(losses_initial, loss_value)
        next!(prog)
    end

    println("Initial parameters evaluated. Optimizing for the best $(selected_initials) initial parameters.")
    optsols = OptimizationSolution[]
    optfunc = OptimizationFunction(loss, AutoForwardDiff())
    prog = Progress(selected_initials; dt=1.0, desc="Optimizing...", color=:blue)
    for param_indx in partialsortperm(losses_initial, 1:selected_initials)
        try 
            optsol_train_2 = _optimize(optfunc, initial_parameters[param_indx], 
                                       models, timepoints, cpeptide_data, number_of_iterations_adam, 
                                       number_of_iterations_lbfgs, learning_rate_adam)
            push!(optsols, optsol_train_2)
        catch
            println("Optimization failed... Skipping")
        end
        next!(prog)
    end

    return optsols

end



"""
select_model(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, neural_network_parameters, betas_train)

Selects the best model based on the data and the neural network parameters. This evaluates the neural network parameters on each individual in the 
validation set and selects the model that performs best on each individual. The model that is most frequently selected as the best model is returned.

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `neural_network_parameters`: The neural network parameters.
- `betas_train`: The training data for the conditional parameters.

# Returns
- `Int`: The index of the best model.
"""
function select_model(
    models::AbstractVector{CPeptideCUDEModel},
    timepoints::AbstractVector{T},
    cpeptide_data::AbstractMatrix{T},
    neural_network_parameters,
    betas_train) where T<:Real

    model_objectives = []
    for (betas, p_nn) in zip(betas_train, neural_network_parameters)
        try
            initial = mean(betas)

            optsols_valid = train( # train con rete definita
                models, timepoints, cpeptide_data, p_nn;
                initial_beta = initial, lbfgs_lower_bound=-Inf,
                lbfgs_upper_bound=Inf
            )
            objectives = [sol.objective for sol in optsols_valid]
            push!(model_objectives, objectives)
        catch
            push!(model_objectives, repeat([Inf], length(models)))
        end
    end

    model_objectives = hcat(model_objectives...)

    # find the model that performs best on each individual
    indices = [idx[2] for idx in argmin(model_objectives, dims=2)[:]]

    # find the amount each model occurs in the best performing models
    frequency = countmap(indices)

    # select the model that is most frequently selected as the best model
    best_model = argmax([frequency[i] for i in sort(unique(indices))])

    return best_model
end
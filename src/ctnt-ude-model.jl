using SimpleChains: SimpleChain, TurboDense, static, init_params
using SciMLBase: ODEProblem, OptimizationSolution
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray

using ProgressMeter: Progress, next!

using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches

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
function neural_network_model(depth::Int, width::Int; input_dims::Int = 7)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

"""
    ctnt_cude!(du, u, p, t)

Compartments. 
- u[1]: sarcomere
- u[2]: citosol
- u[3]: plasma
- p: model parameters
"""
function ctnt_cude!(du, u, p, t, chain::SimpleChain)
    # Esempio di termini dinamici (da adattare al modello specifico)
    # Termini base (senza correzione)
    # p.ode = [a, b, Cs0, Cc0]
    # Cs_ctnt = u[1]
    # Cc_ctnt = u[2]
    # Cp_ctnt = u[3]

    β = exp(p.ode[5])

    # a = 10 ^ θ[1]
    # b = 10 ^ θ[2]

    a = p.ode[1]
    b = p.ode[2]

    correction = chain([u[1], t, p.ode[1:4], β], p.neural)[1]

    du[1] = - (u[1] - u[2] + correction)
    du[2] = (u[1] - u[2] + correction) - a*(u[2] - u[3])
    du[3] = a*(u[2] - u[3]) - b*u[3]

end

struct ctntCUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
ctntCUDEModel

# Arguments

# Returns
- `ctntCUDEModel`: A ctnt model with conditional neural network for module flux between sarcomere and cytosol

TODO: Add docs
"""
function ctntCUDEModel(
    ctnt_timepoints::AbstractVector{T},
    chain::SimpleChain,
    θ::AbstractVector{T}
    ) where T <: Real

    # construct the ude function PROVA GIT
    cude!(du, u, p, t) = ctnt_cude!(du, u, p, t, chain)

    tspan = (ctnt_timepoints[1], ctnt_timepoints[end])

    u0 = [θ[3], θ[4], 0];

    # ode = ODEProblem(cude!, u0, tspan, θ)
    ode = ODEProblem(cude!, u0, tspan)

    return ctntCUDEModel(ode, chain)
end


########################## LOSS FUNCTIONS ##########################################


"""
loss(θ, (model, timepoints, ctnt_data))

Sum of squared errors loss function for the ctnt release model.

# Arguments
- `θ`: The parameter vector.
- `model::ctntCUDEModel`: The ctnt release model.
- `timepoints::AbstractVector{T}`: The timepoints.
- `ctnt_data::AbstractVector{T}`: The ctnt release data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, ctnt_data)::Tuple{M, AbstractVector{T}, AbstractVector{T}}) where T <: Real where M <: ctntCUDEModel

    # solve the ODE problem
    sol = Array(solve(model.problem, p=θ, saveat=timepoints))
    # Calculate the mean squared error
    return sum(abs2, sol[3,:] - ctnt_data)
end




function create_progressbar_callback(its, run)
    prog = Progress(its; dt=1, desc="Optimizing run $(run) ", showspeed=true, color=:blue)
    function callback(_, _)
        next!(prog)
        false
    end

    return callback
end


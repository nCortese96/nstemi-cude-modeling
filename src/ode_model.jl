using QuasiMonteCarlo: LatinHypercubeSample, sample
using OrdinaryDiffEq
using StableRNGs, Random, Plots

function ctnt_ode!(du, u, p, t)

    Cs_ctnt = u[1]
    Cc_ctnt = u[2]
    Cp_ctnt = u[3]

    # a = 10 ^ θ[1]
    # b = 10 ^ θ[2]

    a = p[1]
    b = p[2]
    Td = p[3]
    # Cs0 = exp(p[4])
    # Cc0 = exp(p[5])

    Jsc_ctnt = (Cs_ctnt - Cc_ctnt)
    Jcp_ctnt = a*(Cc_ctnt - Cp_ctnt)
    Jpm_ctnt = b*Cp_ctnt

    G_sc = (t^3)/(t^3 + (Td^3));

    du[1] = - Jsc_ctnt * G_sc #Sarcomere
    du[2] = Jsc_ctnt * G_sc - Jcp_ctnt #Cytosol
    du[3] = Jcp_ctnt - Jpm_ctnt #Plasma

end

# parameter_init = [0.005 0.005 30 0.1 0.001];
# lb = [0.001 0.001 20 0.01 0.001];
# ub = [5 5 300 200 400];

function parameter_sample_montecarlo(
    n_samples::Int64,
    tspan::Tuple{T, T}=(0.0, 200.0),
    lb::AbstractVector{T}=[0.001, 0.005, 100.0, 0.5, 0.5],
    ub::AbstractVector{T}=[1, 1, 300.0, 400.0, 400.0],
    rng::AbstractRNG=StableRNG(42)
    ) where T <: Real
    synth = []
    params_samples = sample(n_samples, lb, ub, LatinHypercubeSample(rng))
    # params_samples = [lb .+ (ub .- lb) .* s for s in eachrow(samples)]
    for params in eachcol(params_samples)
        u0 = [params[end], params[end-1], 0]
        # prob = ODEProblem(ctnt_ode!, u0, tspan, log.(params))
        prob = ODEProblem(ctnt_ode!, u0, tspan, params)
        sol = solve(prob, Tsit5(); saveat=0.01)
        pred = [u[3] for u in sol.u]
        # pred = pred .+ 0.01 * randn.(rng, length(pred))
        push!(synth, pred)
    end
    return synth, params_samples
end

synth_data, params_data = parameter_sample_montecarlo(100);
t_vec = 0.0:0.01:200.0
# length(t_vec)
# length(synth_data[2])
sample_n = 9;
println(params_data[:,sample_n])
plot(t_vec, synth_data[sample_n])
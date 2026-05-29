"""
profile_likelihood.jl

Profile likelihood numerical helpers for workflow step 03b.

Sections:
- Objectives: ODE and fixed-NN cUDE log-Gaussian negative log-likelihoods.
- Profile Utilities: endpoint parsing, classification, bounds, and retries.
- Profile Computation: one-patient and target-level PLA execution.
"""

using ComponentArrays: ComponentArray
using DataFrames: DataFrame, nrow
using Distributions: Chisq, quantile
using LikelihoodProfiler
using LineSearches
using Logging
using Optimization
using OptimizationLBFGSB
using OptimizationOptimJL
using OrdinaryDiffEq: Tsit5
using ProgressMeter
using SciMLBase: ODEProblem, successful_retcode, solve, remake

# =============================================================================
# Objectives
# =============================================================================

"""
    profile_likelihood_threshold()

Return the 95% chi-square threshold used by the legacy profile likelihood
workflow.
"""
# Used by: scripts/03b_run_profile_likelihood.jl and src/plotting.jl PLA helpers.
profile_likelihood_threshold() = quantile(Chisq(1), 0.95)

"""
    patient_nll_log_gaussian(θ, data)

Return the legacy concentrated log-Gaussian objective for one patient. The two
methods dispatch between fixed-NN cUDE and mechanistic ODE targets.
"""
# Used by: src/profile_likelihood.jl.
function patient_nll_log_gaussian(
    θ,
    data::Tuple{ctntUDEModel,<:AbstractVector,<:AbstractVector,<:AbstractVector},
)
    model, timepoints, ctnt_data, fixed_nn_params = data
    p = ComponentArray(ode=θ, neural=fixed_nn_params)

    u0 = initial_conditions_from_log_params(θ)
    prob = remake(model.problem; u0=u0, p=p)

    sol = solve(prob, Tsit5(); p=p, saveat=timepoints, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || return Inf

    plasm = sol[3, :]
    resid = log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)
    rss = sum(abs2, resid)
    n = length(resid)

    return n * log(rss / n)
end

function patient_nll_log_gaussian(
    θ,
    data::Tuple{ODEProblem,<:AbstractVector,<:AbstractVector},
)
    problem, timepoints, ctnt_data = data

    u0 = initial_conditions_from_log_params(θ)
    prob = remake(problem; u0=u0, p=θ)

    sol = solve(prob, Tsit5(); p=θ, saveat=timepoints, abstol=1e-8, reltol=1e-6)
    successful_retcode(sol) || return Inf

    plasm = sol[3, :]
    resid = log.(plasm .+ DELTA) .- log.(ctnt_data .+ DELTA)
    rss = sum(abs2, resid)
    n = length(resid)

    return n * log(rss / n)
end

# =============================================================================
# Profile Utilities
# =============================================================================

"""
    empty_profile_likelihood_profiles_df()

Return the canonical long-format profile DataFrame schema.
"""
# Used by: src/profile_likelihood.jl.
function empty_profile_likelihood_profiles_df()
    return DataFrame(
        patient_id=String[],
        patient_idx=Int[],
        param_idx=Int[],
        param_name=String[],
        class_label=String[],
        step_idx=Int[],
        branch_side=String[],
        x_theta=Float64[],
        x_exp=Float64[],
        delta_theta=Float64[],
        objective=Float64[],
        plr=Float64[],
    )
end

"""
    empty_profile_likelihood_summary_df()

Return the canonical per-parameter profile summary DataFrame schema.
"""
# Used by: src/profile_likelihood.jl.
function empty_profile_likelihood_summary_df()
    return DataFrame(
        patient_id=String[],
        patient_idx=Int[],
        param_idx=Int[],
        param_name=String[],
        class_label=String[],
        theta_hat=Float64[],
        theta_hat_exp=Float64[],
        threshold=Float64[],
        left_endpoint=Float64[],
        right_endpoint=Float64[],
        retcode_left=String[],
        retcode_right=String[],
    )
end

"""
    clamp_strictly_inside(theta, lb, ub; eps_bound)

Clamp log-scale parameters strictly inside optimizer bounds.
"""
# Used by: src/profile_likelihood.jl.
function clamp_strictly_inside(theta, lb, ub; eps_bound::Real)
    θ = Float64.(copy(theta))
    for k in eachindex(θ)
        θ[k] = clamp(θ[k], lb[k] + eps_bound, ub[k] - eps_bound)
    end
    return θ
end

"""
    extract_endpoint_value(x)

Extract a numeric endpoint from LikelihoodProfiler endpoint objects.
"""
# Used by: src/profile_likelihood.jl.
function extract_endpoint_value(x)
    x isa Number && return Float64(x)
    if x !== nothing && hasproperty(x, :value) && getproperty(x, :value) isa Number
        return Float64(getproperty(x, :value))
    end
    return NaN
end

"""
    classify_profile(ep, rc)

Classify one parameter profile from LikelihoodProfiler endpoints and retcodes.
"""
# Used by: src/profile_likelihood.jl.
function classify_profile(ep, rc)
    left_raw = hasproperty(ep, :left) ? getproperty(ep, :left) : nothing
    right_raw = hasproperty(ep, :right) ? getproperty(ep, :right) : nothing

    left_ep = extract_endpoint_value(left_raw)
    right_ep = extract_endpoint_value(right_raw)

    rc_left = lowercase(string(hasproperty(rc, :left) ? getproperty(rc, :left) : rc))
    rc_right = lowercase(string(hasproperty(rc, :right) ? getproperty(rc, :right) : rc))

    left_identifiable = rc_left == "identifiable"
    right_identifiable = rc_right == "identifiable"

    class_label = if left_identifiable && right_identifiable
        "Identifiable"
    elseif left_identifiable || right_identifiable
        "Practically identifiable"
    else
        "Unidentifiable"
    end

    return class_label, left_ep, right_ep, rc_left, rc_right
end

"""
    branch_failed_flag(rc_side)

Return whether a profiler branch retcode indicates failure or max-iteration
termination.
"""
# Used by: src/profile_likelihood.jl.
function branch_failed_flag(rc_side)
    s = lowercase(string(rc_side))
    return occursin("max", s) || occursin("fail", s) || occursin("error", s)
end

"""
    make_profile_window(theta_hat, lb, ub, j, span; eps_bound)

Build a single-parameter bounded profile window around `theta_hat[j]`.
"""
# Used by: src/profile_likelihood.jl.
function make_profile_window(theta_hat, lb, ub, j::Integer, span::Real; eps_bound::Real)
    plb = copy(lb)
    pub = copy(ub)

    lo = max(lb[j] + eps_bound, theta_hat[j] - span)
    hi = min(ub[j] - eps_bound, theta_hat[j] + span)

    if !(lo < theta_hat[j] < hi)
        lo = lb[j] + eps_bound
        hi = ub[j] - eps_bound
    end

    plb[j] = lo
    pub[j] = hi
    return plb, pub
end

"""
    profile_hits_window_edge(pr, j, plb, pub; tol_frac=0.02)

Return true when a profile reaches the temporary profile window boundary.
"""
# Used by: src/profile_likelihood.jl.
function profile_hits_window_edge(pr, j::Integer, plb, pub; tol_frac::Real=0.02)
    df = DataFrame(pr)
    x = Float64.(df[!, Symbol("x$j")])
    x = x[isfinite.(x)]

    isempty(x) && return false

    tol = tol_frac * max(pub[j] - plb[j], 1e-8)
    return (minimum(x) <= plb[j] + tol) || (maximum(x) >= pub[j] - tol)
end

"""
    profile_score(pr)

Return the legacy tuple score used to choose between retry profiles.
"""
# Used by: src/profile_likelihood.jl.
function profile_score(pr)
    ep = endpoints(pr)
    rc = retcodes(pr)

    left_ep = extract_endpoint_value(hasproperty(ep, :left) ? getproperty(ep, :left) : nothing)
    right_ep = extract_endpoint_value(hasproperty(ep, :right) ? getproperty(ep, :right) : nothing)

    rc_left = hasproperty(rc, :left) ? getproperty(rc, :left) : rc
    rc_right = hasproperty(rc, :right) ? getproperty(rc, :right) : rc

    left_failed = branch_failed_flag(rc_left)
    right_failed = branch_failed_flag(rc_right)

    nfinite = Int(isfinite(left_ep)) + Int(isfinite(right_ep))
    nfailed = Int(left_failed) + Int(right_failed)
    npts = nrow(DataFrame(pr))

    return (nfinite, -nfailed, npts)
end

"""
    choose_better_profile(pr_old, pr_new)

Return the better retry profile according to the legacy tuple score.
"""
# Used by: src/profile_likelihood.jl.
choose_better_profile(pr_old, pr_new) =
    profile_score(pr_new) > profile_score(pr_old) ? pr_new : pr_old

"""
    profile_needs_retry(pr, j, plb, pub)

Return true when a profile should be retried with a larger temporary window.
"""
# Used by: src/profile_likelihood.jl.
function profile_needs_retry(pr, j::Integer, plb, pub)
    rc = retcodes(pr)

    rc_left = hasproperty(rc, :left) ? getproperty(rc, :left) : rc
    rc_right = hasproperty(rc, :right) ? getproperty(rc, :right) : rc

    left_failed = branch_failed_flag(rc_left)
    right_failed = branch_failed_flag(rc_right)

    return left_failed || right_failed || profile_hits_window_edge(pr, j, plb, pub)
end

"""
    solve_profile(optprob, theta_hat, plb, pub; step_scale, maxiters)

Run LikelihoodProfiler jointly for all five parameters.
"""
# Used by: src/profile_likelihood.jl.
function solve_profile(optprob, theta_hat, plb, pub; step_scale::Real, maxiters::Integer)
    plprob = ProfileLikelihoodProblem(
        optprob,
        theta_hat;
        idxs=1:5,
        profile_lower=plb,
        profile_upper=pub,
        conf_level=0.95,
        df=1,
    )

    profiler = OptimizationProfiler(
        optimizer=LBFGSB(),
        optimizer_opts=(maxiters=maxiters,),
        stepper=FixedStep(
            initial_step=(pars, idx) -> step_scale * max(abs(pars[idx]), 1e-3),
        ),
    )

    return solve(plprob, profiler; parallel_type=:threads, verbose=false)
end

"""
    solve_profile_for_param(optprob, theta_hat, j, plb, pub; step_scale, maxiters)

Run LikelihoodProfiler for one parameter. This path is kept for the legacy
`separate=true` mode.
"""
# Used by: src/profile_likelihood.jl.
function solve_profile_for_param(optprob, theta_hat, j::Integer, plb, pub; step_scale::Real, maxiters::Integer)
    plprob_j = ProfileLikelihoodProblem(
        optprob,
        theta_hat;
        idxs=[j],
        profile_lower=[plb[j]],
        profile_upper=[pub[j]],
        conf_level=0.95,
        df=1,
    )

    profiler_j = OptimizationProfiler(
        optimizer=LBFGSB(),
        optimizer_opts=(maxiters=maxiters,),
        stepper=FixedStep(
            initial_step=(pars, idx) -> step_scale * max(abs(pars[idx]), 1e-3),
        ),
    )

    sol_j = solve(plprob_j, profiler_j; parallel_type=:threads, verbose=false)
    return sol_j[1]
end

# =============================================================================
# Profile Computation
# =============================================================================

"""
    profile_likelihood_data_tuple(spec, patient, theta_start; chain, neural_params)

Build the fixed model data tuple consumed by the PLA objective.
"""
# Used by: src/profile_likelihood.jl.
function profile_likelihood_data_tuple(spec, patient::PatientData, theta_start; chain=nothing, neural_params=nothing)
    if spec.model_kind == :cude
        chain !== nothing || error("cUDE PLA requires a neural network chain.")
        neural_params !== nothing || error("cUDE PLA requires fixed neural parameters.")
        θ_model = ComponentArray(ode=theta_start, neural=neural_params)
        model = ctntCUDEModel(θ_model, chain, patient.timepoints)
        return (model, patient.timepoints, patient.ctnt_data, neural_params)
    elseif spec.model_kind == :ode
        u0_init = initial_conditions_from_log_params(theta_start)
        tspan = (0.0, patient.timepoints[end] + 10.0)
        problem = ODEProblem(troponin_ode!, u0_init, tspan)
        return (problem, patient.timepoints, patient.ctnt_data)
    end

    error("Unsupported PLA model kind: $(spec.model_kind)")
end

"""
    compute_profile_likelihood_patient(patient, patient_idx, theta_start, spec, settings; ...)

Compute PLA profiles and summary rows for one patient.
"""
# Used by: src/profile_likelihood.jl (compute_profile_likelihood_target).
function compute_profile_likelihood_patient(
    patient::PatientData,
    patient_idx::Integer,
    theta_start,
    spec,
    settings;
    chain=nothing,
    neural_params=nothing,
)
    lower = spec.lower
    upper = spec.upper
    threshold = profile_likelihood_threshold()
    theta0 = clamp_strictly_inside(theta_start, lower, upper; eps_bound=settings.eps_bound)
    data_tpl = profile_likelihood_data_tuple(spec, patient, theta0; chain=chain, neural_params=neural_params)

    optf = OptimizationFunction(
        (θ, data) -> patient_nll_log_gaussian(θ, data),
        AutoFiniteDiff(),
    )
    optprob = OptimizationProblem(optf, theta0, data_tpl, lb=lower, ub=upper)

    opt_sol = Optimization.solve(
        optprob,
        LBFGS(linesearch=LineSearches.BackTracking()),
        maxiters=settings.refit_maxiters,
    )

    if !(isfinite(opt_sol.objective) && all(isfinite, opt_sol.u))
        @warn "Non-finite PLA refit; falling back to clamped start." patient=patient.id target=spec.target_name
        optpars = clamp_strictly_inside(theta0, lower, upper; eps_bound=settings.eps_bound)
        objective_hat = patient_nll_log_gaussian(optpars, data_tpl)
    else
        optpars = clamp_strictly_inside(opt_sol.u, lower, upper; eps_bound=settings.eps_bound)
        objective_hat = opt_sol.objective
    end

    profiles = Vector{Any}(undef, settings.n_params)
    if settings.separate
        for j in 1:settings.n_params
            span = settings.span
            best_profile = nothing

            for _ in 1:settings.expand_tries
                plb, pub = make_profile_window(optpars, lower, upper, j, span; eps_bound=settings.eps_bound)
                candidate = solve_profile_for_param(
                    optprob,
                    optpars,
                    j,
                    plb,
                    pub;
                    step_scale=settings.step_scale,
                    maxiters=settings.profile_maxiters,
                )

                best_profile = isnothing(best_profile) ? candidate : choose_better_profile(best_profile, candidate)
                profile_needs_retry(candidate, j, plb, pub) || break

                full_window =
                    isapprox(plb[j], lower[j] + settings.eps_bound; atol=1e-10, rtol=0.0) &&
                    isapprox(pub[j], upper[j] - settings.eps_bound; atol=1e-10, rtol=0.0)

                full_window && break
                span = min((upper[j] - lower[j]) / 2, span * settings.expand_factor)
            end

            profiles[j] = best_profile
        end
    else
        profiles = solve_profile(
            optprob,
            optpars,
            lower,
            upper;
            step_scale=settings.step_scale,
            maxiters=settings.profile_maxiters,
        )
    end

    patient_profiles = empty_profile_likelihood_profiles_df()
    patient_summary = empty_profile_likelihood_summary_df()
    patient_id = String(patient.id)

    for j in 1:settings.n_params
        ep = endpoints(profiles[j])
        rc = retcodes(profiles[j])
        class_label, left_ep, right_ep, rc_left, rc_right = classify_profile(ep, rc)

        profile_df = DataFrame(profiles[j])
        x_theta = Float64.(profile_df[!, Symbol("x$j")])
        y_obj = Float64.(profile_df[!, :objective])

        keep = isfinite.(x_theta) .& isfinite.(y_obj)
        x_theta = x_theta[keep]
        y_obj = y_obj[keep]

        x_exp = exp.(x_theta)
        delta_theta = x_theta .- optpars[j]
        y_plr = y_obj .- objective_hat

        branch_side = map(delta_theta) do dθ
            if isapprox(dθ, 0.0; atol=1e-10, rtol=0.0)
                "center"
            elseif dθ < 0
                "left"
            else
                "right"
            end
        end

        order = sortperm(delta_theta)
        x_theta = x_theta[order]
        x_exp = x_exp[order]
        delta_theta = delta_theta[order]
        y_obj = y_obj[order]
        y_plr = y_plr[order]
        branch_side = branch_side[order]

        for k in eachindex(x_theta)
            push!(patient_profiles, (
                patient_id=patient_id,
                patient_idx=patient_idx,
                param_idx=j,
                param_name=spec.param_names[j],
                class_label=class_label,
                step_idx=k,
                branch_side=branch_side[k],
                x_theta=x_theta[k],
                x_exp=x_exp[k],
                delta_theta=delta_theta[k],
                objective=y_obj[k],
                plr=y_plr[k],
            ))
        end

        push!(patient_summary, (
            patient_id=patient_id,
            patient_idx=patient_idx,
            param_idx=j,
            param_name=spec.param_names[j],
            class_label=class_label,
            theta_hat=optpars[j],
            theta_hat_exp=exp(optpars[j]),
            threshold=threshold,
            left_endpoint=left_ep,
            right_endpoint=right_ep,
            retcode_left=rc_left,
            retcode_right=rc_right,
        ))
    end

    return (
        profiles=patient_profiles,
        summary=patient_summary,
        theta_hat=optpars,
        objective=objective_hat,
    )
end

"""
    compute_profile_likelihood_target(patients, reshaped_params, spec, settings, paths; ...)

Compute PLA for all patients in one target and write canonical CSV artifacts.
"""
# Used by: scripts/03b_run_profile_likelihood.jl.
function compute_profile_likelihood_target(
    patients,
    reshaped_params,
    spec,
    settings,
    paths;
    chain=nothing,
    neural_params=nothing,
)
    size(reshaped_params, 1) == length(patients) ||
        error("PLA parameter row count $(size(reshaped_params, 1)) does not match patient count $(length(patients)).")

    profiles_long = empty_profile_likelihood_profiles_df()
    summary_df = empty_profile_likelihood_summary_df()

    progress = settings.progress_bars ? Progress(length(patients); desc="PLA $(spec.target_name)", showspeed=true) : nothing

    for i in eachindex(patients)
        patient = patients[i]
        patient_tag = "patient_$(lpad(string(i), 4, '0'))_$(safe_patient_id(patient.id))"

        result = compute_profile_likelihood_patient(
            patient,
            i,
            reshaped_params[i, :],
            spec,
            settings;
            chain=chain,
            neural_params=neural_params,
        )

        append!(profiles_long, result.profiles)
        append!(summary_df, result.summary)
        save_profile_likelihood_patient_csvs(paths, patient_tag, result.profiles, result.summary)

        @info "Completed PLA patient $(i)/$(length(patients))." target=spec.target_name patient=patient.id
        progress !== nothing && next!(progress)
    end

    progress !== nothing && finish!(progress)
    save_profile_likelihood_global_csvs(paths, profiles_long, summary_df)

    return (
        profiles_long=profiles_long,
        summary=summary_df,
        paths=paths,
    )
end

"""
fitting.jl

Optimization, patient-level fitting, and training helpers.

Sections:
- Multi-Start Optimization: reproducible bounded optimization from multiple starts.
- Initialization Utilities: reusable neural and ODE parameter initialization.
- Prediction Helpers: reusable ODE prediction utilities.
- cUDE Training: width-level cUDE training helpers for workflow step 02a.
- cUDE Evaluation: patient-level cUDE evaluation helpers for workflow step 02b.
- ODE Td-Sigmoid Fitting: step 01 fitting and output helpers.
"""

using Base.Threads: @threads
using ComponentArrays: ComponentArray
using LineSearches
using Logging
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using OrdinaryDiffEq: Tsit5
using ProgressMeter
using QuasiMonteCarlo: LatinHypercubeSample, sample
using Random: AbstractRNG
using SciMLBase: ODEProblem, successful_retcode, solve, remake
using SimpleChains: SimpleChain, init_params
using StableRNGs: StableRNG
import Base.Threads

# =============================================================================
# Multi-Start Optimization
# =============================================================================

"""
    run_multistart(loss, N; lower, upper, optimizer, rng, maxiters, maxtime, prescreen, topk, show_progress)

Run Latin-hypercube multi-start optimization and return `(best_solution,
all_results)`.
"""
# Used by: src/fitting.jl (fit_ode_patient). Planned use: scripts/02b_evaluate_cude_nn.jl, scripts/02d_evaluate_cude_nn_external_test.jl, scripts/05_run_systematic_truncation.jl, scripts/07_evaluate_symbolic_formula.jl.
function run_multistart(
    loss::Function,
    N::Int;
    lower::Union{Nothing,Vector{<:Real}}=nothing,
    upper::Union{Nothing,Vector{<:Real}}=nothing,
    optimizer=Fminbox(LBFGS()),
    verbose::Bool=true,
    callback::Function=(x, f) -> nothing,
    rng::AbstractRNG=StableRNG(42),
    maxiters::Int=1000,
    maxtime::Float64=80.0,
    prescreen::Bool=false,
    topk::Int=8,
    show_progress::Bool=true
)
    lhs_starts = sample(N, lower, upper, LatinHypercubeSample(rng))
    starts = [Vector(lhs_starts[:, i]) for i in 1:N]

    selected_idx = collect(1:N)
    prescreen_losses = fill(NaN, N)

    if prescreen
        prescreen_losses = [loss(s) for s in starts]
        finite_idx = findall(isfinite, prescreen_losses)
        isempty(finite_idx) && error("Prescreen failed: no finite starting point found.")

        keep_n = min(topk, length(finite_idx))
        ord = sortperm(prescreen_losses[finite_idx])
        selected_idx = finite_idx[ord[1:keep_n]]
        starts = starts[selected_idx]
    end

    results = Vector{Any}(undef, length(starts))
    optf = OptimizationFunction((x, _) -> loss(x), AutoForwardDiff())

    done = Threads.Atomic{Int}(0)
    best_loss_atomic = Threads.Atomic{Float64}(Inf)
    progress = show_progress ? Progress(length(starts); desc="MultiStart", dt=0.1, showspeed=true) : nothing

    monitor = if show_progress
        Threads.@spawn begin
            last = 0
            while true
                d = done[]
                if d > last
                    ProgressMeter.update!(progress, d; showvalues=[(:best_loss, best_loss_atomic[])])
                    last = d
                end
                d >= length(starts) && break
                sleep(0.1)
            end
        end
    else
        nothing
    end

    best_lock = ReentrantLock()
    best_loss = Inf
    best_sol = nothing

    @threads for i in eachindex(starts)
        p0 = starts[i]

        optprob = isnothing(lower) ?
                  OptimizationProblem(optf, p0) :
                  OptimizationProblem(optf, p0; lb=lower, ub=upper)

        result = solve(optprob, optimizer; maxiters=maxiters, maxtime=maxtime)
        results[i] = result

        if result !== nothing
            lock(best_lock) do
                if result.minimum < best_loss
                    best_loss = result.minimum
                    best_sol = result
                    best_loss_atomic[] = best_loss
                    callback(result.u, result.minimum)
                end
            end
        end

        done[] += 1
    end

    if show_progress
        wait(monitor)
        finish!(progress)
    end

    if verbose
        nfail = count(r -> r === nothing, results)
        println("Finished MultiStart: best_loss=$(best_loss), failed=$(nfail)/$(length(starts))")
    end

    best_sol === nothing && error("MultiStart failed: no valid solution found among $(length(starts)) starts.")
    return best_sol, results
end

# =============================================================================
# Initialization Utilities
# =============================================================================

"""
    sample_initial_neural_parameters(n_initials, chain, rng)

Return reproducible neural-network parameter initializations for a SimpleChain.
"""
# Planned use: scripts/02a_run_cude_training.jl.
function sample_initial_neural_parameters(n_initials::Int, chain::SimpleChain, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

"""
    sample_initial_parameters(n_patients, n_initials, lhs_lb, lhs_ub, rng)

Return Latin-hypercube ODE parameter initializations for all patients.
"""
# Planned use: scripts/02a_run_cude_training.jl.
function sample_initial_parameters(n_patients::Int, n_initials::Int, lhs_lb::AbstractVector{T}, lhs_ub::AbstractVector{T}, rng::AbstractRNG) where T<:Real
    return sample(n_initials, repeat(lhs_lb, n_patients), repeat(lhs_ub, n_patients), LatinHypercubeSample(rng))
end

# =============================================================================
# Prediction Helpers
# =============================================================================

"""
    predict_patient_curve(problem, theta; saveat=1.0, abstol=1e-8, reltol=1e-6)

Solve a patient ODE problem after remaking its initial condition and parameters.
"""
# Planned use: scripts/05_run_systematic_truncation.jl and diagnostic scripts.
function predict_patient_curve(problem::ODEProblem, theta; saveat=1.0, abstol=1e-8, reltol=1e-6)
    u0 = initial_conditions_from_log_params(theta)
    prob = remake(problem; u0=u0, p=theta)
    sol = solve(prob, Tsit5(); p=theta, saveat=saveat, abstol=abstol, reltol=reltol)
    successful_retcode(sol) || error("Prediction solve failed with retcode=$(sol.retcode)")
    return sol
end

"""
    predict_cude_patient_curve(patient, log_params, chain, nn_params; saveat=1.0)

Reconstruct a cUDE patient trajectory from saved log-scale ODE parameters and
fixed neural-network parameters without running a new fit.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl (`plots` mode).
function predict_cude_patient_curve(
    patient::PatientData,
    log_params::AbstractVector,
    chain::SimpleChain,
    nn_params;
    saveat=1.0,
    abstol::Real=1e-8,
    reltol::Real=1e-6,
)
    model = ctntCUDEModel(log_params, chain, patient.timepoints)
    component_params = ComponentArray(ode=Vector(log_params), neural=nn_params)
    u0 = initial_conditions_from_log_params(log_params)
    prob = remake(model.problem; u0=u0, p=component_params)
    sol = solve(prob, Tsit5(); p=component_params, saveat=saveat, abstol=abstol, reltol=reltol)
    successful_retcode(sol) || error("cUDE prediction solve failed with retcode=$(sol.retcode)")
    return sol
end

# =============================================================================
# cUDE Training
# =============================================================================

"""
    make_optimization_progress_callback!(pbar, losses; offset=0, every_iters=10, every_secs=1.0)

Build an Optimization.jl callback that records every reported loss and updates
an optional progress bar at controlled intervals.
"""
# Used by: src/fitting.jl (train_cude_initialization).
function make_optimization_progress_callback!(pbar, losses; offset::Int=0, every_iters::Int=10, every_secs::Real=1.0)
    last_t = Ref(time())
    last_k = Ref(0)

    return (state, loss) -> begin
        push!(losses, loss)

        k = offset + state.iter
        if pbar !== nothing && ((k - last_k[] >= every_iters) || (time() - last_t[] > every_secs))
            ProgressMeter.update!(pbar, k; showvalues=() -> [(:iter, k), (:loss, loss)])
            last_k[] = k
            last_t[] = time()
        end

        return false
    end
end

"""
    cude_local_training_models(training_dataset, chain, theta, n_params)

Build reusable cUDE ODE problem containers for the training cohort using
patient-specific time spans and the shared neural chain.
"""
# Used by: src/fitting.jl (select_cude_initial_candidates, train_cude_width).
function cude_local_training_models(training_dataset::AbstractVector{PatientData}, chain::SimpleChain, theta, n_params::Integer)
    return [
        ctntCUDEModel(
            theta.ode[n_params * (j - 1) + 1:n_params * j],
            chain,
            training_dataset[j].timepoints,
        )
        for j in eachindex(training_dataset)
    ]
end

"""
    generate_cude_initial_candidates(training_dataset, chain, settings; rng)

Generate neural and patient-level ODE initial candidates with deterministic
sampling order.
"""
# Used by: src/fitting.jl (train_cude_width).
function generate_cude_initial_candidates(training_dataset::AbstractVector{PatientData}, chain::SimpleChain, settings; rng::AbstractRNG)
    initial_nn = sample_initial_neural_parameters(settings.initial_guesses, chain, rng)
    initial_ode = sample_initial_parameters(length(training_dataset), settings.initial_guesses, settings.lower, settings.upper, rng)
    n_conditional = hasproperty(settings, :n_conditional) ? settings.n_conditional : 1

    return [
        ComponentArray(
            neural=initial_nn[i],
            ode=repeat(initial_ode[:, i], 1, n_conditional),
        )
        for i in eachindex(initial_nn)
    ]
end

"""
    select_cude_initial_candidates(initial_parameters, training_dataset, chain, settings; width, show_progress)

Evaluate all initial candidates with the cohort training loss and return the
best starts selected by `partialsortperm`.
"""
# Used by: src/fitting.jl (train_cude_width).
function select_cude_initial_candidates(
    initial_parameters,
    training_dataset::AbstractVector{PatientData},
    chain::SimpleChain,
    settings;
    width::Integer,
    show_progress::Bool=true,
)
    isempty(initial_parameters) && error("No cUDE initial parameters were generated.")
    settings.selected_initials <= length(initial_parameters) ||
        error("selected_initials=$(settings.selected_initials) exceeds initial_guesses=$(length(initial_parameters)).")

    local_models = cude_local_training_models(training_dataset, chain, first(initial_parameters), settings.n_params)
    losses_initial = Vector{Float64}(undef, length(initial_parameters))

    progress = show_progress ?
               Progress(length(initial_parameters); dt=1, desc="Evaluating initial guesses width $(width)", showspeed=true, color=:firebrick) :
               nothing

    for k in eachindex(initial_parameters)
        losses_initial[k] = par_training_loss(
            initial_parameters[k],
            (local_models, training_dataset);
            n_params=settings.n_params,
            lb_param=settings.lower,
            ub_param=settings.upper,
            κ_bounds=settings.kappa_bounds,
            λ_back=settings.lambda_back,
        )

        progress !== nothing && next!(progress)
    end

    progress !== nothing && finish!(progress)

    selected_indices = partialsortperm(losses_initial, 1:settings.selected_initials)
    out_params = initial_parameters[selected_indices]

    return (
        out_params=out_params,
        losses_initial=losses_initial,
        selected_indices=selected_indices,
        selected_losses=losses_initial[selected_indices],
        local_models=local_models,
    )
end

"""
    train_cude_initialization(theta_init, local_models, training_dataset, settings; model_index, show_progress)

Train one selected cUDE initialization through the configured ADAM phase followed
by the configured LBFGS phase.
"""
# Used by: src/fitting.jl (train_cude_width).
function train_cude_initialization(
    theta_init,
    local_models,
    training_dataset::AbstractVector{PatientData},
    settings;
    model_index::Integer,
    show_progress::Bool=true,
)
    losses_this = Float64[]

    optfunc = OptimizationFunction(
        (p, data) -> par_training_loss(
            p,
            data;
            n_params=settings.n_params,
            lb_param=settings.lower,
            ub_param=settings.upper,
            κ_bounds=settings.kappa_bounds,
            λ_back=settings.lambda_back,
        ),
        AutoForwardDiff(),
    )

    optprob = OptimizationProblem(optfunc, theta_init, (local_models, training_dataset))

    adam_bar = show_progress ?
               Progress(settings.adam_maxiters; dt=1, desc="ADAM phase theta $(model_index)", showspeed=true, color=:firebrick) :
               nothing
    cb_adam = make_optimization_progress_callback!(adam_bar, losses_this; offset=0, every_iters=10, every_secs=10.0)

    opt_result1 = solve(
        optprob,
        Optimisers.Adam(settings.adam_eta);
        maxiters=settings.adam_maxiters,
        callback=cb_adam,
    )

    adam_bar !== nothing && finish!(adam_bar)

    optprob2 = OptimizationProblem(optfunc, opt_result1.u, (local_models, training_dataset))

    lbfgs_bar = show_progress ?
                Progress(settings.lbfgs_maxiters; dt=1, desc="LBFGS phase theta $(model_index)", showspeed=true, color=:firebrick) :
                nothing
    cb_lbfgs = make_optimization_progress_callback!(lbfgs_bar, losses_this; offset=0, every_iters=10, every_secs=10.0)

    opt_result2 = solve(
        optprob2,
        LBFGS(linesearch=LineSearches.BackTracking());
        maxiters=settings.lbfgs_maxiters,
        g_abstol=settings.lbfgs_tolerances.g,
        f_abstol=settings.lbfgs_tolerances.f,
        x_abstol=settings.lbfgs_tolerances.x,
        callback=cb_lbfgs,
    )

    lbfgs_bar !== nothing && finish!(lbfgs_bar)

    return (
        solution=opt_result2,
        losses=losses_this,
        adam_retcode=opt_result1.retcode,
        lbfgs_retcode=opt_result2.retcode,
        final_loss=opt_result2.objective,
    )
end

"""
    train_cude_width(training_dataset, settings; width, initial_parameters, initial_parameters_source)

Run the complete step 02a cUDE training pipeline for one neural-network width.
When `initial_parameters` is provided, skip the initial candidate generation
and screening phase and train directly from those selected starts.
"""
# Used by: scripts/02a_run_cude_training.jl.
function train_cude_width(
    training_dataset::AbstractVector{PatientData},
    settings;
    width::Integer,
    initial_parameters=nothing,
    initial_parameters_source=nothing,
)
    chain = neural_network_model(settings.nn_depth, width; input_dims=settings.input_dim)

    if initial_parameters === nothing
        rng = StableRNG(settings.rng_seed)
        generated_parameters = generate_cude_initial_candidates(training_dataset, chain, settings; rng=rng)
        selection = select_cude_initial_candidates(
            generated_parameters,
            training_dataset,
            chain,
            settings;
            width=width,
            show_progress=settings.progress_bars,
        )
        initial_source = :generated
    else
        out_params = collect(initial_parameters)
        isempty(out_params) && error("Existing cUDE initial-parameter file contains no out_params.")

        local_models = cude_local_training_models(training_dataset, chain, first(out_params), settings.n_params)
        selected_losses = [
            par_training_loss(
                theta,
                (local_models, training_dataset);
                n_params=settings.n_params,
                lb_param=settings.lower,
                ub_param=settings.upper,
                κ_bounds=settings.kappa_bounds,
                λ_back=settings.lambda_back,
            )
            for theta in out_params
        ]

        selection = (
            out_params=out_params,
            losses_initial=selected_losses,
            selected_indices=nothing,
            selected_losses=selected_losses,
            local_models=local_models,
        )
        initial_source = :loaded
    end

    n_start = length(selection.out_params)
    optsols = Vector{Any}(undef, n_start)
    losses_per_model = Vector{Vector{Float64}}(undef, n_start)
    adam_retcodes = Vector{Any}(undef, n_start)
    lbfgs_retcodes = Vector{Any}(undef, n_start)
    final_losses = Vector{Float64}(undef, n_start)

    for (i, theta_init) in enumerate(selection.out_params)
        @info "Training cUDE width $(width), selected initialization $(i)/$(n_start)."
        trained = train_cude_initialization(
            theta_init,
            selection.local_models,
            training_dataset,
            settings;
            model_index=i,
            show_progress=settings.progress_bars,
        )

        optsols[i] = trained.solution
        losses_per_model[i] = trained.losses
        adam_retcodes[i] = trained.adam_retcode
        lbfgs_retcodes[i] = trained.lbfgs_retcode
        final_losses[i] = trained.final_loss

        @info "Completed cUDE width $(width), initialization $(i)/$(n_start): retcode=$(trained.lbfgs_retcode), final_loss=$(trained.final_loss)"
    end

    neural_network_parameters = [optsol.u.neural[:] for optsol in optsols]
    ode_params = [optsol.u.ode[:] for optsol in optsols]

    return (
        width=width,
        chain=chain,
        initial_source=initial_source,
        initial_parameters_source=initial_parameters_source,
        initial_losses=selection.losses_initial,
        selected_indices=selection.selected_indices,
        selected_losses=selection.selected_losses,
        out_params=selection.out_params,
        optsols=optsols,
        losses_per_model=losses_per_model,
        neural_network_parameters=neural_network_parameters,
        ode_params=ode_params,
        adam_retcodes=adam_retcodes,
        lbfgs_retcodes=lbfgs_retcodes,
        final_losses=final_losses,
    )
end

# =============================================================================
# cUDE Evaluation
# =============================================================================

"""
    fit_cude_patient(patient, chain, nn_params, pguess, lower, upper, settings; rng)

Fit one patient's cUDE ODE parameters with the fixed neural correction from a
trained candidate model.
"""
# Used by: src/fitting.jl (evaluate_cude_model).
function fit_cude_patient(
    patient::PatientData,
    chain::SimpleChain,
    nn_params,
    pguess::AbstractVector,
    lower::AbstractVector,
    upper::AbstractVector,
    settings;
    rng::AbstractRNG,
)
    model = ctntCUDEModel(pguess, chain, patient.timepoints)
    patient_data = (model, patient.timepoints, patient.ctnt_data, nn_params)
    loss_fun = θ -> patient_loss(θ, patient_data; λ_back=settings.lambda_back)
    use_multistart = settings.n_multistart > 0

    if use_multistart
        settings.bounds ||
            error("cUDE multi-start evaluation requires bounds=true because starts are sampled inside [lower, upper].")

        best_result, _ = run_multistart(
            loss_fun,
            settings.n_multistart;
            lower=lower,
            upper=upper,
            rng=rng,
            verbose=false,
            maxiters=settings.maxiters,
            maxtime=Float64(settings.maxtime),
            prescreen=settings.prescreen,
            topk=settings.topk,
            show_progress=settings.progress_bars,
        )

        best_ode_params = Vector(best_result.u)
        best_objective = best_result.minimum
    else
        optfunc = OptimizationFunction(
            (p, data) -> patient_loss(p, data; λ_back=settings.lambda_back),
            AutoForwardDiff(),
        )

        optprob = settings.bounds ?
                  OptimizationProblem(optfunc, pguess, patient_data; lb=lower, ub=upper) :
                  OptimizationProblem(optfunc, pguess, patient_data)

        optsol = solve(
            optprob,
            LBFGS(linesearch=LineSearches.BackTracking());
            maxiters=settings.maxiters,
        )

        best_ode_params = Vector(optsol.u)
        best_objective = optsol.objective
    end

    p_opt = ComponentArray(ode=best_ode_params, neural=nn_params)
    u0_new = initial_conditions_from_log_params(p_opt.ode)
    prob = remake(model.problem; u0=u0_new, p=p_opt)
    opt_model = ctntUDEModel(prob, chain)

    pred = solve(prob, Tsit5(); p=p_opt, saveat=patient.timepoints)
    successful_retcode(pred) || error("Prediction solve failed with retcode=$(pred.retcode)")

    sol = solve(prob, Tsit5(); p=p_opt, saveat=1)
    successful_retcode(sol) || error("Full trajectory solve failed with retcode=$(sol.retcode)")

    return (
        patient=patient.id,
        smape=smape(pred[3, :], patient.ctnt_data),
        rmsle=rmsle(patient.ctnt_data, pred[3, :]),
        loss=best_objective,
        params=best_ode_params,
        component_params=p_opt,
        model=opt_model,
        pred=pred,
        sol=sol,
    )
end

"""
    evaluate_cude_model(patients, chain, nn_params, pguess, settings; dataset_name, width, model_idx)

Evaluate one trained cUDE candidate on an ordered patient cohort.
"""
# Used by: scripts/02b_evaluate_cude_nn.jl, scripts/02d_evaluate_cude_nn_external_test.jl.
function evaluate_cude_model(
    patients::AbstractVector{PatientData},
    chain::SimpleChain,
    nn_params,
    pguess::AbstractVector,
    settings;
    dataset_name::AbstractString,
    width::Integer,
    model_idx::Integer,
)
    rng = StableRNG(settings.rng_seed)
    results = Any[]
    successful_patients = PatientData[]
    successful_indices = Int[]
    validation_params = Vector{Vector{Float64}}()

    progress = settings.progress_bars ?
               Progress(length(patients); desc="Evaluating $(dataset_name) width $(width) model $(model_idx)", color=:cyan, showspeed=true) :
               nothing

    for (i, patient) in enumerate(patients)
        @info "Evaluating cUDE patient $(i)/$(length(patients)) for $(dataset_name), width=$(width), model=$(model_idx): $(patient.id)"

        result = try
            fit_cude_patient(
                patient,
                chain,
                nn_params,
                pguess,
                settings.lower,
                settings.upper,
                settings;
                rng=rng,
            )
        catch err
            @warn "Skipping patient $(patient.id) for $(dataset_name), width=$(width), model=$(model_idx): $(err)"
            nothing
        end

        if result !== nothing
            push!(results, result)
            push!(successful_patients, patient)
            push!(successful_indices, i)
            push!(validation_params, result.params)
            @info "Completed $(dataset_name) cUDE patient $(i)/$(length(patients)): $(patient.id) | SMAPE=$(result.smape), RMSLE=$(result.rmsle), loss=$(result.loss)"
        end

        progress !== nothing && next!(progress)
    end

    progress !== nothing && finish!(progress)

    return (
        results=results,
        successful_patients=successful_patients,
        successful_indices=successful_indices,
        patient_ids=[patient.id for patient in successful_patients],
        ode_params_val=isempty(validation_params) ? Float64[] : reduce(vcat, validation_params),
        smape_values=[result.smape for result in results],
        rmsle_values=[result.rmsle for result in results],
        loss_values=[result.loss for result in results],
    )
end

# =============================================================================
# Symbolic Formula Evaluation
# =============================================================================

"""
    fit_symbolic_formula_patient(patient, pguess, lower, upper, settings; rng)

Fit one patient's ODE parameters for the promoted symbolic surrogate using the
configured patient loss and multi-start strategy.
"""
# Used by: src/fitting.jl (evaluate_symbolic_formula_dataset).
function fit_symbolic_formula_patient(
    patient::PatientData,
    pguess::AbstractVector,
    lower::AbstractVector,
    upper::AbstractVector,
    settings;
    rng::AbstractRNG,
)
    problem = symbolic_formula_problem(pguess, patient)
    patient_data = (problem, patient.timepoints, patient.ctnt_data)
    loss_fun = θ -> patient_loss_formula(θ, patient_data; λ_back=settings.lambda_back)

    if settings.n_multistart > 0
        best_result, _ = run_multistart(
            loss_fun,
            settings.n_multistart;
            lower=lower,
            upper=upper,
            rng=rng,
            verbose=false,
            maxiters=settings.maxiters,
            maxtime=Float64(settings.maxtime),
            prescreen=settings.prescreen,
            topk=settings.topk,
            show_progress=settings.progress_bars,
        )

        best_params = Vector(best_result.u)
        best_objective = best_result.minimum
    else
        optfunc = OptimizationFunction(
            (p, data) -> patient_loss_formula(p, data; λ_back=settings.lambda_back),
            AutoForwardDiff(),
        )
        optprob = OptimizationProblem(optfunc, pguess, patient_data; lb=lower, ub=upper)
        optsol = solve(
            optprob,
            LBFGS(linesearch=LineSearches.BackTracking());
            maxiters=settings.maxiters,
        )

        best_params = Vector(optsol.u)
        best_objective = optsol.objective
    end

    new_problem = remake(problem; u0=initial_conditions_from_log_params(best_params), p=best_params)
    pred = solve(new_problem, Tsit5(); p=best_params, saveat=patient.timepoints)
    successful_retcode(pred) || error("Prediction solve failed with retcode=$(pred.retcode)")

    sol = solve(new_problem, Tsit5(); p=best_params, saveat=1)
    successful_retcode(sol) || error("Full trajectory solve failed with retcode=$(sol.retcode)")

    return (
        patient=patient.id,
        smape=smape(pred[3, :], patient.ctnt_data),
        rmsle=rmsle(patient.ctnt_data, pred[3, :]),
        loss=best_objective,
        params=best_params,
        pred=pred,
        sol=sol,
    )
end

"""
    evaluate_symbolic_formula_dataset(patients, settings; dataset_name)

Evaluate the promoted symbolic surrogate formula on an ordered patient cohort.
The shared RNG is advanced patient-by-patient to keep multi-start evaluation
deterministic in cohort order.
"""
# Used by: scripts/04b_evaluate_symbolic_formula.jl.
function evaluate_symbolic_formula_dataset(
    patients::AbstractVector{PatientData},
    settings;
    dataset_name::AbstractString,
)
    rng = StableRNG(settings.rng_seed)
    results = Any[]
    successful_patients = PatientData[]
    successful_indices = Int[]
    params_list = Vector{Vector{Float64}}()

    progress = settings.progress_bars ?
               Progress(length(patients); desc="Evaluating formula $(dataset_name)", color=:cyan, showspeed=true) :
               nothing

    for (i, patient) in enumerate(patients)
        @info "Evaluating symbolic formula patient $(i)/$(length(patients)) for $(dataset_name): $(patient.id)"

        result = try
            fit_symbolic_formula_patient(
                patient,
                settings.pguess,
                settings.lower,
                settings.upper,
                settings;
                rng=rng,
            )
        catch err
            @warn "Skipping symbolic formula patient $(patient.id) for $(dataset_name): $(err)"
            nothing
        end

        if result !== nothing
            push!(results, result)
            push!(successful_patients, patient)
            push!(successful_indices, i)
            push!(params_list, result.params)
            @info "Completed $(dataset_name) symbolic formula patient $(i)/$(length(patients)): $(patient.id) | SMAPE=$(result.smape), RMSLE=$(result.rmsle), loss=$(result.loss)"
        end

        progress !== nothing && next!(progress)
    end

    progress !== nothing && finish!(progress)

    return (
        results=results,
        successful_patients=successful_patients,
        successful_indices=successful_indices,
        patient_ids=[patient.id for patient in successful_patients],
        params_list=params_list,
        params_list_flat=isempty(params_list) ? Float64[] : reduce(vcat, params_list),
        smape_values=[result.smape for result in results],
        rmsle_values=[result.rmsle for result in results],
        loss_values=[result.loss for result in results],
    )
end

# =============================================================================
# ODE Td-Sigmoid Fitting
# =============================================================================

"""
    fit_ode_patient(patient, pguess, lower, upper; ...)

Fit one patient's Td-sigmoid ODE parameters with bounded multi-start
optimization.
"""
# Used by: src/fitting.jl (fit_ode_dataset).
function fit_ode_patient(patient::PatientData, pguess::AbstractVector, lower::AbstractVector, upper::AbstractVector;
    lambda_back::Real=1.0,
    n_multistart::Int=40,
    rng_seed::Integer=1234,
    maxiters::Int=1000,
    maxtime::Real=80.0,
    prescreen::Bool=false,
    topk::Int=8,
    show_progress::Bool=true)

    t_data = patient.timepoints
    x_data = patient.ctnt_data
    tspan = (0.0, t_data[end] + 10.0)
    u0 = initial_conditions_from_log_params(pguess)
    prob = ODEProblem(troponin_ode!, u0, tspan, pguess)
    data = (prob, t_data, x_data)
    loss = theta -> patient_loss_formula(theta, data; λ_back=lambda_back)

    best_result, _ = run_multistart(
        loss,
        n_multistart;
        lower=lower,
        upper=upper,
        rng=StableRNG(rng_seed),
        verbose=false,
        maxiters=maxiters,
        maxtime=Float64(maxtime),
        prescreen=prescreen,
        topk=topk,
        show_progress=show_progress,
    )

    best_params = Vector(best_result.u)
    newprob = remake(prob; p=best_params, u0=initial_conditions_from_log_params(best_params))
    pred = solve(newprob, Tsit5(); saveat=t_data)
    sol = solve(newprob, Tsit5())

    return (
        patient=patient.id,
        smape=smape(pred[3, :], x_data),
        rmsle=rmsle(x_data, pred[3, :]),
        loss=best_result.minimum,
        params=best_params,
        sol=sol,
        pred=pred,
    )
end

"""
    fit_ode_dataset(patients, settings; dataset_name)

Fit all patients for one step 01 dataset and return numeric fit results.
"""
# Used by: scripts/01_run_ode_tdsigmoid_fit.jl.
function fit_ode_dataset(patients::AbstractVector{PatientData}, settings; dataset_name::AbstractString)
    results = Vector{Any}(undef, length(patients))

    for (i, patient) in enumerate(patients)
        @info "Fitting ODE patient $(i)/$(length(patients)): $(patient.id)"
        result = fit_ode_patient(
            patient,
            settings.pguess,
            settings.lower,
            settings.upper;
            lambda_back=settings.lambda_back,
            n_multistart=settings.n_multistart,
            rng_seed=settings.rng_seed,
            maxiters=settings.maxiters,
            maxtime=settings.maxtime,
            prescreen=settings.prescreen,
            topk=settings.topk,
            show_progress=settings.progress_bars,
        )

        @info "Completed $(dataset_name) patient $(i)/$(length(patients)): $(patient.id) | SMAPE=$(result.smape), RMSLE=$(result.rmsle), loss=$(result.loss)"
        results[i] = result
    end

    return results
end

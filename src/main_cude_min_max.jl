using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics
using Dates
using StaticArrays
using Logging
using Base.Threads: @threads, nthreads

using Revise
includet("ctnt-ude-model.jl")

println("⚠️ Algorithm started $(now())")

############################
# 1. Caricamento del dataset
############################
println("Dataset loading...")
# Percorso del file Excel
# file_path = "data/STEMI_merged.xlsx";
# sheet_times = "Tempi cleaned";
# sheet_values = "Misurazioni cleaned";

# 0 - MIMIC-IV NSTEMI
# 1 - UMG NSTEMI
# 2 - UMG STEMI

dataset_id = 0; # change here for different datasets
# plotting = true; # set true to enable plotting of each patient during optimization and residual calculation

if dataset_id == 0
    dataset_name = "MIMIC-IV"
    dataset_path = "MIMIC-IV/NSTEMI_reorganized_skipped.xlsx"
    column_letter = "B"
elseif dataset_id == 1
    dataset_name = "UMG"
    dataset_path = "UMG_NSTEMI_Dataset.xlsx"
    column_letter = "A"
    # elseif dataset_id == 2
    #     dataset_name = "UMG_STEMI";
    #     dataset_path = "UMG_STEMI_Dataset.xlsx";
    #     column_letter = "A";
end

file_path = "data/$(dataset_path)" # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

input_dim = 2;
nn_depth = 2;
nn_width = 4;
inputs_str = "τ, β";
if input_dim == 4
    inputs_str = "Cs0, Cc0, τ, β"
elseif input_dim == 6
    inputs_str = "τ, a, b, Cs0, Cc0, β"
end

# USE_GPU = true;
T_SCALE = 240.0;
N_params = 5;
λ_back = 1;  # test: 1e-4, 1e-3, 1e-2 # best 1
κ_bounds = 0.05 # best 0.05
# dt = 0.1;

# norm_type = 0; # 0 t_scale division; 1 min-max normalization; 2 log normalization

# if norm_type == 0
#     norm_msg = "Using t scale normalization with T_SCALE = $T_SCALE"
#     norm_label = "$T_SCALE"
# elseif norm_type == 1
#     norm_msg = "Using min-max normalization"
#     norm_label = "min_max"
# elseif norm_type == 2
#     norm_msg = "Using log normalization"
#     norm_label = "log"
# end

# @info norm_msg

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

experiment = "NSTEMI_cUDE_$(dataset_name)_MSE_$(nn_depth)$(nn_width)_sigmoid_regback"
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
mkpath(fig_path)
mkpath(models_path)
open("res/$(experiment)/info_output.txt", "w") do io
    println(io, "Experiment $(experiment) log file")
    # println(io, "be = bounds edited")
    println(io, "Neural network settings:")
    println(io, "dept: $(nn_depth); width: $(nn_width); inputs($(input_dim)): $(inputs_str)")
    println(io, "dataset: $(file_path)")
    # println(io, norm_msg)
end

xf = XLSX.readxlsx(file_path);
# Caricamento dei fogli in DataFrame
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

println("Patient loaded: ", nrow(ids))
println("Initialize...")

patients = [row2Patient(ids[i, :], timepoints_df[i, :], troponin_df[i, :]) for i in 1:nrow(ids)];

dup_counts = patient_duplicate_time_counts(patients; tol=0.0)
(nmod, nrm) = collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

# Trimming to T_SCALE
trimmed_p = trim_time(patients, T_SCALE);
patient_dims(trimmed_p)

# 0. Pre-processing
meas_min_number = 5;
min_acq_time_before = 12.0;
min_acq_n_before = 1;
min_acq_time_after = 48.0;
min_acq_n_after = 1;
min_time = 72.0;
max_gap = 72.0;
anoms = find_anomalies(trimmed_p,
    meas_min_number,
    min_acq_time_before, min_acq_n_before,
    min_acq_time_after, min_acq_n_after,
    min_time;
    max_gap_h=max_gap,
    verbose=false
);

println("Removed sample: $(length(anoms))")

cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p);
patient_dims(cleaned_patients)
println("Total sample: $(length(cleaned_patients))")

all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(cleaned_patients);
display(dist)
savefig("$(fig_path)/dataset_distributions.svg")

plt = scutter_patients(cleaned_patients)
display(plt)
savefig("$(fig_path)/scatter_post.svg")

Random.seed!(1234);
rng = StableRNG(42);
shuffle!(cleaned_patients);
n_train = Int(round(length(cleaned_patients) * 0.8));
training_dataset = cleaned_patients[1:n_train];
test_dataset = cleaned_patients[n_train+1:end];
# n_patients = length(training_dataset);
println("Training split: ", length(training_dataset))
println("Validation split: ", length(test_dataset))

training_id = [patient.id for patient in training_dataset];
test_id = [patient.id for patient in test_dataset];

# check = [];
# check = load("$(models_path)/testsetNSTEMI_MIMIC_0706log.jld2", "test_dataset")
# @load "$(models_path)/testsetNSTEMI_MIMIC_0706log.jld2" check;

@save "$(models_path)/trainingsetNSTEMI_$(experiment).jld2" training_dataset;
@save "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;

# nn_params_init = init_params(chain);
# println("NN parameter len: $(length(nn_params_init))")

n_conditional = 1;
lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1]); # 5.0, 5.0, 200.0, 400.0, 3
# [a, b, Cs0, Cc0 ... last one is conditional parameter β]
initial_guesses = dataset_id == 0 ? 25000 : 1000; # number of initial guesses

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Patient loaded: ", nrow(ids))
    println(io, "Clean dataset info: ", length(cleaned_patients), " patients")
    println(io, "Time info: min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h")
    println(io, "cTnT info: min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL")

    println(io, "\nDataset filter settings: ")
    println(io, "meas_min_number: ", meas_min_number)
    println(io, "min_acq_time_before: ", min_acq_time_before)
    println(io, "min_acq_n_before: ", min_acq_n_before)
    println(io, "min_acq_time_after: ", min_acq_time_after)
    println(io, "min_acq_n_after: ", min_acq_n_after)
    println(io, "min_time: ", min_time)
    println(io, "max_gap: ", max_gap)

    println(io, "\nTraining split: ", length(training_dataset))
    println(io, "Validation split: ", length(test_dataset))

    println(io, "\nLB exp: ", exp.(lhs_lb))
    println(io, "UB exp: ", exp.(lhs_ub))

    println(io, "\nRegularization λ_back: ", λ_back)
end

initial_nn = sample_initial_neural_parameters(initial_guesses, chain, rng);
initial_ode = sample_initial_parameters(length(training_dataset), initial_guesses, lhs_lb, lhs_ub, rng);

initial_parameters = [ComponentArray(
    neural=initial_nn[i],
    ode=repeat(initial_ode[:, i], 1, n_conditional)
) for i in eachindex(initial_nn)];

init_bar = Progress(initial_guesses; dt=1, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick);

losses_initial = Vector{Float64}(undef, initial_guesses);
models = Vector{Vector{ctntUDEModel}}(undef, initial_guesses);

θ_dummy = initial_parameters[1];
local_models = [
    ctntCUDEModel(θ_dummy.ode[N_params*(j-1)+1:N_params*j], chain,
        training_dataset[j].timepoints;
        # norm_type=norm_type
    )
    for j in eachindex(training_dataset)
];

# @threads for k in eachindex(initial_parameters)
for k in eachindex(initial_parameters)
    p = initial_parameters[k]
    # local_models = [
    #     ctntCUDEModel(p.ode[5*(j-1)+1:5*j], chain,
    #                   (0.0, training_dataset[j].timepoints[end]))
    #     for j in eachindex(training_dataset)
    # ]
    # models[k] = local_models
    losses_initial[k] =
    # serial_training_loss(p, (local_models, training_dataset); n_params=N_params)
        par_training_loss(p, (local_models, training_dataset);
            n_params=N_params,
            lb_param=lhs_lb, ub_param=lhs_ub,
            κ_bounds=κ_bounds,
            λ_back=λ_back
        )

    next!(init_bar)      # thread-safe
end

selected_initials = 4;# Threads.nthreads(); # 2 number of best initializations to select
@info "Selecting $selected_initials best initializations out of $initial_guesses"

param_indxs = partialsortperm(losses_initial, 1:selected_initials);
out_params = initial_parameters[param_indxs];

# println("Calculated losses")
# println(losses_initial)
println("Starting points: ", selected_initials)
println("Best starting points loss indexes: ", param_indxs)
println("Best starting points losses: ", losses_initial[param_indxs])

# models = local_models[param_indxs];

@save "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;
# @load "$(models_path)/out_paramsNSTEMI_$(experiment).jld2" out_params;

#### TRAINING ####

adam_maxiters = 500;
lbfgs_maxiters = 400;

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Starting points: ", selected_initials)
    println(io, "Best starting points loss indexes: ", param_indxs)
    println(io, "Best starting points losses: ", losses_initial[param_indxs])

    println(io, "\nadam_maxiters: ", adam_maxiters)
    println(io, "lbfgs_maxiters: ", lbfgs_maxiters)
end

# optsols = OptimizationSolution[];
# losses_per_model = Vector{Vector{Float64}}()

n_start = length(out_params);            # = selected_initials

optsols = Vector{OptimizationSolution}(undef, n_start);
losses_per_model = Vector{Vector{Float64}}(undef, n_start);

function make_progress_cb!(pbar, losses; offset=0, every_iters=10, every_secs=1.0)
    last_t = Ref(time())
    last_k = Ref(0)   # ultimo "global iter" scritto sulla barra

    return (state, l) -> begin
        push!(losses, l)

        k = offset + state.iter  # <-- contatore globale monotono

        if (k - last_k[] >= every_iters) || (time() - last_t[] > every_secs)
            ProgressMeter.update!(pbar, k;
                showvalues=() -> [(:iter, k), (:loss, l)]
            )
            last_k[] = k
            last_t[] = time()
        end
        return false
    end
end

# lb_ode = repeat(lhs_lb, length(training_dataset))
# ub_ode = repeat(lhs_ub, length(training_dataset))

# lb_train = ComponentArray(neural = fill(-Inf, length(θ_init.neural)), ode = lb_ode),
# ub_train = ComponentArray(neural = fill( Inf, length(θ_init.neural)), ode = ub_ode),

optfunc = OptimizationFunction(
    (p, data) -> par_training_loss(p, data;
        n_params=N_params,
        lb_param=lhs_lb, ub_param=lhs_ub,
        κ_bounds=κ_bounds,
        λ_back=λ_back),
    # (p, data) -> serial_training_loss(p, data; n_params=N_params),
    AutoForwardDiff()
); # training_loss AutoForwardDiff() AutoZygote()

for (i, θ_init) in enumerate(out_params)
    # @threads for i in eachindex(out_params)
    θ_init = out_params[i]
    train_bar = Progress(adam_maxiters; dt=1, desc="ADAM phase param set θ$(i)", showspeed=true, color=:firebrick)
    # train_bar.desc = "ADAM phase param set θ$(i)"

    losses_this = Float64[]
    # try

    # cb = make_progress_cb!(train_bar, losses_this)

    cb_adam = make_progress_cb!(train_bar, losses_this; offset=0, every_iters=10, every_secs=10.0)

    println("ADAM for parameter set: θ$(i)")

    # local_models = [
    #     ctntCUDEModel(
    #         θ_init.ode[5*(j-1)+1:5*j], chain,
    #         (0.0, training_dataset[j].timepoints[end])
    #     ) for j in eachindex(training_dataset)
    # ]

    optprob = Optimization.OptimizationProblem(
        optfunc, θ_init,
        (local_models, training_dataset);
    ) # models[i]
    # optprob = Optimization.OptimizationProblem(optfunc, θ_init, (training_dataset, USE_GPU));

    # cb = make_callback(losses_this, state0)

    # opt_adamw = AdamW(η, betas, λdecay)

    opt_result1 = Optimization.solve(
        optprob,
        Optimisers.Adam(0.01),
        maxiters=adam_maxiters,
        callback=cb_adam
        # callback = (state, l) -> begin
        #                 push!(losses_this, l)
        #                 next!(train_bar; showvalues = [(:iter, state.iter), (:loss, l)]);
        #                 # if length(losses_this) % 10 == 0
        #                 #     println("Current loss after $(length(losses_this)) iterations: $(losses_this[end])")
        #                 # end
        #                 return false
        #             end
    ) # Optimisers.Adam(0.01)
    println("LBFGS for parameter set: θ$(i)")
    println(opt_result1.retcode)

    println("Adam iterations: $(length(losses_this))")
    # push!(adam_iters_per_model, length(losses_this))

    finish!(train_bar) # Reset Adam per LBFGS

    # train_bar.desc = "LBFGS phase param set θ$(i)"
    train_bar = Progress(lbfgs_maxiters; dt=1, desc="LBFGS phase param set θ$(i)", showspeed=true, color=:firebrick)
    cb_lbfgs = make_progress_cb!(train_bar, losses_this; offset=0, every_iters=10, every_secs=10.0)

    optprob2 = Optimization.OptimizationProblem(
        optfunc, opt_result1.u,
        (local_models, training_dataset)
    ) # models[i]
    # optprob2 = Optimization.OptimizationProblem(optfunc, opt_result1.u, (training_dataset, USE_GPU));
    opt_result2 = Optimization.solve(
        optprob2,
        LBFGS(linesearch=LineSearches.BackTracking()),
        maxiters=lbfgs_maxiters,
        g_abstol=1e-6,
        f_abstol=1e-6,
        x_abstol=1e-6,
        callback=cb_lbfgs
        # callback = (state, l) -> begin
        #                 push!(losses_this, l)
        #                 next!(train_bar; showvalues = [(:iter, state.iter), (:loss, l)]);
        #                 # if length(losses_this) % 10 == 0
        #                 #     println("Current loss after $(length(losses_this)) iterations: $(losses_this[end])")
        #                 # end
        #                 return false
        #             end
    )

    # push!(optsols, opt_result2)
    optsols[i] = opt_result2

    println("Solutions: $(length(optsols))/$selected_initials")
    println(opt_result2.retcode)

    println("LBFGS iterations: $(length(losses_this))")

    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Returncode model θ$(i): ", opt_result2.retcode)
        println(io, "Final loss model θ$(i): ", opt_result2.objective)
    end

    finish!(train_bar)

    # push!(losses_per_model, losses_this)
    losses_per_model[i] = losses_this

    # catch
    # println("Optimization failed... Skipping")
    # end
    # next!(global_prog)
end

@showprogress desc = "Plottng loss" for (k, loss_vec) in enumerate(losses_per_model)
    # n_adam = adam_iters_per_model[k]      # confine reale

    # Adam
    Plots.plot(1:adam_maxiters, loss_vec[1:adam_maxiters];
        yaxis=:log10, xaxis=:log10,
        label="Adam", color=:blue)

    # LBFGS (solo se c’è qualcosa dopo)
    if adam_maxiters < length(loss_vec)
        Plots.plot!(adam_maxiters+1:length(loss_vec),
            loss_vec[adam_maxiters+1:end];
            label="LBFGS", color=:red)
    end

    savefig("$(fig_path)/loss_$(experiment)_$(k).svg")
end

@save "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses_per_model;
# @load "$(models_path)/lossesNSTEMI_$(experiment).jld2" losses_per_model;
# @save "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;
# @load "$(models_path)/optsolsNSTEMI_$(experiment).jld2" optsols;

neural_network_parameters = [optsol.u.neural[:] for optsol in optsols];
ode_params = [optsol.u.ode[:] for optsol in optsols];

@save "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
# @load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@save "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;
# @load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

# lb = log.([0.001, 0.001, 0.001, 0.01, 0.001]);
# ub = log.([5.0, 5.0, 300.0, 400.0, 3]);

#### Preliminary Evaluation, not used in the final pipeline to select the final model. ####
#### The final model was selected after running test_NN.jl, that extract all the metrics after optimizing through the multi-start optimization. ####
#### The final model was therefore selected through grid_search.jl ####

# n_models = length(optsols)                      # = selected_initials
n_optsol = length(neural_network_parameters);

opt_solutions = Vector{Vector{OptimizationSolution}}(undef, n_optsol);
opt_models = Vector{Vector{ctntUDEModel}}(undef, n_optsol);
opt_objectives = Vector{Vector{Float64}}(undef, n_optsol);
opt_smapes = Vector{Vector{Float64}}(undef, n_optsol);
opt_rmsles = Vector{Vector{Float64}}(undef, n_optsol);

@load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;

ev_bar = Progress(n_optsol * length(test_dataset); desc="Validating", color=:cyan, showspeed=true);
optfunc_val = OptimizationFunction(patient_loss, AutoForwardDiff());
# @showprogress desc="Evaluating" for (k, opt_sol) in enumerate(optsols)

optfunc_val = OptimizationFunction(
    (p, data) -> patient_loss(p, data; λ_back=λ_back),
    # (p, data) -> serial_training_loss(p, data; n_params=N_params),
    AutoForwardDiff()
); # training_loss AutoForwardDiff() AutoZygote()

for k in 1:n_optsol # Solutions level
    # opt_sol = optsols[k]
    # nn_p = opt_sol.u.neural;
    # ode_p = opt_sol.u.ode;
    nn_p = neural_network_parameters[k]
    ode_p = ode_params[k]
    # try
    println("Optsolution n: $k")
    models_valid = [
        ctntCUDEModel(
            ode_p[N_params*(j-1)+1:N_params*j], chain,
            test_dataset[j].timepoints;
            # norm_type=norm_type
        )
        for j in eachindex(test_dataset)]

    # println(models_valid)
    reshaped_params = permutedims(reshape(ode_p, N_params, :))
    initial = vec(mean(reshaped_params, dims=1))
    println("Initial: ", exp.(initial))
    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Initial: ", exp.(initial))
    end

    models_optsol_valid = Vector{OptimizationSolution}(undef, length(models_valid))
    optimized_models = Vector{ctntUDEModel}(undef, length(models_valid))
    models_smape = Vector{Float64}(undef, length(models_valid))
    models_rmsle = Vector{Float64}(undef, length(models_valid))

    for (i, model) in enumerate(models_valid) # Patient level
        patient = test_dataset[i]
        @info "Patient $(patient.id) optimization started..."
        # mean_params = mean ode params and β
        optprob = OptimizationProblem(optfunc_val, initial,
            (model, patient.timepoints, patient.ctnt_data, nn_p),
            lb=lhs_lb, ub=lhs_ub
        )

        optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)

        # tspan = (0.0, patient.timepoints[end]+10);

        p_opt = ComponentArray(ode=optsol_lbfgs.u, neural=nn_p)

        u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]

        prob = remake(model.problem; u0=u0_new, p=p_opt)

        optimized_model = ctntUDEModel(prob, chain)

        pred = solve(optimized_model.problem, Tsit5(); p=p_opt, saveat=patient.timepoints)

        @info "Return code: $(pred.retcode)"

        sol = solve(optimized_model.problem, Tsit5(); p=p_opt, saveat=1)
        p_loss_val = patient_loss(p_opt.ode, (optimized_model, patient.timepoints, patient.ctnt_data, p_opt.neural))

        smape_val = smape(pred[3, :], patient.ctnt_data)
        rmsle_val = rmsle(patient.ctnt_data, pred[3, :])

        # smape_val = smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural));

        models_optsol_valid[i] = optsol_lbfgs
        optimized_models[i] = optimized_model
        models_smape[i] = smape_val
        models_rmsle[i] = rmsle_val

        println("For $(patient.id), params: ", optsol_lbfgs.u)
        println("Params: ", exp.(optsol_lbfgs.u))

        println("Patient loss solution $(k): ", p_loss_val)
        println("Objective solution $(k): ", optsol_lbfgs.objective)

        println("sMAPE: ", smape_val)
        println("RMSLE: ", rmsle_val)

        open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
            println(io, "For $(patient.id), params: ", optsol_lbfgs.u)
            println(io, "Params: ", exp.(optsol_lbfgs.u))
            println(io, "Patient loss solution $(k): ", p_loss_val)
            # println(io, "Compute loss model $(k): ", compute_loss(p_opt, (model, patient.timepoints, patient.ctnt_data)))
            println(io, "Objective solution $(k):    ", optsol_lbfgs.objective)
            # println(io, "sMAPE: ", smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
            println(io, "sMAPE: ", smape_val)
            println(io, "RMSLE: ", rmsle_val)
        end

        pl = Plots.plot(sol[1, :]; lw=2, label="Sarcomere")
        Plots.plot!(pl, sol[2, :]; lw=2, label="Cytosol")
        Plots.plot!(pl, sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
        Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

        pl_plasm = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
        Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

        display(pl)
        display(pl_plasm)

        save("$(fig_path)/$(experiment)_model_$(k)_$(patient.id).svg", pl)
        save("$(fig_path)/$(experiment)_model_$(k)_$(patient.id)_plasm.svg", pl_plasm)
        next!(ev_bar)
    end

    opt_solutions[k] = models_optsol_valid
    opt_models[k] = optimized_models
    objs = [sol.objective for sol in models_optsol_valid]
    opt_objectives[k] = objs
    opt_smapes[k] = models_smape
    opt_rmsles[k] = models_rmsle

    println("Median obj: ", median(objs))
    println("Average obj: ", mean(objs))
    println("Median sMAPE: ", median(models_smape))
    println("Average sMAPE: ", mean(models_smape))
    println("Median RMSLE: ", median(models_rmsle))
    println("Average RMSLE: ", mean(models_rmsle))
    # catch
    #     push!(model_objectives, repeat([Inf], length(models)))
    # end
end
finish!(ev_bar)
@info "Validation completed."

# model_objectives = model_objectives[2]
# find the model that performs best on each individual
objectives = hcat(opt_objectives...);
solutions = hcat(opt_solutions...);
new_models = hcat(opt_models...);
smapes = hcat(opt_smapes...);
rmsles = hcat(opt_rmsles...);

@save "$(models_path)/objectivesNSTEMI_$(experiment).jld2" objectives;

println("Average obj in validation: ", mean(objectives, dims=1)[:])
println("Median obj in validation: ", median(objectives, dims=1)[:])

println("Average sMAPE in validation: ", mean(smapes, dims=1)[:])
println("Median sMAPE in validation: ", median(smapes, dims=1)[:])

println("Average RMSLE in validation: ", mean(rmsles, dims=1)[:])
println("Median RMSLE in validation: ", median(rmsles, dims=1)[:])

println("Best average model id: $(argmin(mean(objectives, dims=1)[:]))")
println("Best median model id: $(argmin(median(objectives, dims=1)[:]))")

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Average in validation: ", mean(objectives, dims=1)[:])
    println(io, "Median in validation: ", median(objectives, dims=1)[:])

    println(io, "Average sMAPE in validation: ", mean(smapes, dims=1)[:])
    println(io, "Median sMAPE in validation: ", median(smapes, dims=1)[:])

    println(io, "Average RMSLE in validation: ", mean(rmsles, dims=1)[:])
    println(io, "Median RMSLE in validation: ", median(rmsles, dims=1)[:])

    println(io, "*Best average model id: $(argmin(mean(objectives, dims=1)[:]))*")
    println(io, "Best median model id: $(argmin(median(objectives, dims=1)[:]))")
end

best_median_model_index = argmin(median(objectives, dims=1)[:]);
best_mean_model_index = argmin(mean(objectives, dims=1)[:]);

t_span_grid = 0.0:0.1:T_SCALE;
β_vals = 0.1:0.1:1;   # alcuni valori tipici del tuo β
# β_vals = [0.1, 0.25, 0.5, 0.75, 1.0]; # valori specifici da testare

p = Plots.plot()
for β in β_vals
    y = [chain([t / T_SCALE, β], neural_network_parameters[best_median_model_index])[1] for t in t_span_grid]
    Plots.plot!(p, t_span_grid, y, label="β = $β", linewidth=2)
end
Plots.plot!(p, xlabel="Time (h)", ylabel="correction f(t_norm,β)", title="Learned sarcomere rupture function", legend=:topright)
savefig("$(fig_path)/correction_function_median.png")
display(p)

p = Plots.plot()
for β in β_vals
    y = [chain([t / T_SCALE, β], neural_network_parameters[best_mean_model_index])[1] for t in t_span_grid]
    Plots.plot!(p, t_span_grid, y, label="β = $β", linewidth=2)
end
Plots.plot!(p, xlabel="Time (h)", ylabel="correction f(t_norm,β)", title="Learned sarcomere rupture function", legend=:topright)
savefig("$(fig_path)/correction_function_mean.png")
display(p)

# best_model_index = argmin(sum(objectives, dims=2)[:])
# best_model = optsols[best_model_index];

# best_nn = best_model.u.neural;
# best_ode_beta = best_model.u.ode
best_nn = neural_network_parameters[best_median_model_index];

# ode_betas_test = [optsol.u for optsol in opt_solutions]
best_solution = opt_solutions[best_median_model_index];
@save "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;
best_models = new_models[best_median_model_index];
@save "$(models_path)/optimized_models_NSTEMI_$(experiment).jld2" new_models;
# losses_test = [optsol.objective for optsol in opt_solutions]
# @save "$(models_path)/lossestestNSTEMI_$(experiment).jld2" losses_test;

@save "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
# @save "$(models_path)/best_ode_beta_NSTEMI_$(experiment).jld2" best_ode_beta

# @load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;

a_dist = [exp(sol.u[1]) for sol in best_solution]
b_dist = [exp(sol.u[2]) for sol in best_solution]
Cs0_dist = [exp(sol.u[3]) for sol in best_solution]
Cc0_dist = [exp(sol.u[4]) for sol in best_solution]
β_dist = [exp(sol.u[end]) for sol in best_solution]

plt_a = histogram(a_dist;
    bins=5,
    xlabel="Value",
    ylabel="#",
    title="Params a",
    legend=false)
display(plt_a)
savefig("$(fig_path)/a_dist.svg")

plt_b = histogram(b_dist;
    bins=5,
    xlabel="Value",
    ylabel="#",
    title="Params b",
    legend=false)
display(plt_b)
savefig("$(fig_path)/b_dist.svg")

plt_Cs0 = histogram(Cs0_dist;
    bins=5,
    xlabel="Value",
    ylabel="#",
    title="Params Cs0",
    legend=false)
savefig("$(fig_path)/Cs0_dist.svg")

plt_Cc0 = histogram(Cc0_dist;
    bins=5,
    xlabel="Value",
    ylabel="#",
    title="Params Cc0",
    legend=false)
display(plt_Cc0)
savefig("$(fig_path)/Cc0_dist.svg")

plt_β = histogram(β_dist;
    bins=5,
    xlabel="Value",
    ylabel="#",
    title="Params β",
    legend=false)
savefig("$(fig_path)/beta_dist.svg")
display(plt_β)
@info "⚠️ Algorithm ended $(now())"

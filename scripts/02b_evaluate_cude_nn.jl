"""
evaluate_cude_nn.jl

Refactored copy of `test_NN.jl`.

Evaluate a trained cUDE neural correction model on a selected dataset.

Pipeline:
1. Configure run settings.
2. Resolve input/output paths.
3. Load required data and model artifacts.
4. Run the main computation.
5. Save metrics, parameters, plots, and logs.

This copy uses `MechanisticAI.jl` as the shared helper entrypoint. The original
script is intentionally left untouched as the legacy baseline.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Shared dependencies and the central refactor entrypoint.
# =============================================================================
using StableRNGs, Random
using Optimization, OptimizationOptimisers, LineSearches
using JLD2, CSV, DataFrames, XLSX, Dates
using Plots, CairoMakie, ProgressMeter, Logging
using Statistics, StatsBase
# using Base.Threads: @threads, nthreads
include("MechanisticAI.jl")
using .MultiStartOptimizer

# =============================================================================
# SCRIPT SETTINGS
# User-editable dataset/model/optimizer flags are preserved from the original
# script in the first executable block below.
# =============================================================================

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run are resolved near the settings that define
# dataset/model identity. Keep load paths explicit during this transition pass.
# =============================================================================

# =============================================================================
# OUTPUT PATHS
# Result directories and output files are created by the preserved pipeline below.
# Future cleanup should move path construction into `build_experiment_paths`.
# =============================================================================

# =============================================================================
# DERIVED SETTINGS
# Values computed from the settings above are kept inline for behavior parity.
# Future cleanup should collect them before the pipeline starts.
# =============================================================================

# =============================================================================
# HELPERS
# Script-local helper functions remain near their original location for now.
# Reusable candidates should migrate to helpers.jl after behavior is validated.
# =============================================================================

# =============================================================================
# PIPELINE
# Main execution flow copied from the original script. This first refactor pass
# changes includes and documentation only; numerical behavior is preserved.
# =============================================================================
# CLI:
# julia --project=. src/evaluate_cude_nn.jl <width> <dataset_id> [bounds=true|false] [λ_back=1.0] [T_SCALE=240.0] [N_multistart=40]
# length(ARGS) >= 2 || error(
#     "Usage: julia --project=. src/evaluate_cude_nn.jl <width> <dataset_id> [bounds=true|false] [λ_back=1.0] [T_SCALE=240.0] [N_multistart=40]"
# )

length(ARGS) >= 2 || @info "REPL execution or default setting selected. 
        For CLI usage, provide at least <width> and <dataset_id>.
        \nUsage: julia --project=. src/evaluate_cude_nn.jl <width> <dataset_id> [bounds=true|false] [λ_back=1.0] [T_SCALE=240.0] [N_multistart=40]"

nn_width = length(ARGS) >= 2 ? parse(Int, ARGS[1]) : 8
dataset_id = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0

bounds = length(ARGS) >= 3 ? parse(Bool, lowercase(ARGS[3])) : true
λ_back = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
T_SCALE = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 240.0
N_multistart = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 40
N_multistart >= 0 || error("N_multistart must be >= 0")

@info "⚠️ Test NN algorithm started $(now())"

# dataset_id = 0; # change here for different datasets
if dataset_id == 0
    dataset_name = "MIMIC-IV"
    UMG_data = false
elseif dataset_id == 1
    dataset_name = "UMG"
    UMG_data = true
else
    error("dataset_id must be 0 (MIMIC-IV) or 1 (UMG)")
end

UDE = false; # false for cUDE
if UDE
    @info "Using UDE model"
    input_dim = 1
    nn_depth = 2
    # nn_width = 8
    N_params = 4
    inputs_str = "τ"
else
    @info "Using cUDE model"
    input_dim = 2
    nn_depth = 2
    # nn_width = 6
    N_params = 5
    inputs_str = "τ, β"
end

# bounds = true;
# λ_back = 1.0;
# T_SCALE = 240.0;
use_multistart = N_multistart > 0
# N_multistart = 40
multistart_maxiters = 1000
multistart_maxtime = 80.0
multistart_rng = StableRNG(1234)
prescreen = false
topk = 12

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

# Copy relative path of the experiment: e.g., "NSTEMI_partrval_UMG_MSE_ts350.0_28_inp2_multipl_softplus"
# experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
# experiment = "NSTEMI_partrval_UMG_MSE_ts350.0_28_inp2_multipl_softplus";
# experiment = "NSTEMI_cUDEabs1_UMG_MSE_ts350.0_28_inp2_multipl_softplus";
# experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus" #REULTS
# experiment = "NSTEMI_cUDE_hi_MIMIC-IV_MSE_ts350.0_28";
# experiment = "NSTEMI_cUDE_filtered_MIMIC-IV_MSE_240.0_28_sigmoid_regback";

experiment = "NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback";

fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

@load "$(models_path)/trainingsetNSTEMI_$(experiment).jld2" training_dataset;

training_ids = [patient.id for patient in training_dataset]

@load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

test_dataset = if UMG_data
    @info "Using UMG dataset for testing"

    try
        @info "Try to load UMG test"

        @load "$(models_path)/UMG_testset.jld2" test_dataset
        @info "UMG test dataset loaded from previous save"
        test_dataset
    catch
        @warn "UMG test dataset not found, loading from Excel file"

        file_path = "data/UMG_NSTEMI_Dataset.xlsx" # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
        sheet_ids = "IDs"
        sheet_times = "times"
        sheet_values = "values"
        xf = XLSX.readxlsx(file_path)
        # Caricamento dei fogli in DataFrame
        # ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
        ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true))
        timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true))
        troponin_df = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true))

        println("Patient loaded: ", nrow(ids))
        patients = [row2Patient(ids[i, :], timepoints_df[i, :], troponin_df[i, :]) for i in 1:nrow(ids)]

        # Trimming to T_SCALE
        dup_counts = patient_duplicate_time_counts(patients; tol=0.0)
        (nmod, nrm) = collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

        # Trimming to T_SCALE
        trimmed_p = trim_time(patients, T_SCALE)
        patient_dims(trimmed_p)

        # 0. Pre-processing
        meas_min_number = 5
        min_acq_time_before = 12.0
        min_acq_n_before = 1
        min_acq_time_after = 48.0
        min_acq_n_after = 1
        min_time = 72.0
        max_gap = 72.0
        anoms = find_anomalies(trimmed_p,
            meas_min_number,
            min_acq_time_before, min_acq_n_before,
            min_acq_time_after, min_acq_n_after,
            min_time;
            max_gap_h=max_gap,
            verbose=false
        )

        # 0. Pre-processing
        # meas_min_number = 5
        # anoms = find_anomalies(trimmed_p, meas_min_number)
        println("Removed: $(length(anoms))")

        cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p)
        patient_dims(cleaned_patients)
        println("Total sample: $(length(cleaned_patients))")

        all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(cleaned_patients)
        display(dist)
        savefig("$(fig_path)/umg_distributions.svg")

        plt = scutter_patients(cleaned_patients)
        # display(plt)
        savefig("$(fig_path)/scatter_post.svg")

        test_dataset = cleaned_patients
        @save "$(models_path)/UMG_testset.jld2" test_dataset
        @info "UMG test dataset saved"
        test_dataset
    end

else
    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset
    @info "Using $dataset_name dataset for testing"
    test_dataset
end

test_ids = [patient.id for patient in test_dataset]

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "********************************")
    println(io, "********************************")
    println(io, "********************************")
    println(io, "********************************")
    println(io, "*************MULTISTART***********")
    println(io, "$dataset_name - Test NN started $(now())")
    println(io, "   Training set: $(length(training_dataset))")
    println(io, "   Test set: $(length(test_dataset))")
    println(io, "********************************")
end

models_summary = DataFrame(
    model_id = String[],
    model_idx = Int[],
    nn_depth = Int[],
    nn_width = Int[],
    n_patients = Int[],

    loss_mean = Float64[],
    loss_std = Float64[],
    loss_median = Float64[],
    loss_q1 = Float64[],
    loss_q3 = Float64[],
    loss_iqr = Float64[],

    smape_mean = Float64[],
    smape_std = Float64[],
    smape_median = Float64[],
    smape_q1 = Float64[],
    smape_q3 = Float64[],
    smape_iqr = Float64[],

    rmsle_mean = Float64[],
    rmsle_std = Float64[],
    rmsle_median = Float64[],
    rmsle_q1 = Float64[],
    rmsle_q3 = Float64[],
    rmsle_iqr = Float64[]
)

# for best_idx in eachindex(neural_network_parameters)
    best_idx = 3

    model_id = "cfg$(nn_depth)$(nn_width)_$(best_idx)"

    figsave_path = "$(fig_path)/$(dataset_name)_test_NN_ab10_$(best_idx)$(use_multistart ? "_ms_test" : "")"
    modelssave_path = "$(models_path)/$(dataset_name)_test_NN_ab10_$(best_idx)$(use_multistart ? "_ms_test" : "")"

    mkpath(figsave_path)
    mkpath(modelssave_path)

    best_nn = neural_network_parameters[best_idx]
    best_ode_p = ode_params[best_idx] # log version where 1 is the index of the best model in info_output
    # reshaped_params = reshape(best_ode_p, :, N_params)
    reshaped_params = permutedims(reshape(best_ode_p, N_params, :)) # reshape to (N_param_sets, N_params) and then permute to (N_param_sets, N_params)
    mean_pguess = vec(mean(reshaped_params, dims=1))
    std_pguess = vec(std(exp.(reshaped_params), dims=1))
    median_pguess = vec(median(exp.(reshaped_params), dims=1))
    q3_pguess = vec([quantile(exp.(reshaped_params[:, i]), 0.75) for i in 1:N_params])
    q1_pguess = vec([quantile(exp.(reshaped_params[:, i]), 0.25) for i in 1:N_params])

    println("Parameters average, STD: ", exp.(mean_pguess), ", ", std_pguess)
    println("Parameters median, [Q1, Q3]: ", median_pguess, " [", q1_pguess, ", ", q3_pguess, "]")

    pguess = log.(median_pguess)

    @info "Selected as pguess: $(exp.(pguess))"

    # t_norm = range(0.0, 1.0, length=200)
    t_span_grid = 0.1:0.1:2500  # alcuni valori tipici del tuo β
    β_vals = 0.1:0.1:1.0
    p = Plots.plot()
    for β in β_vals
        y = [chain([t / T_SCALE, β], best_nn)[1] for t in t_span_grid]
        Plots.plot!(p, t_span_grid, y, label="β = $β", linewidth=2)
    end
    Plots.plot!(p, xlabel="Time (h)", ylabel="rupture f(t_norm,β)", title="Learned sarcomere rupture function")
    display(p)
    # save("$(figsave_path)/correction_function.png", p)
    savefig(p, "$(figsave_path)/correction_function_2500h.png")
    # Plots.plot!(p, legend=false)

    a_train, b_train, Cs0_train, Cc0_train, β_train, f_train = params_extraction(
        training_dataset, best_ode_p;
        N_params=N_params, data_label="training", 
        dataset=dataset_name, figsave_path=figsave_path, 
        show_outliers=true, savefigure=true)
    display(f_train)

    CSV.write("$(modelssave_path)/patients_params_train.csv", DataFrame(
        patient_id=training_ids,
        a=a_train,
        b=b_train,
        Cs0=Cs0_train,
        Cc0=Cc0_train,
        beta=β_train
    ))

    ΔCsc0_train = [Cs0_train[i] - Cc0_train[i] for i in eachindex(Cs0_train)]
    @info "Median [Q1-Q3] in training param ΔCsc0: $(median(ΔCsc0_train)) [$(quantile(ΔCsc0_train, 0.25)) - $(quantile(ΔCsc0_train, 0.75))]"
    @info "Average, STD in training param ΔCsc0: $(mean(ΔCsc0_train)) std: $(std(ΔCsc0_train))"

    if N_params == 5
        lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001]) # 0.001, 0.001, 0.01, 0.01, 0.001
        # lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1.0]); # 5.0, 5.0, 300.0, 400.0, 1
        lhs_ub = log.([10.0, 10.0, 500.0, 500.0, 1.0]) # 5.0, 5.0, 300.0, 400.0, 1
    else
        lhs_lb = log.([0.001, 0.001, 0.001, 0.001]) # 0.001, 0.001, 0.01, 0.01
        lhs_ub = log.([10.0, 10.0, 500.0, 500.0]) # 5.0, 5.0, 300.0, 400.0
    end

    # ode_p = best_solution;
    # pguess = mean([optsol.u for optsol in ode_p]);

    # optfunc = OptimizationFunction(patient_loss, AutoForwardDiff());
    # optfunc = OptimizationFunction(
    #     (p, data) -> patient_loss(p, data; λ_back=λ_back),
    #     # (p, data) -> serial_training_loss(p, data; n_params=N_params),
    #     AutoForwardDiff()
    # ) # training_loss AutoForwardDiff() AutoZygote()

    println("*********************************")
    println("$dataset_name - Evaluating NN with sMAPE (idx:$(best_idx))")

    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "*********************************")
        println(io, "$dataset_name - Evaluating NN with sMAPE (idx:$(best_idx))$(bounds ? " larger bounds" : " no bounds")")
    end

    # smape_values = []
    # rmsle_values = []
    # loss_values = []
    # validation_params = []

    successful_idx = Int[]
    smape_values = Float64[]
    rmsle_values = Float64[]
    loss_values = Float64[]
    validation_params = Vector{typeof(pguess)}()

    # u0_init = [exp(pguess[3]), exp(pguess[4]), 0.0];

    ev_bar = Progress(length(test_dataset); desc="Validating", color=:cyan, showspeed=true)
    for (i, patient) in enumerate(test_dataset)

        # tspan = (0.0, patient.timepoints[end] + 10.0);
        # model = UDE ? ctntUDEModel(pguess, chain, tspan) : ctntCUDEModel(pguess, chain, tspan); # check i cUDE or UDE

        @info "Patient $(patient.id)"

        model = UDE ? ctntUDEModel(pguess, chain, patient.timepoints) :
                      ctntCUDEModel(pguess, chain, patient.timepoints)

        # if bounds
        #     optprob = OptimizationProblem(optfunc, pguess,
        #         (model, patient.timepoints, patient.ctnt_data, best_nn),
        #         lb=lhs_lb, ub=lhs_ub
        #     )
        # else
        #     optprob = OptimizationProblem(optfunc, pguess,
        #         (model, patient.timepoints, patient.ctnt_data, best_nn)
        #     )
        # end
        # optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()),
        #     maxiters=1000)
        # p_opt = ComponentArray(ode=optsol_lbfgs.u, neural=best_nn)

        patient_data = (model, patient.timepoints, patient.ctnt_data, best_nn)

        loss_fun = θ -> patient_loss(θ, patient_data; λ_back=λ_back)

        if use_multistart
            if !bounds
                error("MultiStart in test_NN currently requires bounds=true, because run_multistart samples starts within [lower, upper].")
            end

            best_result, all_results = MultiStartOptimizer.run_multistart(
                loss_fun,
                N_multistart;
                lower = lhs_lb,
                upper = lhs_ub,
                rng = multistart_rng,
                verbose = false,
                maxiters = multistart_maxiters,
                maxtime = multistart_maxtime,
                prescreen = prescreen,
                topk = topk
            )

            if best_result === nothing
                @warn "No multistart solution found for patient $(patient.id)"
                next!(ev_bar)
                continue
            end

            best_ode_params = best_result.u
            best_objective = best_result.minimum
        else
            optfunc = OptimizationFunction(
                (p, data) -> patient_loss(p, data; λ_back=λ_back),
                # (p, data) -> serial_training_loss(p, data; n_params=N_params),
                AutoForwardDiff()
            )

            if bounds
                optprob = OptimizationProblem(
                    optfunc,
                    pguess,
                    patient_data;
                    lb = lhs_lb,
                    ub = lhs_ub
                )
            else
                optprob = OptimizationProblem(
                    optfunc,
                    pguess,
                    patient_data
                )
            end

            optsol_lbfgs = Optimization.solve(
                optprob,
                LBFGS(linesearch = LineSearches.BackTracking());
                maxiters = 1000
            )

            best_ode_params = optsol_lbfgs.u
            best_objective = optsol_lbfgs.objective
        end

        p_opt = ComponentArray(ode = best_ode_params, neural = best_nn)

        println("For $(patient.id), params: ", p_opt.ode)
        println("Params: ", exp.(p_opt.ode))
        push!(validation_params, p_opt.ode)
        push!(successful_idx, i)
        # push!(optsols_valid, optsol_lbfgs);

        u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]
        prob = remake(model.problem; u0=u0_new, p=p_opt)

        opt_model = ctntUDEModel(prob, chain)

        # pred = solve(prob, Tsit5(); p=p_opt, saveat=patient.timepoints)
        pred = try
            solve(prob, Tsit5(); p = p_opt, saveat = patient.timepoints)
        catch
            @warn "Prediction solve failed for patient $(patient.id)"
            open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
                println(io, "WARN: Prediction solve failed for patient $(patient.id)")
            end
            pop!(validation_params)
            pop!(successful_idx)
            next!(ev_bar)
            continue
        end
        println(pred.retcode)
        # sol = solve(prob, Tsit5(); p=p_opt, saveat=1)
        sol = try
            solve(prob, Tsit5(); p = p_opt, saveat = 1)
        catch
            @warn "Full trajectory solve failed for patient $(patient.id)"
            open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
                println(io, "WARN: Full trajectory solve failed for patient $(patient.id)")
            end
            pop!(validation_params)
            pop!(successful_idx)
            next!(ev_bar)
            continue
        end

        println("Patient loss: ", patient_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
        # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
        
        # push!(loss_values, optsol_lbfgs.objective) # after multistart
        # println("Objective:    ", optsol_lbfgs.objective) # after multistart
        push!(loss_values, best_objective)
        println("Objective:    ", best_objective)
        # validation_metric = smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural));

        smape_val = smape(pred[3, :], patient.ctnt_data)
        rmsle_val = rmsle(patient.ctnt_data, pred[3, :])

        println("sMAPE: ", smape_val)
        println("RMSLE: ", rmsle_val)

        open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
            println(io, "Patient $(patient.id) sMAPE NN validation: $smape_val")
        end
        push!(smape_values, smape_val)
        push!(rmsle_values, rmsle_val)
        # pred = sol[3,:];

        # pl = Plots.plot(pred; lw=2, label="Model with NN Prediction", xlabel="Time", ylabel="Troponin", title="Patient $(patient.id)")
        pl = Plots.plot(sol[1, :]; lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title="cUDE NN Patient $(patient.id)")
        Plots.plot!(pl, sol[2, :]; lw=2, label="Cytosol")
        Plots.plot!(pl, sol[3, :]; lw=2, label="Blood")
        Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

        pl_plasm = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
        Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

        display(pl)
        display(pl_plasm)

        # save("$(figsave_path)/patient_$(patient.id)$(dataset_name).svg", pl)
        # save("$(figsave_path)/patient_$(patient.id)$(dataset_name)_plasm.svg", pl_plasm)
        savefig(pl, "$(figsave_path)/patient_$(patient.id)_$(dataset_name).svg")
        savefig(pl_plasm, "$(figsave_path)/patient_$(patient.id)_$(dataset_name)_plasm.svg")

        next!(ev_bar)
    end

    if isempty(successful_idx)
        @warn "No successful patients for model $(model_id). Skipping summary and CSV export."
        open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
            println(io, "WARN: No successful patients for model $(model_id). Skipping summary and CSV export.")
        end
        # continue
    end

    used_test_dataset = test_dataset[successful_idx]
    used_test_ids = test_ids[successful_idx]

    ode_params_val = vcat(validation_params...)
    @save "$(modelssave_path)/best_params_val_$(dataset_name).jld2" ode_params_val

    a, b, Cs0, Cc0, β, fig = params_extraction(
        used_test_dataset,
        ode_params_val;
        N_params = N_params,
        data_label = "validation",
        dataset = dataset_name,
        figsave_path = figsave_path,
        show_outliers = true,
        savefigure = true
    )
    display(fig)

    CSV.write("$(modelssave_path)/patients_params_val.csv", DataFrame(
        patient_id=used_test_ids,
        a=a,
        b=b,
        Cs0=Cs0,
        Cc0=Cc0,
        beta=β
    ))

    loss_q1  = quantile(loss_values, 0.25)
    loss_q3  = quantile(loss_values, 0.75)
    smape_q1 = quantile(smape_values, 0.25)
    smape_q3 = quantile(smape_values, 0.75)
    rmsle_q1 = quantile(rmsle_values, 0.25)
    rmsle_q3 = quantile(rmsle_values, 0.75)

    push!(models_summary, (
        model_id = model_id,
        model_idx = best_idx,
        nn_depth = nn_depth,
        nn_width = nn_width,
        n_patients = length(used_test_dataset),

        loss_mean = mean(loss_values),
        loss_std = std(loss_values),
        loss_median = median(loss_values),
        loss_q1 = loss_q1,
        loss_q3 = loss_q3,
        loss_iqr = loss_q3 - loss_q1,

        smape_mean = mean(smape_values),
        smape_std = std(smape_values),
        smape_median = median(smape_values),
        smape_q1 = smape_q1,
        smape_q3 = smape_q3,
        smape_iqr = smape_q3 - smape_q1,

        rmsle_mean = mean(rmsle_values),
        rmsle_std = std(rmsle_values),
        rmsle_median = median(rmsle_values),
        rmsle_q1 = rmsle_q1,
        rmsle_q3 = rmsle_q3,
        rmsle_iqr = rmsle_q3 - rmsle_q1
    ))

    println("--> Average - STD loss: $(mean(loss_values)) - $(std(loss_values))")
    println("--> Median [Q1-Q3] loss: $(median(loss_values)) [$(quantile(loss_values, 0.25)) - $(quantile(loss_values, 0.75))]")
    println("--> Average - STD sMAPE: $(mean(smape_values)) - $(std(smape_values))")
    println("--> Median [Q1-Q3] sMAPE: $(median(smape_values)) [$(quantile(smape_values, 0.25)) - $(quantile(smape_values, 0.75))]")
    println("--> Average - STD RMSLE: $(mean(rmsle_values)) - $(std(rmsle_values))")
    println("--> Median [Q1-Q3] RMSLE: $(median(rmsle_values)) [$(quantile(rmsle_values, 0.25)) - $(quantile(rmsle_values, 0.75))]")

    open("res/$(experiment)/info_output.txt", "a") do io
        println(io, "--> Average - STD loss: $(mean(loss_values)) - $(std(loss_values))")
        println(io, "--> Median [Q1-Q3] loss: $(median(loss_values)) [$(quantile(loss_values, 0.25)) - $(quantile(loss_values, 0.75))]")
        println(io, "--> Average - STD sMAPE: $(mean(smape_values)) - $(std(smape_values))")
        println(io, "--> Median [Q1-Q3] sMAPE: $(median(smape_values)) [$(quantile(smape_values, 0.25)) - $(quantile(smape_values, 0.75))]")
        println(io, "--> Average - STD RMSLE: $(mean(rmsle_values)) - $(std(rmsle_values))")
        println(io, "--> Median [Q1-Q3] RMSLE: $(median(rmsle_values)) [$(quantile(rmsle_values, 0.25)) - $(quantile(rmsle_values, 0.75))]")
    end

    CSV.write("$(modelssave_path)/patients_metrics_val.csv", DataFrame(
        patient_id=used_test_ids,
        smape=smape_values,
        rmsle=rmsle_values,
        loss=loss_values
    ))
# end

sort!(models_summary, :smape_median)
CSV.write("$(models_path)/models_summary_$(dataset_name).csv", models_summary)

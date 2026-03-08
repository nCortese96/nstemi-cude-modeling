using DataFrames, XLSX, CSV
using Logging, Dates
using Plots, JLD2
using DataInterpolations
using StatsBase, Random, StableRNGs
using Optimization, OptimizationOptimisers, LineSearches
using ProgressMeter, Statistics

using Revise
includet("ctnt-ude-model.jl")

rmsle(y_true, y_pred) = sqrt(mean((log.(y_pred .+ 1) .- log.(y_true .+ 1)).^2))

dataset_id = 0;
T_SCALE = 350.0;
plotting = true;

if dataset_id == 0
    dataset = "MIMIC-IV";
    dataset_name = "MIMIC_IV";
    dataset_path = "MIMIC-IV/NSTEMI_reorganized_skipped.xlsx";
    column_letter = "B";
elseif dataset_id == 1
    dataset_name = "UMG";
    dataset_path = "UMG_NSTEMI_Dataset.xlsx";
    column_letter = "A";
end

file_path = "data/$(dataset_path)"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped UMG_STEMI_Dataset
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

xf = XLSX.readxlsx(file_path);
# Caricamento dei fogli in DataFrame
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];
patients = trim_time(patients, T_SCALE);
patient_dims(patients)

if dataset_id == 0
    @info "All eligible $dataset validation"
    ae_ids = CSV.read("res/ids_all_eligible_$(dataset)_val.csv", DataFrame);
    all_eligible_idxs = findall(p -> p.id in ae_ids.patient, patients);
    patients = patients[all_eligible_idxs];
end

# High information characteristics
meas_min_number = 8;
min_acq_time_before=12.0;
min_acq_n_before=2;
min_acq_time_after=36.0;
min_acq_n_after=1;
min_time=12.0;
max_gap=24.0;

anoms = find_anomalies(
    patients, 
    meas_min_number, 
    min_acq_time_before, min_acq_n_before, 
    min_acq_time_after, min_acq_n_after, 
    min_time; 
    max_gap_h=max_gap,
    verbose=false
    );
@info "Removed sample number for high information: $(length(anoms))"
gold_std_patients = filter(p -> !haskey(anoms, p.id), patients);
patient_dims(gold_std_patients)
@info "Total sample number for high information: $(length(gold_std_patients))"

df_gold_ids = DataFrame(patient = [p.id for p in gold_std_patients]);
# CSV.write(joinpath("./res", "ids_gold_std_patients_$(dataset_name).csv"), df_gold_ids)

for (i, p) in enumerate(gold_std_patients)
    @info "index: $(i), $(p.id); N: [$(length(p.ctnt_data))]"
    x_data = [0.0, p.ctnt_data...];
    t_data = [0.0, p.timepoints...];

    interp_linear = LinearInterpolation(x_data, t_data)
    # interp_quadratic = QuadraticInterpolation(x_data, t_data)
    # interp_cubic = CubicSpline(x_data, t_data)

    # println("Valore al tempo 1.5 (Lineare): ", interp_linear(1.5))
    # println("Valore al tempo 1.5 (Cubica): ", interp_cubic(1.5))

    t_fine = 0.0:0.1:t_data[end]

    plt = Plots.plot(p.timepoints, p.ctnt_data, seriestype=:scatter, label="Measurements")
    Plots.plot!(plt, t_fine, interp_linear.(t_fine), label="Linear", linestyle=:dash)
    # plot!(t_fine , interp_cubic.(t_fine), label="Quadratic Interpolation", linewidth=1)
    # plot!(t_fine , interp_cubic.(t_fine), label="Cubic Spline", linewidth=2)
    display(plt)
    savefig(plt, "$(fig_path)/interp_$(p.id).svg")
end

# base_choise_idx = 4;
UMG_data = false; # true for UDE with UMG data, false for cUDE with MIMIC-IV data

UDE = false; # false for cUDE

best_idx = 4; # index of the best model to test 

retro = false; # false for forecasting analysis

if UDE
    @info "Using UDE model"
    input_dim = 1;
    nn_depth = 2;
    nn_width = 8;
    N_params = 4;
    inputs_str = "τ";
else
    @info "Using cUDE model"
    input_dim = 2;
    nn_depth = 2;
    nn_width = 8;
    N_params = 5;
    inputs_str = "τ, β";
end 

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus";

fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

figsave_path = "$(fig_path)/$(dataset)_test_$(retro ? "rs" : "fr")_NN_$(best_idx)";
modelssave_path = "$(models_path)/$(dataset)_test_$(retro ? "rs" : "fr")_NN_$(best_idx)";

mkpath(figsave_path)
mkpath(modelssave_path)

@load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;

best_nn = neural_network_parameters[best_idx];
best_ode_p = ode_params[best_idx]; # log version where 1 is the index of the best model in info_output
pguess = vec(mean(reshape(best_ode_p, :, N_params), dims=1));

@info "Initial: $(exp.(pguess))"

if N_params == 5
    lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
    # lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1.0]); # 5.0, 5.0, 300.0, 400.0, 1
    lhs_ub = log.([10.0, 10.0, 500.0, 500.0, 1.0]); # 5.0, 5.0, 300.0, 400.0, 1
else
    lhs_lb = log.([0.001, 0.001, 0.001, 0.001]); # 0.001, 0.001, 0.01, 0.01
    lhs_ub = log.([5.0, 5.0, 500.0, 500.0]); # 5.0, 5.0, 300.0, 400.0
end

optfunc = OptimizationFunction(patient_loss, AutoForwardDiff());

min_keep_sample = 6;

Random.seed!(42); 

u0_init = [exp(pguess[3]), exp(pguess[4]), 0.0];

for base_choise_idx in eachindex(gold_std_patients)
    base_patient = gold_std_patients[base_choise_idx];
    base_patient_id = "s42$(base_patient.id)";

    @info "Selected patient $(base_patient.id) with $(length(base_patient.timepoints)) measurements"

    # n_sample = length(base_patient.timepoints)-5; # Retrospec
    n_sample = length(base_patient.timepoints) - min_keep_sample; # Forecasting
    @info "N sample: $n_sample"
    # sample_s = [sample(2:Int64(round(length(base_patient.timepoints)/2)), 2, replace=false) for _ in 1:n_sample]
    # sample_s = [sort(sample(1:length(base_patient.timepoints), 6, replace=false)) for _ in 1:n_sample]
    if retro
        sample_s = StatsBase.sample(1:length(base_patient.timepoints)-min_keep_sample, n_sample, replace=false) # Retrospec
    else
        sample_s = StatsBase.sample(length(base_patient.timepoints)-min_keep_sample:length(base_patient.timepoints), n_sample, replace=false) # Forecast
    end

    println(sample_s)

    rec_patients = PatientData[];
    for (i, samp) in enumerate(sample_s)

        # selected_p = [1, samp..., collect(length(base_patient.timepoints)-4:length(base_patient.timepoints))...];
        # selected_p = [samp..., length(base_patient.timepoints)];
        if retro
            selected_p = [samp, collect(length(base_patient.timepoints)-min_keep_sample+1:length(base_patient.timepoints))...]; # Retrospec
        else
            selected_p = [collect(1:length(base_patient.timepoints)-n_sample)..., samp]; # Forecast
        end
        selected_p
        push!(rec_patients, PatientData("$(base_patient_id)_$(i)", base_patient.timepoints[selected_p], base_patient.ctnt_data[selected_p]))

    end

    exp_run = "res/$(experiment)/$(retro ? "RetrospectiveSubsamplingValidation" : "ForecastingValidation")/$(base_patient_id)";
    exp_models = "$exp_run/models";
    exp_figs = "$exp_run/figs";

    mkpath(exp_models)
    mkpath(exp_figs)

    save_df = fromPatientData2DataFrame(rec_patients; save=true, save_path="$(exp_models)/df_$(base_patient_id).csv")

    open("$(exp_run)/info_output.txt", "w") do io
        println(io, "Experiment $(exp_run) log file")
        println(io, "dataset: $(file_path)")
        println(io, "UB: $(exp.(lhs_ub))")
        println(io, "Selected patient for sampling: $(base_patient.id)")
        println(io, "Gold standard characteristics:
        meas_min_number = $meas_min_number;
        min_acq_time_before = $min_acq_time_before;
        min_acq_n_before = $min_acq_n_before;
        min_acq_time_after = $min_acq_n_before;
        min_acq_n_after = $min_acq_n_after;
        min_time = $min_time;
        max_gap = $max_gap")
    end

    @info "Estimating curves and params for test set..."

    smape_values = [];
    rmsle_values = [];
    loss_values = [];
    validation_params = [];

    ev_bar = Progress(length(rec_patients); desc = "Validating", color = :cyan, showspeed = true);
    for (i, patient) in enumerate(rec_patients)

        # tspan = (0.0, base_patient.timepoints[end] + 20.0);
        # model = UDE ? ctntUDEModel(pguess, chain, tspan) : ctntCUDEModel(pguess, chain, tspan); # check i cUDE or UDE

        model = UDE ? ctntUDEModel(pguess, chain, base_patient.timepoints) : ctntCUDEModel(pguess, chain, base_patient.timepoints); # check i cUDE or UDE

        optprob = OptimizationProblem(optfunc, pguess,
            (model, patient.timepoints, patient.ctnt_data, best_nn),
            lb = lhs_lb, ub = lhs_ub
            );
            
        optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()),
                    maxiters=10000);

        p_opt = ComponentArray(ode = optsol_lbfgs.u, neural = best_nn);

        @info "For $(patient.id), params: $(p_opt.ode))"
        @info "Params: $(exp.(p_opt.ode)))"
        push!(validation_params, p_opt.ode)
        # push!(optsols_valid, optsol_lbfgs);

        u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]
        prob   = remake(model.problem; u0 = u0_new, p = p_opt)

        opt_model = ctntUDEModel(prob, chain);

        sol = Array(solve(prob, Tsit5(); p=p_opt, saveat=1));
        @info "Patient loss: $(patient_loss(p_opt.ode, (opt_model, base_patient.timepoints, base_patient.ctnt_data, p_opt.neural))))"
        push!(loss_values, optsol_lbfgs.objective)
        @info "Objective: $(optsol_lbfgs.objective))"

        validation_metric = smape_loss(p_opt.ode, (opt_model, base_patient.timepoints, base_patient.ctnt_data, p_opt.neural));
        @info "sMAPE: $(validation_metric)"

        open("$(exp_run)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
            println(io, "Patient $(patient.id) sMAPE NN validation: $validation_metric")
        end

        push!(smape_values, validation_metric)

        pred = sol[3,:];

        rmsle_res = rmsle(pred[3,:], base_patient.ctnt_data)
        push!(rmsle_values, rmsle_res)
        @info "RMSLE: $(rmsle_res)"

        plt = Plots.plot(pred; lw=2, label="cTnT simulation patient $(patient.id)", xlabel="Time", ylabel="Troponin", title="Patient $(patient.id)")
        Plots.scatter!(plt, base_patient.timepoints, base_patient.ctnt_data, label = "Removed datapoints", markershape = :x)
        Plots.scatter!(plt, patient.timepoints, patient.ctnt_data, ms=5, label="Used measurements")
        Plots.plot!(plt, legend = false)

        if plotting
            display(plt)
        end

        save("$(exp_figs)/patient_$(patient.id)$(UMG_data ? "_test" : "").svg", plt)
        next!(ev_bar)
    end

    ode_params_val = vcat(validation_params...);

    @info "Process ended succesfully"
end

    # for patient in rec_patients
    #     smape_res = 0;
    #     id = patient.id
    #     @info "Patient: $(id)"
    #     t_data = patient.timepoints
    #     x_data = patient.ctnt_data

    #     # placeholder
    #     tspan = (0.0, base_patient.timepoints[end] + 20.0)

    #     u0 = [pguess[end], pguess[end-1], 0]
    #     prob = ODEProblem(troponin_ode!, u0, tspan, pguess)

    #     # Solve the ODE problem with our guessed parameter values
    #     initial_sol = solve(prob)

    #     loss = RobustLosses.make_log_loss(prob, x_data, t_data)

    #     N=40

    #     # println("Treads: ", Threads.nthreads())
    #     best_result, all_results = MultiStartOptimizer.run_multistart(loss, N;
    #         lower=lower, upper=upper,
    #         # callback=callback,
    #         rng=StableRNG(1234),
    #         verbose=false,
    #         maxiters=1000,
    #         maxtime=80.0
    #         )

    #     best_result
    #     if best_result !== nothing
    #         best_params = best_result.u
    #         u0 = [best_params[end], best_params[end-1], 0]
    #         newprob = remake(prob, p = best_params, u0=u0)
    #         pred = solve(newprob, Tsit5(); saveat=base_patient.timepoints, abstol=1e-10, reltol=1e-8) # abstol=1e-10, reltol=1e-8
    #         sol = solve(newprob, Tsit5(); abstol=1e-10, reltol=1e-8) # abstol=1e-10, reltol=1e-8
    #         # pred = [sol(t)[3] for t in t_data]
    #         plt = Plots.plot(sol; idxs=3, label = "cTnT simulation patient $id")
    #         Plots.scatter!(plt, base_patient.timepoints, base_patient.ctnt_data, label = "Removed datapoints", markershape = :x)
    #         Plots.scatter!(plt, t_data, x_data, label = "Used measurements")
    #         Plots.plot!(plt, legend = false)

    #         # interp_linear = LinearInterpolation([0.0, base_patient.ctnt_data...], [0.0, base_patient.timepoints...])
    #         # Plots.plot!(plt, tspan, interp_linear.(tspan), label="Original inear interpolation", linestyle=:dash)

    #         if plotting
    #             display(plt)
    #         end
    #         savefig(plt, "$(fig_path)/$(patient.id).svg")

    #         smape_res = smape(pred[3,:], base_patient.ctnt_data);
    #         rmsle_res = rmsle(pred[3,:], base_patient.ctnt_data)
    #         # smape_res = smape(pred, x_data);
    #         push!(optimized_params, best_params)
    #     else
    #         @warn "No result set found."
    #         best_params = nothing
    #     end
    #     println("Patient $id - sMAPE: $smape_res")
    #     println("Patient $id - RMSLE: $rmsle_res")
    #     push!(smape_values, smape_res)
    #     push!(rmsle_values, rmsle_res)
    #     open("$(exp_run)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    #         println(io, "Patient: $id   |   smape: $smape_res   |   rmsle: $rmsle_res   |   params: $(best_params)")
    #     end
    # end
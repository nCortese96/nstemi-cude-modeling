using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter
using Statistics
using Dates
using Logging
using CairoMakie
# using Base.Threads: @threads, nthreads

@info "⚠️ Test NN algorithm started $(now())"

include("ctnt-ude-model.jl")

########### SET THIS PARAMETER FOR VALIDATION/TEST as FALSE/TRUE ###############################################
UMG_data = false;
########### SET THIS PARAMETER FOR VALIDATION/TEST as FALSE/TRUE ###############################################
dataset = "MIMIC-IV"; # "MIMIC-IV"; # "UMG" # used only for logging
if UMG_data
    dataset = "UMG"
end
UDE = false; # false for cUDE
best_idx = 1; # index of the best model to test 
bounds = true; # true for bounds, false for no bounds
λ_back = 0.0;

if UDE
    @info "Using UDE model"
    input_dim = 1
    nn_depth = 2
    nn_width = 8
    N_params = 4
    inputs_str = "τ"
else
    @info "Using cUDE model"
    input_dim = 2
    nn_depth = 2
    nn_width = 8
    N_params = 5
    inputs_str = "τ, β"
end

T_SCALE = 240.0;

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

figsave_path = "$(fig_path)/$(dataset)_test_NN_$(best_idx)";
modelssave_path = "$(models_path)/$(dataset)_test_NN_$(best_idx)";

# if bounds
#     figsave_path = "$(figsave_path)_larger_bounds"
#     modelssave_path = "$(modelssave_path)_larger_bounds"
# end

mkpath(figsave_path)
mkpath(modelssave_path)

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Test NN algorithm Started")
#     println(io, "********************************")
# end

test_dataset = if UMG_data
    @info "Using UMG dataset for testing"
    dataset = "UMG"

    # figsave_path = "$(fig_path)/umg_test_nn_$(best_idx)";
    # modelssave_path = "$(models_path)/umg_test_nn_$(best_idx)";   

    # mkpath(figsave_path)
    # mkpath(modelssave_path)
    try
        @info "Try to load UMG test"

        @load "$(modelssave_path)/UMG_testset.jld2" test_dataset
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
        savefig("$(figsave_path)/umg_distributions.svg")

        open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
            println(io, "$dataset - Test NN started $(now())")
            println(io, "   Patient loaded: ", nrow(ids))
            println(io, "   Patients after cleaning: $(length(cleaned_patients))")
            println(io, "   Time: min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h")
            println(io, "   cTnT: min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL")
        end

        plt = scutter_patients(cleaned_patients)
        # display(plt)
        savefig("$(figsave_path)/scatter_post.svg")

        test_dataset = cleaned_patients
        @save "$(modelssave_path)/UMG_testset.jld2" test_dataset
        @info "UMG test dataset saved"
        test_dataset
    end

else
    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset
    @info "Using $dataset dataset for testing"
    test_dataset
end;

@load "$(models_path)/trainingsetNSTEMI_$(experiment).jld2" training_dataset;

# @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
@load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters;
@load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;
# @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;

best_nn = neural_network_parameters[best_idx];
best_ode_p = ode_params[best_idx]; # log version where 1 is the index of the best model in info_output
reshaped_params = reshape(best_ode_p, :, N_params);
mean_pguess = vec(mean(reshaped_params, dims=1));
std_pguess = vec(std(exp.(reshaped_params), dims=1));
median_pguess = vec(median(exp.(reshaped_params), dims=1));
q3_pguess = vec([quantile(exp.(reshaped_params[:, i]), 0.75) for i in 1:N_params]);
q1_pguess = vec([quantile(exp.(reshaped_params[:, i]), 0.25) for i in 1:N_params]);

# println("Parameters average, STD: ", exp.(mean_pguess), ", ", std_pguess)
# println("Parameters median, [Q1, Q3]: ", median_pguess, " [", q1_pguess, ", ", q3_pguess, "]")

pguess = log.(median_pguess);

@info "Selected as pguess: $(exp.(pguess))"

a = []
b = []
Cs0 = []
Cc0 = []
β = []
ΔCsc0 = []

@showprogress desc = "Training prams extraction..." for (i, patient) in enumerate(training_dataset)

    idx1 = N_params * (i - 1) + 1
    idx2 = N_params * i
    ode_p = best_ode_p[idx1:idx2]
    p = ComponentArray(ode=ode_p, neural=best_nn)

    push!(a, exp(p.ode[1]))
    push!(b, exp(p.ode[2]))
    push!(Cs0, exp(p.ode[3]))
    push!(Cc0, exp(p.ode[4]))
    if N_params == 5
        push!(β, exp(p.ode[5]))
    end
    push!(ΔCsc0, exp(p.ode[3]) - exp(p.ode[4]))

end

# t_norm = range(0.0, 1.0, length=200)
t_span_grid = 0.1:0.1:400  # alcuni valori tipici del tuo β
β_vals = 0.1:0.1:1.0
p = Plots.plot()
for β in β_vals
    y = [chain([t / T_SCALE, β], best_nn)[1] for t in t_span_grid]
    Plots.plot!(p, t_span_grid, y, label="β = $β", linewidth=2)
end
Plots.plot!(p, xlabel="Time (h)", ylabel="rupture f(t_norm,β)", title="Learned sarcomere rupture function")
save("$(figsave_path)/correction_function.png", p)
# Plots.plot!(p, legend=false)
display(p)

# @info "Max : Min value in training param a: $(maximum(a)) : $(minimum(a))"
# @info "Max : Min value in training param b: $(maximum(b)) : $(minimum(b))"
# @info "Max : Min value in training param Cs0: $(maximum(Cs0)) : $(minimum(Cs0))"
# @info "Max : Min value in training param Cc0: $(maximum(Cc0)) : $(minimum(Cc0))"
# if N_params == 5
#     @info "Max : Min value in training param β: $(maximum(β)) : $(minimum(β))"
# end
# @info "Max : Min value in training param ΔCsc0: $(maximum(ΔCsc0)) : $(minimum(ΔCsc0))"

@info "Median [Q1-Q3] in training param a: $(median(a)) [$(quantile(a, 0.25)) - $(quantile(a, 0.75))]"
@info "Median [Q1-Q3] in training param b: $(median(b)) [$(quantile(b, 0.25)) - $(quantile(b, 0.75))]"
@info "Median [Q1-Q3] in training param Cs0: $(median(Cs0)) [$(quantile(Cs0, 0.25)) - $(quantile(Cs0, 0.75))]"
@info "Median [Q1-Q3] in training param Cc0: $(median(Cc0)) [$(quantile(Cc0, 0.25)) - $(quantile(Cc0, 0.75))]"
if N_params == 5
    @info "Median [Q1-Q3] in training param β: $(median(β)) [$(quantile(β, 0.25)) - $(quantile(β, 0.75))]"
end
@info "Median [Q1-Q3] in training param ΔCsc0: $(median(ΔCsc0)) [$(quantile(ΔCsc0, 0.25)) - $(quantile(ΔCsc0, 0.75))]"

@info "Average, STD in training param a: $(mean(a)) std: $(std(a))"
@info "Average, STD in training param b: $(mean(b)) std: $(std(b))"
@info "Average, STD in training param Cs0: $(mean(Cs0)) std: $(std(Cs0))"
@info "Average, STD in training param Cc0: $(mean(Cc0)) std: $(std(Cc0))"
if N_params == 5
    @info "Average, STD in training param β: $(mean(β)) std: $(std(β))"
end
@info "Average, STD in training param ΔCsc0: $(mean(ΔCsc0)) std: $(std(ΔCsc0))"

params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β];

@info "Boxplotting params"

par_names = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"];

x = vcat([fill(1, length(a))]...);

f = Figure(
    size=(1400, 700), # input
);

Label(
    f[0, 1:length(par_names)],
    "Parameter distributions training — $dataset dataset", ;
    fontsize=22,
    tellwidth=false
);

axes = [];

# max_y_values = [2, 1.5, 200.0, 250.0, 1.0];

@showprogress desc = "Generating axes..." for (i, p) in enumerate(par_names)
    ax = Axis(f[1, i],
        title=p,
        xticklabelsvisible=false, # Nasconde i numeri (1, 2...)
        xticksvisible=false,      # Nasconde le tacchette
        # limits = (nothing, nothing, nothing, max_y_values[i])
    )
    push!(axes, ax)
    # push!(axes, (Axis(f[1, i], title = p)))
end

my_colors = [:skyblue, :orange, :lightgreen, :pink, :violet];
@showprogress desc = "Generating boxplots..." for (i, (ax, p)) in enumerate(zip(axes, params))
    current_color = my_colors[mod1(i, length(my_colors))]
    # i = (i-1)+1;
    CairoMakie.boxplot!(
        ax, x, p;
        # color = x, 
        # width = 0.5,
        # mediancolor = :red,
        # whiskercolor = :gray,
        # outliercolor = :green,
        # show_notch = true
        # color = :skyblue,
        color=current_color,
        whiskerwidth=0.3,
        strokewidth=0.3,
        show_outliers=true
    )
    # ax.xticks = (1:length(exps), exps_names);
    # ax.xticklabelrotation = pi/3;
    # autolimits!(ax)
end

display(f)
save("$(figsave_path)/training_params_distribution.svg", f)

if N_params == 5
    lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001]) # 0.001, 0.001, 0.01, 0.01, 0.001
    # lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1.0]); # 5.0, 5.0, 300.0, 400.0, 1
    lhs_ub = log.([5.0, 5.0, 500.0, 500.0, 1.0]) # 5.0, 5.0, 300.0, 400.0, 1
else
    lhs_lb = log.([0.001, 0.001, 0.001, 0.001]) # 0.001, 0.001, 0.01, 0.01
    lhs_ub = log.([5.0, 5.0, 500.0, 500.0]) # 5.0, 5.0, 300.0, 400.0
end

# ode_p = best_solution;
# pguess = mean([optsol.u for optsol in ode_p]);

# optfunc = OptimizationFunction(patient_loss, AutoForwardDiff());
optfunc = OptimizationFunction(
    (p, data) -> patient_loss(p, data; λ_back=λ_back),
    # (p, data) -> serial_training_loss(p, data; n_params=N_params),
    AutoForwardDiff()
); # training_loss AutoForwardDiff() AutoZygote()

println("*********************************")
println("$dataset - Evaluating NN with sMAPE (idx:$(best_idx))")

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "*********************************")
    println(io, "$dataset - Evaluating NN with sMAPE (idx:$(best_idx))$(bounds ? " larger bounds" : " no bounds")")
end

smape_values = [];
rmsle_values = [];
loss_values = [];
validation_params = [];

# u0_init = [exp(pguess[3]), exp(pguess[4]), 0.0];

ev_bar = Progress(length(test_dataset); desc="Validating", color=:cyan, showspeed=true);
for (i, patient) in enumerate(test_dataset)

    # tspan = (0.0, patient.timepoints[end] + 10.0);
    # model = UDE ? ctntUDEModel(pguess, chain, tspan) : ctntCUDEModel(pguess, chain, tspan); # check i cUDE or UDE

    @info "Patient $(patient.id)"

    model = ctntCUDEModel(
        pguess,
        chain,
        patient.timepoints;
        # norm_type = 0
    ) # check i cUDE or UDE

    if bounds
        optprob = OptimizationProblem(optfunc, pguess,
            (model, patient.timepoints, patient.ctnt_data, best_nn),
            lb=lhs_lb, ub=lhs_ub
        )
    else
        optprob = OptimizationProblem(optfunc, pguess,
            (model, patient.timepoints, patient.ctnt_data, best_nn)
        )
    end
    optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()),
        maxiters=1000)

    p_opt = ComponentArray(ode=optsol_lbfgs.u, neural=best_nn)

    println("For $(patient.id), params: ", p_opt.ode)
    println("Params: ", exp.(p_opt.ode))
    push!(validation_params, p_opt.ode)
    # push!(optsols_valid, optsol_lbfgs);

    u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]
    prob = remake(model.problem; u0=u0_new, p=p_opt)

    opt_model = ctntUDEModel(prob, chain)

    pred = solve(prob, Tsit5(); p=p_opt, saveat=patient.timepoints)
    println(pred.retcode)
    sol = solve(prob, Tsit5(); p=p_opt, saveat=1)

    println("Patient loss: ", patient_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural)))
    # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
    push!(loss_values, optsol_lbfgs.objective)
    println("Objective:    ", optsol_lbfgs.objective)
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
    Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

    pl_plasm = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
    Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

    display(pl)
    display(pl_plasm)

    save("$(figsave_path)/patient_$(patient.id)$(dataset).svg", pl)
    save("$(figsave_path)/patient_$(patient.id)$(dataset)_plasm.svg", pl_plasm)
    next!(ev_bar)
end

ode_params_val = vcat(validation_params...);
@save "$(modelssave_path)/best_params_val_$(dataset).jld2" ode_params_val;

a = []
b = []
Cs0 = []
Cc0 = []
β = []

@showprogress desc = "Training prams extraction..." for (i, patient) in enumerate(test_dataset)

    idx1 = N_params * (i - 1) + 1
    idx2 = N_params * i
    local ode_p = ode_params_val[idx1:idx2]
    local p = ComponentArray(ode=ode_p, neural=best_nn)

    push!(a, exp(p.ode[1]))
    push!(b, exp(p.ode[2]))
    push!(Cs0, exp(p.ode[3]))
    push!(Cc0, exp(p.ode[4]))
    if N_params == 5
        push!(β, exp(p.ode[5]))
    end

end

@info "Average, STD in validation param a: $(mean(a)) std: $(std(a))"
@info "Average, STD in validation param b: $(mean(b)) std: $(std(b))"
@info "Average, STD in validation param Cs0: $(mean(Cs0)) std: $(std(Cs0))"
@info "Average, STD in validation param Cc0: $(mean(Cc0)) std: $(std(Cc0))"
if N_params == 5
    @info "Average, STD in validation param β: $(mean(β)) std: $(std(β))"
end

@info "Median [Q1-Q3] in validation param a: $(median(a)) [$(quantile(a, 0.25)) - $(quantile(a, 0.75))]"
@info "Median [Q1-Q3] in validation param b: $(median(b)) [$(quantile(b, 0.25)) - $(quantile(b, 0.75))]"
@info "Median [Q1-Q3] in validation param Cs0: $(median(Cs0)) [$(quantile(Cs0, 0.25)) - $(quantile(Cs0, 0.75))]"
@info "Median [Q1-Q3] in validation param Cc0: $(median(Cc0)) [$(quantile(Cc0, 0.25)) - $(quantile(Cc0, 0.75))]"
if N_params == 5
    @info "Median [Q1-Q3] in validation param β: $(median(β)) [$(quantile(β, 0.25)) - $(quantile(β, 0.75))]"
end

params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β];

@info "Boxplotting params"

par_names = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"];

x = vcat([fill(1, length(a))]...);

f = Figure(
    size=(1400, 700), # input
);

Label(
    f[0, 1:length(par_names)],
    "Parameter distributions validation — $dataset dataset", ;
    fontsize=22,
    tellwidth=false
);

axes = [];

# max_y_values = [15, 5, 200.0, 200.0, 3.0];

@showprogress desc = "Generating axes..." for (i, p) in enumerate(par_names)
    ax = Axis(f[1, i],
        title=p,
        xticklabelsvisible=false, # Nasconde i numeri (1, 2...)
        xticksvisible=false,      # Nasconde le tacchette
        # limits=(nothing, nothing, nothing, max_y_values[i])
    )
    push!(axes, ax)
    # push!(axes, (Axis(f[1, i], title = p)))
end

my_colors = [:skyblue, :orange, :lightgreen, :pink, :violet];
@showprogress desc = "Generating boxplots..." for (i, (ax, p)) in enumerate(zip(axes, params))
    current_color = my_colors[mod1(i, length(my_colors))]
    # i = (i-1)+1;
    CairoMakie.boxplot!(
        ax, x, p;
        # color = x, 
        # width = 0.5,
        # mediancolor = :red,
        # whiskercolor = :gray,
        # outliercolor = :green,
        # show_notch = true
        # color = :skyblue,
        color=current_color,
        whiskerwidth=0.3,
        strokewidth=0.3,
        show_outliers=true
    )
    # ax.xticks = (1:length(exps), exps_names);
    # ax.xticklabelrotation = pi/3;
    # autolimits!(ax)
end

display(f)
save("$(figsave_path)/validation_params_distribution_$(dataset).svg", f)

println("--> Average - STD sMAPE: $(mean(smape_values)) - $(std(smape_values))")
println("--> Median [Q1-Q3] sMAPE: $(median(smape_values)) [$(quantile(smape_values, 0.25)) - $(quantile(smape_values, 0.75))]")
println("--> Average - STD loss: $(mean(loss_values)) - $(std(loss_values))")
println("--> Median [Q1-Q3] loss: $(median(loss_values)) [$(quantile(loss_values, 0.25)) - $(quantile(loss_values, 0.75))]")

println("--> Average - STD RMSLE: $(mean(rmsle_values)) - $(std(rmsle_values))")
println("--> Median [Q1-Q3] RMSLE: $(median(rmsle_values)) [$(quantile(rmsle_values, 0.25)) - $(quantile(rmsle_values, 0.75))]")

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Average - STD sMAPE: $(mean(smape_values)) - $(std(smape_values))")
    println(io, "--> Median [Q1-Q3] sMAPE: $(median(smape_values)) [$(quantile(smape_values, 0.25)) - $(quantile(smape_values, 0.75))]")
    println(io, "--> Average - STD loss: $(mean(loss_values)) - $(std(loss_values))")
    println(io, "--> Median [Q1-Q3] loss: $(median(loss_values)) [$(quantile(loss_values, 0.25)) - $(quantile(loss_values, 0.75))]")
    println(io, "--> Average - STD RMSLE: $(mean(rmsle_values)) - $(std(rmsle_values))")
    println(io, "--> Median [Q1-Q3] RMSLE: $(median(rmsle_values)) [$(quantile(rmsle_values, 0.25)) - $(quantile(rmsle_values, 0.75))]")
end


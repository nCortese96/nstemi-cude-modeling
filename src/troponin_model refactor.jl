using Optimization, SciMLBase, SciMLSensitivity
# using OptimizationPolyalgorithms
# using ForwardDiff, 
using QuasiMonteCarlo
using StableRNGs, Plots
using StatsBase
using DataFrames, XLSX, CSV
using Logging, Dates
using Revise
# using SciMLBase: successful_retcode

includet("MultiStartOptimizer.jl")
includet("ctnt-ude-model.jl")
# using .MultiStartOptimizer
# using .DataUtils

# 0 - MIMIC-IV NSTEMI
# 1 - UMG NSTEMI
# 2 - UMG STEMI

println("⚠️ Algorithm ODE started $(now())")

dataset_id = 1; # change here for different datasets
plotting = true; # set true to enable plotting of each patient during optimization and residual calculation
λ_back = 1.0; # regularization weight for backward compatibility with previous versions (set to 0.0 to disable)

if dataset_id == 0
    dataset_name = "MIMIC-IV";
    dataset_path = "MIMIC-IV/NSTEMI_reorganized_skipped.xlsx";
    column_letter = "B";
elseif dataset_id == 1
    dataset_name = "UMG";
    dataset_path = "UMG_NSTEMI_Dataset.xlsx";
    column_letter = "A";
# elseif dataset_id == 2
#     dataset_name = "UMG_STEMI";
#     dataset_path = "UMG_STEMI_Dataset.xlsx";
#     column_letter = "A";
end

file_path = "data/$(dataset_path)"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped UMG_STEMI_Dataset
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

xf = XLSX.readxlsx(file_path);
# Caricamento dei fogli in DataFrame
# ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true)); #for UMG_NSTEMI_Dataset use sheet_times
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$column_letter:$column_letter", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

pguess = log.([0.005, 0.005, 0.01, 0.01, 30.0]); # [a, b, Cs0, Cc0, Td]
lower = log.([0.001, 0.001, 0.001, 0.001, 0.001]);
upper = log.([10.0, 10.0, 500.0, 500.0, 500.0]);

# experiment = "Exp$(exp_n)_$(dataset_name)_bounds";
experiment = "NSTEMI_ODE_TdSigmoid/$(dataset_name)_opt_lambda$(Int(λ_back))";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
mkpath(fig_path)
mkpath(models_path)
open("res/$(experiment)/info_output.txt", "w") do io
    println(io, "Experiment $(experiment) log file")
    println(io, "dataset: $(file_path)")
    println(io, "UB: $(upper)")
end

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

dup_counts = patient_duplicate_time_counts(patients; tol=0.0)
(nmod, nrm) = collapse_duplicates!(patients, dup_counts; agg=mean, tol=0.0)

# Trimming to T_SCALE
T_SCALE = 240.0;
trimmed_p = trim_time(patients, T_SCALE);
patient_dims(trimmed_p)

#0. Pre-processing
# scripts/ae_hi_splitting.jl

# 0. Select patients from ID file produced by ae_hi_splitting
ids_csv_path = joinpath("res", "ids_all_eligible_$(dataset_name).csv")
@assert isfile(ids_csv_path) "ID file not found: $(ids_csv_path). Run scripts/ae_hi_splitting.jl first"

ids_df = CSV.read(ids_csv_path, DataFrame)
allowed_ids = Set(string.(ids_df.patient))

cleaned_patients = filter(p -> p.id in allowed_ids, trimmed_p)

println("Total IDs in file: $(length(allowed_ids))")
println("Total samples after trim + ID filter: $(length(cleaned_patients))")
patient_dims(cleaned_patients)

missing_after_trim = setdiff(allowed_ids, Set(p.id for p in trimmed_p))
if !isempty(missing_after_trim)
    @warn "IDs present in file but missing after trim_time: $(length(missing_after_trim))"
end

# println("Campioni rimossi in totale: $(length(anoms))")

# cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p);
# patient_dims(cleaned_patients)
println("total sample: $(length(cleaned_patients)) patients.")

all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(cleaned_patients);
display(dist)
# savefig("$(fig_path)/dataset_distributions.svg")

@info "Dataset Loaded. Patients: $(length(cleaned_patients))"

@info "Estimating curves and params for $(dataset_name) processing..."

function callback(p, l)
    display(l)
    u0 = [exp(p[3]), exp(p[4]), 0.0]
    newprob = remake(prob, p = p, u0=u0)
    sol = solve(newprob, Tsit5(); saveat = t_data, abstol=1e-10, reltol=1e-8) # abstol=1e-10, reltol=1e-8
    plt = plot(sol[3,:], label = "cTnT simulation")
    scatter!(plt, t_data, x_data, label = "Data")
    display(plt)
    return false
end

softplus_stable(x) = log1p(exp(-abs(x))) + max(x, 0)
relu_smooth(x; κ=0.05) = κ * softplus_stable(x / κ)

successful_idx = Int[]
smape_values = Float64[]
rmsle_values = Float64[]
loss_values = Float64[]

optimized_params = [];
residuals = [];

N_multistart=40

for (i, patient) in enumerate(cleaned_patients) #subset
    smape_res = NaN
    rmsle_res = NaN
    loss_res = NaN
    best_params = fill(NaN, length(pguess))
    id = patient.id
    @info "Patient: $(id)"
    t_data = patient.timepoints
    x_data = patient.ctnt_data

    # placeholder
    tspan = (0.0, t_data[end] + 10.0)
    
    u0 = [exp(pguess[3]), exp(pguess[4]), 0.0]
    prob = ODEProblem(troponin_ode!, u0, tspan, pguess)

    data = (prob, t_data, x_data)
    loss = θ -> patient_loss_formula(θ, data; λ_back=λ_back)

    best_result, all_results = MultiStartOptimizer.run_multistart(
        loss, N_multistart;
        lower = lower, upper = upper,
        # callback=callback,
        rng=StableRNG(1234),
        verbose=false,
        maxiters=1000,
        maxtime=80.0,
        prescreen = false,
        topk=8
        )

    best_result
    if best_result !== nothing
        best_params = best_result.u
        u0 = [exp(best_params[3]), exp(best_params[4]), 0.0] # Cs0, Cc0, Cp0 # default
        newprob = remake(prob, p = best_params, u0=u0)
        pred = solve(newprob, Tsit5(); saveat=t_data,
        # abstol=1e-10, reltol=1e-8
        )
        sol = solve(newprob, Tsit5();
        # abstol=1e-10, reltol=1e-8
        )
        # pred = [sol(t)[3] for t in t_data]

        pl = Plots.plot(sol; idxs=1, lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title="ODE - Patient $(patient.id)")
        Plots.plot!(pl, sol; idxs=2, lw=2, label="Cytosol")
        Plots.plot!(pl, sol; idxs=3, lw=2, label="Blood")
        Plots.scatter!(pl, t_data, x_data, ms=5, label="Observed Data", legend=:best)

        pl_plasm = Plots.plot(sol; idxs=3, lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
        Plots.scatter!(pl_plasm, t_data, x_data, ms=5, label="Observed Data", legend=:best)

        display(pl)
        display(pl_plasm)

        # save("$(figsave_path)/patient_$(patient.id)$(dataset_name).svg", pl)
        # save("$(figsave_path)/patient_$(patient.id)$(dataset_name)_plasm.svg", pl_plasm)
        savefig(pl, "$(fig_path)/patient_$(patient.id)_$(dataset_name).svg")
        savefig(pl_plasm, "$(fig_path)/patient_$(patient.id)_$(dataset_name)_plasm.svg")

        # plt = Plots.plot(sol; idxs=3, label = "cTnT simulation patient $id")
        # Plots.scatter!(plt, t_data, x_data, label = "Data")
        # if plotting
        #     display(plt)
        # end
        # savefig(plt, "$(fig_path)/$(patient.id).svg")

        smape_res = smape(pred[3,:], x_data);
        rmsle_res = rmsle(x_data, pred[3,:]);
        loss_res = best_result.minimum;

        push!(optimized_params, best_params)
        push!(successful_idx, i)
    else
        @warn "No result set found."
        # best_params = nothing
    end
    println("Patient $id - sMAPE: $smape_res")
    push!(smape_values, smape_res)
    push!(rmsle_values, rmsle_res)
    push!(loss_values, loss_res)
    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Patient: $id   |   smape: $smape_res   |   rmsle: $rmsle_res   |   loss: $loss_res   |   params: $(best_params)")
    end

end
@info "Finish!"
smape_values = filter(isfinite, smape_values)
rmsle_values = filter(isfinite, rmsle_values)
loss_values = filter(isfinite, loss_values)

if isempty(smape_values) || isempty(rmsle_values) || isempty(loss_values)
    error("No finite metrics available. Check optimization failures in info_output.txt")
end

smape_avg_val = mean(smape_values)
smape_std_val = std(smape_values)
smape_median_val = median(smape_values)
smape_q1_val = quantile(smape_values, 0.25)
smape_q3_val = quantile(smape_values, 0.75)
smape_iqr_val = smape_q3_val - smape_q1_val

rmsle_avg_val = mean(rmsle_values)
rmsle_std_val = std(rmsle_values)
rmsle_median_val = median(rmsle_values)
rmsle_q1_val = quantile(rmsle_values, 0.25)
rmsle_q3_val = quantile(rmsle_values, 0.75)
rmsle_iqr_val = rmsle_q3_val - rmsle_q1_val

loss_avg_val = mean(loss_values)
loss_std_val = std(loss_values)
loss_median_val = median(loss_values)
loss_q1_val = quantile(loss_values, 0.25)
loss_q3_val = quantile(loss_values, 0.75)
loss_iqr_val = loss_q3_val - loss_q1_val

println("Average - STD loss: $loss_avg_val")
println("Median [Q1-Q3] loss: $loss_median_val [$loss_q1_val - $loss_q3_val]")

println("Average - STD sMAPE: $smape_avg_val")
println("Median [Q1-Q3] sMAPE: $smape_median_val [$smape_q1_val - $smape_q3_val]")

println("Average - STD RMSLE: $rmsle_avg_val")
println("Median [Q1-Q3] RMSLE: $rmsle_median_val [$rmsle_q1_val - $rmsle_q3_val]")

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "Average - STD loss: $loss_avg_val")
    println(io, "Median [Q1-Q3] loss: $loss_median_val [$loss_q1_val - $loss_q3_val]")

    println(io, "Average - STD sMAPE: $smape_avg_val")
    println(io, "Median [Q1-Q3] sMAPE: $smape_median_val [$smape_q1_val - $smape_q3_val]")

    println(io, "Average - STD RMSLE: $rmsle_avg_val")
    println(io, "Median [Q1-Q3] RMSLE: $rmsle_median_val [$rmsle_q1_val - $rmsle_q3_val]")
end


info_path = "res/$(experiment)/info_output.txt"
csv_out_path = "$(models_path)/params_out.csv";
meta_path = "$(models_path)/meta.csv";

df = parse_log_to_csv(info_path; out_csv=csv_out_path, meta_csv=meta_path);
using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using Statistics
using StatsPlots
using Dates
include("ctnt-ude-model.jl")

# This file is not for optimization, is only for recalculate curves

################################# DATASET LOAD ####################################

# file_path = "data/UMG_NSTEMI_Dataset.xlsx"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
# sheet_ids = "IDs";
# sheet_times = "times";
# sheet_values = "values";

# xf = XLSX.readxlsx(file_path);
# # Caricamento dei fogli in DataFrame
# # ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
# ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true));
# timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
# troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

# patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

# ################################# DEFINED TEST SET LOAD ####################################

# test_ids = ["n34","n100","n62","n16","n6",
#             "n91","n78","n87","n10","n53",
#             "n8","n95","n92","n45","n38",
#             "n63","n85","n46","n79","n61"]


# test_set = filter(p -> p.id in test_ids, patients)

# lookup = Dict(p.id => p for p in patients)          # id → PatientData
# ordered_test_set = [lookup[id] for id in test_ids if haskey(lookup, id)]

# for p in ordered_test_set
#     println(p.id)
# end

################################# EXPERIMENT LOADING ####################################

input_dim = 2;
nn_depth = 2;
nn_width = 8;
inputs_str = "t, β";
if input_dim == 3
    inputs_str = "u[1], t, β";
elseif input_dim == 7
    inputs_str = "u[1], t, a, b, Cs0, Cc0, β";
end

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

T_SCALE = 350.0;

experiment = "NSTEMI_partrvalMIMIC_logSSEf_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

@load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;
@load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;

# ########### SET THIS PARAMETER FOR VALIDATION/TEST as FALSE/TRUE ###############################################
# UMG_data = false;
UMG = "";
# ########### SET THIS PARAMETER FOR VALIDATION/TEST as FALSE/TRUE ###############################################

# if UMG_data
#     UMG = "UMG";
#     figsave_path = "$(fig_path)/umg_test_nn";
#     modelssave_path = "$(models_path)/umg_test_nn";   
    
#     mkpath(figsave_path)
#     mkpath(modelssave_path)

#     file_path = "data/UMG_NSTEMI_Dataset.xlsx"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
#     sheet_ids = "IDs";
#     sheet_times = "times";
#     sheet_values = "values";
#     xf = XLSX.readxlsx(file_path);
#     # Caricamento dei fogli in DataFrame
#     # ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
#     ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true));
#     timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
#     troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

#     println("Patient loaded: ", nrow(ids))
#     patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

#     # Trimming to T_SCALE
#     trimmed_p = trim_time(patients, T_SCALE);
#     patient_dims(trimmed_p)

#     # 0. Pre-processing
#     meas_min_number = 5;
#     anoms = find_anomalies(trimmed_p, meas_min_number);
#     println("Removed: $(length(anoms))")

#     cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p);
#     patient_dims(cleaned_patients)
#     println("Total sample: $(length(cleaned_patients))")

#     all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(cleaned_patients);
#     display(dist)
#     savefig("$(figsave_path)/umg_distributions.svg")

#     open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
#         println(io, "UMG - Test NN started $(now())")
#         println(io, "   Patient loaded: ", nrow(ids))
#         println(io, "   Time: min = $(round(t_min, digits=2)) h   max = $(round(t_max, digits=2)) h")
#         println(io, "   cTnT: min = $(round(c_min, digits=4)) ng/mL   max = $(round(c_max, digits=2)) ng/mL")
#     end

#     plt = scutter_patients(cleaned_patients)
#     # display(plt)
#     savefig("$(figsave_path)/scatter_post.svg")

#     test_dataset = cleaned_patients;

# else
#     @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;
# end


open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "*********************************")
    println(io, "Evaluating NN with sMAPE")
end

# mkpath("$fig_path/evaluation")

smape_values = [];
# for (i, patient) in enumerate(ordered_test_set)
for (i, patient) in enumerate(test_dataset)
# patient = ordered_test_set[i]

    p_opt = ComponentArray(ode = best_solution[i].u, neural = best_nn);

    tspan = (0.0, patient.timepoints[end] + 10);

    opt_model = ctntCUDEModel(p_opt, chain, tspan);

    validation_metric = smape_loss(p_opt.ode, (opt_model, patient.timepoints, patient.ctnt_data, p_opt.neural))

    u0_new = [exp(p_opt.ode[3]), exp(p_opt.ode[4]), 0.0]

    prob   = remake(opt_model.problem; u0 = u0_new, p = p_opt)
    opt_new = ctntCUDEModel(prob, chain);

    sol = solve(opt_new.problem, AutoTsit5(Rosenbrock23()); p=p_opt, saveat=1.0)  

    pred = solve(opt_new.problem, AutoTsit5(Rosenbrock23()); p=p_opt, saveat=patient.timepoints) 

    pl = plot(sol[3, :]; lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
    scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

    display(pl)

    println("sMAPE: $validation_metric")
    open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
        println(io, "Patient $(patient.id) sMAPE NN validation: $validation_metric")
    end
    push!(smape_values, validation_metric)
end

println(median(smape_values))

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "--> Median in sMAPE NN validation: ", median(smape_values))
end

a_dist = [exp(sol.u[1]) for sol in best_solution]
b_dist = [exp(sol.u[2]) for sol in best_solution]
Cs0_dist = [exp(sol.u[3]) for sol in best_solution]
Cc0_dist = [exp(sol.u[4]) for sol in best_solution]
β_dist = [exp(sol.u[5]) for sol in best_solution]

dist_path = "distributions$(UMG)"

mkpath("$fig_path/$dist_path")

plt_a = histogram(a_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param a distribution",
                 legend = false)
savefig("$(fig_path)/$dist_path/a_dist.svg")

plt_b = histogram(b_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param b distribution",
                 legend = false)
savefig("$(fig_path)/$dist_path/b_dist.svg")

plt_Cs0 = histogram(Cs0_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param Cs0 distribution",
                 legend = false)
savefig("$(fig_path)/$dist_path/plt_Cs0_dist.svg")

plt_Cc0 = histogram(Cc0_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param Cc0 distribution",
                 legend = false)
savefig("$(fig_path)/$dist_path/plt_Cc0_dist.svg")

plt_β = histogram(β_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param β distribution",
                 legend = false)
savefig("$(fig_path)/$dist_path/plt_β_dist.svg")

boxplot([a_dist, b_dist, Cs0_dist, Cc0_dist, β_dist];
        labels = ["a" "b" "Cs0" "Cc0" "β"],
        title  = "Parameter distribution",
        ylabel = "Values")

p1 = boxplot([a_dist,b_dist,β_dist];
             labels = ["a","b","β"], outliers    = false,     # do not plot fliers
            whisker     = 1.5,       # Tukey default
            ylabel      = "Values",
            fillalpha   = 0.6)
p2 = boxplot([Cs0_dist,Cc0_dist];
             labels = ["Cs0","Cc0"],
             outliers    = false,     # do not plot fliers
            whisker     = 1.5,
            fillalpha   = 0.6)
plot(p1, p2; layout=(1,2), size=(800,400))
savefig("$(fig_path)/$dist_path/distbox.svg")
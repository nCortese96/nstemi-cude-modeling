using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using Statistics
using StatsPlots
include("ctnt-ude-model.jl")

################################# DATASET LOAD ####################################

file_path = "data/UMG_NSTEMI_Dataset.xlsx"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

xf = XLSX.readxlsx(file_path);
# Caricamento dei fogli in DataFrame
# ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

################################# DEFINED TEST SET LOAD ####################################

test_ids = ["n34","n100","n62","n16","n6",
            "n91","n78","n87","n10","n53",
            "n8","n95","n92","n45","n38",
            "n63","n85","n46","n79","n61"]


test_set = filter(p -> p.id in test_ids, patients)

lookup = Dict(p.id => p for p in patients)          # id → PatientData
ordered_test_set = [lookup[id] for id in test_ids if haskey(lookup, id)]

for p in ordered_test_set
    println(p.id)
end

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

experiment = "NSTEMI_logSSE_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
println(experiment)
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

@load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
@load "$(models_path)/best_solutionNSTEMI_$(experiment).jld2" best_solution;

smape_values = [];
for (i, patient) in enumerate(ordered_test_set)
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
        println(io, "Patient $(patient.id) sMAPE: $validation_metric")
    end
    push!(smape_values, validation_metric)
end

println(median(smape_values))

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "Median in sMAPE validation: ", median(smape_values))
end

a_dist = [exp(sol.u[1]) for sol in best_solution]
b_dist = [exp(sol.u[2]) for sol in best_solution]
Cs0_dist = [exp(sol.u[3]) for sol in best_solution]
Cc0_dist = [exp(sol.u[4]) for sol in best_solution]
β_dist = [exp(sol.u[5]) for sol in best_solution]

mkpath("$fig_path/distributions")

plt_a = histogram(a_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param a distribution",
                 legend = false)
savefig("$(fig_path)/distributions/a_dist.svg")

plt_b = histogram(b_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param b distribution",
                 legend = false)
savefig("$(fig_path)/distributions/b_dist.svg")

plt_Cs0 = histogram(Cs0_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param Cs0 distribution",
                 legend = false)
savefig("$(fig_path)/distributions/plt_Cs0_dist.svg")

plt_Cc0 = histogram(Cc0_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param Cc0 distribution",
                 legend = false)
savefig("$(fig_path)/distributions/plt_Cc0_dist.svg")

plt_β = histogram(β_dist;
                 bins = 10,
                 xlabel = "Values",
                 ylabel = "#",
                 title = "Param β distribution",
                 legend = false)
savefig("$(fig_path)/distributions/plt_β_dist.svg")

boxplot([a_dist, b_dist, Cs0_dist, Cc0_dist, β_dist];
        labels = ["a" "b" "Cs0" "Cc0" "β"],
        title  = "Parameter distributions",
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
savefig("$(fig_path)/distributions/distbox.svg")
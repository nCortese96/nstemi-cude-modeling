using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2

include("ctnt-ude-model.jl")

@load "theta_opt.jld2" θ_opt

println("Caricamento del dataset...")
# Percorso del file Excel
file_path = "data/STEMI_merged.xlsx";
sheet_times = "Tempi cleaned";
sheet_values = "Misurazioni cleaned";

# Caricamento dei fogli in DataFrame
ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "B:X", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "B:X", header=false, infer_eltypes=true));

initial_params = [log(0.005), log(0.005), log(0.1), log(0.001), log(0.1)];

# Costruisci l'array di PatientData iterando su tutte le righe.
n_rows = nrow(ids);
patients = [row_to_patient(i, ids, timepoints_df, troponin_df, initial_params) for i in 1:n_rows];

println("Numero di pazienti caricati: ", length(patients))

##############################################
# 2. Suddivisione in training e validation
##############################################
Random.seed!(1234);
shuffle!(patients);
n_train = Int(round(length(patients) * 0.8));
training_dataset = patients[1:n_train];
test_dataset = patients[n_train+1:end];

N_nn = 97 # oppure un valore noto, ad esempio 97
fixed_nn_params = θ_opt[1:N_nn]

# Supponiamo di prendere il primo paziente dal training_dataset:
i = 6
patient = training_dataset[i]
idx_start = N_nn + 5*(i-1) + 1  # per il primo paziente
idx_end   = N_nn + 5*i
patient_params = θ_opt[idx_start:idx_end]  # usa il guess iniziale, per esempio
tspan = (patient.timepoints[1], patient.timepoints[end])

chain = neural_network_model(2, 6; input_dims=7);

# Costruisci il modello per questo paziente:
model = ctntCUDEModel(patient_params, chain, tspan)
# p = ComponentArray(ode = patient_params, neural = fixed_nn_params)

optfunc = OptimizationFunction((patient_params, x) -> patient_loss(patient_params, model, patient.timepoints, patient.ctnt_data, fixed_nn_params), AutoForwardDiff())
lbfgs_maxiters = 1000
optprob2 = Optimization.OptimizationProblem(optfunc, patient_params);
opt_result2 = Optimization.solve(optprob2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=lbfgs_maxiters);
opt_result2

# Prova a risolvere l'ODE:
p = ComponentArray(ode = opt_result2.u, neural = fixed_nn_params)
sol = solve(model.problem, p=p, saveat=patient.timepoints)
pred = [u[3] for u in sol.u]
sum((pred .- patient.ctnt_data).^2)
# println("Dimensioni sol: ", size(Array(sol)))
println("Patient ctnt_data: ", patient.ctnt_data)

plot(patient.timepoints, pred, lw=2, label="Model Prediction", xlabel="Time", ylabel="CTNT", title="Patient $(patient.id)")
scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")
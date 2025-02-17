# Supponiamo di prendere il primo paziente dal training_dataset:
patient = training_dataset[1]
N_nn = length(nn_params_init)
idx_start = N_nn + 5*(1-1) + 1  # per il primo paziente
idx_end   = N_nn + 5*1
patient_params = θ_intermediate[idx_start:idx_end]  # usa il guess iniziale, per esempio
tspan = (patient.timepoints[1], patient.timepoints[end])

# Costruisci il modello per questo paziente:
model = ctntCUDEModel(patient_params, chain, tspan)
p = ComponentArray(ode = patient_params, neural = nn_params_init)

# Prova a risolvere l'ODE:
sol = solve(model.problem, p=p, saveat=patient.timepoints)
sum(abs2, sol[3,:] - patient.ctnt_data)
println("Dimensioni sol: ", size(Array(sol)))
println("Patient ctnt_data: ", patient.ctnt_data)
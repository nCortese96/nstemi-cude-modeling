"""
TODO: 
1. load and setup dataset
2. Split train/test
3. Create a model for each Patients
4. Training phase
4.1. Split in train/validation
4.2. solve the model -> nn_params, ode_params
4.3. select best model
5. Optimize specific parameters with fixed nn -> params from training
6. Optimize specific parameters with fixed nn -> params from test

"""

# test the code

using StableRNGs, DataFrames, StatsBase, XLSX, Random

include("ctnt-ude-model.jl")

rng = StableRNG(42)

# Load and Split - trin/test
file_path = "data/STEMI_merged.xlsx";
sheet_times = "Tempi cleaned";
sheet_values = "Misurazioni cleaned";

times_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:X", header=false, infer_eltypes=true));
ids = times_df[:, :A];
select!(times_df, Not(:A));
values_df = DataFrame(XLSX.readtable(file_path, sheet_values, "B:X", header=false, infer_eltypes=true));

println("Patients: ", nrow(values_df))

train_idx, test_idx = split_data(rng, nrow(values_df), 0.8)

train_ids = ids[train_idx, :]
train_times = times_df[train_idx, :]
train_values = values_df[train_idx, :]

test_ids = ids[test_idx, :]
test_times = times_df[test_idx, :]
test_values = values_df[test_idx, :]

train_model = true

chain = neural_network_model(2, 6)


# create the models

θ_init = [0.005, 0.005, 0.1, 0.001]

models_train = [
    ctntCUDEModel(collect(skipmissing(train_times[i,:])), chain, θ_init)
    for i in axes(train_times, 1)
]

if train_model

    indices_train, indices_validation = split_data(rng, length(train_ids), 0.8)
    
    optsols_train = train(models_train[indices_train], train_data.timepoints, train_data.cpeptide[indices_train,:], rng)
end
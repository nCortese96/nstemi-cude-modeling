using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter: Progress, next!

include("ctnt-ude-model.jl")

# Dataset loading

file_path = "data/ANN_dataset_IX.xlsx";
sheet_ids = "id";
sheet_times = "times";
sheet_values = "values";

# Caricamento dei fogli in DataFrame
# ids = DataFrame(XLSX.readtable(file_path, sheet_times, "A:A", header=false, infer_eltypes=true));
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "A:A", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "B:N", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "B:N", header=false, infer_eltypes=true));

perm = collect(1:nrow(ids))
Random.seed!(1234);
rng = StableRNG(42);
shuffle!(rng, perm)
n_train = Int(round(nrow(ids) * 0.8))

train_ids     = ids[1:n_train]
train_times   = timepoints_df[1:n_train]
train_values  = troponin_df[1:n_train]

test_ids      = ids[n_train, end]
test_times    = timepoints_df[n_train, end]
test_values   = troponin_df[n_train, end]
using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2
using ProgressMeter

println("Model selection started")

include("ctnt-ude-model.jl")

@load "res/models/optsolsNSTEMI_MAPE_0606log.jld2" optsols;
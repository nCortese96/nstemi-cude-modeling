using StableRNGs, DataFrames, StatsBase, Random
using Optimization, OptimizationOptimisers, LineSearches
using JLD2, CSV, XLSX
using ProgressMeter
using Statistics
using Dates
using Logging

using Revise

includet("ctnt-ude-model.jl")

@info "Starting residual calculation script"

smapes = hcat(opt_smapes...);
rmsles = hcat(opt_rmsles...);
objectives = hcat(opt_objectives...);

n_patients, n_models = size(smapes)

wins_smape = zeros(Int, n_models)
wins_rmsle = zeros(Int, n_models)

for i in 1:n_patients
    wins_smape[argmin(view(smapes, i, :))] += 1
    wins_rmsle[argmin(view(rmsles, i, :))] += 1
end

median_smape = vec(median(smapes, dims=1))
median_rmsle = vec(median(rmsles, dims=1))
median_obj = vec(median(objectives, dims=1))

println("Validation wins by sMAPE: ", wins_smape)
println("Validation wins by RMSLE: ", wins_rmsle)
println("Median sMAPE: ", median_smape)
println("Median RMSLE: ", median_rmsle)
println("Median objective: ", median_obj)

# ranking primario: più wins_sMAPE, poi mediana sMAPE, poi mediana RMSLE, poi objective
candidate_idxs = findall(==(maximum(wins_smape)), wins_smape)

if length(candidate_idxs) == 1
    best_model_index = only(candidate_idxs)
else
    med_smape_sub = median_smape[candidate_idxs]
    candidate_idxs = candidate_idxs[findall(==(minimum(med_smape_sub)), med_smape_sub)]

    if length(candidate_idxs) == 1
        best_model_index = only(candidate_idxs)
    else
        med_rmsle_sub = median_rmsle[candidate_idxs]
        candidate_idxs = candidate_idxs[findall(==(minimum(med_rmsle_sub)), med_rmsle_sub)]

        if length(candidate_idxs) == 1
            best_model_index = only(candidate_idxs)
        else
            obj_sub = median_obj[candidate_idxs]
            best_model_index = candidate_idxs[argmin(obj_sub)]
        end
    end
end

println("Best model selected by validation rule: ", best_model_index)
# @save "$(models_path)/best_model_index_$(experiment).jld2" best_model_index
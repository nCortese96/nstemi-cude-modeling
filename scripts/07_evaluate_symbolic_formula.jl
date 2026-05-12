"""
evaluate_symbolic_formula.jl

Refactored copy of `test_formula.jl`.

Evaluate the symbolic-regression surrogate formula against patient trajectories.

Pipeline:
1. Configure run settings.
2. Resolve input/output paths.
3. Load required data and model artifacts.
4. Run the main computation.
5. Save metrics, parameters, plots, and logs.

This copy uses `MechanisticAI.jl` as the shared helper entrypoint. The original
script is intentionally left untouched as the legacy baseline.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Shared dependencies and the central refactor entrypoint.
# =============================================================================
using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using Plots, JLD2, CSV
using ProgressMeter
using Statistics
using Dates
using Logging
include("MechanisticAI.jl")
using .MultiStartOptimizer

# =============================================================================
# SCRIPT SETTINGS
# User-editable dataset/model/optimizer flags are preserved from the original
# script in the first executable block below.
# =============================================================================

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run are resolved near the settings that define
# dataset/model identity. Keep load paths explicit during this transition pass.
# =============================================================================

# =============================================================================
# OUTPUT PATHS
# Result directories and output files are created by the preserved pipeline below.
# Future cleanup should move path construction into `build_experiment_paths`.
# =============================================================================

# =============================================================================
# DERIVED SETTINGS
# Values computed from the settings above are kept inline for behavior parity.
# Future cleanup should collect them before the pipeline starts.
# =============================================================================

# =============================================================================
# HELPERS
# Script-local helper functions remain near their original location for now.
# Reusable candidates should migrate to helpers.jl after behavior is validated.
# =============================================================================

# =============================================================================
# PIPELINE
# Main execution flow copied from the original script. This first refactor pass
# changes includes and documentation only; numerical behavior is preserved.
# =============================================================================
println("⚠️ Test formula algorithm started $(now())")

dataset_id = 0;
if dataset_id == 0
    dataset_name = "MIMIC-IV"
    UMG_data = false
elseif dataset_id == 1
    dataset_name = "UMG"
    UMG_data = true
else
    error("dataset_id must be 0 (MIMIC-IV) or 1 (UMG)")
end

UDE = false; # false for cUDE
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

λ_back = 1.0;
best_idx = 3; # index of the best model to test
T_SCALE = 240.0;
N_multistart = 40
use_multistart = N_multistart > 0
multistart_maxiters = 1000
multistart_maxtime = 80.0
multistart_rng = StableRNG(1234)

# experiment = "NSTEMI_partrvalMIMIC_SSEf_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
# experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus"
experiment = "NSTEMI_cUDE_MIMIC-IV_MSE_2$(nn_width)_sigmoid_regback";
fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";

run_tag = "$(dataset_name)_test_formula_$(best_idx)$(use_multistart ? "_ms_test" : "")"
figsave_path = "$(fig_path)/$(run_tag)"
modelssave_path = "$(models_path)/$(run_tag)"
mkpath(figsave_path)
mkpath(modelssave_path)

formula_fig_save = "$(figsave_path)/formula_04_28"
formula_models_save = "$(modelssave_path)/formula_04_28"

mkpath(formula_fig_save)
mkpath(formula_models_save)

# figsave_path = "$(fig_path)/test_formula";
# modelssave_path = "$(models_path)/test_formula";

# mkpath(figsave_path)
# mkpath(modelssave_path)

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "********************************")
#     println(io, "Test formula algorithm Started")
#     println(io, "********************************")
# end

# @load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn
@info "Loading dataset"
test_dataset = if UMG_data
    @info "Using UMG dataset for testing"

    @load "$(models_path)/UMG_testset.jld2" test_dataset
    @info "$dataset_name test dataset loaded from previous save"
    test_dataset
else
    @load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset
    @info "Using $dataset_name dataset for testing"
    test_dataset
end

@info "Dataset $dataset_name loaded with $(length(test_dataset)) patients"
patient_dims(test_dataset)
all_times, all_ctnt, t_min, t_max, c_min, c_max, dist = plot_distribution(test_dataset);
display(dist)

lhs_lb = log.([0.001, 0.001, 0.001, 0.001, 0.001]); # 0.001, 0.001, 0.01, 0.01, 0.001
lhs_ub = log.([10.0, 10.0, 500.0, 500.0, 1]); # 5.0, 5.0, 300.0, 400.0, 3

# ode_p = best_solution;

# corr(t_norm, β) = inv((inv(t_norm * t_norm) * (β + 0.03258268522354614)) + 0.12298368204885249) * 0.03353526705206174

# const C1 = 0.03258268522354614
# const C2 = 0.12298368204885249
# const C3 = 0.03353526705206174

# corr(t_norm, β) = begin
#     t2 = t_norm * t_norm
#     (C3 * t2) / (β + C1 + C2 * t2)
# end

# const K_A = 0.27268062309876695
# const K_BETA = 8.131160031481027
# const K_OFFSET = 0.2649350278080258

# corr_tnorm(t_norm, beta) = K_A * t_norm^2 / (t_norm^2 + K_BETA * beta + K_OFFSET)

# # const K_A = 0.27268062309876695
# const K_BETA_H = 468354.81781330716
# const K_OFFSET_H = 15260.257601742287

# corr_t(t, beta) = K_A * t^2 / (t^2 + K_BETA_H * beta + K_OFFSET_H)

# # const K_A = 0.27268062309876695
# # const K_BETA = 8.131160031481027
# # const K_OFFSET = 0.2649350278080258
# const K_NORM = 240.0

# T_eff(beta) = K_NORM * sqrt(K_BETA * beta + K_OFFSET)

# corr_t_eff(t, beta) = K_A * t^2 / (t^2 + T_eff(beta)^2)

@info "Loading trained neural network"
chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim)

@load "$(models_path)/nnNSTEMI_$(experiment).jld2" neural_network_parameters
best_nn = neural_network_parameters[best_idx]

@info "Network loaded"

#from sr_outputs_extended_2026-03-24_16-42-27
# corr_t(t_norm, β) = inv(t_norm + (inv(((t_norm * t_norm) * t_norm) * 0.0006774434799252378) * (β * β))) * t_norm
# const c = 0.0006774434799252378
# T_eff(β) = (β^2 / c)
# # function corr_t(t_norm, β)
# #     c = 0.0006774434799252378
# #     num = t_norm^4
# #     den = t_norm^4 + (β^2 / c)
# #     return num / den
# # end
# function corr_t(t_norm, t_eff)
#     num = t_norm^4
#     den = t_norm^4 + t_eff
#     return num / den
# end

# from sr_outputs_extended_2026-03-27_12-53-41
const c1 = 0.0007780399162888297
const c2 = 1.0553531103104006
const c = 1 / c2
# corr_t(t_norm, β) = inv((inv((t_norm * t_norm) * (t_norm * (t_norm * c1))) * (β * β)) + c2)
const c_beta = c2 * c1
sqrt(sqrt(c_beta))
T_eff(β) = (β^2 / c_beta) # power 4
function corr_t(t_norm, t_eff)
    num = c * t_norm^4
    den = t_norm^4 + t_eff
    return num / den
end

function fτ(t, Td)
    return t^3 / (t^3 + Td^3)
end

t_span_grid = 0.1:0.1:2400  # alcuni valori tipici del tuo β
β_vals = 0.1:0.1:1.0

p_teacher = Plots.plot()
for β in β_vals
    y = [chain([t / T_SCALE, β], best_nn)[1] for t in t_span_grid]
    Plots.plot!(p_teacher, t_span_grid, y, label="β = $β", linewidth=2)
end
Plots.plot!(p_teacher,
    xlabel="Time (h)",
    # ylabel="NN(t_norm,β)",
    legend=false
)
display(p_teacher)

# p = Plots.plot()
# for β in β_vals
#     y = [corr_t(t / T_SCALE, β) for t in t_span_grid]
#     Plots.plot!(p, t_span_grid, y, label="β = $β", linewidth=2)
# end
# Plots.plot!(p, xlabel="Time (h)", ylabel="SR(t_norm,β)", title="Learned sarcomere rupture function", legend=:bottomright)
# display(p)

# p1 = Plots.plot()
# for β in β_vals
#     y = [corr_t(t / T_SCALE, β) for t in t_span_grid]
#     Plots.plot!(p1, t_span_grid, y, label="β = $β", linewidth=2)
# end
# Plots.plot!(p1, xlabel="Time (h)", ylabel="SR(t_norm,β)", legend=:bottomright)
# display(p1)

T_eff_values = [T_eff(β) for β in β_vals]
p = Plots.plot()
for T_eff in T_eff_values
    y = [corr_t(t / T_SCALE, T_eff) for t in t_span_grid]
    Plots.plot!(p, t_span_grid, y, label="T_eff = $(round(T_eff, digits=2))", linewidth=2)
end
Plots.plot!(p,
    xlabel="Time (h)",
    ylabel="SR(τ,T_eff)",
    title="Learned sarcomere rupture function",
    legend=:bottomright,
    # xticks=([0, 2400], ["0", "2400"]),
    # yticks=([0, 1], ["0", "1"]),
    # tickfontsize=14
)
display(p)

p1 = Plots.plot()
for T_eff in T_eff_values
    y = [corr_t(t / T_SCALE, T_eff) for t in t_span_grid]
    Plots.plot!(p1, t_span_grid, y, label="T_eff = $(round(T_eff, digits=2))", linewidth=2)
end
Plots.plot!(p1,
    xlabel="Time (h)",
    #  ylabel="SR(τ,T_eff)",
    #  title="Learned sarcomere rupture function",
    legend=false,
    # xticks=([0, 2400], ["0", "2400"]),
    # yticks=([0, 1], ["0", "1"]),
    # tickfontsize=14
)
display(p1)

# save("$(figsave_path)/correction_function.png", p)

Td_vals = collect(round.(range(20, 500.0, length=10), digits=0))
p2 = Plots.plot()
for Td in Td_vals
    y = [fτ(t, Td) for t in t_span_grid]
    Plots.plot!(p2, t_span_grid, y, label="Td = $(Int(Td))", linewidth=2)
end
Plots.plot!(p2,
    xlabel="Time (h)",
    #  ylabel="ϕ(t,Td)",
    legend=false,
    # xticks=([0, 2400], ["0", "2400"]),
    # yticks=([0, 1], ["0", "1"]),
    # tickfontsize=14
)
display(p2)

Td_vals_ext = collect(round.(range(500, 1000.0, length=10), digits=0))
p2_ext = Plots.plot()
for Td in Td_vals_ext
    y = [fτ(t, Td) for t in t_span_grid]
    Plots.plot!(p2_ext, t_span_grid, y, label="Td = $(Int(Td))", linewidth=2)
end
Plots.plot!(p2_ext,
    xlabel="Time (h)",
    #  ylabel="ϕ(t,Td)",
    legend=false,
    # xticks=([0, 2400], ["0", "2400"]),
    # yticks=([0, 1], ["0", "1"]),
    # tickfontsize=14
)
display(p2_ext)

savefig(p_teacher, "$(formula_fig_save)/correction_NN.svg")
savefig(p, "$(formula_fig_save)/correction_surrogate_with_title.svg")
savefig(p1, "$(formula_fig_save)/correction_surrogate.svg")
savefig(p2, "$(formula_fig_save)/correction_sigmoid.svg")
savefig(p2_ext, "$(formula_fig_save)/correction_sigmoid_ext.svg")

function ctnt_ode!(du, u, p, t)
    Cs = u[1]
    Cc = u[2]
    Cp = u[3]
    # println("ode")
    # println(p)
    β = exp(p[5]) # Positive conditional parameter

    a = exp(p[1])
    b = exp(p[2])
    # Cs0 = exp(p.ode[3])
    # Cc0 = exp(p.ode[4])

    # correction = chain([u[1], t, p.ode[1:4]..., β], p.neural)[1]

    # correction = chain([u[1], t, a, b, Cs0, Cc0, β], p.neural)[1]

    # correction = chain([u[1], t, β], p.neural)[1]

    t_norm = t / T_SCALE

    correction = corr_t(t_norm, T_eff(β))

    du[1] = -(Cs - Cc) * correction
    du[2] = (Cs - Cc) * correction - a * (Cc - Cp)
    du[3] = a * (Cc - Cp) - b * Cp
end

sy_ode!(du, u, p, t) = ctnt_ode!(du, u, p, t)

# @load "$(models_path)/odebetasNSTEMI_$(experiment).jld2" ode_params;
# pguess = vec(mean(reshape(best_ode_p, :, N_params), dims=1));
# println("Initial: ", exp.(pguess))

params_path = "$(models_path)/$(dataset_name)_test_NN_$(best_idx)$(use_multistart ? "_ms_test" : "")"
isfile("$(params_path)/best_params_val_$(dataset_name).jld2") || error("Missing NN validation params at $(params_path)")
@load "$(params_path)/best_params_val_$(dataset_name).jld2" ode_params_val

# reshaped_params = reshape(ode_params_val, :, N_params)
reshaped_params = permutedims(reshape(ode_params_val, N_params, :)) # reshape to (N_param_sets, N_params) and then permute to (N_param_sets, N_params)
mean_pguess = vec(mean(reshaped_params, dims=1))
std_pguess = vec(std(exp.(reshaped_params), dims=1))
median_pguess = vec(median(exp.(reshaped_params), dims=1))
q3_pguess = vec([quantile(exp.(reshaped_params[:, i]), 0.75) for i in 1:N_params])
q1_pguess = vec([quantile(exp.(reshaped_params[:, i]), 0.25) for i in 1:N_params])

# [0.005, 0.005, 30.0, 0.01, 0.01]
pguess = log.(median_pguess)
# pguess = log.([0.005, 0.005, 0.1, 0.01, 0.5])

all(isfinite, ode_params_val) || error("ode_params_val contains non-finite values")
all(isfinite, pguess) || error("pguess contains non-finite values")

# optfunc = OptimizationFunction(patient_loss_formula, AutoForwardDiff());
optfunc = OptimizationFunction(
    (p, data) -> patient_loss_formula(p, data; λ_back=λ_back),
    # (p, data) -> serial_training_loss(p, data; n_params=N_params),
    AutoForwardDiff()
)

open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
    println(io, "*********************************")
    println(io, "Evaluating $(dataset_name) Formula from SymReg with sMAPE")
end

successful_idx = Int[]
smape_values = Float64[]
rmsle_values = Float64[]
loss_values = Float64[]
params_list = Vector{Vector{Float64}}()

ev_bar = Progress(length(test_dataset); desc="Validating", color=:cyan, showspeed=true, dt=0.1);

for (i, patient) in enumerate(test_dataset)

    u0_init = [exp(pguess[3]), exp(pguess[4]), 0.0]
    tspan = (0.0, patient.timepoints[end] + 10.0)
    problem = ODEProblem(sy_ode!, u0_init, tspan)

    patient_data = (problem, patient.timepoints, patient.ctnt_data)
    loss_fun = θ -> patient_loss_formula(θ, patient_data; λ_back=λ_back)

    @info "Patient $(patient.id)"

    if use_multistart
        best_result, all_results = MultiStartOptimizer.run_multistart(
            loss_fun,
            N_multistart;
            lower=lhs_lb,
            upper=lhs_ub,
            rng=multistart_rng,
            verbose=false,
            maxiters=multistart_maxiters,
            maxtime=multistart_maxtime,
            prescreen=false,
            topk=8
        )

        if best_result === nothing
            @warn "No multistart solution found for patient $(patient.id)"
            open("res/$(experiment)/info_output.txt", "a") do io
                println(io, "WARN: No multistart solution found for patient $(patient.id)")
            end
            next!(ev_bar)
            continue
        end

        p_opt = Vector(best_result.u)
        best_objective = best_result.minimum
    else
        optprob = OptimizationProblem(optfunc, pguess, patient_data; lb=lhs_lb, ub=lhs_ub)
        optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()); maxiters=1000)
        p_opt = Vector(optsol_lbfgs.u)
        if any(x -> !isfinite(x), p_opt)
            @warn "Non-finite p_opt for patient $(patient.id)"
            next!(ev_bar)
            continue
        end
        best_objective = optsol_lbfgs.objective
        println("Objective: ", best_objective)
    end

    prob = remake(problem; u0=[exp(p_opt[3]), exp(p_opt[4]), 0.0], p=p_opt)
    # pred = solve(prob, Tsit5(); p = p_opt, saveat = patient.timepoints)
    pred = try
        solve(prob, Tsit5(); p=p_opt, saveat=patient.timepoints)
    catch e
        @warn "Prediction solve failed for patient $(patient.id): $e"
        open("res/$(experiment)/info_output.txt", "a") do io
            println(io, "WARN: Prediction solve failed for patient $(patient.id)")
        end
        next!(ev_bar)
        continue
    end
    println(pred.retcode)
    if !successful_retcode(pred)
        @warn "Non-success retcode for patient $(patient.id): $(pred.retcode)"
        open("res/$(experiment)/info_output.txt", "a") do io
            println(io, "WARN: Non-success retcode for patient $(patient.id): $(pred.retcode)")
        end
        next!(ev_bar)
        continue
    end

    push!(successful_idx, i)
    push!(params_list, p_opt)

    sol = try
        solve(prob, Tsit5(); p=p_opt, saveat=1)
    catch
        @warn "Full trajectory solve failed for patient $(patient.id)"
        open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
            println(io, "WARN: Full trajectory solve failed for patient $(patient.id)")
        end
        pop!(params_list)
        pop!(successful_idx)
        next!(ev_bar)
        continue
    end

    smape_val = smape(pred[3, :], patient.ctnt_data)
    rmsle_val = rmsle(patient.ctnt_data, pred[3, :])

    println("sMAPE: ", smape_val)
    println("RMSLE: ", rmsle_val)

    open("res/$(experiment)/info_output.txt", "a") do io
        println(io, "Patient $(patient.id) sMAPE Formula validation: $(smape_val)")
        # println(io, "Patient $(patient.id) RMSLE Formula validation: $(rmsle_val)")
    end

    push!(loss_values, best_objective)
    push!(smape_values, smape_val)
    push!(rmsle_values, rmsle_val)

    pl = Plots.plot(sol[1, :]; lw=2, label="Sarcomere", xlabel="Time", ylabel="CTNT", title="Surrogate - Patient $(patient.id)")
    Plots.plot!(pl, sol[2, :]; lw=2, label="Cytosol")
    Plots.plot!(pl, sol[3, :]; lw=2, label="Blood")
    Plots.scatter!(pl, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    pl_plasm = Plots.plot(sol[3, :]; lw=2, label="Blood", xlabel="Time", ylabel="cTnT [ng/mL]", title="Patient $(patient.id)")
    Plots.scatter!(pl_plasm, patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data", legend=:best)

    display(pl)
    display(pl_plasm)

    savefig(pl, "$(formula_fig_save)/patient_$(patient.id)$(dataset_name).svg")
    savefig(pl_plasm, "$(formula_fig_save)/patient_$(patient.id)$(dataset_name)_plasm.svg")
    @info "Patient $(patient.id) formula plot saved"
    next!(ev_bar)
end

if isempty(successful_idx)
    @warn "No successful patients in formula evaluation. Aborting summary/export."
    open("res/$(experiment)/info_output.txt", "a") do io
        println(io, "WARN: No successful patients in formula evaluation. Aborting summary/export.")
    end
    error("No successful patients in test_formula")
end

println("--> Average - STD loss: $(mean(loss_values)) - $(std(loss_values))")
println("--> Median [Q1-Q3] loss: $(median(loss_values)) [$(quantile(loss_values, 0.25)) - $(quantile(loss_values, 0.75))]")
println("--> Average - STD sMAPE: $(mean(smape_values)) - $(std(smape_values))")
println("--> Median [Q1-Q3] sMAPE: $(median(smape_values)) [$(quantile(smape_values, 0.25)) - $(quantile(smape_values, 0.75))]")
println("--> Average - STD RMSLE: $(mean(rmsle_values)) - $(std(rmsle_values))")
println("--> Median [Q1-Q3] RMSLE: $(median(rmsle_values)) [$(quantile(rmsle_values, 0.25)) - $(quantile(rmsle_values, 0.75))]")

open("res/$(experiment)/info_output.txt", "a") do io
    println(io, "--> Average - STD loss: $(mean(loss_values)) - $(std(loss_values))")
    println(io, "--> Median [Q1-Q3] loss: $(median(loss_values)) [$(quantile(loss_values, 0.25)) - $(quantile(loss_values, 0.75))]")
    println(io, "--> Average - STD sMAPE: $(mean(smape_values)) - $(std(smape_values))")
    println(io, "--> Median [Q1-Q3] sMAPE: $(median(smape_values)) [$(quantile(smape_values, 0.25)) - $(quantile(smape_values, 0.75))]")
    println(io, "--> Average - STD RMSLE: $(mean(rmsle_values)) - $(std(rmsle_values))")
    println(io, "--> Median [Q1-Q3] RMSLE: $(median(rmsle_values)) [$(quantile(rmsle_values, 0.25)) - $(quantile(rmsle_values, 0.75))]")
end

used_test_dataset = test_dataset[successful_idx]
used_test_ids = [p.id for p in used_test_dataset]
println("Successfully evaluated formula on $(length(successful_idx)) out of $(length(test_dataset)) patients")

CSV.write("$(formula_models_save)/patients_metrics_val_formula.csv", DataFrame(
    patient_id=used_test_ids,
    smape=smape_values,
    rmsle=rmsle_values,
    loss=loss_values
))
params_list_flat = vcat(params_list...)
@save "$(formula_models_save)/best_params_val_formula_$(dataset_name).jld2" params_list_flat

a, b, Cs0, Cc0, β, fig = params_extraction(
    used_test_dataset,
    params_list_flat;
    N_params=N_params,
    data_label="formula",
    dataset=dataset_name,
    figsave_path=figsave_path,
    show_outliers=true,
    savefigure=true
)
display(fig)

CSV.write("$(formula_models_save)/patients_params_val_formula.csv", DataFrame(
    patient_id=used_test_ids,
    a=a,
    b=b,
    Cs0=Cs0,
    Cc0=Cc0,
    beta=β
))

model_summary_formula = DataFrame(
    model_id=["formula_cfg$(nn_depth)$(nn_width)_$(best_idx)"],
    model_idx=[best_idx],
    nn_depth=[nn_depth],
    nn_width=[nn_width],
    n_patients=[length(used_test_dataset)],
    loss_mean=[mean(loss_values)],
    loss_std=[std(loss_values)],
    loss_median=[median(loss_values)],
    loss_q1=[quantile(loss_values, 0.25)],
    loss_q3=[quantile(loss_values, 0.75)],
    loss_iqr=[quantile(loss_values, 0.75) - quantile(loss_values, 0.25)],
    smape_mean=[mean(smape_values)],
    smape_std=[std(smape_values)],
    smape_median=[median(smape_values)],
    smape_q1=[quantile(smape_values, 0.25)],
    smape_q3=[quantile(smape_values, 0.75)],
    smape_iqr=[quantile(smape_values, 0.75) - quantile(smape_values, 0.25)],
    rmsle_mean=[mean(rmsle_values)],
    rmsle_std=[std(rmsle_values)],
    rmsle_median=[median(rmsle_values)],
    rmsle_q1=[quantile(rmsle_values, 0.25)],
    rmsle_q3=[quantile(rmsle_values, 0.75)],
    rmsle_iqr=[quantile(rmsle_values, 0.75) - quantile(rmsle_values, 0.25)]
)

CSV.write("$(models_path)/model_summary_formula_$(dataset_name).csv", model_summary_formula)

@info "Processing $(experiment) residuals data"

residuals_ae, smape_ae = compute_plot_residuals(
    used_test_dataset,
    params_list,
    ctnt_ode!;
    N_params=N_params,
    UDE=UDE,
    hi=false,
    show_plots=true,
    figsave_path=formula_fig_save,
    modelssave_path=formula_models_save,
    dataset_label="$(dataset_name)_FORMULA"
)

@info "Computed residuals for AE test dataset. Median sMAPE: $(median(smape_ae.smape))"

@info "Test formula calculation script completed"

# for (i, patient) in enumerate(test_dataset)

#     sy_ode!(du, u, p, t) = ctnt_ode!(du, u, p, t; norm_type = norm_type, pat_times = patient.timepoints)

#     u0_init = [exp(pguess[3]), exp(pguess[4]), 0.0];

#     tspan = (0.0, patient.timepoints[end] + 10.0);

#     problem = ODEProblem(sy_ode!, u0_init, tspan);

#     optprob = OptimizationProblem(optfunc, pguess, (problem, patient.timepoints, patient.ctnt_data), lb = lhs_lb, ub = lhs_ub);

#     optsol_lbfgs = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000);

#     p_opt = optsol_lbfgs.u;

#     println("For $(patient.id), params: ", p_opt)
#     println("Params: ", exp.(p_opt))
#     # push!(optsols_valid, optsol_lbfgs);
#     push!(params_list, p_opt);

#     u0_new = [exp(p_opt[3]), exp(p_opt[4]), 0.0]
#     prob   = remake(problem; u0 = u0_new, p = p_opt)

#     # sol = Array(solve(prob, Tsit5(); p=p_opt, saveat=1));
#     sol = solve(prob, Tsit5(); p=p_opt, saveat=1);
#     println("Patient loss: ", patient_loss_formula(p_opt, (prob, patient.timepoints, patient.ctnt_data)))
#     # println("Compute loss: ", compute_loss(p_opt, (opt_model, patient.timepoints, patient.ctnt_data)))
#     push!(loss_values, optsol_lbfgs.objective)
#     println("Objective:    ", optsol_lbfgs.objective)
#     validation_metric = smape_loss_formula(p_opt, (prob, patient.timepoints, patient.ctnt_data));
#     println("sMAPE: ", validation_metric)
#     open("res/$(experiment)/info_output.txt", "a") do io          # "w" = write (sovrascrive)
#         println(io, "Patient $(patient.id) sMAPE Formula validation - test: $validation_metric")
#     end
#     push!(smape_values, validation_metric)
#     pred = sol[3,:];

#     pl = Plots.plot(pred; lw=2, label="Model with Formula Prediction", xlabel="Time", ylabel="Troponin", title="Patient $(patient.id)")
#     Plots.scatter!(patient.timepoints, patient.ctnt_data, ms=5, label="Observed Data")

#     display(pl)

#     save("$(figsave_path)/formula/patient_$(patient.id).svg", pl)
#     next!(ev_bar)
# end

# hi_ids = if !UMG_data

#     hi_ids = try
#         CSV.read("res/ids_high_information_$(dataset_name)_val.csv", DataFrame);
#     catch e
#         @warn "Error loading high information ids: $e"
#         @info "Calculating high information ids from dataset"

#         meas_min_number = 8;
#         min_acq_time_before=24.0;
#         min_acq_n_before=2;
#         min_acq_time_after=36.0;
#         min_acq_n_after=1;
#         min_time=12.0;
#         max_gap=36.0;

#         anoms = find_anomalies(
#             test_dataset, 
#             meas_min_number, 
#             min_acq_time_before, min_acq_n_before, 
#             min_acq_time_after, min_acq_n_after, 
#             min_time; 
#             max_gap_h=max_gap
#             );

#         @info "Removed sample number for high information: $(length(anoms))"
#         high_information_val = filter(p -> !haskey(anoms, p.id), test_dataset);
#         patient_dims(high_information_val)
#         @info "Total sample number for high information: $(length(high_information_val))"

#         hi_ids = DataFrame(patient = [p.id for p in high_information_val]);
#         CSV.write(joinpath("./res", "ids_high_information_$(dataset_name)_val.csv"), hi_ids)
#         hi_ids;
#     end

#     hi_ids;

# else

#     hi_ids = CSV.read("res/ids_high_information_$(dataset_name)_minafter.csv", DataFrame); # calcuated in TroponinReleaseDiffEqs.jl
#     # high_information = filter(p -> p.id in hi_ids.patient, test_dataset);
#     hi_ids;

# end

# # if UMG_data
# @info "Best model and parameters loaded successfully"
# @info "Loaded $(length(params_list)) parameters"

# hi_ids = CSV.read("res/ids_high_information_$(dataset_name)_minafter.csv", DataFrame);
# high_information = filter(p -> p.id in hi_ids.patient, test_dataset);
# high_information_idxs = findall(p -> p.id in hi_ids.patient, test_dataset);
# high_information = test_dataset[high_information_idxs];
# patient_dims(high_information)

# params_list_hi = [params_list[i] for i in high_information_idxs];
# # ode_params_val_hi = vcat(ode_params_val_hi...);

# @info "Total sample number for high information: $(length(high_information))"

# residuals_ae, smape_ae = compute_plot_residuals(test_dataset, params_list, ctnt_ode!;
#     N_params = N_params, UDE = UDE, hi = false, show_plots = true,
#     figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label=dataset
#     );

# residuals_hi, smape_hi = compute_plot_residuals(high_information, params_list_hi, ctnt_ode!;
#     N_params = N_params, UDE = UDE, hi = true, show_plots = true,
#     figsave_path=figsave_path, modelssave_path=modelssave_path, dataset_label="$(dataset)_HI"
#     );

# @info "Computed residuals for HI test dataset. Median sMAPE: $(median(smape_hi.smape))"

# open("res/$(experiment)/info_output.txt", "a") do io
#     println(io, "--> Average sMAPE with formula on HI dataset - test: $(mean(smape_hi.smape)) - $(std(smape_hi.smape))")
#     println(io, "--> Median sMAPE with formula on HI dataset - test: $(median(smape_hi.smape)) [$(quantile(smape_hi.smape, 0.25)) - $(quantile(smape_hi.smape, 0.75))]")
# end
# end

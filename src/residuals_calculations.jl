using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using JLD2, CSV
using ProgressMeter
using Statistics
using Dates
using Logging

using Revise

includet("ctnt-ude-model.jl")

@info "Starting residual calculation script"

UDE = false; # false for cUDE

N_params = UDE ? 4 : 5; # number of UDE parameters 5 for cUDE

input_dim = 2;
nn_depth = 2;
nn_width = 8;
inputs_str = "t, β";
inputs_str = input_dim == 2 ? "τ, Cs0" : "τ";

T_SCALE = 350.0;

const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, 350.0];

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

# experiment = "NSTEMI_UDE_UMG_MSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus";
# experiment = "NSTEMI_partrval_UMG_MSE_ts350.0_28_inp2_multipl_softplus";

fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
modelssave_path = "res/$(experiment)/models/test_NN";

@load "$(models_path)/best_nn_NSTEMI_$(experiment).jld2" best_nn;
@load "$(modelssave_path)/best_params_val_$(experiment).jld2" ode_params_val;
@load "$(models_path)/testsetNSTEMI_$(experiment).jld2" test_dataset;

out = DataFrame(id=String[], t=Float64[], y=Float64[], yhat=Float64[], res=Float64[]);

@info "Processing $(experiment) residuals data"

a = []
b = []
Cs0 = []
Cc0 = []
β = []

@showprogress desc="Computing residuals..." for (i, patient) in enumerate(test_dataset)

    idx1 = N_params*(i-1) + 1;
    idx2 = N_params*i;
    ode_p = ode_params_val[idx1:idx2];
    p = ComponentArray(ode = ode_p, neural = best_nn);

    push!(a, exp(p.ode[1]))
    push!(b, exp(p.ode[2]))
    push!(Cs0, exp(p.ode[3]))
    push!(Cc0, exp(p.ode[4]))
    if N_params == 5
        push!(β, exp(p.ode[5]))
    end

    # u0 = [p.ode[end], p.ode[end-1], 0.0]
    tspan = (0.0, patient.timepoints[end] + 10.0);

    model = UDE ? ctntUDEModel(p, chain, tspan) : ctntCUDEModel(p, chain, tspan);

    y, yhat, res = compute_residuals_patient(model, patient, p;
                                            plotting=true)

    append!(out, DataFrame(id = fill(patient.id, length(y)),
                               t  = patient.timepoints,
                               y  = y,
                               yhat = yhat,
                               res  = res))
end

params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β];

@info "Saving residuals data to CSV and plotting"

CSV.write("res/$(experiment)/residuals.csv", out)

add_time_bins!(out, EDGES)

fig_vs_time = plot_residuals_vs_time(
    out, 
    EDGES; 
    title="Residuals vs time - UMG", TMAX=350.0, nmin=1);
display(fig_vs_time)
CairoMakie.save("$(fig_path)/residuals_vs_time.png", fig_vs_time)

fig_vs_fitted = plot_residuals_vs_fitted(out; title="Residuals vs fitted - UMG")
display(fig_vs_fitted)
CairoMakie.save("$(fig_path)/residuals_vs_fitted.png", fig_vs_fitted)

@info "Boxplotting params"

par_names = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"];

x = vcat([fill(1,length(a))]...);

f = Figure(
    size = (1400, 700), # input
    );

Label(
    f[0, 1:length(par_names)],
    "Parameter distributions — UMG";
    fontsize = 22,
    tellwidth = false
);

axes = [];

@showprogress desc="Generating axes..." for (i, p) in enumerate(par_names)
    push!(axes, (Axis(f[1, i], title = p)))
end

@showprogress desc="Generating boxplots..." for (ax, p) in zip(axes, params)
    i = (i-1)+1;
    CairoMakie.boxplot!(
        ax, x, p;
        color = x, 
        # width = 0.5,
        # mediancolor = :red,
        # whiskercolor = :gray,
        # outliercolor = :green,
        # show_notch = true
        );
    # ax.xticks = (1:length(exps), exps_names);
    # ax.xticklabelrotation = pi/3;
end
display(f)
save("$(fig_path)/boxplots.png", f)

@info "Residual calculation script completed"
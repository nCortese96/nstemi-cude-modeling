using StableRNGs, DataFrames, StatsBase, XLSX, Random
using Optimization, OptimizationOptimisers, LineSearches
using JLD2, CSV
using ProgressMeter
using Statistics
using Dates
using Logging

using Revise

includet("ctnt-ude-model.jl")

@info "Starting comprensive cude boxplot script"

# UMG_data = false; # true for UDE with UMG data, false for cUDE with MIMIC-IV data

UDE = false; # false for cUDE

best_idx = 4; # index of the best model to test 

if UDE
    @info "Using UDE model"
    input_dim = 1;
    nn_depth = 2;
    nn_width = 8;
    N_params = 4;
    inputs_str = "τ";
else
    @info "Using cUDE model"
    input_dim = 2;
    nn_depth = 2;
    nn_width = 8;
    N_params = 5;
    inputs_str = "τ, β";
end 

T_SCALE = 350.0;

const EDGES = [0.0, 12.0, 24.0, 48.0, 72.0, 120.0, 200.0, 350.0];

chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim);

# experiment = "NSTEMI_UDE_UMG_MSE_ts$(T_SCALE)_$(nn_depth)$(nn_width)_inp$(input_dim)_multipl_softplus";
experiment = "NSTEMI_partrval_MIMIC-IV_MSE_ts350.0_28_inp2_multipl_softplus";
# experiment = "NSTEMI_partrval_UMG_MSE_ts350.0_28_inp2_multipl_softplus";

fig_path = "res/$(experiment)/figs";
models_path = "res/$(experiment)/models";
# modelssave_path = "res/$(experiment)/models/test_NN";

figsave_path = "$(fig_path)/comprensive_cude_box_$(best_idx)";
modelssave_path = "$(models_path)/comprensive_cude_box_$(best_idx)";   
    
mkpath(figsave_path)
mkpath(modelssave_path)

@info "Loading datasets"

mimic_ae = JLD2.load("$(models_path)/testsetNSTEMI_$(experiment).jld2", "test_dataset");
@info "MIMIC-IV AE test dataset loaded with $(length(mimic_ae)) patients"
hi_ids_mimic = CSV.read("res/ids_high_information_MIMIC-IV_val.csv", DataFrame);
mimic_hi_idxs = findall(p -> p.id in hi_ids_mimic.patient, mimic_ae);
mimic_hi = mimic_ae[mimic_hi_idxs];
@info "MIMIC-IV HI test dataset loaded with $(length(mimic_hi)) patients"

umg_ae = JLD2.load("$(models_path)/umg_test_nn_$(best_idx)/UMG_testset.jld2", "test_dataset");
@info "UMG test dataset loaded with $(length(umg_ae)) patients"
hi_ids_umg = CSV.read("res/ids_high_information_UMG_minafter.csv", DataFrame);
umg_hi_idxs = findall(p -> p.id in hi_ids_umg.patient, umg_ae);
umg_hi = umg_ae[umg_hi_idxs];
@info "UMG HI test dataset loaded with $(length(umg_hi)) patients"


@info "Loading best model and parameters"

best_nn = JLD2.load("$(models_path)/best_nn_NSTEMI_$(experiment).jld2", "best_nn");
ode_p_val_mimic_ae = JLD2.load("$(models_path)/test_NN_$(best_idx)/best_params_val_MIMIC-IV.jld2", "ode_params_val");
ode_p_val_umg_ae = JLD2.load("$(models_path)/umg_test_nn_$(best_idx)/best_params_val_UMG.jld2", "ode_params_val");

@info "Best model and parameters loaded successfully"
@info "Loaded $(length(ode_p_val_mimic_ae)) parameters for MIMIC-IV"
@info "Loaded $(length(ode_p_val_umg_ae)) parameters for UMG"

ode_p_mimic_hi = [ode_p_val_mimic_ae[N_params * (i-1) + 1:N_params * i] for i in mimic_hi_idxs];
ode_p_mimic_hi = vcat(ode_p_mimic_hi...);

ode_p_umg_hi = [ode_p_val_umg_ae[N_params * (i-1) + 1:N_params * i] for i in umg_hi_idxs];
ode_p_umg_hi = vcat(ode_p_umg_hi...);

@info "Loaded $(length(ode_p_mimic_hi)) parameters for MIMIC-IV HI"
@info "Loaded $(length(ode_p_umg_hi)) parameters for UMG HI"

@info "Processing $(experiment) residuals data"

function extract_p_from_df(patients::Vector{PatientData}, ode_params_val; N_params::Int = 5, UDE::Bool = false)

    a = []
    b = []
    Cs0 = []
    Cc0 = []
    β = []

    @showprogress desc="Computing residuals..." for (i, patient) in enumerate(patients)

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
    end

    params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β];

    @info "Boxplotting params"

    return params
    
end

par_names = UDE ? ["a", "b", "Cs0", "Cc0"] : ["a", "b", "Cs0", "Cc0", "β"];

params_mimic_ae = extract_p_from_df(mimic_ae, ode_p_val_mimic_ae; N_params = N_params, UDE = UDE);
params_mimic_hi = extract_p_from_df(mimic_hi, ode_p_mimic_hi; N_params = N_params, UDE = UDE);
params_umg_ae = extract_p_from_df(umg_ae, ode_p_val_umg_ae; N_params = N_params, UDE = UDE);
params_umg_hi = extract_p_from_df(umg_hi, ode_p_umg_hi; N_params = N_params, UDE = UDE);

exps = [];
push!(exps, params_mimic_ae)
push!(exps, params_mimic_hi)
push!(exps, params_umg_ae)
push!(exps, params_umg_hi)

exps_names = ["MIMIC-IV AE", "MIMIC-IV HI", "UMG AE", "UMG HI"];

a = vcat(params_mimic_ae[1]..., params_mimic_hi[1]..., params_umg_ae[1]..., params_umg_hi[1]...);
b = vcat(params_mimic_ae[2]..., params_mimic_hi[2]..., params_umg_ae[2]..., params_umg_hi[2]...);
Cs0 = vcat(params_mimic_ae[3]..., params_mimic_hi[3]..., params_umg_ae[3]..., params_umg_hi[3]...);
Cc0 = vcat(params_mimic_ae[4]..., params_mimic_hi[4]..., params_umg_ae[4]..., params_umg_hi[4]...);
if !UDE 
    β = vcat(params_mimic_ae[5]..., params_mimic_hi[5]..., params_umg_ae[5]..., params_umg_hi[5]...);
end

params = UDE ? [a, b, Cs0, Cc0] : [a, b, Cs0, Cc0, β];

# x = vcat([fill(1,length(a))]...);

x = vcat([fill(i, length(exp[1])) for (i, exp) in enumerate(exps)]...);

f = Figure(
    size = (1400, 700), # input
    );

Label(
    f[0, 1:length(par_names)],
    "Parameter distributions — cUDE on MIMIC-IV and UMG datasets",;
    fontsize = 22,
    tellwidth = false
);

axes = [];

@showprogress desc="Generating axes..." for (i, p) in enumerate(par_names)
    push!(axes, (Axis(f[1, i], title = p)))
end

@showprogress desc="Generating boxplots..." for (ax, p) in zip(axes, params)
    # i = (i-1)+1;
    CairoMakie.boxplot!(
        ax, x, p;
        color = x, 
        # width = 0.5,
        # mediancolor = :red,
        # whiskercolor = :gray,
        # outliercolor = :green,
        # show_notch = true
        );
    ax.xticks = (1:length(exps), exps_names);
    ax.xticklabelrotation = pi/3;
end

display(f)
CairoMakie.save("$(figsave_path)/comprensive_boxplots.png", f)

if show_plots
    display(fig_vs_time)
    display(fig_vs_fitted)
    
end
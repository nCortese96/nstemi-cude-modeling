using CairoMakie
using CSV
using DataFrames
using Statistics: mean, std

# function main()
# Paths
cude_mimic_path = "res/NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback/models/MIMIC-IV_test_NN_3_ms_test/patients_metrics_val.csv"
cude_umg_path = "res/NSTEMI_cUDE_MIMIC-IV_MSE_28_sigmoid_regback/models/UMG_test_NN_ab10_3_ms_test/patients_metrics_val.csv"
ode_mimic_path = "res/NSTEMI_ODE_TdSigmoid/MIMIC-IV_opt_lambda1/models/params_out.csv"
ode_umg_path = "res/NSTEMI_ODE_TdSigmoid/UMG_opt_lambda1/models/params_out.csv"

# Load Data
df_cude_mimic = dropmissing(CSV.read(cude_mimic_path, DataFrame), [:smape, :rmsle])
df_cude_umg = dropmissing(CSV.read(cude_umg_path, DataFrame), [:smape, :rmsle])
df_ode_mimic = dropmissing(CSV.read(ode_mimic_path, DataFrame), [:smape, :rmsle])
df_ode_umg = dropmissing(CSV.read(ode_umg_path, DataFrame), [:smape, :rmsle])

# Filter MIMIC-IV ODE parameters to match the validation subset
val_ids_mimic = string.(CSV.read("res/ids_all_eligible_MIMIC-IV_val.csv", DataFrame).patient)
filter!(row -> string(row.patient) in val_ids_mimic, df_ode_mimic)

# Structure data for Makie
datasets = Int[]
models = Int[]
smapes = Float64[]
rmsles = Float64[]

function append_data!(df, dataset_id, model_id)
    n = nrow(df)
    append!(datasets, fill(dataset_id, n))
    append!(models, fill(model_id, n))
    append!(smapes, df.smape)
    append!(rmsles, df.rmsle)
end

# dataset_id: 1=MIMIC, 2=UMG
# model_id: 1=cUDE, 2=ODE
append_data!(df_cude_mimic, 1, 1)
append_data!(df_ode_mimic, 1, 2)
append_data!(df_cude_umg, 2, 1)
append_data!(df_ode_umg, 2, 2)

# Plot settings
CairoMakie.activate!(type="svg") # professional quality
fig = Figure(size=(1000, 500), fontsize=18)

ax_smape = Axis(fig[1, 1], title="sMAPE Distribution", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="sMAPE (%)")
ax_rmsle = Axis(fig[1, 2], title="RMSLE Distribution", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="RMSLE")

# Colors for violin: cUDE -> orange, ODE -> blue
colors = [m == 1 ? :darkorange : :royalblue for m in models]

CairoMakie.violin!(ax_smape, datasets, smapes, dodge=models, color=colors, show_median=true, mediancolor=:black)
CairoMakie.violin!(ax_rmsle, datasets, rmsles, dodge=models, color=colors, show_median=true, mediancolor=:black)

# Handle Y-axis limits just to be sure we don't have weird scale issues, typically violins figure this out automatically
# but some outliers can stretch the view. We'll leave the default CairoMakie scaling as it usually handles outliers.

# Legend
elem_cude = PolyElement(color=:darkorange, strokecolor=:transparent)
elem_ode = PolyElement(color=:royalblue, strokecolor=:transparent)
Legend(fig[1, 3], [elem_cude, elem_ode], ["cUDE", "ODE"], "Models", framevisible=false)

# Save
out_dir = "res/patient_paper_selections/metrics_comparison"
mkpath(out_dir)
save_path_svg = joinpath(out_dir, "violin_metrics_cUDE_vs_ODE.svg")
save(save_path_svg, fig)

# Save a high-res PNG as well
# CairoMakie.activate!(type="png")
# save_path_png = joinpath(out_dir, "violin_metrics_cUDE_vs_ODE.png")
# save(save_path_png, fig, px_per_unit=3)

println("Plots successfully saved in: $(out_dir)")
println(" - $(basename(save_path_svg))")
# println(" - $(basename(save_path_png))")

# --- Plot Mean ± Std (Barplot + Errorbars) ---
groups_df = [df_cude_mimic, df_ode_mimic, df_cude_umg, df_ode_umg]
group_datasets = [1, 1, 2, 2]
group_models = [1, 2, 1, 2] # 1 -> cUDE, 2 -> ODE

smape_means = [mean(df.smape) for df in groups_df]
smape_stds = [std(df.smape) for df in groups_df]

@info "sMAPE means +- STD: $smape_means +- $smape_stds"

rmsle_means = [mean(df.rmsle) for df in groups_df]
rmsle_stds = [std(df.rmsle) for df in groups_df]

@info "RMSLE means +- STD: $rmsle_means +- $rmsle_stds"

fig2 = Figure(size=(1000, 500), fontsize=18)
ax2_smape = Axis(fig2[1, 1], title="sMAPE (Mean ± STD)", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="sMAPE (%)")
ax2_rmsle = Axis(fig2[1, 2], title="RMSLE (Mean ± STD)", xticks=(1:2, ["MIMIC", "UMG"]), ylabel="RMSLE")

group_colors = [m == 1 ? :darkorange : :royalblue for m in group_models]

# Manual dodging to align barplot and errorbars since errorbars! lacks `dodge` kwarg
x_dodged = [Float64(d) + (m == 1 ? -0.2 : 0.2) for (d, m) in zip(group_datasets, group_models)]

CairoMakie.barplot!(ax2_smape, x_dodged, smape_means, color=group_colors, width=0.35)
CairoMakie.errorbars!(ax2_smape, x_dodged, smape_means, smape_stds, color=:black, whiskerwidth=10)

CairoMakie.barplot!(ax2_rmsle, x_dodged, rmsle_means, color=group_colors, width=0.35)
CairoMakie.errorbars!(ax2_rmsle, x_dodged, rmsle_means, rmsle_stds, color=:black, whiskerwidth=10)

Legend(fig2[1, 3], [elem_cude, elem_ode], ["cUDE", "ODE"], "Models", framevisible=false)

save_path2_svg = joinpath(out_dir, "barplot_mean_std_metrics_cUDE_vs_ODE.svg")
save(save_path2_svg, fig2)

# Save a high-res PNG as well
CairoMakie.activate!(type="png")
save_path2_png = joinpath(out_dir, "barplot_mean_std_metrics_cUDE_vs_ODE.png")
save(save_path2_png, fig2, px_per_unit=3)

println(" - $(basename(save_path2_svg))")
# end

# main()

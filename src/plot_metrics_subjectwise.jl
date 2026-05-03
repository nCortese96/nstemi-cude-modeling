using CairoMakie
using CSV
using DataFrames

# Helper to plot subject-wise comparison
function plot_subjectwise!(ax, x, y, title, xlabel, ylabel)
    max_val = max(maximum(x), maximum(y))
    min_val = min(minimum(x), minimum(y))

    # Calculate both percentages
    total = length(x)
    better_cude = sum(y .< x)
    better_ode = sum(y .> x)
    perc_cude = round(100 * better_cude / total, digits=1)
    perc_ode = round(100 * better_ode / total, digits=1)

    # Symmetrical limits to enforce the diagonal meaning properly
    pad = (max_val - min_val) * 0.05 + 1e-4
    lim_min = max(0.0, min_val - pad) # don't go below 0 for error metrics
    lim_max = max_val + pad

    # Draw colored bands (shading) for regions
    # Upper-left half (y > x) - ODE better
    band!(ax, [lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max], color=(:royalblue, 0.08))
    # Lower-right half (y < x) - cUDE better
    band!(ax, [lim_min, lim_max], [lim_min, lim_min], [lim_min, lim_max], color=(:darkorange, 0.08))

    scatter!(ax, x, y, color=:black, markersize=8, alpha=0.6)
    ablines!(ax, 0, 1, color=:red, linestyle=:dash, linewidth=2)

    # Axis style
    ax.title = title
    ax.xlabel = xlabel
    ax.ylabel = ylabel

    xlims!(ax, lim_min, lim_max)
    ylims!(ax, lim_min, lim_max)

    # Identity line annotation - upper left (ODE better)
    x_ode_text = lim_min + (lim_max - lim_min) * 0.05
    y_ode_text = lim_max - (lim_max - lim_min) * 0.05
    text!(ax, x_ode_text, y_ode_text, text="ODE better:\n$(perc_ode)%", align=(:left, :top), color=:royalblue, fontsize=16, font=:bold)

    # Identity line annotation - lower right (cUDE better)
    x_cude_text = lim_max - (lim_max - lim_min) * 0.05
    y_cude_text = lim_min + (lim_max - lim_min) * 0.05
    text!(ax, x_cude_text, y_cude_text, text="cUDE better:\n$(perc_cude)%", align=(:right, :bottom), color=:darkorange, fontsize=16, font=:bold)
end

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

# Filter MIMIC-IV ODE parameters to properly match the validation subset
val_ids_mimic = string.(CSV.read("res/ids_all_eligible_MIMIC-IV_val.csv", DataFrame).patient)
filter!(row -> string(row.patient) in val_ids_mimic, df_ode_mimic)

# Rename patient column for ODE to match cUDE
rename!(df_ode_mimic, :patient => :patient_id)
rename!(df_ode_umg, :patient => :patient_id)

# Pre-rename metrics to avoid conflicts
rename!(df_cude_mimic, :smape => :smape_cude, :rmsle => :rmsle_cude)
rename!(df_cude_umg, :smape => :smape_cude, :rmsle => :rmsle_cude)
rename!(df_ode_mimic, :smape => :smape_ode, :rmsle => :rmsle_ode)
rename!(df_ode_umg, :smape => :smape_ode, :rmsle => :rmsle_ode)

# Cast IDs to strings for robust merging
df_cude_mimic.patient_id = string.(df_cude_mimic.patient_id)
df_cude_umg.patient_id = string.(df_cude_umg.patient_id)
df_ode_mimic.patient_id = string.(df_ode_mimic.patient_id)
df_ode_umg.patient_id = string.(df_ode_umg.patient_id)

# Merge on patient_id
merged_mimic = innerjoin(df_cude_mimic, df_ode_mimic, on=:patient_id, makeunique=true)
merged_umg = innerjoin(df_cude_umg, df_ode_umg, on=:patient_id, makeunique=true)

CairoMakie.activate!(type="svg")
fig = Figure(size=(1000, 1000), fontsize=18)

ax1 = Axis(fig[1, 1], aspect=1)
plot_subjectwise!(ax1, merged_mimic.smape_ode, merged_mimic.smape_cude, "MIMIC-IV: sMAPE", "ODE sMAPE (%)", "cUDE sMAPE (%)")

ax2 = Axis(fig[1, 2], aspect=1)
plot_subjectwise!(ax2, merged_mimic.rmsle_ode, merged_mimic.rmsle_cude, "MIMIC-IV: RMSLE", "ODE RMSLE", "cUDE RMSLE")

ax3 = Axis(fig[2, 1], aspect=1)
plot_subjectwise!(ax3, merged_umg.smape_ode, merged_umg.smape_cude, "UMG: sMAPE", "ODE sMAPE (%)", "cUDE sMAPE (%)")

ax4 = Axis(fig[2, 2], aspect=1)
plot_subjectwise!(ax4, merged_umg.rmsle_ode, merged_umg.rmsle_cude, "UMG: RMSLE", "ODE RMSLE", "cUDE RMSLE")

out_dir = "res/patient_paper_selections/metrics_comparison"
mkpath(out_dir)
save_path_svg = joinpath(out_dir, "scatter_subjectwise_cUDE_vs_ODE.svg")
save(save_path_svg, fig)

CairoMakie.activate!(type="png")
save_path_png = joinpath(out_dir, "scatter_subjectwise_cUDE_vs_ODE.png")
save(save_path_png, fig, px_per_unit=3)

println("Subject-wise Scatter Plots successfully saved in: $(out_dir)")
println(" - $(basename(save_path_svg))")
# end

# main()

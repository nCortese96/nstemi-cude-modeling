using CSV
using DataFrames
using Statistics

# =========================
# File input
# =========================
ode_metrics_file = "ODE_trunc_metrics_all.csv"
cude_metrics_file = "CUDE_trunc_metrics_all.csv"

ode_params_file = "ODE_trunc_params_all.csv"
cude_params_file = "CUDE_trunc_params_all.csv"

# =========================
# File output
# =========================
metrics_out_file = "trunc_metrics_summary.csv"
params_out_file = "trunc_params_summary.csv"

# =========================
# Helpers
# =========================
scenario_label(section, setnum) = string(uppercasefirst(section), " (S", setnum, ")")

function median_iqr_string(x; digits=2)
    v = collect(skipmissing(x))
    med = median(v)
    q1 = quantile(v, 0.25)
    q3 = quantile(v, 0.75)
    return string(round(med, digits=digits), " [", round(q1, digits=digits), ", ", round(q3, digits=digits), "]")
end

# =========================
# 1) Metrics summary table
# =========================
ode_metrics = CSV.read(ode_metrics_file, DataFrame)
cude_metrics = CSV.read(cude_metrics_file, DataFrame)

sections = ["start", "middle", "end"]
sets = [1, 2]

metrics_rows = DataFrame(
    Scenario=String[],
    ODE_sMAPE_full=String[],
    ODE_RMSLE_full=String[],
    cUDE_sMAPE_full=String[],
    cUDE_RMSLE_full=String[]
)

for sec in sections
    for s in sets
        ode_sub = filter(r -> r.trunc_section == sec && r.trunc_set == s, ode_metrics)
        cude_sub = filter(r -> r.trunc_section == sec && r.trunc_set == s, cude_metrics)

        push!(metrics_rows, (
            scenario_label(sec, s),
            median_iqr_string(ode_sub.smape_full),
            median_iqr_string(ode_sub.rmsle_full),
            median_iqr_string(cude_sub.smape_full),
            median_iqr_string(cude_sub.rmsle_full)
        ))
    end
end

CSV.write(metrics_out_file, metrics_rows)

# =========================
# 2) Parameter ratio summary table
# =========================
ode_params = CSV.read(ode_params_file, DataFrame)
cude_params = CSV.read(cude_params_file, DataFrame)

params_rows = DataFrame(
    Model=String[],
    Scenario=String[],
    a=String[],
    b=String[],
    Cs0=String[],
    Cc0=String[],
    Modulation_parameter=String[]
)

# ODE
for sec in sections
    for s in sets
        sub = filter(r -> r.trunc_section == sec && r.trunc_set == s, ode_params)

        push!(params_rows, (
            "ODE",
            scenario_label(sec, s),
            median_iqr_string(sub.a_ratio_vs_full),
            median_iqr_string(sub.b_ratio_vs_full),
            median_iqr_string(sub.Cs0_ratio_vs_full),
            median_iqr_string(sub.Cc0_ratio_vs_full),
            "Td: " * median_iqr_string(sub.Td_ratio_vs_full)
        ))
    end
end

# cUDE
for sec in sections
    for s in sets
        sub = filter(r -> r.trunc_section == sec && r.trunc_set == s, cude_params)

        push!(params_rows, (
            "cUDE",
            scenario_label(sec, s),
            median_iqr_string(sub.a_ratio_vs_full),
            median_iqr_string(sub.b_ratio_vs_full),
            median_iqr_string(sub.Cs0_ratio_vs_full),
            median_iqr_string(sub.Cc0_ratio_vs_full),
            "β: " * median_iqr_string(sub.beta_ratio_vs_full)
        ))
    end
end

CSV.write(params_out_file, params_rows)

println("Saved:")
println(" - ", metrics_out_file)
println(" - ", params_out_file)
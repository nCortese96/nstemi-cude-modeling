# ==============================================================================
#  Structural Identifiability Analysis
#  Models: (1) cUDE with NN correction (SR surrogate as rational proxy, β explicit)
#          (2) Surrogate (lumped parameter K_surr)
#          (3) ODE pure (sigmoidal release function with Td)
#
#  Configuration: Model 3 / cfg 2×8 / idx 3
#  Parameters per model:
#    cUDE:      {a, b, Cs0, Cc0, β}
#    Surrogate: {a, b, Cs0, Cc0, K_surr}
#    ODE pure:  {a, b, Cs0, Cc0, Td}
# ==============================================================================
#
# --- How StructuralIdentifiability.jl Works ---
#
# StructuralIdentifiability.jl assesses *structural* (a priori) identifiability
# of parametric ODE models from input-output data, BEFORE any parameter estimation.
#
# The package implements two complementary approaches:
#
# (A) LOCAL IDENTIFIABILITY (assess_local_identifiability):
#     Uses the *rank test on the Jacobian* of the coefficient map. Starting from
#     the ODE system, it computes a truncated power-series solution at a randomly
#     chosen parameter point and checks whether the map from parameters to
#     output coefficients has full rank. This is equivalent to checking that
#     the parameters can be locally (i.e., up to a finite number of solutions)
#     recovered from the output. A parameter is locally identifiable if the
#     Jacobian of the observability/identifiability matrix is full rank.
#     Complexity: polynomial in model size; very fast.
#
# (B) GLOBAL IDENTIFIABILITY (assess_identifiability):
#     Uses *differential algebra elimination* (Ritt's algorithm / characteristic
#     sets) to derive input-output equations that relate only the output, its
#     derivatives, the input, and the parameters—eliminating all state variables.
#     From these polynomial equations, it assesses whether the parameter-to-output
#     map is generically injective (globally identifiable), finite-to-one (locally
#     identifiable), or not injective (non-identifiable). This is a symbolic
#     computation carried out over the field of rational functions.
#     Complexity: potentially exponential in the worst case, but efficient for
#     small-to-medium models.
#
# IMPORTANT: The analysis is *structural*, meaning it depends only on the model
# equations and the observation structure, NOT on specific data or parameter values.
# It tells us whether parameters CAN be uniquely determined from perfect,
# noise-free, continuous-time input-output data.
#
# NOTE ON INITIAL CONDITIONS: StructuralIdentifiability.jl reports state
# identifiability as e.g. "Cs(t) => globally". This means the ENTIRE state
# trajectory (including the initial condition Cs(0)) is uniquely determined.
# Therefore, "Cs(t) globally identifiable" ⟹ Cs(0) is globally identifiable.
# We map Cs(t) → Cs0 and Cc(t) → Cc0 in the results tables below.
#
# Reference:
#   Dong, R., Goodbrake, C., Harrington, H., & Pogudin, G. (2023).
#   "Differential elimination for dynamical models via projections with
#   applications to structural identifiability." SIAM Journal on Applied Algebra
#   and Geometry, 7(1), 194-235.
#
#   Hong, H., Ovchinnikov, A., Pogudin, G., & Yap, C. (2019).
#   "SIAN: software for structural identifiability analysis of ODE models."
#   Bioinformatics, 35(16), 2873-2874.
#
# ==============================================================================

using StructuralIdentifiability
using Dates
using CSV
using DataFrames

# ==============================================================================
# CONSTANTS
# ==============================================================================
# Constant from symbolic regression (from sr_outputs_extended_2026-03-24)
const c_val = 0.0006774434799252378
const T_SCALE_SI = 240.0

report_dir = "res/structural"
mkpath(report_dir)

println("=" ^ 70)
println("Structural Identifiability Analysis")
println("Generated on: $(now())")
println("=" ^ 70)

# ==============================================================================
# MODEL 1: cUDE — Neural Network Correction with Explicit β
# ==============================================================================
#
# In the cUDE framework, the sarcomere rupture dynamics are governed by a
# neural network f_θ(τ, β) with FIXED, pre-trained weights θ* (cfg 2×8, idx 3).
#
# --- Justification for the rational surrogate representation ---
#
# StructuralIdentifiability.jl operates via differential algebra over fields
# of rational functions. Neural networks with nonlinear activations (tanh,
# sigmoid) are transcendental and cannot be directly encoded in @ODEmodel.
#
# To assess the identifiability of ALL model parameters including β, we
# represent the NN correction using its symbolic regression (SR) surrogate:
#
#   f_θ*(τ, β) ≈ τ^4 / (τ^4 + β²/c)
#
# where c = 0.0006774434799252378 is a known constant from SR fitting.
# This surrogate was shown to faithfully approximate the NN output across
# the training domain (see test_formula.jl validation results).
#
# Since c is a KNOWN constant (not a free parameter), it does not affect
# the structural identifiability of β. Formally:
#   β is identifiable in τ^4/(τ^4 + β²/c) ⟺ β is identifiable in τ^4/(τ^4 + β²)
# because dividing by a known nonzero constant is an invertible transformation.
# We therefore use the simplified form τ^4/(τ^4 + beta²) in the @ODEmodel.
#
# The @ODEmodel macro requires RHS to be rational in states and parameters
# only (no explicit time). We introduce an auxiliary clock state:
#   tau'(t) = 1, tau(0) = 0  ⟹  tau(t) = t
#
# ALTERNATIVE APPROACHES (not used here, documented for completeness):
#   - Treat the NN as a generic known input u(t). This tests identifiability
#     of {a, b, Cs0, Cc0} but β cannot be assessed since it is absorbed into
#     the generic function. See Ljung & Glad (1994), Villaverde et al. (2016).
#   - Use numerical local identifiability methods (profile likelihood, Fisher
#     information matrix) that handle the NN directly but provide only local
#     (not global) guarantees.
#   - Use a polynomial/rational approximation of the NN with fixed numerical
#     coefficients. This tests identifiability for that specific approximation.
# ==============================================================================

println("\n--- Model 1: cUDE (SR surrogate with explicit β) ---")

# Parameters: a, b, beta. States: Cs, Cc, Cp, tau.
# Initial conditions Cs(0)=Cs0, Cc(0)=Cc0 are implicitly tested via state identifiability.
# Correction: tau^4 / (tau^4 + beta^2), where the known constant 1/c is absorbed.

ode_cude = @ODEmodel(
    Cs'(t) = -(Cs(t) - Cc(t)) * tau(t)^4 / (tau(t)^4 + beta^2),
    Cc'(t) = (Cs(t) - Cc(t)) * tau(t)^4 / (tau(t)^4 + beta^2) - a * (Cc(t) - Cp(t)),
    Cp'(t) = a * (Cc(t) - Cp(t)) - b * Cp(t),
    tau'(t) = 1,
    y(t) = Cp(t)
)

print("  Assessing Local Identifiability...")
res_local_cude = assess_local_identifiability(ode_cude)
println(" Done.")

print("  Assessing Global Identifiability...")
res_global_cude = assess_identifiability(ode_cude)
println(" Done.")

# ==============================================================================
# MODEL 2: Surrogate — Symbolic Regression Formula (Lumped K_surr)
# ==============================================================================
#
# The surrogate replaces the NN with a closed-form rational function derived
# via symbolic regression (SymbolicRegression.jl / PySR):
#
#   correction(t, β) = (t/T_SCALE)^4 / ((t/T_SCALE)^4 + β²/c)
#
# where c = 0.0006774434799252378 is a fixed constant from SR.
#
# Here we use a LUMPED parameter K_surr that absorbs β, c, and T_SCALE:
#   K_surr = (β²/c) * T_SCALE^4
# The correction simplifies to: tau^4 / (tau^4 + K_surr)
#
# This tests whether K_surr (as a single lumped parameter) is identifiable.
# Unlike Model 1 which uses β² (introducing sign ambiguity ±β), K_surr is
# expected to be globally identifiable since K_surr > 0 by construction.
#
# Same clock trick: tau'(t) = 1.
# ==============================================================================

println("\n--- Model 2: Surrogate (lumped K_surr) ---")

ode_surrogate = @ODEmodel(
    Cs'(t) = -(Cs(t) - Cc(t)) * tau(t)^4 / (tau(t)^4 + K_surr),
    Cc'(t) = (Cs(t) - Cc(t)) * tau(t)^4 / (tau(t)^4 + K_surr) - a * (Cc(t) - Cp(t)),
    Cp'(t) = a * (Cc(t) - Cp(t)) - b * Cp(t),
    tau'(t) = 1,
    y(t) = Cp(t)
)

print("  Assessing Local Identifiability...")
res_local_surr = assess_local_identifiability(ode_surrogate)
println(" Done.")

print("  Assessing Global Identifiability...")
res_global_surr = assess_identifiability(ode_surrogate)
println(" Done.")

# ==============================================================================
# MODEL 3: ODE Pure — Sigmoidal Release Function
# ==============================================================================
#
# The mechanistic ODE model uses a Hill-type sigmoidal release function:
#
#   φ(t, Td) = t^3 / (t^3 + Td^3)
#
# where Td is a delay (half-activation) time parameter.
# Same clock trick: tau'(t) = 1.
#
# Parameters: {a, b, Td} + initial conditions {Cs0, Cc0}.
# ==============================================================================

println("\n--- Model 3: ODE pure (sigmoidal release τ³/(τ³+Td³)) ---")

ode_pure = @ODEmodel(
    Cs'(t) = -(Cs(t) - Cc(t)) * tau(t)^3 / (tau(t)^3 + Td^3),
    Cc'(t) = (Cs(t) - Cc(t)) * tau(t)^3 / (tau(t)^3 + Td^3) - a * (Cc(t) - Cp(t)),
    Cp'(t) = a * (Cc(t) - Cp(t)) - b * Cp(t),
    tau'(t) = 1,
    y(t) = Cp(t)
)

print("  Assessing Local Identifiability...")
res_local_ode = assess_local_identifiability(ode_pure)
println(" Done.")

print("  Assessing Global Identifiability...")
res_global_ode = assess_identifiability(ode_pure)
println(" Done.")

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

println("\n" * "=" ^ 70)
println("Generating reports...")

# --- Helper: rename state variables to meaningful parameter names ---
# Cs(t) → Cs0, Cc(t) → Cc0 (initial conditions), remove auxiliary states
const PARAM_RENAME = Dict(
    "Cs(t)" => "Cs0",
    "Cc(t)" => "Cc0",
)
const PARAM_SKIP = Set(["Cp(t)", "tau(t)"])  # auxiliary/always-observable states

function results_to_df(res_local, res_global)
    params = collect(keys(res_global))
    rows = []
    for p in params
        pname = string(p)
        pname in PARAM_SKIP && continue
        display_name = get(PARAM_RENAME, pname, pname)
        push!(rows, (
            Parameter = display_name,
            Local     = res_local[p],
            Global    = string(res_global[p])
        ))
    end
    DataFrame(rows)
end

df_cude = results_to_df(res_local_cude, res_global_cude)
df_surr = results_to_df(res_local_surr, res_global_surr)
df_ode  = results_to_df(res_local_ode, res_global_ode)

# --- CSV per model ---
CSV.write(joinpath(report_dir, "identifiability_results_cude.csv"), df_cude)
CSV.write(joinpath(report_dir, "identifiability_results_surrogate.csv"), df_surr)
CSV.write(joinpath(report_dir, "identifiability_results_ode.csv"), df_ode)

# --- Combined CSV ---
df_cude_tagged = copy(df_cude); df_cude_tagged.Model .= "cUDE"
df_surr_tagged = copy(df_surr); df_surr_tagged.Model .= "Surrogate_SR"
df_ode_tagged  = copy(df_ode);  df_ode_tagged.Model  .= "ODE_pure"
df_combined = vcat(df_cude_tagged, df_surr_tagged, df_ode_tagged)
select!(df_combined, :Model, :Parameter, :Local, :Global)
CSV.write(joinpath(report_dir, "identifiability_results_combined.csv"), df_combined)

# --- Text report ---
report_path = joinpath(report_dir, "identifiability_report.txt")
open(report_path, "w") do io
    println(io, "=" ^ 70)
    println(io, "STRUCTURAL IDENTIFIABILITY ANALYSIS REPORT")
    println(io, "Generated on: $(now())")
    println(io, "=" ^ 70)

    println(io, "\n## Method")
    println(io, "Tool: StructuralIdentifiability.jl")
    println(io, "Local identifiability: Jacobian rank test on power-series coefficients")
    println(io, "Global identifiability: Differential algebra elimination (characteristic sets)")
    println(io, "Note: Cs(t) and Cc(t) state identifiability imply Cs(0)=Cs0 and Cc(0)=Cc0")
    println(io, "      initial condition identifiability (mapped as Cs0, Cc0 in tables).")

    println(io, "\n" * "-" ^ 70)
    println(io, "MODEL 1: cUDE (SR surrogate with explicit β)")
    println(io, "-" ^ 70)
    println(io, "Structure: Cs' = -(Cs-Cc)*φ, Cc' = (Cs-Cc)*φ - a*(Cc-Cp), Cp' = a*(Cc-Cp) - b*Cp")
    println(io, "Correction: φ(t,β) = t⁴ / (t⁴ + β²)")
    println(io, "  (known constant c = $c_val absorbed; does not affect β identifiability)")
    println(io, "Parameters: {a, b, β, Cs0, Cc0}")
    println(io, "Observation: y(t) = Cp(t)\n")
    for row in eachrow(df_cude)
        println(io, "  $(rpad(row.Parameter, 12)) local=$(row.Local)  global=$(row.Global)")
    end

    println(io, "\n" * "-" ^ 70)
    println(io, "MODEL 2: Surrogate (Symbolic Regression, lumped K_surr)")
    println(io, "-" ^ 70)
    println(io, "Correction: φ(t) = t⁴ / (t⁴ + K_surr)")
    println(io, "  where K_surr = (β²/c) · T_SCALE⁴, c = $c_val, T_SCALE = $T_SCALE_SI h")
    println(io, "Parameters: {a, b, K_surr, Cs0, Cc0}")
    println(io, "Observation: y(t) = Cp(t)\n")
    for row in eachrow(df_surr)
        println(io, "  $(rpad(row.Parameter, 12)) local=$(row.Local)  global=$(row.Global)")
    end

    println(io, "\n" * "-" ^ 70)
    println(io, "MODEL 3: ODE Pure (Sigmoidal Release)")
    println(io, "-" ^ 70)
    println(io, "Correction: φ(t,Td) = t³ / (t³ + Td³)")
    println(io, "Parameters: {a, b, Td, Cs0, Cc0}")
    println(io, "Observation: y(t) = Cp(t)\n")
    for row in eachrow(df_ode)
        println(io, "  $(rpad(row.Parameter, 12)) local=$(row.Local)  global=$(row.Global)")
    end

    println(io, "\n" * "=" ^ 70)
    println(io, "END OF REPORT")
    println(io, "=" ^ 70)
end

# --- Console summary ---
println("\nResults saved to: $report_dir/")
println("  - identifiability_report.txt")
println("  - identifiability_results_cude.csv")
println("  - identifiability_results_surrogate.csv")
println("  - identifiability_results_ode.csv")
println("  - identifiability_results_combined.csv")

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("\nModel 1 — cUDE (params: a, b, β, Cs0, Cc0):")
display(df_cude)

println("\nModel 2 — Surrogate (params: a, b, K_surr, Cs0, Cc0):")
display(df_surr)

println("\nModel 3 — ODE pure (params: a, b, Td, Cs0, Cc0):")
display(df_ode)

println("\nAnalysis complete.")

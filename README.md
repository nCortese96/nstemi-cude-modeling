# MechanisticAI.jl

MechanisticAI.jl is a Julia research codebase for mechanistic and hybrid
modeling of cardiac troponin T (cTnT) trajectories.

The repository provides a sequential, config-driven workflow for:

- preprocessing patient-level cTnT cohorts;
- fitting a mechanistic ODE model;
- training and evaluating cUDE hybrid models;
- selecting the best cUDE candidate;
- running comparison analyses such as diagnostics, profile likelihood analysis,
  and systematic truncation;
- fitting and evaluating a symbolic surrogate.

The refactored workflow is designed to preserve the equations, parameter
bounds, losses, metrics, patient ordering, and output conventions used by the
original analyses.

## Repository Layout

```text
.
├── config/
│   └── workflow_config.jl        # Central user-editable workflow settings
├── scripts/                      # Numbered executable workflow steps
├── src/                          # Reusable Julia helpers
├── data/                         # Input datasets expected by the workflow
├── results/                      # Official output tree
└── results_test/                 # Test-mode output tree
```

Important helper files in `src/` include:

```text
data_io.jl                 # paths, artifact IO, cohort loading, CLI parsers
preprocessing.jl           # preprocessing and cohort report generation
models.jl                  # model definitions, metrics, losses, formula logic
fitting.jl                 # optimization, ODE/cUDE/formula fitting
model_selection.jl         # cUDE model-selection helpers
diagnostics.jl             # model-comparison diagnostics
profile_likelihood.jl      # profile likelihood numerical workflow
systematic_truncation.jl   # systematic truncation numerical workflow
symbolic_regression.jl     # symbolic regression helpers
plotting.jl                # shared plotting helpers
```

## Installation

Run commands from the repository root.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Input datasets are expected under `data/`. Some patient-level datasets may be
restricted and are not necessarily distributed with the public repository.

## Configuration

Workflow settings live in:

```text
config/workflow_config.jl
```

The central object is `WORKFLOW_CONFIG`. The most frequently edited settings
are:

- `WORKFLOW_CONFIG.run.test_mode`: when `true`, outputs are written to
  `results_test/`; when `false`, outputs are written to `results/`.
- `WORKFLOW_CONFIG.run.progress_bars`: enables or disables progress bars in
  supported long-running steps.
- `WORKFLOW_CONFIG.model.t_scale`: shared analysis/model time scale in hours.
- `WORKFLOW_CONFIG.datasets`: dataset registry.
- Step-specific settings such as `cude_training`, `cude_evaluation`,
  `profile_likelihood`, `systematic_truncation`, and
  `symbolic_formula_evaluation`.

The `results/` and `results_test/` trees are intentionally mirrored. Use
`results_test/` for trial runs and validation; use `results/` for official
outputs.

## CLI Convention

All refactored workflow scripts are executable from the command line.

The plain command runs the full step:

```bash
julia --project=. scripts/<step_script>.jl
```

For computationally expensive scripts, use Julia threads:

```bash
JULIA_NUM_THREADS=auto julia --project=. scripts/<step_script>.jl
JULIA_NUM_THREADS=8 julia --project=. scripts/<step_script>.jl
```

For scripts that support plot-only regeneration, use:

```bash
julia --project=. scripts/<step_script>.jl plots
```

Plot-only modes reuse existing CSV/JLD2 artifacts and do not rerun fitting,
training, profile likelihood computation, or truncation optimization. If a full
step is rerun, plots are regenerated and overwritten as part of the workflow.

## Workflow Steps

Run the scripts in numerical order when building results from scratch.

| Step | Script | Purpose |
| --- | --- | --- |
| 00 | `scripts/00_run_preprocessing.jl` | Build preprocessed cohorts, reports, split artifacts, and gold-standard IDs. |
| 01 | `scripts/01_run_ode_tdsigmoid_fit.jl` | Fit the mechanistic ODE Td-sigmoid model. |
| 02a | `scripts/02a_run_cude_training.jl` | Train cUDE candidates. |
| 02b | `scripts/02b_evaluate_cude_nn.jl` | Evaluate cUDE candidates on MIMIC-IV validation/test data. |
| 02c | `scripts/02c_grid_search.jl` | Select the cUDE model from validation summaries. |
| 02d | `scripts/02d_evaluate_cude_nn_external_test.jl` | Evaluate the selected cUDE model on the external UMG cohort. |
| 03a | `scripts/03a_run_model_diagnostics.jl` | Generate ODE/cUDE diagnostics and comparison figures. |
| 03b | `scripts/03b_run_profile_likelihood.jl` | Run profile likelihood analysis for ODE and cUDE targets. |
| 03c | `scripts/03c_run_systematic_truncation.jl` | Run systematic truncation stress tests and overlays. |
| 04a | `scripts/04a_run_symbolic_regression.jl` | Fit a symbolic surrogate for the selected cUDE correction function. |
| 04b | `scripts/04b_evaluate_symbolic_formula.jl` | Evaluate the fixed symbolic surrogate formula on test cohorts. |

## Command Reference

### Core Pipeline

```bash
julia --project=. scripts/00_run_preprocessing.jl
JULIA_NUM_THREADS=auto julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl
JULIA_NUM_THREADS=auto julia --project=. scripts/02a_run_cude_training.jl
JULIA_NUM_THREADS=auto julia --project=. scripts/02b_evaluate_cude_nn.jl
julia --project=. scripts/02c_grid_search.jl
JULIA_NUM_THREADS=auto julia --project=. scripts/02d_evaluate_cude_nn_external_test.jl
```

### Comparison Analyses

```bash
JULIA_NUM_THREADS=auto julia --project=. scripts/03a_run_model_diagnostics.jl
JULIA_NUM_THREADS=auto julia --project=. scripts/03b_run_profile_likelihood.jl
JULIA_NUM_THREADS=auto julia --project=. scripts/03c_run_systematic_truncation.jl
```

Plot-only regeneration:

```bash
julia --project=. scripts/03a_run_model_diagnostics.jl plots
julia --project=. scripts/03b_run_profile_likelihood.jl plots
julia --project=. scripts/03c_run_systematic_truncation.jl plots
```

Current 03a plot-only mode regenerates residual diagnostic plots from existing
residual CSV files.

### PLA Targets And Plot Modes

Profile likelihood analysis supports target and plot-mode arguments.

Targets:

```text
all
cude_mimic
cude_umg
ode_mimic
ode_umg
```

Plot-only modes:

```text
plots
plots_patients
plots_aggregate
```

Examples:

```bash
JULIA_NUM_THREADS=auto julia --project=. scripts/03b_run_profile_likelihood.jl cude_mimic
julia --project=. scripts/03b_run_profile_likelihood.jl plots_aggregate
julia --project=. scripts/03b_run_profile_likelihood.jl cude_mimic plots
julia --project=. scripts/03b_run_profile_likelihood.jl plots ode_umg
```

### Systematic Truncation Targets

Systematic truncation supports:

```text
all
ode
cude
summary
overlay
plots
```

Examples:

```bash
JULIA_NUM_THREADS=auto julia --project=. scripts/03c_run_systematic_truncation.jl ode
JULIA_NUM_THREADS=auto julia --project=. scripts/03c_run_systematic_truncation.jl cude
julia --project=. scripts/03c_run_systematic_truncation.jl summary
julia --project=. scripts/03c_run_systematic_truncation.jl overlay
julia --project=. scripts/03c_run_systematic_truncation.jl plots
```

The `plots` target regenerates patient-level truncation figures and ODE-vs-cUDE
overlays from existing truncation CSV artifacts. Overlay generation also writes
an axis-title-free mirror under `truncation_overlay_comparison/no_labels/`,
while preserving numeric tick labels.

### Symbolic Surrogate

```bash
JULIA_NUM_THREADS=auto julia --project=. scripts/04a_run_symbolic_regression.jl
JULIA_NUM_THREADS=auto julia --project=. scripts/04b_evaluate_symbolic_formula.jl
```

Symbolic regression can be computationally expensive and stochastic. The fixed
surrogate formula evaluated by step 04b is defined explicitly in `src/models.jl`
for reproducibility.

## Output Trees

Canonical output roots:

```text
results/
results_test/
```

Current step-level directories:

```text
results*/00_cohorts
results*/01_ode_evaluation
results*/02_cude_workflow
results*/03_comparison_analyses
results*/04_symbolic_surrogate
```

The active root is selected by `WORKFLOW_CONFIG.run.test_mode`.

## Reproducibility Notes

The workflow is intended to preserve the scientific behavior of the original
analysis code. Exact numerical reproduction can still depend on:

- access to the same patient-level input datasets;
- Julia and package versions;
- stochastic initialization in cUDE training and symbolic regression;
- whether previously saved initial parameters are reused where supported.

For long-running steps, run from a clean terminal with an explicit
`JULIA_NUM_THREADS` value and keep `config/workflow_config.jl` under version
control for the run being reproduced.

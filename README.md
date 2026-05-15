# MechanisticAI.jl

MechanisticAI.jl is a Julia research codebase for mechanistic and hybrid
modeling of cardiac troponin T (cTnT) trajectories.

The project organizes a sequential workflow for preprocessing patient-level
cTnT data, fitting mechanistic ODE models, training hybrid cUDE models,
selecting models, and running downstream analyses such as diagnostics, profile
likelihood analysis, systematic truncation, symbolic regression, and surrogate
evaluation.

The current codebase is being consolidated into a reproducible, config-driven
workflow while preserving the scientific equations, parameter bounds, metrics,
and output conventions used by the original analyses.

## Repository Layout

```text
.
├── config/
│   └── workflow_config.jl        # Central settings for refactored workflow steps
├── scripts/
│   ├── 00_run_preprocessing.jl   # Cohort preprocessing and artifact export
│   ├── 01_run_ode_tdsigmoid_fit.jl
│   ├── 02a_run_cude_training.jl
│   ├── 02b_evaluate_cude_nn.jl
│   ├── 02c_grid_search.jl
│   ├── 02d_evaluate_cude_nn_external_test.jl
│   ├── 03_model_diagnostic.jl
│   ├── 04_simple_pla_afs_multimodel.jl
│   ├── 05_run_systematic_truncation.jl
│   ├── 06_sym_reg_controlled.jl
│   └── 07_evaluate_symbolic_formula.jl
├── src/
│   ├── MechanisticAI.jl          # Shared include entrypoint
│   ├── data_io.jl                # Workflow IO, patient data IO, cohort loading
│   ├── preprocessing.jl          # Preprocessing, reporting, and cohort export
│   ├── models.jl                 # Model definitions, metrics, and losses
│   ├── fitting.jl                # Optimization and fitting helpers
│   └── helpers.jl                # Remaining helpers pending full consolidation
├── data/                         # Input datasets expected by workflow scripts
├── results/                      # Official output tree for refactored runs
└── results_test/                 # Test-mode output tree
```

## Workflow Overview

The refactored workflow is represented by numbered scripts in `scripts/`.
Numbers describe the intended execution order.

| Step | Script | Purpose |
| --- | --- | --- |
| 00 | `00_run_preprocessing.jl` | Load raw datasets, collapse duplicate timepoints, trim by time, apply eligibility filters, write reports and cohort artifacts. |
| 01 | `01_run_ode_tdsigmoid_fit.jl` | Fit the mechanistic ODE model with Td-sigmoid release on preprocessed cohorts. |
| 02a | `02a_run_cude_training.jl` | Train cUDE models. |
| 02b | `02b_evaluate_cude_nn.jl` | Evaluate cUDE models during validation. |
| 02c | `02c_grid_search.jl` | Select candidate models from validation summaries. |
| 02d | `02d_evaluate_cude_nn_external_test.jl` | Evaluate the selected cUDE model on an external dataset. |
| 03 | `03_model_diagnostic.jl` | Model diagnostics, including parameter and residual analyses. |
| 04 | `04_simple_pla_afs_multimodel.jl` | Profile likelihood analysis for ODE and cUDE models. |
| 05 | `05_run_systematic_truncation.jl` | Systematic truncation analysis. |
| 06 | `06_sym_reg_controlled.jl` | Symbolic regression based on the selected cUDE model. |
| 07 | `07_evaluate_symbolic_formula.jl` | Surrogate formula fitting and evaluation. |

## Configuration

Workflow settings live in:

```text
config/workflow_config.jl
```

The central configuration object is `WORKFLOW_CONFIG`. It includes:

```julia
WORKFLOW_CONFIG.paths
WORKFLOW_CONFIG.run
WORKFLOW_CONFIG.outputs
WORKFLOW_CONFIG.datasets
WORKFLOW_CONFIG.model
WORKFLOW_CONFIG.preprocessing
WORKFLOW_CONFIG.ode_tdsigmoid
```

Important settings include:

- `WORKFLOW_CONFIG.run.test_mode`: writes outputs to `results_test/` when
  enabled.
- `WORKFLOW_CONFIG.run.progress_bars`: enables or disables progress bars in
  long-running steps that support them.
- `WORKFLOW_CONFIG.outputs.cohorts`: step 00 cohort output directory.
- `WORKFLOW_CONFIG.outputs.ode_evaluation`: step 01 ODE evaluation output
  directory.
- `WORKFLOW_CONFIG.model.t_scale`: model-domain time scale in hours.
- `WORKFLOW_CONFIG.datasets`: dataset registry used by workflow scripts.

The `t_scale` setting is shared by preprocessing and model code. It defines the
analysis window during preprocessing and is also used by cUDE models to
normalize time.

## Running The Workflow

Run commands from the repository root with the project environment active.

### Step 00: Preprocessing

```bash
julia --project=. scripts/00_run_preprocessing.jl
```

This step reads raw Excel datasets from `data/`, writes preprocessing reports,
exports all-eligible patient IDs, and saves JLD2 cohort artifacts under the
configured cohort output directory.

When `WORKFLOW_CONFIG.run.test_mode=true`, outputs are written under
`results_test/00_cohorts`. Otherwise they are written under
`results/00_cohorts`.

### Step 01: ODE Td-Sigmoid Fit

```bash
julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl
```

This step loads the preprocessed cohorts produced by step 00 and fits one
mechanistic ODE model per patient. Outputs are written under
`results*/01_ode_evaluation`.

For threaded execution, start Julia with a thread count before running the
script:

```bash
JULIA_NUM_THREADS=auto julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl
```

or:

```bash
JULIA_NUM_THREADS=8 julia --project=. scripts/01_run_ode_tdsigmoid_fit.jl
```

Progress bars and plotting behavior are controlled from
`config/workflow_config.jl`.

## Output Trees

Refactored runs use two mirrored output roots:

```text
results/
results_test/
```

Use `results/` for official outputs and `results_test/` for exploratory or
validation runs. The active root is selected by `WORKFLOW_CONFIG.run.test_mode`.

Current step-level output directories include:

```text
results*/00_cohorts
results*/01_ode_evaluation
results*/02_cude_workflow
results*/03_comparison_analyses
```

## Notes

The repository is still being consolidated. Numbered workflow scripts are the
active execution path, while helper code in `src/` is being progressively split
into focused modules.

Later workflow steps may still contain code inherited from the original
analysis scripts and will be cleaned as their steps are refactored.

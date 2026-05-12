# MechanisticAI.jl

MechanisticAI.jl is a Julia research codebase for mechanistic and hybrid
modeling of cardiac troponin T (cTnT) trajectories. The repository currently
contains the original analysis workflow and an ongoing refactor whose goal is
to make the pipeline easier to read, reproduce, and extend.

The refactor is intentionally conservative: scientific equations, parameter
bounds, metrics, and output conventions are being preserved while scripts are
renamed, reorganized, and connected to shared helpers.

## Current Repository Structure

```text
.
├── config/
│   └── workflow_config.jl        # Central workflow settings used by refactored scripts
├── scripts/
│   ├── 00_run_preprocessing.jl   # Dataset loading, cleaning, eligibility report, split export
│   ├── 01_run_ode_tdsigmoid_fit.jl
│   ├── 02a_run_cude_training.jl
│   ├── 02b_evaluate_cude_nn.jl
│   ├── 02c_grid_search.jl
│   ├── 02d_evaluate_cude_nn_external_test.jl
│   ├── 03_model_diagnostic.jl    # Placeholder for the diagnostics workflow
│   ├── 04_simple_pla_afs_multimodel.jl
│   ├── 05_run_systematic_truncation.jl
│   ├── 06_sym_reg_controlled.jl
│   └── 07_evaluate_symbolic_formula.jl
├── src/
│   ├── MechanisticAI.jl          # Shared entrypoint for refactored scripts
│   └── helpers.jl                # Consolidated helpers during the transition
├── data/                         # Input datasets, not part of the refactor itself
├── res/                          # Legacy output tree from previous runs
├── results/                      # New output root for refactored runs
└── .legacy/src/                  # Original scripts kept as reference baselines
```

Some plotting and diagnostic scripts are still present in `scripts/` while the
step `03_model_diagnostic.jl` is being designed. They have not yet been folded
into the numbered workflow.

## Refactored Workflow

The refactored workflow is represented by numbered scripts in `scripts/`.
Numbers describe execution order; names keep the original script intent visible.

| Step | Script | Purpose |
| --- | --- | --- |
| 00 | `00_run_preprocessing.jl` | Load datasets, collapse duplicate timepoints, trim by time, apply eligibility filters, export reports and patient sets. |
| 01 | `01_run_ode_tdsigmoid_fit.jl` | Fit the mechanistic ODE with Td-sigmoid release. |
| 02a | `02a_run_cude_training.jl` | Train the cUDE model. |
| 02b | `02b_evaluate_cude_nn.jl` | Evaluate cUDE models during validation/model comparison. |
| 02c | `02c_grid_search.jl` | Select candidate models from validation summaries. |
| 02d | `02d_evaluate_cude_nn_external_test.jl` | Evaluate the selected cUDE model on an external dataset. |
| 03 | `03_model_diagnostic.jl` | Placeholder for parameter, residual, and plotting diagnostics. |
| 04 | `04_simple_pla_afs_multimodel.jl` | Profile likelihood analysis for ODE/cUDE models. |
| 05 | `05_run_systematic_truncation.jl` | Systematic truncation analysis. |
| 06 | `06_sym_reg_controlled.jl` | Symbolic regression on the selected cUDE model. |
| 07 | `07_evaluate_symbolic_formula.jl` | Surrogate formula fitting/evaluation. |

## Configuration

Shared settings for the refactored workflow live in:

```text
config/workflow_config.jl
```

The public configuration object is `WORKFLOW_CONFIG`, currently structured as:

```julia
WORKFLOW_CONFIG.paths
WORKFLOW_CONFIG.datasets
WORKFLOW_CONFIG.model
WORKFLOW_CONFIG.preprocessing
```

Important current settings:

- `WORKFLOW_CONFIG.paths.data_root = "data"`
- `WORKFLOW_CONFIG.paths.results_root = "results"`
- `WORKFLOW_CONFIG.model.t_scale = 240.0`
- `WORKFLOW_CONFIG.preprocessing.dataset_keys = (:mimic_iv, :umg)`
- `WORKFLOW_CONFIG.preprocessing.output_dir = "results"`

`t_scale` is a model-domain time scale in hours. It is used by preprocessing to
define the analysis window and by cUDE models to normalize time as `t / t_scale`.
It is therefore stored under `WORKFLOW_CONFIG.model`, not under preprocessing.

## Current Refactor Progress

Completed in the current refactor pass:

- Created `src/MechanisticAI.jl` as the shared include entrypoint.
- Consolidated model, data IO, preprocessing, reporting, loss, diagnostics,
  PLA, log parsing, and multi-start optimization helpers into `src/helpers.jl`.
- Integrated `MultiStartOptimizer` into `helpers.jl` while preserving the
  `MultiStartOptimizer.run_multistart` call pattern.
- Moved legacy baseline scripts into `.legacy/src/`.
- Created the numbered workflow scripts in `scripts/`.
- Created `config/workflow_config.jl`.
- Refactored `scripts/00_run_preprocessing.jl` to read settings directly from
  `WORKFLOW_CONFIG` and to run as a simple top-level pipeline.
- Created `results/` as the root for future refactored outputs. It is currently
  intended to stay empty until the workflow is executed.

## How To Run The Current Preprocessing Step

From the repository root:

```bash
julia --project=. scripts/00_run_preprocessing.jl
```

This step reads Excel inputs from `data/`, writes preprocessing reports and
patient-set artifacts to `results/`, and uses the settings in
`config/workflow_config.jl`.

The downstream scripts are present but have not all been fully cleaned to the
same standard as step 00 yet.

## Notes For Continuing The Refactor

Recommended next work:

1. Apply the same config-driven, top-level pipeline style used in
   `00_run_preprocessing.jl` to `01_run_ode_tdsigmoid_fit.jl`.
2. Move step-specific constants from scripts into `WORKFLOW_CONFIG` only when
   they are shared, user-editable, or needed for reproducibility.
3. Keep `helpers.jl` modular by section until the helper surface stabilizes;
   then split it into dedicated files that mirror its section structure.
4. Design `03_model_diagnostic.jl` before moving the remaining diagnostic and
   plotting scripts out of their provisional state.
5. Keep `.legacy/src/` untouched unless a legacy script is needed as a reference
   while refactoring its numbered replacement.

## Development Status

This repository is not yet in its final public form. The README is already
written in the intended public style, but the implementation is mid-refactor.
Until the workflow is fully validated, treat `.legacy/src/` as the source of
historical behavior and `scripts/` as the active refactor line.

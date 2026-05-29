"""
02c_grid_search.jl

Select the best cUDE candidate model from the step 02b validation summaries.

Pipeline:
1. Load shared helpers and workflow configuration.
2. Resolve step 02b summary inputs and step 02c output paths.
3. Load and validate all configured width-level model summaries.
4. Rank candidates with the configured model-selection rule.
5. Save the ranked general summary, selected model, per-width winners, report, and plots.

Command line:
    julia --project=. scripts/02c_grid_search.jl

Use `config/workflow_config.jl` to switch between `results/` and
`results_test/`, change widths, disable plots, or change the selection rule.
"""

# =============================================================================
# IMPORTS AND SHARED HELPERS
# Minimal dependencies used directly by this executable workflow script.
# =============================================================================

using Logging

include(joinpath(@__DIR__, "..", "src", "data_io.jl"))
include(joinpath(@__DIR__, "..", "src", "model_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
include(joinpath(@__DIR__, "..", "config", "workflow_config.jl"))

# =============================================================================
# SCRIPT SETTINGS
# User-editable settings live in `config/workflow_config.jl`.
# =============================================================================

config = WORKFLOW_CONFIG
settings = config.cude_model_selection
dataset_config = config.datasets[settings.dataset_key]

# =============================================================================
# INPUT PATHS
# Files and folders loaded by this run.
# =============================================================================

dataset_name = dataset_config.dataset_name
input_dir = settings.input_dir
widths = settings.widths

# =============================================================================
# OUTPUT PATHS
# Files and folders produced by this run.
# =============================================================================

output_root = settings.output_dir
paths = cude_model_selection_output_paths(output_root, dataset_name)

# =============================================================================
# DERIVED SETTINGS
# Values derived from config and loaded artifacts. No heavy side effects here.
# =============================================================================

selection_rule = settings.selection_rule
ranking_columns = cude_model_selection_sort_columns(selection_rule)

# =============================================================================
# PIPELINE
# =============================================================================

@info "cUDE model-selection workflow started."
log_workflow_context(
    config;
    script_name="02c_grid_search.jl",
    output_paths=(input_dir=input_dir, output_root=output_root),
)

@info "Dataset: $(dataset_name)"
@info "Widths: $(collect(widths))"
@info "Selection rule: $(selection_rule)"
@info "Ranking columns: $(ranking_columns)"

ensure_output_dirs!(output_root; header="Ensured step 02c output root")
log_output_paths(
    (
        general_summary=paths.general_summary,
        selected_model=paths.selected_model,
        best_by_width=paths.best_by_width,
        report=paths.report,
        figures=paths.fig_dir,
    );
    header="cUDE model-selection output files",
)

@info "Loading step 02b model summaries."
general_summary = load_cude_model_summaries(input_dir, widths, dataset_name)
validate_cude_model_summary!(general_summary; source_path=input_dir)
@info "Loaded $(size(general_summary, 1)) cUDE candidate rows."

selection = select_cude_models(general_summary, selection_rule)
save_cude_model_selection_outputs(paths, selection)
write_cude_model_selection_report(
    paths.report,
    selection;
    dataset_name=dataset_name,
    widths=widths,
    input_dir=input_dir,
    output_dir=output_root,
)

save_model_selection_plots(paths, selection; plotting=settings.plotting)

selected = selection.selected_model
@info "Selected cUDE model: $(selected.model_id[1]) | width=$(selected.nn_width[1]) | model_idx=$(selected.model_idx[1])"
@info "cUDE model-selection workflow completed."

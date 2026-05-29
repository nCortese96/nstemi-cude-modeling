"""
model_selection.jl

cUDE model-selection helpers shared by workflow step 02c and downstream scripts.

Sections:
- Summary Validation: canonical 02b model-summary schema checks.
- Ranking Rules: configurable global and per-width candidate ranking.
"""

using DataFrames

# =============================================================================
# Summary Validation
# =============================================================================

const CUDE_MODEL_SUMMARY_REQUIRED_COLUMNS = [
    :model_id, :model_idx, :nn_depth, :nn_width, :n_patients,
    :loss_mean, :loss_std, :loss_median, :loss_q1, :loss_q3, :loss_iqr,
    :smape_mean, :smape_std, :smape_median, :smape_q1, :smape_q3, :smape_iqr,
    :rmsle_mean, :rmsle_std, :rmsle_median, :rmsle_q1, :rmsle_q3, :rmsle_iqr,
]

"""
    validate_cude_model_summary!(df; source_path="")

Validate that a step 02b `models_summary` table exposes the canonical columns
needed by step 02c model selection.
"""
# Used by: scripts/02c_grid_search.jl.
function validate_cude_model_summary!(df::DataFrame; source_path::AbstractString="")
    missing_columns = setdiff(CUDE_MODEL_SUMMARY_REQUIRED_COLUMNS, Symbol.(names(df)))
    isempty(missing_columns) ||
        error("Missing columns in $(source_path): $(missing_columns)")

    isempty(df) && error("Empty cUDE model summary: $(source_path)")
    return df
end

# =============================================================================
# Ranking Rules
# =============================================================================

"""
    cude_model_selection_sort_columns(selection_rule)

Return the ordered ranking columns for a configured cUDE model-selection rule.
"""
# Used by: src/model_selection.jl and scripts/02c_grid_search.jl.
function cude_model_selection_sort_columns(selection_rule::Symbol)
    if selection_rule == :robust_loss_mean
        return [:loss_mean, :loss_std, :smape_mean, :smape_std, :rmsle_mean, :rmsle_std]
    elseif selection_rule == :robust_smape_median
        return [:smape_median, :smape_iqr, :rmsle_median, :rmsle_iqr, :loss_median, :loss_iqr]
    else
        error("Unsupported cUDE model-selection rule: $(selection_rule). Supported rules: :robust_loss_mean, :robust_smape_median.")
    end
end

"""
    rank_cude_models(df, selection_rule)

Rank cUDE candidate rows by the configured selection rule.
"""
# Used by: src/model_selection.jl and scripts/02c_grid_search.jl.
function rank_cude_models(df::DataFrame, selection_rule::Symbol)
    validate_cude_model_summary!(df)
    sort_columns = cude_model_selection_sort_columns(selection_rule)

    return sort(copy(df), sort_columns, rev=fill(false, length(sort_columns)))
end

"""
    add_legacy_selection_columns!(df, selection_columns)

Attach legacy-style selection metadata columns to the selected-model row.
"""
# Used by: src/model_selection.jl.
function add_legacy_selection_columns!(df::DataFrame, selection_columns)
    isempty(selection_columns) && return df

    df[!, :selection_primary] = fill(String(selection_columns[1]), nrow(df))
    for (idx, column) in enumerate(selection_columns[2:end])
        df[!, Symbol("selection_tiebreak_$(idx)")] = fill(String(column), nrow(df))
    end

    return df
end

"""
    legacy_best_by_width_table(general_summary, selection_rule)

Build the compact legacy `robust_best_by_width` table for step 02c.
"""
# Used by: src/model_selection.jl.
function legacy_best_by_width_table(general_summary::DataFrame, selection_rule::Symbol)
    rows = DataFrame[]

    for width_group in groupby(general_summary, :nn_width)
        ranked_width = rank_cude_models(DataFrame(width_group), selection_rule)
        best = ranked_width[1, :]

        push!(
            rows,
            DataFrame(
                nn_width=[best.nn_width],
                best_model_id=[best.model_id],
                best_model_idx=[best.model_idx],
                best_loss_mean=[best.loss_mean],
                best_loss_std=[best.loss_std],
                best_smape_mean=[best.smape_mean],
                best_smape_std=[best.smape_std],
                best_rmsle_mean=[best.rmsle_mean],
                best_rmsle_std=[best.rmsle_std],
                best_loss_median=[best.loss_median],
                best_loss_iqr=[best.loss_iqr],
                best_smape_median=[best.smape_median],
                best_smape_iqr=[best.smape_iqr],
                best_rmsle_median=[best.rmsle_median],
                best_rmsle_iqr=[best.rmsle_iqr],
                pool_size=[nrow(ranked_width)],
            ),
        )
    end

    best_by_width = vcat(rows...)
    sort!(best_by_width, :nn_width)
    return best_by_width
end

"""
    select_cude_models(general_summary, selection_rule)

Select the best global cUDE candidate and the best candidate for each neural
width.
"""
# Used by: scripts/02c_grid_search.jl.
function select_cude_models(general_summary::DataFrame, selection_rule::Symbol)
    ranked = rank_cude_models(general_summary, selection_rule)
    selection_columns = cude_model_selection_sort_columns(selection_rule)
    selected_model = add_legacy_selection_columns!(copy(ranked[1:1, :]), selection_columns)
    best_by_width = legacy_best_by_width_table(general_summary, selection_rule)

    return (
        general_summary=ranked,
        selected_model=selected_model,
        best_by_width=best_by_width,
        selection_rule=selection_rule,
        selection_columns=selection_columns,
    )
end

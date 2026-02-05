using CSV, JLD2
using DataFrames
using Statistics
using Printf
using ProgressMeter

using Revise
includet(joinpath(@__DIR__, "ctnt-ude-model.jl"))
# -----------------------------
# Helpers
# -----------------------------
function parse_bracket_vector(s::AbstractString)::Vector{Float64}
    s2 = strip(s)
    s2 = replace(s2, '[' => "", ']' => "")
    s2 = strip(s2)
    isempty(s2) && return Float64[]
    return parse.(Float64, strip.(split(s2, ',')))
end

function safe_mkdir(path::AbstractString)
    isdir(path) || mkpath(path)
end

# -----------------------------
# Parse metadata + best model id
# -----------------------------
function parse_log_metadata(lines::Vector{String})
    meta = Dict{String,Any}()

    for ln in lines
        if (m = match(r"^Experiment\s+(.*)\s+log file\s*$", ln)) !== nothing
            meta["experiment"] = m.captures[1]
        elseif (m = match(r"^dept:\s*(\d+);\s*width:\s*(\d+);\s*inputs\((\d+)\):", ln)) !== nothing
            meta["nn_depth"] = parse(Int, m.captures[1])
            meta["nn_width"] = parse(Int, m.captures[2])
            meta["input_dim"] = parse(Int, m.captures[3])
        elseif (m = match(r"^dataset:\s*(.*)\s*$", ln)) !== nothing
            meta["dataset_path"] = m.captures[1]
        elseif (m = match(r"^Training split:\s*(\d+)\s*$", ln)) !== nothing
            meta["n_train"] = parse(Int, m.captures[1])
        elseif (m = match(r"^Validation split:\s*(\d+)\s*$", ln)) !== nothing
            meta["n_val"] = parse(Int, m.captures[1])
        elseif (m = match(r"^Best model id:\s*(\d+)\s*$", ln)) !== nothing
            meta["best_model_id"] = parse(Int, m.captures[1])
        end
    end

    return meta
end

# -----------------------------
# Parse per-patient params + loss in validation (MIMIC block)
# These are the lines:
#   For nXXXX, params: [...]
#   Params: [...]
#   Patient loss model k: ...
# Count should match n_val for each k.
# -----------------------------
function parse_validation_params_losses(lines::Vector{String})
    df = DataFrame(
        dataset = String[],
        model_id = Int[],
        patient_id = String[],
        log_a = Float64[],
        log_b = Float64[],
        log_Cs0 = Float64[],
        log_Cc0 = Float64[],
        log_beta = Float64[],
        a = Float64[],
        b = Float64[],
        Cs0 = Float64[],
        Cc0 = Float64[],
        beta = Float64[],
        loss = Float64[]
    )

    cur_patient = nothing
    cur_log = nothing
    cur_exp = nothing
    cur_dataset = "MIMIC-IV"  # nel log i blocchi loss/params sono relativi alla validation interna di MIMIC

    for ln in lines
        # stop parsing params/loss when UMG test starts (non hai stampato params/loss per UMG in questo log)
        if startswith(ln, "UMG - Test NN started")
            break
        end

        if (m = match(r"^For\s+(n\d+),\s*params:\s*(\[.*\])\s*$", ln)) !== nothing
            cur_patient = m.captures[1]
            cur_log = parse_bracket_vector(m.captures[2])
            cur_exp = nothing
            continue
        end

        if (m = match(r"^Params:\s*(\[.*\])\s*$", ln)) !== nothing
            cur_exp = parse_bracket_vector(m.captures[1])
            continue
        end

        if (m = match(r"^Patient loss model\s+(\d+):\s*([0-9eE\+\-\.]+)\s*$", ln)) !== nothing
            model_id = parse(Int, m.captures[1])
            loss = parse(Float64, m.captures[2])

            if cur_patient === nothing || cur_log === nothing || cur_exp === nothing
                @warn "Skipping incomplete triple around: $ln"
                continue
            end

            if length(cur_log) < 5 || length(cur_exp) < 5
                @warn "Unexpected param length for $(cur_patient): log=$(length(cur_log)) exp=$(length(cur_exp))"
                continue
            end

            push!(df, (
                cur_dataset,
                model_id,
                String(cur_patient),
                cur_log[1], cur_log[2], cur_log[3], cur_log[4], cur_log[5],
                cur_exp[1], cur_exp[2], cur_exp[3], cur_exp[4], cur_exp[5],
                loss
            ))

            cur_patient = nothing
            cur_log = nothing
            cur_exp = nothing
        end
    end

    return df
end

# -----------------------------
# Parse sMAPE blocks (MIMIC validation, UMG test)
# Headers:
#   MIMIC-IV - Evaluating NN with sMAPE (idx:K)
#   UMG - Evaluating NN with sMAPE (idx:K)
# Lines:
#   Patient nXXXX sMAPE NN validation: VAL
# We store run_id because you have repeated blocks (UMG twice, MIMIC multiple idx).
# -----------------------------
function parse_smape_blocks(lines::Vector{String}, best_model_id::Union{Missing,Int})
    df = DataFrame(
        dataset = String[],
        phase = String[],      # "validation" (MIMIC) o "test" (UMG)
        run_id = Int[],
        model_id = Union{Missing,Int}[],
        patient_id = String[],
        smape = Float64[]
    )

    cur_dataset = nothing
    cur_phase = nothing
    cur_model_id = missing
    run_id = 0

    for ln in lines
        if (m = match(r"^(MIMIC-IV|UMG)\s+-\s+Evaluating NN with sMAPE(?:\s+\(idx:(\d+)\))?\s*$", ln)) !== nothing
            cur_dataset = m.captures[1]
            cur_phase = (cur_dataset == "UMG") ? "test" : "validation"
            cur_model_id = (m.captures[2] === nothing) ? missing : parse(Int, m.captures[2])
            run_id += 1
            continue
        end

        if cur_dataset === nothing
            continue
        end

        if (m = match(r"^Patient\s+(n\d+)\s+sMAPE NN validation:\s*([0-9eE\+\-\.]+)\s*$", ln)) !== nothing
            pid = m.captures[1]
            sval = parse(Float64, m.captures[2])
            push!(df, (String(cur_dataset), String(cur_phase), run_id, cur_model_id, String(pid), sval))
        end
    end

    # se esiste un blocco senza idx per MIMIC, lo interpretiamo come "best model"
    if best_model_id !== missing
        mask = (df.dataset .== "MIMIC-IV") .& (df.phase .== "validation") .& ismissing.(df.model_id)
        df.model_id[mask] .= best_model_id
    end

    return df
end

# -----------------------------
# Optional: compute training sMAPE from saved JLD2
# Needs:
#   res/<experiment>/models/trainingsetNSTEMI_<experiment>.jld2
#   res/<experiment>/models/nnNSTEMI_<experiment>.jld2
#   res/<experiment>/models/odebetasNSTEMI_<experiment>.jld2
# and ctnt-ude-model.jl in project root.
# -----------------------------
function compute_training_smape_from_jld2(experiment_dir::String, experiment::String, best_model_id::Int, nn_depth::Int, nn_width::Int, input_dim::Int)
    models_path = joinpath(experiment_dir, "models")
    train_file = joinpath(models_path, "trainingsetNSTEMI_$(experiment).jld2")
    nn_file    = joinpath(models_path, "nnNSTEMI_$(experiment).jld2")
    ode_file   = joinpath(models_path, "odebetasNSTEMI_$(experiment).jld2")

    if !(isfile(train_file) && isfile(nn_file) && isfile(ode_file))
        @warn "JLD2 files not found. Skipping training sMAPE computation." train_file nn_file ode_file
        return nothing
    end

    # IMPORTANT: define PatientData + model functions before loading
    # include(joinpath(@__DIR__, "ctnt-ude-model.jl"))

    training_dataset = JLD2.load(train_file, "training_dataset")
    neural_network_parameters = JLD2.load(nn_file, "neural_network_parameters")
    ode_params = JLD2.load(ode_file, "ode_params")

    chain = neural_network_model(nn_depth, nn_width; input_dims=input_dim)

    best_nn  = neural_network_parameters[best_model_id]
    best_ode = ode_params[best_model_id]

    N_params = 5
    n_pat = length(training_dataset)

    if length(best_ode) != N_params * n_pat
        @warn "Unexpected ode vector length vs training patients" length(best_ode) n_pat
    end

    out = DataFrame(
        patient_id = String[],
        smape = Float64[],
        loss = Float64[],
        a = Float64[], b = Float64[], Cs0 = Float64[], Cc0 = Float64[], beta = Float64[]
    )

    for (j, pat) in enumerate(training_dataset)
        θ = best_ode[N_params*(j-1)+1 : N_params*j]  # log-params

        model = ctntCUDEModel(θ, chain, (0.0, pat.timepoints[end]))

        # loss (log-MSE) e sMAPE (come da funzioni nel tuo ctnt-ude-model.jl)
        l = patient_loss(θ, (model, pat.timepoints, pat.ctnt_data, best_nn))
        s = smape_loss(θ, (model, pat.timepoints, pat.ctnt_data, best_nn))

        push!(out, (
            pat.id,
            s,
            l,
            exp(θ[1]), exp(θ[2]), exp(θ[3]), exp(θ[4]), exp(θ[5])
        ))
    end

    return out
end

# -----------------------------
# Main
# -----------------------------
function main()
    log_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "info_output.txt")
    out_dir  = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "extracted_metrics")

    @info "Reading log" log_path
    lines = readlines(log_path)

    meta = parse_log_metadata(lines)
    experiment = String(get(meta, "experiment", "UNKNOWN_EXPERIMENT"))
    best_model_id = get(meta, "best_model_id", missing)
    nn_depth = get(meta, "nn_depth", 2)
    nn_width = get(meta, "nn_width", 8)
    input_dim = get(meta, "input_dim", 2)

    safe_mkdir(out_dir)

    # 1) validation params/loss (MIMIC)
    df_val_params = parse_validation_params_losses(lines)
    CSV.write(joinpath(out_dir, "validation_params_loss_MIMIC.csv"), df_val_params)

    # 2) sMAPE blocks (MIMIC validation + UMG test)
    df_smape = parse_smape_blocks(lines, best_model_id)
    CSV.write(joinpath(out_dir, "smape_patientlevel_allblocks.csv"), df_smape)

    # 3) join (where possible): MIMIC validation params/loss + smape by (patient_id, model_id)
    df_smape_mimic = df_smape[(df_smape.dataset .== "MIMIC-IV") .& (df_smape.phase .== "validation") .& .!ismissing.(df_smape.model_id), :]
    rename!(df_smape_mimic, :model_id => :model_id)
    df_join = leftjoin(df_val_params, df_smape_mimic[:, [:run_id, :model_id, :patient_id, :smape]],
                       on=[:model_id, :patient_id])
    CSV.write(joinpath(out_dir, "validation_MIMIC_join_loss_params_smape.csv"), df_join)

    # 4) summary tables
    df_sum_smape = combine(groupby(df_smape, [:dataset, :phase, :run_id, :model_id]),
        :smape => (x -> median(skipmissing(x))) => :median_smape,
        :smape => (x -> mean(skipmissing(x)))   => :mean_smape,
        :smape => (x -> quantile(collect(skipmissing(x)), 0.25)) => :q1_smape,
        :smape => (x -> quantile(collect(skipmissing(x)), 0.75)) => :q3_smape,
    nrow => :n_patients
)
    CSV.write(joinpath(out_dir, "smape_summary_by_block.csv"), df_sum_smape)

    df_sum_loss = combine(groupby(df_val_params, [:dataset, :model_id]),
        :loss => (x -> median(skipmissing(x))) => :median_loss,
        :loss => (x -> mean(skipmissing(x)))   => :mean_loss,
        :loss => (x -> quantile(collect(skipmissing(x)), 0.25)) => :q1_loss,
        :loss => (x -> quantile(collect(skipmissing(x)), 0.75)) => :q3_loss,
    nrow => :n_patients
)
    CSV.write(joinpath(out_dir, "validation_loss_summary_MIMIC_by_model.csv"), df_sum_loss)

    # 5) training sMAPE (optional, from JLD2)
    experiment_dir_guess = dirname(log_path)
    @info "Training: experiment_dir_guess (from log dirname)" experiment_dir_guess

    # fallback: res/<experiment>
    if !isdir(joinpath(experiment_dir_guess, "models"))
        experiment_dir_guess = joinpath(@__DIR__, "res", experiment)
        @info "Training: fallback experiment_dir_guess" experiment_dir_guess
    end

    models_path = joinpath(experiment_dir_guess, "models")
    @info "Training: models_path" models_path isdir(models_path)

    # stampa best_model_id
    @info "Training: best_model_id" best_model_id

    if best_model_id === missing
        @error "Training skipped: best_model_id is missing (not parsed from log)."
    else
        # prova a calcolare; se fallisce vogliamo stacktrace
        try
            df_train = compute_training_smape_from_jld2(experiment_dir_guess, experiment, best_model_id, nn_depth, nn_width, input_dim)

            if df_train === nothing
                @error "Training computation returned nothing (likely missing/invalid JLD2 inside compute_training_smape_from_jld2)."
            else
                @info "Training computed" n_patients = nrow(df_train)

                train_csv = joinpath(out_dir, "training_patientlevel_smape_loss.csv")
                sum_csv   = joinpath(out_dir, "training_summary.csv")

                CSV.write(train_csv, df_train)

                df_train_sum = DataFrame(
                    median_smape = median(df_train.smape),
                    q1_smape = quantile(df_train.smape, 0.25),
                    q3_smape = quantile(df_train.smape, 0.75),
                    mean_smape = mean(df_train.smape),
                    median_loss = median(df_train.loss),
                    mean_loss = mean(df_train.loss),
                    n_patients = nrow(df_train)
                )
                CSV.write(sum_csv, df_train_sum)

                @info "Training CSV written" train_csv sum_csv
            end
        catch e
            @error "Training computation failed" exception=(e, catch_backtrace())
        end
    end

end

main()

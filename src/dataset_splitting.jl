using DataFrames, XLSX, CSV
using Logging
using Revise

# includet("../src/DataUtils.jl")
# using .DataUtils

includet("ctnt-ude-model.jl")

@info "Starting AE, HI and LI splitting script"

T_SCALE = 240.0# 350.0;

dataset_id = 1; # change here for different datasets

if dataset_id == 0
    dataset_name = "MIMIC-IV";
    dataset_path = "MIMIC-IV/NSTEMI_reorganized_skipped.xlsx";
    column_letter = "B";
elseif dataset_id == 1
    dataset_name = "UMG";
    dataset_path = "UMG_NSTEMI_Dataset.xlsx";
    column_letter = "A";
elseif dataset_id == 2
    dataset_name = "UMG_STEMI";
    dataset_path = "UMG_STEMI_Dataset.xlsx";
    column_letter = "A";
end;

file_path = "data/$(dataset_path)"; # UMG_NSTEMI_Dataset MIMIC-IV/NSTEMI_reorganized_skipped UMG_STEMI_Dataset
sheet_ids = "IDs";
sheet_times = "times";
sheet_values = "values";

xf = XLSX.readxlsx(file_path);
# Caricamento dei fogli in DataFrame
ids = DataFrame(XLSX.readtable(file_path, sheet_ids, "$(column_letter):$(column_letter)", header=false, infer_eltypes=true));
timepoints_df = DataFrame(XLSX.readtable(file_path, sheet_times, "A:Z", header=false, infer_eltypes=true));
troponin_df  = DataFrame(XLSX.readtable(file_path, sheet_values, "A:Z", header=false, infer_eltypes=true));

patients = [row2Patient(ids[i,:], timepoints_df[i,:], troponin_df[i,:]) for i in 1:nrow(ids)];

dup_counts = patient_duplicate_time_counts(patients; tol=0.0)
(nmod, nrm) = collapse_duplicates!(patients, dup_counts;)

# Trimming to T_SCALE
trimmed_p = trim_time(patients, T_SCALE);
patient_dims(trimmed_p)

# 0. Pre-processing
meas_min_number = 5;
min_acq_time_before = 12.0;
min_acq_n_before = 1;
min_acq_time_after = 48.0;
min_acq_n_after = 1;
min_time = 72.0;
max_gap = 72.0;
anoms = find_anomalies(trimmed_p,
    meas_min_number,
    min_acq_time_before, min_acq_n_before,
    min_acq_time_after, min_acq_n_after,
    min_time;
    max_gap_h=max_gap,
    verbose=false
);

println("Removed sample: $(length(anoms))")

# cleaned_patients = filter(p -> !haskey(anoms, p.id), trimmed_p);
# patient_dims(cleaned_patients)
# println("Total sample: $(length(cleaned_patients))")

# Trimming to T_SCALE
# trimmed_p = trim_time(patients, T_SCALE);
# patient_dims(trimmed_p)

# Alle eligible characteristics
# meas_min_number = 6;

# anoms = find_anomalies(trimmed_p, meas_min_number);
# @info "Removed sample number for all eligible: $(length(anoms))"

all_eligible = filter(p -> !haskey(anoms, p.id), trimmed_p);
patient_dims(all_eligible)
@info "Total sample number for all eligible: $(length(all_eligible))"

df_ae_ids = DataFrame(patient = [p.id for p in all_eligible]);
CSV.write(joinpath("./res", "ids_all_eligible_$(dataset_name).csv"), df_ae_ids)

# # High information characteristics
# meas_min_number = 8;
# min_acq_time_before=24.0;
# min_acq_n_before=2;
# min_acq_time_after=36.0;
# min_acq_n_after=1;
# min_time=12.0;
# max_gap=36.0;

# anoms = find_anomalies(
#     trimmed_p, 
#     meas_min_number, 
#     min_acq_time_before, min_acq_n_before, 
#     min_acq_time_after, min_acq_n_after, 
#     min_time; 
#     max_gap_h=max_gap
#     );
# @info "Removed sample number for high information: $(length(anoms))"
# high_information = filter(p -> !haskey(anoms, p.id), trimmed_p);
# patient_dims(high_information)
# @info "Total sample number for high information: $(length(high_information))"

# df_hi_ids = DataFrame(patient = [p.id for p in high_information]);
# CSV.write(joinpath("./res", "ids_high_information_$(dataset_name).csv"), df_hi_ids)

# # Low information characteristics
# # maximum 6 measurements

# meas_min_number = 3;

# anoms_temp = find_anomalies(
#     trimmed_p, 
#     meas_min_number
#     );

# @info "Removed sample number for low information: $(length(anoms_temp))"
# temp = filter(p -> !haskey(anoms_temp, p.id), trimmed_p);
# patient_dims(temp)
# @info "Total sample number for low information: $(length(temp))"

# # df_temp_ids = DataFrame(patient = [p.id for p in temp]);

# meas_min_number = 6;

# anoms = find_anomalies(
#     temp, 
#     meas_min_number
#     );

# @info "Removed sample number for low information: $(length(anoms))"
# low_information = filter(p -> haskey(anoms, p.id), temp);
# patient_dims(low_information)
# @info "Total sample number for low information: $(length(low_information))"

# df_li_ids = DataFrame(patient = [p.id for p in low_information]);
# CSV.write(joinpath("./res", "ids_low_information_$(dataset_name).csv"), df_li_ids)

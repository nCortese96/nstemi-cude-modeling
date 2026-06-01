WITH filtered_labs AS (
    SELECT le.hadm_id
    FROM `physionet-data.mimiciv_3_1_hosp.labevents` AS le
    JOIN (
        SELECT DISTINCT hadm_id
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
        WHERE icd_code LIKE '41071'
    ) AS diag
    ON le.hadm_id = diag.hadm_id
    WHERE le.itemid = 51003
),

hadm_filtered AS (
    SELECT hadm_id
    FROM filtered_labs
    GROUP BY hadm_id
    HAVING COUNT(*) >= 4
)

SELECT subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, edregtime, edouttime
FROM `physionet-data.mimiciv_3_1_hosp.admissions`
WHERE hadm_id IN (SELECT hadm_id FROM hadm_filtered);

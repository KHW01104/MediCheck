# Feature Definition

## Prediction Target

- `target_cvd`: binary label indicating whether `conditions.DESCRIPTION` contains one of the following cardiovascular keywords: `cardiovascular`, `coronary`, `myocardial`, `heart disease`, or `stroke`

## Patient-Level Features

- `patient_id`: patient identifier from `patients.Id`
- `age`: age in years calculated from `patients.BIRTHDATE` using reference date `2026-01-01`
- `gender_encoded`: `F = 0`, `M = 1`, and unknown or missing values as `NaN`
- `bmi`: patient-level mean BMI from `observations.VALUE` where `observations.DESCRIPTION` contains `BMI` or `Body Mass Index`
- `has_diabetes`: 1 if `conditions.DESCRIPTION` contains `diabetes`
- `has_hypertension`: 1 if `conditions.DESCRIPTION` contains `hypertension` or `high blood pressure`
- `has_obesity`: 1 if `conditions.DESCRIPTION` contains `obesity`
- `medication_count`: number of medication rows for the patient
- `encounter_count`: number of encounter rows for the patient
- `condition_count`: number of condition rows for the patient
- `abnormal_bmi_flag`: 1 if any BMI value for the patient is below 10 or above 80
- `target_cvd`: cardiovascular disease target label

## Notes

- The current version uses rule-based keyword/code matching for condition grouping.
- Rules are intentionally simple so they can be explained and extended during research discussions.
- Future versions can replace these rules with OMOP/CDM mappings, richer terminology standards, or temporal features.
- BMI values outside the expected range are preserved in the feature table so the quality evaluation layer can inspect them.

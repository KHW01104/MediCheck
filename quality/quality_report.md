# MediCheck Data Quality Report

The quality_score is a MediCheck portfolio heuristic for synthetic-data demos. It is not a clinical data quality standard.

Generated at: `2026-05-21T15:21:15.099147+00:00`

Quality score: **99.74 / 100**

## Dataset Summary

| Item | Rows |
| --- | --- |
| patients | 1000 |
| conditions | 1743 |
| medications | 787 |
| encounters | 2823 |
| observations | 4052 |
| feature_rows | 1000 |

## Missingness

| Feature | Missing Count | Missing Rate |
| --- | --- | --- |
| bmi | 5 | 0.50% |
| age | 0 | 0.00% |
| gender_encoded | 0 | 0.00% |
| has_diabetes | 0 | 0.00% |
| has_hypertension | 0 | 0.00% |
| medication_count | 0 | 0.00% |
| encounter_count | 0 | 0.00% |

## Outliers

| Rule | Outlier Count | Outlier Rate |
| --- | --- | --- |
| age | 0 | 0.00% |
| bmi | 3 | 0.30% |
| medication_count | 0 | 0.00% |
| encounter_count | 0 | 0.00% |

## Duplicates

| Target | Duplicate Count | Duplicate Rate |
| --- | --- | --- |
| patient_id | 0 | 0.00% |
| encounters.Id | 5 | 0.18% |

## Code Coverage

| Table | Rows | Missing CODE or DESCRIPTION | Missing Rate | Coverage Rate |
| --- | --- | --- | --- | --- |
| conditions | 1743 | 18 | 1.03% | 98.97% |
| medications | 787 | 4 | 0.51% | 99.49% |

## Temporal Consistency

| Rule | Issue Count | Rows Checked | Issue Rate |
| --- | --- | --- | --- |
| conditions_start_before_birth | 8 | 1743 | 0.46% |
| encounters_start_before_birth | 0 | 2823 | 0.00% |

## Score Penalties

| Component | Penalty |
| --- | --- |
| missingness_penalty | 0.02 |
| outlier_penalty | 0.01 |
| duplicate_penalty | 0.02 |
| code_quality_penalty | 0.15 |
| temporal_penalty | 0.06 |

## Safety Note

MediCheck is a research and education prototype using synthetic EHR-style data. This report does not replace real clinical data quality assessment.

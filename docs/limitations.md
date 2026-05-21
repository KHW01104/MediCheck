# Limitations

- Uses synthetic EHR data rather than real hospital data.
- Condition grouping is rule-based and not terminology-complete.
- The feature set is intentionally compact for a 2 to 3 week prototype scope.
- Temporal disease trajectories are simplified into patient-level aggregates.
- External validation and calibration are not included in this version.
- The local explanation output is model-dependent and should be interpreted as supportive, not causal.
- Current features and labels are derived from the same synthetic condition table, so label leakage is possible. This is acceptable for a portfolio prototype but should be redesigned with temporal cutoffs before any serious predictive modeling claim.
- Data quality assessment is based on synthetic data and simple heuristic checks. It does not replace clinical data quality standards or real-world hospital data governance review.
- If `xgboost` is unavailable or fails to load its runtime dependency such as `libomp`, the code falls back to a scikit-learn tree ensemble for local development.
- No OMOP full pipeline, MIMIC-IV workflow, deep learning, RAG, or production infrastructure is included in this version.

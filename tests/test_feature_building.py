"""Tests for MediCheck feature building."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd


def test_build_features_creates_patient_features(tmp_path: Path) -> None:
    """Verify that patient_features.csv is generated with required feature columns."""
    from data_pipeline.generate_sample_data import generate_sample_data
    from data_pipeline.build_features import build_feature_table

    data_dir = tmp_path / "sample"
    output_path = tmp_path / "processed" / "patient_features.csv"
    generate_sample_data(n_patients=50, out_dir=data_dir, seed=42)

    feature_table = build_feature_table(data_dir=data_dir, output_path=output_path)
    saved_table = pd.read_csv(output_path)

    required_columns = {
        "patient_id",
        "age",
        "gender_encoded",
        "bmi",
        "has_diabetes",
        "has_hypertension",
        "has_obesity",
        "medication_count",
        "encounter_count",
        "condition_count",
        "abnormal_bmi_flag",
        "target_cvd",
    }

    assert output_path.exists()
    assert required_columns.issubset(feature_table.columns)
    assert required_columns.issubset(saved_table.columns)
    assert set(saved_table["target_cvd"].dropna().unique()).issubset({0, 1})

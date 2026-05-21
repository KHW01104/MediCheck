from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medicheck.data.loader import load_ehr_tables
from medicheck.data.sample_generator import generate_sample_ehr
from medicheck.features.feature_builder import build_patient_feature_table


def test_feature_table_contains_required_columns(tmp_path: Path) -> None:
    generate_sample_ehr(tmp_path, n_patients=30, seed=11)
    tables = load_ehr_tables(tmp_path)
    feature_table = build_patient_feature_table(tables)

    expected_columns = {
        "PATIENT_ID",
        "age",
        "sex",
        "diabetes_flag",
        "hypertension_flag",
        "obesity_flag",
        "bmi_latest",
        "visit_count",
        "medication_count",
        "diagnosis_count",
        "cvd_risk_label",
    }
    assert expected_columns.issubset(set(feature_table.columns))
    assert len(feature_table) == 30


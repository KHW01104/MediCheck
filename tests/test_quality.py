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
from medicheck.quality.profiler import build_quality_report


def test_quality_report_has_expected_sections(tmp_path: Path) -> None:
    generate_sample_ehr(tmp_path, n_patients=25, seed=17)
    tables = load_ehr_tables(tmp_path)
    feature_table = build_patient_feature_table(tables)
    report = build_quality_report(tables, feature_table)

    assert "missingness" in report
    assert "duplicates" in report
    assert "temporal_consistency" in report
    assert "outliers" in report


"""Tests for MediCheck data quality checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_quality_checks_create_reports(tmp_path: Path) -> None:
    """Verify that quality checks create JSON and Markdown reports."""
    from data_pipeline.build_features import build_feature_table
    from data_pipeline.generate_sample_data import generate_sample_data
    from quality.quality_check import run_quality_checks

    data_dir = tmp_path / "sample"
    feature_path = tmp_path / "processed" / "patient_features.csv"
    output_dir = tmp_path / "quality"

    generate_sample_data(n_patients=50, out_dir=data_dir, seed=123)
    build_feature_table(data_dir=data_dir, output_path=feature_path)
    report = run_quality_checks(data_dir=data_dir, feature_path=feature_path, output_dir=output_dir)

    assert (output_dir / "quality_report.json").exists()
    assert (output_dir / "quality_report.md").exists()
    assert "score" in report
    assert "quality_score" in report["score"]
    assert 0 <= report["score"]["quality_score"] <= 100
    assert "missingness" in report
    assert "temporal_consistency" in report

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medicheck.data.loader import load_ehr_tables
from medicheck.data.sample_generator import generate_sample_ehr
from medicheck.features.feature_builder import build_patient_feature_table
from medicheck.modeling.train import train_models


def test_training_returns_metrics(tmp_path: Path) -> None:
    generate_sample_ehr(tmp_path, n_patients=80, seed=23)
    tables = load_ehr_tables(tmp_path)
    feature_table = build_patient_feature_table(tables)
    bundle = train_models(feature_table)

    assert "baseline" in bundle.metrics
    assert "main" in bundle.metrics
    assert bundle.feature_columns


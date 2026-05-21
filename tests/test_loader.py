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


def test_load_generated_sample(tmp_path: Path) -> None:
    generate_sample_ehr(tmp_path, n_patients=20, seed=7)
    tables = load_ehr_tables(tmp_path)

    assert len(tables.patients) == 20
    assert "PATIENT_ID" in tables.patients.columns
    assert not tables.encounters.empty


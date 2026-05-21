"""Tests for MediCheck synthetic sample data generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_generate_sample_data_creates_required_csv_files(tmp_path: Path) -> None:
    """Verify that sample Synthea-style CSV files are generated with required columns."""
    from data_pipeline.generate_sample_data import generate_sample_data

    output_dir = tmp_path / "sample"
    generate_sample_data(n_patients=50, out_dir=output_dir, seed=42)

    expected_columns = {
        "patients.csv": ["Id", "BIRTHDATE", "SEX"],
        "conditions.csv": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
        "medications.csv": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
        "encounters.csv": ["Id", "START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
        "observations.csv": ["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS"],
    }

    for filename, columns in expected_columns.items():
        path = output_dir / filename
        assert path.exists(), f"{filename} was not generated"
        dataframe = pd.read_csv(path)
        assert columns == dataframe.columns.tolist()
        assert len(dataframe) > 0


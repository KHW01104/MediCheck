"""Load Synthea-style CSV tables for MediCheck."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_FILES = {
    "patients": "patients.csv",
    "conditions": "conditions.csv",
    "medications": "medications.csv",
    "encounters": "encounters.csv",
    "observations": "observations.csv",
}


@dataclass
class SyntheaTables:
    """Container for MediCheck Synthea-style source tables."""

    patients: pd.DataFrame
    conditions: pd.DataFrame
    medications: pd.DataFrame
    encounters: pd.DataFrame
    observations: pd.DataFrame


def validate_input_files(data_dir: str | Path) -> dict[str, Path]:
    """Validate that all required Synthea-style CSV files exist."""
    base_dir = Path(data_dir)
    missing_files: list[Path] = []
    file_paths: dict[str, Path] = {}

    for table_name, filename in REQUIRED_FILES.items():
        path = base_dir / filename
        file_paths[table_name] = path
        if not path.exists():
            missing_files.append(path)

    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "MediCheck could not find the required Synthea-style CSV files.\n"
            f"Expected directory: {base_dir}\n"
            f"Missing files:\n{missing_text}\n"
            "You can generate sample data with:\n"
            "python data_pipeline/generate_sample_data.py --n-patients 1000 --out-dir data/sample"
        )

    return file_paths


def read_csv_table(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    """Read one CSV table and parse known date columns when available."""
    dataframe = pd.read_csv(path)
    for column in date_columns or []:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")
    return dataframe


def load_synthea_csv(data_dir: str | Path) -> SyntheaTables:
    """Load required Synthea-style CSV tables from a directory."""
    file_paths = validate_input_files(data_dir)
    return SyntheaTables(
        patients=read_csv_table(file_paths["patients"], date_columns=["BIRTHDATE"]),
        conditions=read_csv_table(file_paths["conditions"], date_columns=["START", "STOP"]),
        medications=read_csv_table(file_paths["medications"], date_columns=["START", "STOP"]),
        encounters=read_csv_table(file_paths["encounters"], date_columns=["START", "STOP"]),
        observations=read_csv_table(file_paths["observations"], date_columns=["DATE"]),
    )


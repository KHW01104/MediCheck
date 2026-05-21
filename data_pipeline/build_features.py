"""Build patient-level MediCheck features from Synthea-style CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .load_synthea import SyntheaTables, load_synthea_csv
except ImportError:  # pragma: no cover - supports direct CLI execution
    from load_synthea import SyntheaTables, load_synthea_csv

REFERENCE_DATE = pd.Timestamp("2026-01-01")
CVD_KEYWORDS = ["cardiovascular", "coronary", "myocardial", "heart disease", "stroke"]


def contains_keywords(series: pd.Series, keywords: list[str]) -> pd.Series:
    """Return a boolean mask for case-insensitive keyword matching."""
    pattern = "|".join(keywords)
    return series.fillna("").astype(str).str.lower().str.contains(pattern, regex=True)


def calculate_age(patients: pd.DataFrame) -> pd.Series:
    """Calculate age from BIRTHDATE using the MediCheck reference date."""
    birthdates = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    return ((REFERENCE_DATE - birthdates).dt.days / 365.25).round(1)


def encode_gender(patients: pd.DataFrame) -> pd.Series:
    """Encode gender as F=0, M=1, and unknown values as NaN."""
    return patients["SEX"].map({"F": 0, "M": 1})


def build_condition_flags(conditions: pd.DataFrame) -> pd.DataFrame:
    """Build patient-level disease flags and the target CVD label from conditions."""
    if conditions.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "has_diabetes",
                "has_hypertension",
                "has_obesity",
                "target_cvd",
                "condition_count",
            ]
        )

    descriptions = conditions["DESCRIPTION"].fillna("").astype(str).str.lower()
    condition_frame = conditions[["PATIENT"]].copy()
    condition_frame["has_diabetes"] = descriptions.str.contains("diabetes", regex=False)
    condition_frame["has_hypertension"] = descriptions.str.contains("hypertension", regex=False) | descriptions.str.contains(
        "high blood pressure",
        regex=False,
    )
    condition_frame["has_obesity"] = descriptions.str.contains("obesity", regex=False)
    condition_frame["target_cvd"] = contains_keywords(conditions["DESCRIPTION"], CVD_KEYWORDS)

    flags = (
        condition_frame.groupby("PATIENT")[["has_diabetes", "has_hypertension", "has_obesity", "target_cvd"]]
        .max()
        .astype(int)
        .reset_index()
        .rename(columns={"PATIENT": "patient_id"})
    )
    counts = conditions.groupby("PATIENT").size().rename("condition_count").reset_index().rename(columns={"PATIENT": "patient_id"})
    return flags.merge(counts, on="patient_id", how="outer")


def build_bmi_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Build mean BMI and abnormal BMI flag from observation rows."""
    if observations.empty:
        return pd.DataFrame(columns=["patient_id", "bmi", "abnormal_bmi_flag"])

    description = observations["DESCRIPTION"].fillna("").astype(str)
    bmi_mask = description.str.contains("BMI", case=False, regex=False) | description.str.contains(
        "Body Mass Index",
        case=False,
        regex=False,
    )
    bmi_rows = observations.loc[bmi_mask, ["PATIENT", "VALUE"]].copy()
    if bmi_rows.empty:
        return pd.DataFrame(columns=["patient_id", "bmi", "abnormal_bmi_flag"])

    bmi_rows["VALUE"] = pd.to_numeric(bmi_rows["VALUE"], errors="coerce")
    bmi_rows["abnormal_value"] = bmi_rows["VALUE"].lt(10) | bmi_rows["VALUE"].gt(80)
    grouped = (
        bmi_rows.groupby("PATIENT")
        .agg(bmi=("VALUE", "mean"), abnormal_bmi_flag=("abnormal_value", "max"))
        .reset_index()
        .rename(columns={"PATIENT": "patient_id"})
    )
    grouped["abnormal_bmi_flag"] = grouped["abnormal_bmi_flag"].fillna(False).astype(int)
    return grouped


def build_count_features(tables: SyntheaTables) -> pd.DataFrame:
    """Build patient-level medication, encounter, and condition row counts."""
    medication_count = tables.medications.groupby("PATIENT").size().rename("medication_count")
    encounter_count = tables.encounters.groupby("PATIENT").size().rename("encounter_count")
    condition_count = tables.conditions.groupby("PATIENT").size().rename("condition_count")
    counts = pd.concat([medication_count, encounter_count, condition_count], axis=1).reset_index()
    return counts.rename(columns={"PATIENT": "patient_id"})


def build_feature_table(data_dir: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    """Create and optionally save the MediCheck patient-level feature table."""
    tables = load_synthea_csv(data_dir)

    features = pd.DataFrame(
        {
            "patient_id": tables.patients["Id"],
            "age": calculate_age(tables.patients),
            "gender_encoded": encode_gender(tables.patients),
        }
    )

    condition_features = build_condition_flags(tables.conditions)
    bmi_features = build_bmi_features(tables.observations)
    count_features = build_count_features(tables)

    features = features.merge(condition_features, on="patient_id", how="left")
    features = features.merge(bmi_features, on="patient_id", how="left")
    features = features.merge(count_features, on="patient_id", how="left", suffixes=("", "_counted"))

    if "condition_count_counted" in features.columns:
        features["condition_count"] = features["condition_count"].fillna(features["condition_count_counted"])
        features = features.drop(columns=["condition_count_counted"])

    flag_columns = ["has_diabetes", "has_hypertension", "has_obesity", "target_cvd", "abnormal_bmi_flag"]
    count_columns = ["medication_count", "encounter_count", "condition_count"]
    for column in flag_columns + count_columns:
        if column not in features.columns:
            features[column] = np.nan

    for column in flag_columns:
        features[column] = features[column].fillna(0).astype(int)
    for column in count_columns:
        features[column] = features[column].fillna(0).astype(int)

    ordered_columns = [
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
    ]
    feature_table = features[ordered_columns].sort_values("patient_id").reset_index(drop=True)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        feature_table.to_csv(output_file, index=False)
        print(f"Saved feature table: {output_file} rows={len(feature_table)}")

    return feature_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature table generation."""
    parser = argparse.ArgumentParser(description="Build MediCheck patient-level features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sample"), help="Directory containing Synthea-style CSV files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/patient_features.csv"),
        help="Output CSV path for patient-level features.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the feature table builder from the command line."""
    args = parse_args()
    build_feature_table(data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()

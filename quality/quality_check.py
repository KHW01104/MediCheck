"""Generate MediCheck data quality reports from synthetic EHR data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.load_synthea import SyntheaTables, load_synthea_csv

MISSINGNESS_FEATURES = [
    "bmi",
    "age",
    "gender_encoded",
    "has_diabetes",
    "has_hypertension",
    "medication_count",
    "encounter_count",
]

HEURISTIC_NOTE = (
    "The quality_score is a MediCheck portfolio heuristic for synthetic-data demos. "
    "It is not a clinical data quality standard."
)


def _rate(count: int, denominator: int) -> float:
    """Return a rounded rate while avoiding division by zero."""
    if denominator <= 0:
        return 0.0
    return round(float(count) / float(denominator), 4)


def _is_blank(series: pd.Series) -> pd.Series:
    """Return whether values are null or blank after string trimming."""
    return series.isna() | series.astype(str).str.strip().eq("")


def load_feature_table(feature_path: str | Path) -> pd.DataFrame:
    """Load the patient-level feature table with a friendly missing-file error."""
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(
            "MediCheck could not find the patient feature table.\n"
            f"Missing file: {path}\n"
            "Build it with:\n"
            "python data_pipeline/build_features.py --data-dir data/sample --output data/processed/patient_features.csv"
        )
    return pd.read_csv(path)


def summarize_missingness(feature_table: pd.DataFrame) -> list[dict[str, Any]]:
    """Calculate missingness rates for key MediCheck features."""
    total_rows = len(feature_table)
    rows: list[dict[str, Any]] = []
    for feature in MISSINGNESS_FEATURES:
        if feature not in feature_table.columns:
            missing_count = total_rows
        else:
            missing_count = int(feature_table[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "missing_rate": _rate(missing_count, total_rows),
            }
        )
    return rows


def summarize_outliers(feature_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate simple outlier counts for patient-level features."""
    rules = {
        "age": lambda values: values.lt(0) | values.gt(120),
        "bmi": lambda values: values.lt(10) | values.gt(80),
        "medication_count": lambda values: values.lt(0),
        "encounter_count": lambda values: values.lt(0),
    }
    total_rows = len(feature_table)
    results: dict[str, dict[str, Any]] = {}

    for feature, rule in rules.items():
        if feature not in feature_table.columns:
            outlier_count = 0
        else:
            values = pd.to_numeric(feature_table[feature], errors="coerce")
            outlier_count = int(rule(values).fillna(False).sum())
        results[feature] = {
            "outlier_count": outlier_count,
            "outlier_rate": _rate(outlier_count, total_rows),
        }

    return results


def summarize_duplicates(feature_table: pd.DataFrame, tables: SyntheaTables) -> dict[str, dict[str, Any]]:
    """Check duplicate patient IDs and encounter IDs."""
    patient_duplicate_count = (
        int(feature_table.duplicated(subset=["patient_id"]).sum()) if "patient_id" in feature_table.columns else 0
    )
    encounter_duplicate_count = (
        int(tables.encounters.duplicated(subset=["Id"]).sum()) if "Id" in tables.encounters.columns else 0
    )
    return {
        "patient_id": {
            "duplicate_count": patient_duplicate_count,
            "duplicate_rate": _rate(patient_duplicate_count, len(feature_table)),
        },
        "encounters.Id": {
            "duplicate_count": encounter_duplicate_count,
            "duplicate_rate": _rate(encounter_duplicate_count, len(tables.encounters)),
        },
    }


def summarize_code_coverage(tables: SyntheaTables) -> dict[str, dict[str, Any]]:
    """Calculate CODE or DESCRIPTION missing rates for conditions and medications."""
    results: dict[str, dict[str, Any]] = {}
    for table_name, dataframe in {
        "conditions": tables.conditions,
        "medications": tables.medications,
    }.items():
        if dataframe.empty:
            missing_count = 0
        else:
            code_missing = _is_blank(dataframe["CODE"]) if "CODE" in dataframe.columns else pd.Series(True, index=dataframe.index)
            description_missing = (
                _is_blank(dataframe["DESCRIPTION"])
                if "DESCRIPTION" in dataframe.columns
                else pd.Series(True, index=dataframe.index)
            )
            missing_count = int((code_missing | description_missing).sum())

        missing_rate = _rate(missing_count, len(dataframe))
        results[table_name] = {
            "rows": int(len(dataframe)),
            "missing_code_or_description_count": missing_count,
            "missing_code_or_description_rate": missing_rate,
            "coverage_rate": round(1.0 - missing_rate, 4),
        }
    return results


def _events_before_birth(events: pd.DataFrame, patients: pd.DataFrame, event_date_column: str) -> dict[str, Any]:
    """Count event rows whose START date is earlier than patient birth date."""
    if events.empty:
        return {"issue_count": 0, "issue_rate": 0.0, "rows_checked": 0}

    birthdate_lookup = patients.set_index("Id")["BIRTHDATE"]
    event_frame = events[["PATIENT", event_date_column]].copy()
    event_frame[event_date_column] = pd.to_datetime(event_frame[event_date_column], errors="coerce")
    event_frame["BIRTHDATE"] = pd.to_datetime(event_frame["PATIENT"].map(birthdate_lookup), errors="coerce")
    issue_mask = event_frame[event_date_column].notna() & event_frame["BIRTHDATE"].notna()
    issue_mask = issue_mask & event_frame[event_date_column].lt(event_frame["BIRTHDATE"])
    issue_count = int(issue_mask.sum())
    return {
        "issue_count": issue_count,
        "issue_rate": _rate(issue_count, len(event_frame)),
        "rows_checked": int(len(event_frame)),
    }


def summarize_temporal_consistency(tables: SyntheaTables) -> dict[str, dict[str, Any]]:
    """Check whether condition and encounter start dates precede birth dates."""
    return {
        "conditions_start_before_birth": _events_before_birth(tables.conditions, tables.patients, "START"),
        "encounters_start_before_birth": _events_before_birth(tables.encounters, tables.patients, "START"),
    }


def calculate_quality_score(report: dict[str, Any]) -> dict[str, Any]:
    """Calculate a simple 100-point heuristic quality score."""
    missing_rates = [row["missing_rate"] for row in report["missingness"]]
    outlier_rates = [row["outlier_rate"] for row in report["outliers"].values()]
    duplicate_rates = [row["duplicate_rate"] for row in report["duplicates"].values()]
    code_missing_rates = [row["missing_code_or_description_rate"] for row in report["code_coverage"].values()]
    temporal_rates = [row["issue_rate"] for row in report["temporal_consistency"].values()]

    penalties = {
        "missingness_penalty": round(float(np.mean(missing_rates or [0.0])) * 25, 2),
        "outlier_penalty": round(float(np.mean(outlier_rates or [0.0])) * 20, 2),
        "duplicate_penalty": round(float(np.mean(duplicate_rates or [0.0])) * 20, 2),
        "code_quality_penalty": round(float(np.mean(code_missing_rates or [0.0])) * 20, 2),
        "temporal_penalty": round(float(np.mean(temporal_rates or [0.0])) * 25, 2),
    }
    total_penalty = round(sum(penalties.values()), 2)
    score = max(0.0, min(100.0, round(100.0 - total_penalty, 2)))
    return {
        "quality_score": score,
        "total_penalty": total_penalty,
        "penalties": penalties,
        "note": HEURISTIC_NOTE,
    }


def build_quality_report(data_dir: str | Path, feature_path: str | Path) -> dict[str, Any]:
    """Build the full MediCheck quality report dictionary."""
    tables = load_synthea_csv(data_dir)
    feature_table = load_feature_table(feature_path)

    report: dict[str, Any] = {
        "project": "MediCheck",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "feature_path": str(feature_path),
        "summary": {
            "patients": int(len(tables.patients)),
            "conditions": int(len(tables.conditions)),
            "medications": int(len(tables.medications)),
            "encounters": int(len(tables.encounters)),
            "observations": int(len(tables.observations)),
            "feature_rows": int(len(feature_table)),
        },
        "missingness": summarize_missingness(feature_table),
        "outliers": summarize_outliers(feature_table),
        "duplicates": summarize_duplicates(feature_table, tables),
        "code_coverage": summarize_code_coverage(tables),
        "temporal_consistency": summarize_temporal_consistency(tables),
        "quality_score_method": HEURISTIC_NOTE,
    }
    report["score"] = calculate_quality_score(report)
    return report


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a compact Markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the MediCheck quality report as Markdown."""
    missingness_rows = [
        [row["feature"], row["missing_count"], f"{row['missing_rate']:.2%}"]
        for row in report["missingness"]
    ]
    outlier_rows = [
        [feature, result["outlier_count"], f"{result['outlier_rate']:.2%}"]
        for feature, result in report["outliers"].items()
    ]
    duplicate_rows = [
        [target, result["duplicate_count"], f"{result['duplicate_rate']:.2%}"]
        for target, result in report["duplicates"].items()
    ]
    code_rows = [
        [
            table,
            result["rows"],
            result["missing_code_or_description_count"],
            f"{result['missing_code_or_description_rate']:.2%}",
            f"{result['coverage_rate']:.2%}",
        ]
        for table, result in report["code_coverage"].items()
    ]
    temporal_rows = [
        [rule, result["issue_count"], result["rows_checked"], f"{result['issue_rate']:.2%}"]
        for rule, result in report["temporal_consistency"].items()
    ]
    penalty_rows = [
        [name, value]
        for name, value in report["score"]["penalties"].items()
    ]

    return "\n\n".join(
        [
            "# MediCheck Data Quality Report",
            report["quality_score_method"],
            f"Generated at: `{report['generated_at']}`",
            f"Quality score: **{report['score']['quality_score']} / 100**",
            "## Dataset Summary",
            _markdown_table(
                ["Item", "Rows"],
                [[key, value] for key, value in report["summary"].items()],
            ),
            "## Missingness",
            _markdown_table(["Feature", "Missing Count", "Missing Rate"], missingness_rows),
            "## Outliers",
            _markdown_table(["Rule", "Outlier Count", "Outlier Rate"], outlier_rows),
            "## Duplicates",
            _markdown_table(["Target", "Duplicate Count", "Duplicate Rate"], duplicate_rows),
            "## Code Coverage",
            _markdown_table(
                ["Table", "Rows", "Missing CODE or DESCRIPTION", "Missing Rate", "Coverage Rate"],
                code_rows,
            ),
            "## Temporal Consistency",
            _markdown_table(["Rule", "Issue Count", "Rows Checked", "Issue Rate"], temporal_rows),
            "## Score Penalties",
            _markdown_table(["Component", "Penalty"], penalty_rows),
            "## Safety Note",
            "MediCheck is a research and education prototype using synthetic EHR-style data. "
            "This report does not replace real clinical data quality assessment.",
        ]
    ) + "\n"


def save_quality_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Save the quality report as JSON and Markdown."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "quality_report.json"
    markdown_path = output_path / "quality_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def run_quality_checks(
    data_dir: str | Path = "data/sample",
    feature_path: str | Path = "data/processed/patient_features.csv",
    output_dir: str | Path = "quality",
) -> dict[str, Any]:
    """Generate and save MediCheck quality reports."""
    report = build_quality_report(data_dir=data_dir, feature_path=feature_path)
    paths = save_quality_outputs(report, output_dir=output_dir)
    print(f"Saved quality JSON: {paths['json']}")
    print(f"Saved quality Markdown: {paths['markdown']}")
    print(f"Quality score: {report['score']['quality_score']} / 100")
    return report


def run_quality_check(data_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Backward-compatible wrapper for earlier MediCheck skeleton tests."""
    feature_path = PROJECT_ROOT / "data" / "processed" / "patient_features.csv"
    output_dir = Path(output_path).parent if output_path is not None else PROJECT_ROOT / "quality"
    return run_quality_checks(data_dir=data_dir, feature_path=feature_path, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MediCheck quality checks."""
    parser = argparse.ArgumentParser(description="Generate MediCheck synthetic EHR data quality reports.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sample"), help="Directory containing source CSV files.")
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path("data/processed/patient_features.csv"),
        help="Path to the patient feature table.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("quality"), help="Directory for report outputs.")
    return parser.parse_args()


def main() -> None:
    """Run the MediCheck quality checker from the command line."""
    args = parse_args()
    run_quality_checks(data_dir=args.data_dir, feature_path=args.feature_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()


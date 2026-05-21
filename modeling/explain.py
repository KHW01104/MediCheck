"""Explain MediCheck model predictions and global feature importance."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib

Path(tempfile.gettempdir(), "medicheck_matplotlib").mkdir(parents=True, exist_ok=True)
Path(tempfile.gettempdir(), "medicheck_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir(), "medicheck_matplotlib")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir(), "medicheck_cache")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_LABELS = {
    "age": "Age",
    "bmi": "BMI",
    "medication_count": "Medication count",
    "encounter_count": "Encounter count",
    "condition_count": "Condition count",
    "gender_encoded": "Gender encoded",
    "has_diabetes": "Diabetes history",
    "has_hypertension": "Hypertension history",
    "has_obesity": "Obesity history",
    "abnormal_bmi_flag": "Abnormal BMI flag",
}

RISK_FACTOR_LABELS = {
    "has_hypertension": "고혈압 이력",
    "has_diabetes": "당뇨 이력",
    "has_obesity": "비만 이력",
    "abnormal_bmi_flag": "비정상 BMI 값",
    "bmi": "높은 BMI",
    "age": "고령",
    "medication_count": "많은 처방 약물 수",
    "encounter_count": "많은 의료기관 방문 횟수",
    "condition_count": "많은 진단 기록 수",
}


def load_feature_columns(feature_columns_path: str | Path) -> list[str]:
    """Load feature columns from a MediCheck feature_columns.json file."""
    path = Path(feature_columns_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("feature_columns", []))


def unwrap_model(model: Any) -> Any:
    """Return the estimator inside a Pipeline, or the model itself."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def get_transformed_feature_names(model: Any, fallback_feature_columns: list[str]) -> list[str]:
    """Return feature names after preprocessing when available."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        try:
            names = list(model.named_steps["preprocessor"].get_feature_names_out())
            return [name.split("__", 1)[-1] for name in names]
        except Exception:
            return fallback_feature_columns
    return fallback_feature_columns


def _collapse_transformed_importance(
    transformed_names: list[str],
    importances: np.ndarray,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Aggregate transformed feature importances back to original feature names."""
    scores = {feature: 0.0 for feature in feature_columns}
    for transformed_name, importance in zip(transformed_names, importances, strict=False):
        matched_feature = next(
            (feature for feature in feature_columns if transformed_name == feature or transformed_name.startswith(f"{feature}_")),
            transformed_name,
        )
        scores[matched_feature] = scores.get(matched_feature, 0.0) + float(abs(importance))

    total = sum(scores.values())
    rows = []
    for feature, importance in scores.items():
        normalized = float(importance / total) if total else 0.0
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "importance": round(float(importance), 6),
                "normalized_importance": round(normalized, 6),
            }
        )
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def compute_global_feature_importance(model: Any, feature_columns: list[str]) -> list[dict[str, Any]]:
    """Compute global feature importance from tree importances or linear coefficients."""
    estimator = unwrap_model(model)
    transformed_names = get_transformed_feature_names(model, feature_columns)

    if hasattr(estimator, "feature_importances_"):
        raw_importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        raw_importance = np.asarray(estimator.coef_[0], dtype=float)
    else:
        raw_importance = np.ones(len(transformed_names), dtype=float)

    return _collapse_transformed_importance(transformed_names, raw_importance, feature_columns)


def save_feature_importance_plot(feature_importance: list[dict[str, Any]], output_path: str | Path, top_n: int = 12) -> Path:
    """Save a horizontal bar chart for MediCheck global feature importance."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(feature_importance).head(top_n)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(data["label"][::-1], data["normalized_importance"][::-1], color="#2f6f73")
    axis.set_xlabel("Normalized importance")
    axis.set_ylabel("Feature")
    axis.set_title("MediCheck Global Feature Importance")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_feature_importance_json(feature_importance: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save global feature importance as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "MediCheck",
        "note": "Feature importance is a model explanation aid for a synthetic-data prototype, not a clinical conclusion.",
        "feature_importance": feature_importance,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None for missing or invalid values."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_stat(feature_stats: dict[str, Any] | None, key: str, default: float) -> float:
    """Read a numeric feature statistic with a default fallback."""
    if not feature_stats:
        return default
    value = feature_stats.get(key, default)
    if isinstance(value, dict):
        value = value.get("threshold", default)
    numeric_value = _safe_float(value)
    return default if numeric_value is None else numeric_value


def _importance_lookup(global_importance: list[dict[str, Any]] | None) -> dict[str, float]:
    """Create a feature-to-importance lookup for ranking local risk factors."""
    if not global_importance:
        return {}
    return {
        str(row["feature"]): float(row.get("normalized_importance", row.get("importance", 0.0)))
        for row in global_importance
    }


def get_top_risk_factors(
    patient_features: dict[str, Any] | pd.Series,
    feature_stats: dict[str, Any] | None = None,
    global_importance: list[dict[str, Any]] | None = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return simple patient-level risk factors without making a medical diagnosis."""
    if isinstance(patient_features, pd.Series):
        features = patient_features.to_dict()
    else:
        features = dict(patient_features)

    importance = _importance_lookup(global_importance)
    medication_threshold = _feature_stat(feature_stats, "medication_count_p75", 3.0)
    encounter_threshold = _feature_stat(feature_stats, "encounter_count_p75", 4.0)
    condition_threshold = _feature_stat(feature_stats, "condition_count_p75", 3.0)

    candidates: list[dict[str, Any]] = []

    checks = [
        ("has_hypertension", _safe_float(features.get("has_hypertension")) == 1, "고혈압 이력이 있는 환자 특성입니다."),
        ("has_diabetes", _safe_float(features.get("has_diabetes")) == 1, "당뇨 이력이 있는 환자 특성입니다."),
        ("has_obesity", _safe_float(features.get("has_obesity")) == 1, "비만 이력이 있는 환자 특성입니다."),
        ("abnormal_bmi_flag", _safe_float(features.get("abnormal_bmi_flag")) == 1, "BMI 값이 일반 범위를 벗어난 기록이 있습니다."),
        ("bmi", (_safe_float(features.get("bmi")) or 0.0) >= 30.0, "BMI가 30 이상으로 기록되었습니다."),
        ("age", (_safe_float(features.get("age")) or 0.0) >= 65.0, "65세 이상 연령대입니다."),
        (
            "medication_count",
            (_safe_float(features.get("medication_count")) or 0.0) >= medication_threshold,
            "학습 데이터 기준 처방 약물 수가 높은 편입니다.",
        ),
        (
            "encounter_count",
            (_safe_float(features.get("encounter_count")) or 0.0) >= encounter_threshold,
            "학습 데이터 기준 의료기관 방문 횟수가 높은 편입니다.",
        ),
        (
            "condition_count",
            (_safe_float(features.get("condition_count")) or 0.0) >= condition_threshold,
            "학습 데이터 기준 진단 기록 수가 높은 편입니다.",
        ),
    ]

    for feature, triggered, explanation in checks:
        if not triggered:
            continue
        value = features.get(feature)
        candidates.append(
            {
                "feature": feature,
                "label": RISK_FACTOR_LABELS.get(feature, feature),
                "value": None if pd.isna(value) else value,
                "importance": round(float(importance.get(feature, 0.0)), 6),
                "message": explanation,
            }
        )

    candidates.sort(key=lambda row: (row["importance"], str(row["feature"])), reverse=True)
    return candidates[:top_n]


def build_feature_stats(feature_table: pd.DataFrame) -> dict[str, float]:
    """Build training-data thresholds used for individual explanation rules."""
    return {
        "medication_count_p75": float(feature_table["medication_count"].quantile(0.75)),
        "encounter_count_p75": float(feature_table["encounter_count"].quantile(0.75)),
        "condition_count_p75": float(feature_table["condition_count"].quantile(0.75)),
    }


def try_save_shap_summary_plot(model: Any, feature_columns: list[str], output_dir: Path) -> str | None:
    """Generate an optional SHAP summary plot when SHAP and compatible data are available."""
    try:
        import shap
    except Exception:
        return None

    sample_path = PROJECT_ROOT / "data" / "processed" / "patient_features.csv"
    if not sample_path.exists():
        return None

    try:
        feature_table = pd.read_csv(sample_path)
        sample = feature_table[feature_columns].head(200)
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            transformed = model.named_steps["preprocessor"].transform(sample)
            estimator = unwrap_model(model)
            transformed_names = get_transformed_feature_names(model, feature_columns)
            explainer = shap.Explainer(estimator)
            shap_values = explainer(transformed)
            plt.figure()
            shap.summary_plot(shap_values, transformed, feature_names=transformed_names, show=False)
        else:
            explainer = shap.Explainer(model)
            shap_values = explainer(sample)
            plt.figure()
            shap.summary_plot(shap_values, sample, show=False)

        output_path = output_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
        return str(output_path)
    except Exception as error:
        print(f"Warning: SHAP summary plot was skipped. Reason: {error}")
        return None


def explain_global_model(
    model_path: str | Path = "models/main_model.pkl",
    feature_columns_path: str | Path = "models/feature_columns.json",
    output_dir: str | Path = "evaluation",
) -> dict[str, Any]:
    """Compute and save global model explanations for MediCheck."""
    model = joblib.load(Path(model_path))
    feature_columns = load_feature_columns(feature_columns_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_importance = compute_global_feature_importance(model, feature_columns)
    json_path = save_feature_importance_json(feature_importance, output_path / "feature_importance.json")
    plot_path = save_feature_importance_plot(feature_importance, output_path / "feature_importance.png")
    shap_path = try_save_shap_summary_plot(model, feature_columns, output_path)

    result = {
        "model_path": str(model_path),
        "feature_columns_path": str(feature_columns_path),
        "feature_importance_json": str(json_path),
        "feature_importance_png": str(plot_path),
        "shap_summary_png": shap_path,
        "feature_importance": feature_importance,
    }
    print(f"Saved feature importance JSON: {json_path}")
    print(f"Saved feature importance plot: {plot_path}")
    if shap_path:
        print(f"Saved SHAP summary plot: {shap_path}")
    else:
        print("SHAP summary plot skipped; feature importance is available.")
    return result


def get_feature_importance(model_pipeline, top_n: int = 10) -> list[dict[str, object]]:
    """Return global feature importance values for a trained MediCheck model."""
    default_columns = [
        "age",
        "bmi",
        "medication_count",
        "encounter_count",
        "condition_count",
        "gender_encoded",
        "has_diabetes",
        "has_hypertension",
        "has_obesity",
        "abnormal_bmi_flag",
    ]
    return compute_global_feature_importance(model_pipeline, default_columns)[:top_n]


def get_patient_risk_factors(
    main_model,
    baseline_model,
    patient_frame,
) -> list[dict[str, object]]:
    """Backward-compatible wrapper for patient-level risk factor explanations."""
    del main_model, baseline_model
    first_row = patient_frame.iloc[0] if hasattr(patient_frame, "iloc") else patient_frame
    return get_top_risk_factors(first_row)


if __name__ == "__main__":
    explain_global_model()

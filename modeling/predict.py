"""Common MediCheck prediction logic for API and Streamlit use."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.explain import build_feature_stats, get_top_risk_factors

SAFETY_NOTICE = (
    "MediCheck is a research and education prototype. "
    "It is not intended for diagnosis, treatment, or prescription."
)

DEFAULT_FEATURE_VALUES = {
    "age": None,
    "bmi": None,
    "medication_count": 0,
    "encounter_count": 0,
    "condition_count": 0,
    "gender_encoded": None,
    "has_diabetes": 0,
    "has_hypertension": 0,
    "has_obesity": 0,
    "abnormal_bmi_flag": 0,
}


def _training_hint(model_dir: Path) -> str:
    """Return a friendly training command for missing model artifacts."""
    return (
        "MediCheck model artifacts were not found. "
        "Please run training first:\n"
        "python modeling/train.py --feature-path data/processed/patient_features.csv "
        f"--model-dir {model_dir} --evaluation-dir evaluation"
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_importance(path: Path) -> list[dict[str, Any]]:
    """Load feature importance rows when available."""
    if not path.exists():
        return []
    payload = _load_json(path)
    if isinstance(payload, dict):
        return list(payload.get("feature_importance", []))
    if isinstance(payload, list):
        return payload
    return []


def _load_feature_stats(feature_path: Path) -> dict[str, float]:
    """Load feature statistics from the training feature table when available."""
    if not feature_path.exists():
        return {}
    try:
        feature_table = pd.read_csv(feature_path)
        return build_feature_stats(feature_table)
    except Exception:
        return {}


def load_model_artifacts(model_dir: str | Path = "models") -> dict[str, Any]:
    """Load MediCheck model artifacts for risk prediction."""
    model_path = Path(model_dir)
    main_model_path = model_path / "main_model.pkl"
    feature_columns_path = model_path / "feature_columns.json"

    missing = [path for path in [main_model_path, feature_columns_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"{_training_hint(model_path)}\nMissing artifacts: {missing_text}")

    feature_payload = _load_json(feature_columns_path)
    feature_columns = list(feature_payload.get("feature_columns", []))
    if not feature_columns:
        raise ValueError(f"MediCheck feature column metadata is empty: {feature_columns_path}")

    evaluation_dir = PROJECT_ROOT / "evaluation"
    feature_path = PROJECT_ROOT / "data" / "processed" / "patient_features.csv"
    return {
        "model": joblib.load(main_model_path),
        "model_path": main_model_path,
        "feature_columns": feature_columns,
        "feature_columns_path": feature_columns_path,
        "feature_importance": _load_feature_importance(evaluation_dir / "feature_importance.json"),
        "feature_stats": _load_feature_stats(feature_path),
        "safety_notice": SAFETY_NOTICE,
    }


def risk_level_from_score(risk_score: float) -> str:
    """Map a risk probability to a low, medium, or high label."""
    if risk_score < 0.33:
        return "low"
    if risk_score < 0.66:
        return "medium"
    return "high"


def _coerce_feature_value(value: Any) -> Any:
    """Normalize empty strings to missing values while preserving numeric values."""
    if value == "":
        return pd.NA
    return value


def prepare_input_features(input_dict: dict[str, Any], feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Align input features to the trained model feature order."""
    missing_features: list[str] = []
    row: dict[str, Any] = {}

    for feature in feature_columns:
        if feature in input_dict:
            row[feature] = _coerce_feature_value(input_dict[feature])
        elif feature in DEFAULT_FEATURE_VALUES:
            row[feature] = DEFAULT_FEATURE_VALUES[feature]
            missing_features.append(feature)
        else:
            raise ValueError(f"Missing required MediCheck feature with no default value: {feature}")

    return pd.DataFrame([row], columns=feature_columns), missing_features


def _predict_probability(model: Any, features: pd.DataFrame) -> float:
    """Predict positive-class probability from a fitted model or pipeline."""
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(features)[:, 1][0])
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(features)[0])
        score = float(1.0 / (1.0 + np.exp(-margin)))
    else:
        score = float(model.predict(features)[0])
    return max(0.0, min(1.0, score))


def predict_risk(input_dict: dict[str, Any], artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Predict cardiovascular risk score, risk level, and top risk factors."""
    loaded_artifacts = artifacts or load_model_artifacts()
    feature_frame, missing_features = prepare_input_features(input_dict, loaded_artifacts["feature_columns"])
    risk_score = _predict_probability(loaded_artifacts["model"], feature_frame)

    patient_features = feature_frame.iloc[0].to_dict()
    top_factor_rows = get_top_risk_factors(
        patient_features,
        feature_stats=loaded_artifacts.get("feature_stats"),
        global_importance=loaded_artifacts.get("feature_importance"),
    )
    top_risk_factors = [row["label"] for row in top_factor_rows]

    result = {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level_from_score(risk_score),
        "top_risk_factors": top_risk_factors,
        "top_risk_factor_details": top_factor_rows,
        "safety_notice": loaded_artifacts.get("safety_notice", SAFETY_NOTICE),
    }
    if missing_features:
        result["missing_features_filled"] = missing_features
        result["missing_feature_note"] = (
            "Some missing features were filled with MediCheck default values. "
            "Model imputers may also handle missing numeric values."
        )
    return result


if __name__ == "__main__":
    artifacts = load_model_artifacts()
    demo_input = {
        "age": 72,
        "bmi": 31.2,
        "medication_count": 5,
        "encounter_count": 6,
        "condition_count": 4,
        "gender_encoded": 1,
        "has_diabetes": 1,
        "has_hypertension": 1,
        "has_obesity": 1,
        "abnormal_bmi_flag": 0,
    }
    print(json.dumps(predict_risk(demo_input, artifacts), indent=2, ensure_ascii=False))

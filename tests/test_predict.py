"""Tests for MediCheck prediction helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_predict_risk_with_tmp_trained_model(tmp_path: Path) -> None:
    """Verify that prediction works with a small model trained in a temp directory."""
    from data_pipeline.build_features import build_feature_table
    from data_pipeline.generate_sample_data import generate_sample_data
    from modeling.predict import load_model_artifacts, predict_risk
    from modeling.train import train_models

    data_dir = tmp_path / "sample"
    feature_path = tmp_path / "processed" / "patient_features.csv"
    model_dir = tmp_path / "models"
    evaluation_dir = tmp_path / "evaluation"

    generate_sample_data(n_patients=50, out_dir=data_dir, seed=42)
    build_feature_table(data_dir=data_dir, output_path=feature_path)
    train_models(feature_path=feature_path, model_dir=model_dir, evaluation_dir=evaluation_dir)

    artifacts = load_model_artifacts(model_dir)
    result = predict_risk(
        {
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
        },
        artifacts,
    )

    assert 0 <= result["risk_score"] <= 1
    assert result["risk_level"] in {"low", "medium", "high"}
    assert isinstance(result["top_risk_factors"], list)
    assert "safety_notice" in result

"""FastAPI service for MediCheck cardiovascular risk prediction."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import ModelInfoResponse, PredictionRequest, PredictionResponse
from modeling.predict import SAFETY_NOTICE, load_model_artifacts, predict_risk

app = FastAPI(
    title="MediCheck API",
    version="0.1.0",
    description=SAFETY_NOTICE,
)


@lru_cache(maxsize=1)
def get_artifacts() -> dict[str, Any]:
    """Load model artifacts lazily and reuse them across requests."""
    try:
        return load_model_artifacts(PROJECT_ROOT / "models")
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"MediCheck failed to load model artifacts: {error}") from error


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "MediCheck"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return loaded MediCheck model metadata."""
    artifacts = get_artifacts()
    model = artifacts["model"]
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        model_type = model.named_steps["model"].__class__.__name__
    else:
        model_type = model.__class__.__name__
    return ModelInfoResponse(
        project_name="MediCheck",
        model_type=model_type,
        feature_columns=artifacts["feature_columns"],
        safety_notice=SAFETY_NOTICE,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict synthetic-data cardiovascular risk without clinical diagnosis claims."""
    try:
        result = predict_risk(payload.model_dump(), get_artifacts())
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"MediCheck prediction failed: {error}") from error

    return PredictionResponse(
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        top_risk_factors=result["top_risk_factors"],
        safety_notice=result["safety_notice"],
    )


@app.get("/quality-report")
def quality_report() -> dict[str, Any]:
    """Return the saved MediCheck quality report."""
    report_path = PROJECT_ROOT / "quality" / "quality_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "MediCheck quality report was not found. "
                "Generate it first with: "
                "python quality/quality_check.py --data-dir data/sample "
                "--feature-path data/processed/patient_features.csv --output-dir quality"
            ),
        )
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail=f"MediCheck quality report is not valid JSON: {error}") from error


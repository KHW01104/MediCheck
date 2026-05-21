"""Pydantic schemas for the MediCheck FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single-patient feature payload for MediCheck risk prediction."""

    age: float | None = Field(None, ge=0, le=120)
    gender_encoded: int | None = Field(None, ge=0, le=1)
    bmi: float | None = Field(None)
    has_diabetes: int = Field(0, ge=0, le=1)
    has_hypertension: int = Field(0, ge=0, le=1)
    has_obesity: int = Field(0, ge=0, le=1)
    medication_count: int = Field(0, ge=0)
    encounter_count: int = Field(0, ge=0)
    condition_count: int = Field(0, ge=0)
    abnormal_bmi_flag: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    """MediCheck prediction response without clinical diagnosis claims."""

    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    top_risk_factors: list[str]
    safety_notice: str


class ModelInfoResponse(BaseModel):
    """Metadata for the loaded MediCheck prediction model."""

    project_name: str
    model_type: str
    feature_columns: list[str]
    safety_notice: str


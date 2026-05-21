from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "MediCheck"


def test_predict_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "age": 62,
            "gender_encoded": 1,
            "bmi": 31.2,
            "has_diabetes": 1,
            "has_hypertension": 1,
            "has_obesity": 1,
            "medication_count": 5,
            "encounter_count": 12,
            "condition_count": 8,
            "abnormal_bmi_flag": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert "top_risk_factors" in body
    assert "safety_notice" in body

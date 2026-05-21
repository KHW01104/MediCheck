"""Tests for MediCheck explanation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_top_risk_factors_returns_triggered_rules() -> None:
    """Verify that rule-based patient risk factors are returned."""
    from modeling.explain import get_top_risk_factors

    patient_features = {
        "age": 72,
        "bmi": 32.5,
        "medication_count": 6,
        "encounter_count": 7,
        "condition_count": 4,
        "has_diabetes": 1,
        "has_hypertension": 1,
        "has_obesity": 1,
        "abnormal_bmi_flag": 0,
    }
    feature_stats = {
        "medication_count_p75": 3,
        "encounter_count_p75": 4,
        "condition_count_p75": 3,
    }

    factors = get_top_risk_factors(patient_features, feature_stats=feature_stats)
    labels = {factor["label"] for factor in factors}

    assert factors
    assert "고혈압 이력" in labels
    assert "당뇨 이력" in labels


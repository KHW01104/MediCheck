"""Tests for MediCheck document-based explanations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_factor_explanation_uses_medical_knowledge_doc() -> None:
    """Verify that known risk factor labels return document explanations."""
    from docs_explainer import get_factor_explanation

    explanation = get_factor_explanation("고혈압 이력")

    assert "고혈압" in explanation
    assert "대체하지 않습니다" in explanation


def test_unknown_factor_returns_safe_fallback() -> None:
    """Verify that unknown factors still return a safe general explanation."""
    from docs_explainer import get_factor_explanation

    explanation = get_factor_explanation("unknown factor")

    assert "진단" in explanation
    assert "처방" in explanation


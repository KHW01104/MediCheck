"""Evaluate MediCheck cardiovascular risk prediction models."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

Path(tempfile.gettempdir(), "medicheck_matplotlib").mkdir(parents=True, exist_ok=True)
Path(tempfile.gettempdir(), "medicheck_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir(), "medicheck_matplotlib")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir(), "medicheck_cache")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_prediction_scores(model, features) -> np.ndarray:
    """Return positive-class probabilities or best available decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(features)


def evaluate_binary_classifier(model, features, labels) -> dict[str, Any]:
    """Compute binary classification metrics for a MediCheck model."""
    scores = get_prediction_scores(model, features)
    predictions = (scores >= 0.5).astype(int)

    roc_auc = None
    if len(set(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, scores))

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "confusion_matrix": matrix.tolist(),
        "support": int(len(labels)),
        "positive_rate": round(float(np.mean(labels)), 4),
    }


def evaluate_model(model, features, labels) -> dict[str, Any]:
    """Backward-compatible wrapper for MediCheck model evaluation."""
    return evaluate_binary_classifier(model, features, labels)


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Save model metrics as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def render_metrics_markdown(metrics: dict[str, Any]) -> str:
    """Render model metrics as a Markdown report."""
    rows = []
    for model_name, values in metrics["models"].items():
        rows.append(
            [
                model_name,
                values.get("roc_auc"),
                values.get("accuracy"),
                values.get("precision"),
                values.get("recall"),
                values.get("f1"),
                values.get("support"),
            ]
        )

    table = "\n".join(
        [
            "| Model | ROC-AUC | Accuracy | Precision | Recall | F1 | Support |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
        ]
    )
    return "\n\n".join(
        [
            "# MediCheck Model Evaluation",
            "MediCheck is a synthetic-data research and education prototype. These metrics do not establish clinical validity.",
            table,
            "## Main Model",
            f"- Selected model: `{metrics['metadata']['main_model_name']}`",
            f"- Test size: `{metrics['metadata']['test_size']}`",
            f"- Random state: `{metrics['metadata']['random_state']}`",
        ]
    ) + "\n"


def save_metrics_markdown(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Save model metrics as a Markdown report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_metrics_markdown(metrics), encoding="utf-8")
    return path


def save_confusion_matrix_plot(model, features, labels, output_path: str | Path, title: str = "MediCheck Confusion Matrix") -> Path:
    """Save a confusion matrix plot for a fitted classifier."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(features)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(matrix, display_labels=["No CVD", "CVD"])
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_roc_curve_plot(
    model_scores: dict[str, np.ndarray],
    labels,
    output_path: str | Path,
    title: str = "MediCheck ROC Curve",
) -> Path:
    """Save ROC curves for one or more fitted classifiers."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))
    for model_name, scores in model_scores.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        axis.plot(false_positive_rate, true_positive_rate, label=f"{model_name} AUC={auc:.3f}")

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path

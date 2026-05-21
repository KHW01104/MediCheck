"""Train MediCheck cardiovascular disease risk prediction models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .evaluate import (
        evaluate_binary_classifier,
        get_prediction_scores,
        save_confusion_matrix_plot,
        save_metrics_json,
        save_metrics_markdown,
        save_roc_curve_plot,
    )
except ImportError:  # pragma: no cover - supports direct CLI execution
    from evaluate import (
        evaluate_binary_classifier,
        get_prediction_scores,
        save_confusion_matrix_plot,
        save_metrics_json,
        save_metrics_markdown,
        save_roc_curve_plot,
    )

NUMERIC_FEATURES = ["age", "bmi", "medication_count", "encounter_count", "condition_count"]
BINARY_FEATURES = ["gender_encoded", "has_diabetes", "has_hypertension", "has_obesity", "abnormal_bmi_flag"]
FEATURE_COLUMNS = [*NUMERIC_FEATURES, *BINARY_FEATURES]
TARGET_COLUMN = "target_cvd"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_feature_table(feature_path: str | Path) -> pd.DataFrame:
    """Load the MediCheck patient feature table."""
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(
            "MediCheck could not find the patient feature table.\n"
            f"Missing file: {path}\n"
            "Build it with:\n"
            "python data_pipeline/build_features.py --data-dir data/sample --output data/processed/patient_features.csv"
        )
    return pd.read_csv(path)


def validate_training_data(feature_table: pd.DataFrame) -> None:
    """Validate required columns and target class availability."""
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in feature_table.columns]
    if missing_columns:
        raise ValueError(f"MediCheck feature table is missing columns: {', '.join(missing_columns)}")

    class_counts = feature_table[TARGET_COLUMN].value_counts(dropna=True)
    if len(class_counts) < 2:
        raise ValueError("MediCheck training requires both target classes in target_cvd.")
    if class_counts.min() < 2:
        raise ValueError("MediCheck training requires at least two rows in each target class for stratified splitting.")


def warn_if_imbalanced(labels: pd.Series) -> None:
    """Print and emit a warning when the target classes are strongly imbalanced."""
    positive_rate = float(labels.mean())
    minority_rate = min(positive_rate, 1.0 - positive_rate)
    if minority_rate < 0.1:
        message = (
            "Warning: target_cvd is imbalanced "
            f"(positive_rate={positive_rate:.3f}, minority_rate={minority_rate:.3f}). "
            "Metrics should be interpreted cautiously for this synthetic prototype."
        )
        print(message)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build a preprocessing transformer for MediCheck model features."""
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Build the Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def create_main_estimator(labels: pd.Series):
    """Create the main model, preferring XGBoost and falling back to RandomForest."""
    try:
        from xgboost import XGBClassifier

        negative_count = int((labels == 0).sum())
        positive_count = int((labels == 1).sum())
        scale_pos_weight = negative_count / positive_count if positive_count else 1.0
        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        )
        return estimator, "xgboost"
    except Exception as error:  # pragma: no cover - depends on optional runtime libraries
        reason_lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        reason = reason_lines[0] if reason_lines else error.__class__.__name__
        print(f"Warning: XGBoost is unavailable, using RandomForestClassifier fallback. Reason: {reason}")
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        return estimator, "random_forest_fallback"


def build_main_pipeline(labels: pd.Series) -> tuple[Pipeline, str]:
    """Build the main tree model pipeline without feature scaling."""
    estimator, model_name = create_main_estimator(labels)
    return (
        Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                ("model", estimator),
            ]
        ),
        model_name,
    )


def save_feature_columns(model_dir: Path) -> Path:
    """Save the feature column list used for training."""
    path = model_dir / "feature_columns.json"
    path.write_text(
        json.dumps(
            {
                "numeric_features": NUMERIC_FEATURES,
                "binary_features": BINARY_FEATURES,
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def train_models(
    feature_path: str | Path = "data/processed/patient_features.csv",
    model_dir: str | Path = "models",
    evaluation_dir: str | Path = "evaluation",
) -> dict[str, Any]:
    """Train MediCheck baseline and main models, then save metrics and artifacts."""
    model_path = Path(model_dir)
    evaluation_path = Path(evaluation_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    evaluation_path.mkdir(parents=True, exist_ok=True)

    feature_table = load_feature_table(feature_path)
    validate_training_data(feature_table)

    x = feature_table[FEATURE_COLUMNS].copy()
    y = feature_table[TARGET_COLUMN].astype(int)
    warn_if_imbalanced(y)

    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline_model = build_baseline_pipeline()
    main_model, main_model_name = build_main_pipeline(train_y)

    baseline_model.fit(train_x, train_y)
    main_model.fit(train_x, train_y)

    baseline_metrics = evaluate_binary_classifier(baseline_model, test_x, test_y)
    main_metrics = evaluate_binary_classifier(main_model, test_x, test_y)

    metadata = {
        "project": "MediCheck",
        "baseline_model_name": "logistic_regression",
        "main_model_name": main_model_name,
        "feature_path": str(feature_path),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target_column": TARGET_COLUMN,
        "clinical_claim": "None. Synthetic-data research and education prototype only.",
    }
    metrics = {
        "metadata": metadata,
        "models": {
            "baseline_logistic": baseline_metrics,
            "main_model": main_metrics,
        },
    }

    baseline_path = model_path / "baseline_logistic.pkl"
    main_model_path = model_path / "main_model.pkl"
    feature_columns_path = save_feature_columns(model_path)
    joblib.dump(baseline_model, baseline_path)
    joblib.dump(main_model, main_model_path)

    confusion_path = save_confusion_matrix_plot(
        main_model,
        test_x,
        test_y,
        evaluation_path / "confusion_matrix.png",
        title=f"MediCheck Confusion Matrix ({main_model_name})",
    )
    roc_path = save_roc_curve_plot(
        {
            "baseline_logistic": get_prediction_scores(baseline_model, test_x),
            "main_model": get_prediction_scores(main_model, test_x),
        },
        test_y,
        evaluation_path / "roc_curve.png",
    )

    metrics_json_path = evaluation_path / "metrics.json"
    metrics_md_path = evaluation_path / "metrics.md"
    outputs = {
        "baseline_model": str(baseline_path),
        "main_model": str(main_model_path),
        "feature_columns": str(feature_columns_path),
        "metrics_json": str(metrics_json_path),
        "metrics_markdown": str(metrics_md_path),
        "confusion_matrix": str(confusion_path),
        "roc_curve": str(roc_path),
    }
    metrics["outputs"] = outputs
    save_metrics_json(metrics, metrics_json_path)
    save_metrics_markdown(metrics, metrics_md_path)

    print("MediCheck model training complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"Main model: {main_model_name}")
    print(f"Main model ROC-AUC: {main_metrics['roc_auc']}")
    return metrics


def train_model(feature_table_path: str | Path = "data/processed/patient_features.csv"):
    """Backward-compatible wrapper for earlier MediCheck skeleton usage."""
    return train_models(feature_path=feature_table_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MediCheck model training."""
    parser = argparse.ArgumentParser(description="Train MediCheck cardiovascular risk models.")
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path("data/processed/patient_features.csv"),
        help="Path to patient_features.csv.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for trained model files.")
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("evaluation"),
        help="Directory for evaluation reports and figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Run MediCheck model training from the command line."""
    args = parse_args()
    train_models(feature_path=args.feature_path, model_dir=args.model_dir, evaluation_dir=args.evaluation_dir)


if __name__ == "__main__":
    main()

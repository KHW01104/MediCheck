"""Streamlit dashboard for the MediCheck digital health AI prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docs_explainer import get_factor_explanation
from modeling.predict import SAFETY_NOTICE, load_model_artifacts, predict_risk

QUALITY_REPORT_PATH = PROJECT_ROOT / "quality" / "quality_report.json"
METRICS_PATH = PROJECT_ROOT / "evaluation" / "metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "evaluation" / "confusion_matrix.png"
ROC_CURVE_PATH = PROJECT_ROOT / "evaluation" / "roc_curve.png"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "evaluation" / "feature_importance.png"
FEATURE_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "patient_features.csv"


st.set_page_config(page_title="MediCheck", layout="wide")


def load_json_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_quality_report() -> dict[str, Any] | None:
    """Load the saved MediCheck data quality report."""
    return load_json_file(QUALITY_REPORT_PATH)


@st.cache_data
def load_metrics() -> dict[str, Any] | None:
    """Load saved MediCheck model metrics."""
    return load_json_file(METRICS_PATH)


@st.cache_data
def load_feature_defaults() -> dict[str, Any]:
    """Load default form values from the processed feature table."""
    if not FEATURE_TABLE_PATH.exists():
        return {
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
        }

    feature_table = pd.read_csv(FEATURE_TABLE_PATH)
    return {
        "age": int(feature_table["age"].median()),
        "gender_encoded": int(feature_table["gender_encoded"].mode().iloc[0]),
        "bmi": float(round(feature_table["bmi"].median(), 1)),
        "has_diabetes": int(feature_table["has_diabetes"].mode().iloc[0]),
        "has_hypertension": int(feature_table["has_hypertension"].mode().iloc[0]),
        "has_obesity": int(feature_table["has_obesity"].mode().iloc[0]),
        "medication_count": int(feature_table["medication_count"].median()),
        "encounter_count": int(feature_table["encounter_count"].median()),
        "condition_count": int(feature_table["condition_count"].median()),
        "abnormal_bmi_flag": int(feature_table["abnormal_bmi_flag"].mode().iloc[0]),
    }


@st.cache_resource
def load_prediction_artifacts() -> dict[str, Any]:
    """Load model artifacts once for Streamlit prediction."""
    return load_model_artifacts(PROJECT_ROOT / "models")


def show_missing_artifact(message: str, commands: list[str]) -> None:
    """Show a friendly setup guide when an artifact is missing."""
    st.warning(message)
    st.code("\n".join(commands), language="bash")


def dataframe_from_mapping(mapping: dict[str, Any], key_name: str, value_name: str) -> pd.DataFrame:
    """Convert a mapping to a two-column dataframe for display."""
    return pd.DataFrame([{key_name: key, value_name: value} for key, value in mapping.items()])


def render_overview() -> None:
    """Render the MediCheck overview page."""
    st.title("MediCheck")
    st.subheader("합성 EHR 기반 만성질환 위험 예측 및 의료데이터 품질평가 대시보드")
    st.info("이 시스템은 연구·교육용 프로토타입이며 진단, 치료, 처방 목적으로 사용할 수 없음.")

    col_data, col_stack = st.columns(2)
    with col_data:
        st.markdown("**사용 데이터**")
        st.write("Synthea-style synthetic EHR")
    with col_stack:
        st.markdown("**사용 기술**")
        st.write("Python, pandas, scikit-learn, XGBoost/RandomForest, FastAPI, Streamlit")

    st.markdown("**Safety Notice**")
    st.write(SAFETY_NOTICE)
    st.caption("MediCheck는 실제 환자 데이터가 아닌 합성 데이터를 사용하는 포트폴리오 및 학습용 프로젝트입니다.")


def render_quality() -> None:
    """Render the data quality report page."""
    report = load_quality_report()
    if report is None:
        show_missing_artifact(
            "품질 리포트가 없습니다. 먼저 데이터 생성/feature 생성/품질평가를 실행하세요.",
            [
                "python data_pipeline/generate_sample_data.py --n-patients 1000 --out-dir data/sample",
                "python data_pipeline/build_features.py --data-dir data/sample --output data/processed/patient_features.csv",
                "python quality/quality_check.py --data-dir data/sample --feature-path data/processed/patient_features.csv --output-dir quality",
            ],
        )
        return

    score = report.get("score", {}).get("quality_score", "N/A")
    st.metric("Quality Score", f"{score} / 100")
    st.caption(report.get("quality_score_method", "Synthetic-data heuristic quality score."))

    st.markdown("**Dataset Summary**")
    st.dataframe(dataframe_from_mapping(report.get("summary", {}), "item", "rows"), use_container_width=True)

    missingness = pd.DataFrame(report.get("missingness", []))
    outliers = pd.DataFrame(
        [
            {"rule": feature, **values}
            for feature, values in report.get("outliers", {}).items()
        ]
    )
    duplicates = pd.DataFrame(
        [
            {"target": target, **values}
            for target, values in report.get("duplicates", {}).items()
        ]
    )
    code_coverage = pd.DataFrame(
        [
            {"table": table, **values}
            for table, values in report.get("code_coverage", {}).items()
        ]
    )
    temporal = pd.DataFrame(
        [
            {"rule": rule, **values}
            for rule, values in report.get("temporal_consistency", {}).items()
        ]
    )

    tab_missing, tab_outlier, tab_duplicate, tab_code, tab_time = st.tabs(
        ["Missingness", "Outliers", "Duplicates", "Code Coverage", "Temporal"]
    )
    with tab_missing:
        st.dataframe(missingness, use_container_width=True)
        if not missingness.empty:
            st.bar_chart(missingness.set_index("feature")["missing_rate"])
    with tab_outlier:
        st.dataframe(outliers, use_container_width=True)
    with tab_duplicate:
        st.dataframe(duplicates, use_container_width=True)
    with tab_code:
        st.dataframe(code_coverage, use_container_width=True)
    with tab_time:
        st.dataframe(temporal, use_container_width=True)


def render_performance() -> None:
    """Render model performance metrics and figures."""
    metrics = load_metrics()
    if metrics is None:
        show_missing_artifact(
            "모델 평가 결과가 없습니다. 먼저 데이터 생성/feature 생성/학습/평가를 실행하세요.",
            [
                "python data_pipeline/generate_sample_data.py --n-patients 1000 --out-dir data/sample",
                "python data_pipeline/build_features.py --data-dir data/sample --output data/processed/patient_features.csv",
                "python modeling/train.py --feature-path data/processed/patient_features.csv --model-dir models --evaluation-dir evaluation",
            ],
        )
        return

    rows = []
    for model_name, values in metrics.get("models", {}).items():
        rows.append(
            {
                "model": model_name,
                "roc_auc": values.get("roc_auc"),
                "accuracy": values.get("accuracy"),
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1": values.get("f1"),
                "support": values.get("support"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("모든 성능 지표는 합성 데이터 기반 프로토타입 결과이며 임상적 유효성을 의미하지 않습니다.")

    col_matrix, col_roc = st.columns(2)
    with col_matrix:
        st.markdown("**Confusion Matrix**")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
        else:
            st.info("confusion_matrix.png가 없습니다. 먼저 학습/평가를 실행하세요.")
    with col_roc:
        st.markdown("**ROC Curve**")
        if ROC_CURVE_PATH.exists():
            st.image(str(ROC_CURVE_PATH), use_container_width=True)
        else:
            st.info("roc_curve.png가 없습니다. 먼저 학습/평가를 실행하세요.")


def render_risk_badge(risk_level: str) -> None:
    """Render a compact text badge for the risk level."""
    st.markdown(f"**Risk Level:** `{risk_level.upper()}`")


def render_prediction() -> None:
    """Render the patient risk prediction form."""
    defaults = load_feature_defaults()
    st.caption("이 결과는 진단이 아니라 합성 데이터 기반 예측 프로토타입의 출력입니다.")

    with st.form("patient-risk-form"):
        col_left, col_right = st.columns(2)
        with col_left:
            age = st.number_input("Age", min_value=0, max_value=120, value=int(defaults["age"]))
            gender_encoded = st.selectbox("Gender encoded", options=[0, 1], index=int(defaults["gender_encoded"]))
            bmi = st.number_input("BMI", min_value=0.0, max_value=120.0, value=float(defaults["bmi"]), step=0.1)
            medication_count = st.number_input(
                "Medication count",
                min_value=0,
                max_value=100,
                value=int(defaults["medication_count"]),
            )
            encounter_count = st.number_input(
                "Encounter count",
                min_value=0,
                max_value=100,
                value=int(defaults["encounter_count"]),
            )
        with col_right:
            condition_count = st.number_input(
                "Condition count",
                min_value=0,
                max_value=100,
                value=int(defaults["condition_count"]),
            )
            has_diabetes = st.checkbox("Diabetes history", value=bool(defaults["has_diabetes"]))
            has_hypertension = st.checkbox("Hypertension history", value=bool(defaults["has_hypertension"]))
            has_obesity = st.checkbox("Obesity history", value=bool(defaults["has_obesity"]))
            abnormal_bmi_flag = st.checkbox("Abnormal BMI flag", value=bool(defaults["abnormal_bmi_flag"]))

        submitted = st.form_submit_button("Predict Risk")

    if not submitted:
        return

    try:
        artifacts = load_prediction_artifacts()
        result = predict_risk(
            {
                "age": age,
                "gender_encoded": gender_encoded,
                "bmi": bmi,
                "has_diabetes": int(has_diabetes),
                "has_hypertension": int(has_hypertension),
                "has_obesity": int(has_obesity),
                "medication_count": medication_count,
                "encounter_count": encounter_count,
                "condition_count": condition_count,
                "abnormal_bmi_flag": int(abnormal_bmi_flag),
            },
            artifacts,
        )
    except FileNotFoundError as error:
        show_missing_artifact(
            str(error),
            [
                "python modeling/train.py --feature-path data/processed/patient_features.csv --model-dir models --evaluation-dir evaluation",
                "python modeling/explain.py",
            ],
        )
        return
    except Exception as error:
        st.error(f"예측 중 오류가 발생했습니다: {error}")
        return

    st.metric("Risk Score", f"{result['risk_score']:.2f}")
    render_risk_badge(result["risk_level"])
    st.markdown("**Top Risk Factors**")
    if result["top_risk_factors"]:
        st.write(", ".join(result["top_risk_factors"]))
        factor_rows = []
        for factor in result["top_risk_factor_details"]:
            factor_rows.append(
                {
                    "risk_factor": factor["label"],
                    "value": factor["value"],
                    "model_message": factor["message"],
                    "document_explanation": get_factor_explanation(factor["label"]),
                }
            )
        st.dataframe(pd.DataFrame(factor_rows), use_container_width=True)
    else:
        st.write("현재 입력에서 표시할 주요 위험 요인이 많지 않습니다.")
    st.caption(result["safety_notice"])


def render_feature_importance() -> None:
    """Render global feature importance artifacts."""
    if not FEATURE_IMPORTANCE_PATH.exists():
        show_missing_artifact(
            "feature_importance.png가 없습니다. 먼저 학습과 설명 산출물 생성을 실행하세요.",
            [
                "python modeling/train.py --feature-path data/processed/patient_features.csv --model-dir models --evaluation-dir evaluation",
                "python modeling/explain.py",
            ],
        )
        return

    st.image(str(FEATURE_IMPORTANCE_PATH), use_container_width=True)
    st.write(
        "Feature importance는 모델이 예측에 사용한 변수의 상대적 영향도를 요약한 설명 보조 지표입니다. "
        "이는 인과관계나 임상적 판단을 의미하지 않습니다."
    )


def main() -> None:
    """Run the MediCheck Streamlit dashboard."""
    overview, quality, performance, prediction, importance = st.tabs(
        ["Overview", "Data Quality", "Model Performance", "Patient Risk Prediction", "Feature Importance"]
    )

    with overview:
        render_overview()
    with quality:
        render_quality()
    with performance:
        render_performance()
    with prediction:
        render_prediction()
    with importance:
        render_feature_importance()


if __name__ == "__main__":
    main()

# MediCheck

MediCheck는 Synthea-style 합성 EHR 데이터를 사용해 만성질환 관련 feature를 만들고, 심혈관질환 위험도를 예측하며, 의료데이터 품질 문제와 주요 위험 요인을 함께 보여주는 디지털헬스 AI 포트폴리오 프로젝트입니다.

## Safety Notice

MediCheck는 연구·교육용 프로토타입입니다.

- 실제 환자 데이터를 사용하지 않습니다.
- 진단, 치료, 처방, 임상 의사결정 목적으로 사용할 수 없습니다.
- 모델 성능과 예측 결과는 합성 데이터 기반 소프트웨어 데모 결과이며 임상적 유효성을 의미하지 않습니다.
- 문서 기반 설명 도우미는 의료전문가 판단을 대체하지 않습니다.

자세한 내용은 [docs/safety_notice.md](docs/safety_notice.md)를 참고하세요.

## Features

- Synthea-style synthetic EHR sample data generation
- Patient-level feature engineering
- Medical data quality report
- Logistic Regression baseline model
- XGBoost main model with RandomForest fallback
- Model evaluation with ROC-AUC, Accuracy, Precision, Recall, F1, and Confusion Matrix
- Global feature importance and patient-level top risk factors
- FastAPI prediction API
- Streamlit digital health dashboard
- Lightweight document explainer using [docs/medical_knowledge.md](docs/medical_knowledge.md)

## Quickstart

아래 순서대로 실행하면 처음 프로젝트를 받은 사람도 sample data 생성부터 API와 dashboard 실행까지 재현할 수 있습니다.

### 1. 가상환경 생성

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

예상 생성 파일:

```text
.venv/
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

예상 생성 파일:

```text
.venv/lib/... 또는 .venv/Lib/...
```

### 3. sample data 생성

```bash
python data_pipeline/generate_sample_data.py --n-patients 1000 --out-dir data/sample
```

예상 생성 파일:

```text
data/sample/patients.csv
data/sample/conditions.csv
data/sample/medications.csv
data/sample/encounters.csv
data/sample/observations.csv
```

### 4. feature table 생성

```bash
python data_pipeline/build_features.py --data-dir data/sample --output data/processed/patient_features.csv
```

예상 생성 파일:

```text
data/processed/patient_features.csv
```

### 5. data quality check 실행

```bash
python quality/quality_check.py --data-dir data/sample --feature-path data/processed/patient_features.csv --output-dir quality
```

예상 생성 파일:

```text
quality/quality_report.json
quality/quality_report.md
```

### 6. model training 실행

```bash
python modeling/train.py --feature-path data/processed/patient_features.csv --model-dir models --evaluation-dir evaluation
```

예상 생성 파일:

```text
models/baseline_logistic.pkl
models/main_model.pkl
models/feature_columns.json
evaluation/metrics.json
evaluation/metrics.md
evaluation/confusion_matrix.png
evaluation/roc_curve.png
```

선택 사항으로 feature importance를 생성하려면 다음 명령을 실행합니다.

```bash
python modeling/explain.py
```

예상 생성 파일:

```text
evaluation/feature_importance.json
evaluation/feature_importance.png
```

### 7. API 실행

```bash
uvicorn api.main:app --reload
```

예상 실행 결과:

```text
http://127.0.0.1:8000
```

주요 endpoint:

```text
GET  /health
GET  /model-info
POST /predict
GET  /quality-report
```

### 8. dashboard 실행

```bash
streamlit run dashboard/app.py
```

예상 실행 결과:

```text
http://localhost:8501
```

대시보드에서 확인할 수 있는 항목:

```text
Overview
Data Quality
Model Performance
Patient Risk Prediction
Feature Importance
```

### 9. test 실행

```bash
pytest -q
```

예상 출력:

```text
13 passed
```

## Project Structure

```text
MediCheck/
├─ api/
├─ dashboard/
├─ data/
│  ├─ sample/
│  └─ processed/
├─ data_pipeline/
├─ docs/
├─ docs_explainer/
├─ evaluation/
├─ modeling/
├─ models/
├─ quality/
├─ tests/
├─ README.md
└─ requirements.txt
```

## Feature Definition

주요 feature는 [docs/feature_definition.md](docs/feature_definition.md)에 정리되어 있습니다.

Target:

```text
target_cvd
```

`target_cvd`는 `conditions.DESCRIPTION`에 `cardiovascular`, `coronary`, `myocardial`, `heart disease`, `stroke` 중 하나가 포함되면 1로 정의합니다.

## Limitations

- 합성 데이터 기반 프로토타입입니다.
- 현재 feature와 label이 같은 `conditions.csv`에서 파생되므로 label leakage 가능성이 있습니다.
- 실제 임상 성능, 진단 정확도, 치료 효과를 주장하지 않습니다.
- 의료데이터 품질평가는 포트폴리오용 heuristic이며 실제 임상 품질평가 기준을 대체하지 않습니다.

자세한 내용은 [docs/limitations.md](docs/limitations.md)를 참고하세요.

## Research Portfolio Positioning

MediCheck는 다음 역량을 보여주기 위한 미니 프로젝트입니다.

- 의료데이터사이언스
- 의료 AI 모델링
- 의료데이터 품질평가
- 설명가능 AI
- FastAPI 기반 추론 API
- Streamlit 기반 디지털헬스 대시보드
- 향후 의료 LLM/RAG, EHR/CDM, 디지털헬스 플랫폼 확장 가능성

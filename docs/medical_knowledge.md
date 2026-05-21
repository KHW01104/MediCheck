# MediCheck Medical Knowledge Notes

MediCheck uses these short notes as a simple explanation helper for synthetic-data demos.
They are general educational summaries and must not be interpreted as diagnosis, treatment, or prescription advice.

## diabetes_cvd_risk

Aliases: diabetes, has_diabetes, 당뇨 이력, 당뇨병

당뇨병은 일반적으로 심혈관질환 위험과 관련이 있을 수 있습니다. 혈당 조절 상태, 동반 질환, 생활습관, 약물 사용 등 여러 요인이 함께 영향을 줄 수 있으므로, MediCheck의 설명은 예측 모델이 참고한 위험 요인을 이해하기 위한 일반 정보입니다.

## hypertension_cvd_risk

Aliases: hypertension, has_hypertension, 고혈압 이력, 고혈압, high blood pressure

고혈압은 일반적으로 심장과 혈관에 부담을 줄 수 있어 심혈관질환 위험과 관련이 있을 수 있습니다. 다만 개인의 실제 위험은 혈압 수준, 기간, 동반 질환, 검사 결과 등에 따라 달라지며, 이 설명은 임상 판단을 대체하지 않습니다.

## obesity_chronic_risk

Aliases: obesity, has_obesity, 비만 이력, 비만

비만은 당뇨병, 고혈압, 이상지질혈증 등 여러 만성질환 위험과 일반적으로 관련이 있을 수 있습니다. MediCheck에서는 합성 EHR feature 중 하나로 사용되며, 개별 환자의 건강 상태를 단정하지 않습니다.

## bmi_meaning

Aliases: bmi, BMI, 높은 BMI, 비정상 BMI 값, abnormal_bmi_flag

BMI는 체중과 키를 이용해 계산하는 지표로, 체중 상태를 대략적으로 살펴보는 데 사용됩니다. 근육량, 체성분, 질환 상태를 모두 반영하지는 못하므로, MediCheck에서는 예측 모델의 입력 feature 중 하나로만 해석해야 합니다.

## ai_model_limitations

Aliases: 의료 AI 예측 모델의 한계, model limitation, risk_score, risk_level, 많은 처방 약물 수, 많은 의료기관 방문 횟수, 많은 진단 기록 수, medication_count, encounter_count, condition_count, 고령, age

의료 AI 예측 모델은 입력 데이터의 패턴을 바탕으로 위험도를 추정하는 도구입니다. 높은 위험도로 예측되었다는 것은 관련 feature 조합이 모델에서 위험 신호로 해석되었다는 의미이며, 특정 질환이 있다고 단정하는 표현이 아닙니다.

## synthetic_data_limitations

Aliases: 합성 의료데이터의 한계, synthetic data, Synthea, synthetic EHR

합성 의료데이터는 소프트웨어 테스트와 교육용 데모에 유용하지만 실제 임상 데이터의 복잡성, 기록 편향, 기관별 차이, 환자 다양성을 완전히 반영하지 못합니다. MediCheck 결과는 실제 임상 성능이나 의료적 유효성을 입증하지 않습니다.


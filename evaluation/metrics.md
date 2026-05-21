# MediCheck Model Evaluation

MediCheck is a synthetic-data research and education prototype. These metrics do not establish clinical validity.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_logistic | 0.9994 | 0.985 | 0.8696 | 1.0 | 0.9302 | 200 |
| main_model | 0.9947 | 0.985 | 0.8696 | 1.0 | 0.9302 | 200 |

## Main Model

- Selected model: `random_forest_fallback`

- Test size: `0.2`

- Random state: `42`

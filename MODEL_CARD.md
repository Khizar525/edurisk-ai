# Model Card — EduRisk AI

## Model Details

**Model Name:** EduRisk AI — Student Academic Risk Predictor
**Version:** 1.0.0
**Date:** July 2026
**Developed by:** Khizar Akram, Safwan Marwat, Syed Mughees, Ifrahim Yousuf
**Organization:** Bahria University Karachi Campus
**Course:** CSL 460 — Data Mining
**License:** MIT
**Repository:** [github.com/Khizar525/edurisk-ai](https://github.com/Khizar525/edurisk-ai)

## Model Type

Multi-class classifier (Random Forest) with SHAP-based explainability.
- **Algorithm:** Random Forest (100 estimators, max_depth=15)
- **Framework:** scikit-learn 1.6+
- **Explainability:** SHAP TreeExplainer

## Intended Use

**Primary Use:** Predict academic risk level (Low/Medium/High) for university students based on self-reported lifestyle, psychological, and academic indicators.

**Intended Users:** University counselors, academic advisors, student welfare departments.

**Out-of-Scope Uses:**
- Clinical diagnosis of mental health conditions
- Automated decision-making without human oversight
- Students under 17 or over 30 years of age

## Training Data

**Dataset:** [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (Kaggle)
**Samples:** 27,901 student records
**Features:** 11 (after selection from 18 original columns)
**Target:** 3-class risk level (Low: 37.5%, Medium: 21.8%, High: 40.7%)

### Features Used

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Gender | Categorical | Male/Female | Student gender |
| Age | Numerical | 17-30 | Student age |
| Academic Pressure | Ordinal | 1-5 | Self-reported academic pressure |
| CGPA | Continuous | 0.0-4.0 | Cumulative GPA |
| Study Satisfaction | Ordinal | 1-5 | Self-reported study satisfaction |
| Sleep Duration | Categorical | 4 categories | Daily sleep duration |
| Dietary Habits | Categorical | Healthy/Moderate/Unhealthy | Dietary quality |
| Work/Study Hours | Numerical | 0-12 | Daily study/work hours |
| Financial Stress | Ordinal | 1-5 | Self-reported financial stress |
| Family History | Categorical | Yes/No | Family history of mental illness |
| Suicidal Thoughts | Categorical | Yes/No | History of suicidal thoughts |

### Data Preprocessing

1. **Imputation:** Median for numerical, mode for categorical (3 missing values in Financial Stress)
2. **Encoding:** LabelEncoder for categorical features (deterministic mapping)
3. **Scaling:** StandardScaler fit on training data only (no leakage)
4. **Risk Engineering:** Composite scoring formula mapping raw indicators to 3-class risk level

## Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **85.58%** |
| **ROC-AUC (OvR)** | **94.92%** |
| **3-Fold CV Mean** | 85.27% |
| **3-Fold CV Std** | 0.39% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low Risk | 0.87 | 0.88 | 0.88 | 2,090 |
| Medium Risk | 0.70 | 0.59 | 0.64 | 1,219 |
| High Risk | 0.91 | 0.97 | 0.94 | 2,272 |
| **Weighted Avg** | **0.85** | **0.86** | **0.85** | **5,581** |

### Model Comparison

| Model | Accuracy | ROC-AUC | CV Mean |
|-------|----------|---------|---------|
| **Random Forest** | **0.8558** | **0.9492** | 0.8527 |
| XGBoost | 0.8524 | 0.9502 | 0.8588 |
| MLP | 0.8518 | 0.9469 | 0.8466 |
| SVM | 0.8212 | 0.9310 | 0.8240 |

### Error Analysis

| Class | Error Rate | Most Common Confusion |
|-------|------------|----------------------|
| Low Risk | 11.9% | → Medium Risk (249 cases) |
| Medium Risk | 40.9% | → Low Risk (274 cases) |
| High Risk | 2.6% | → Medium Risk (57 cases) |

**Key Finding:** Medium Risk is the hardest class to predict due to overlap with adjacent classes. High Risk is the easiest to predict with only 2.6% error rate.

## Fairness and Bias

### Known Limitations

1. **Label Encoding:** Categorical features are label-encoded, which may introduce artificial ordinal relationships for linear models (tree-based models are unaffected)
2. **Risk Engineering:** The composite scoring formula uses fixed thresholds that may not generalize across different student populations
3. **Class Imbalance:** Medium Risk class is underrepresented (21.8%), leading to lower recall for that class
4. **Self-Reported Data:** All features are self-reported, subject to response bias
5. **Demographic Bias:** The model was trained on a specific student population and may not generalize to other demographics

### Recommendations

- Use as a screening tool, not a diagnostic instrument
- Always involve human counselors in final decisions
- Monitor prediction distribution for demographic bias in deployment
- Retrain periodically with updated data

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| Train/Test Split | 80/20 stratified |
| Cross-Validation | 3-fold stratified |
| Tuning Method | GridSearchCV |
| Scoring Metric | Accuracy |

### Hyperparameters (Best Model — Random Forest)

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 15 |
| min_samples_split | 5 |
| random_state | 42 |

## Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Model | `models/best_model.pkl` | Trained Random Forest classifier |
| Scaler | `models/scaler.pkl` | StandardScaler (fit on training data) |
| Encoders | `models/encoders.pkl` | LabelEncoders for categorical features |
| Features | `models/feature_names.pkl` | Ordered feature name list |

## How to Use

### Python

```python
from src.inference.predictor import RiskPredictor

predictor = RiskPredictor()
result = predictor.predict(
    gender="Male", age=20, academic_pressure=4,
    cgpa=2.2, study_satisfaction=2,
    sleep_duration="Less than 5 hours",
    dietary_habits="Unhealthy",
    work_study_hours=8, financial_stress=4,
    family_history="Yes", suicidal_thoughts="Yes",
)
print(result["risk_level"])  # ❌ High Risk
print(result["confidence"])  # 87.3%
```

### REST API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "Male", "age": 20, ...}'
```

### Gradio UI

```bash
python -m app.main
```

## Citation

```bibtex
@misc{edurisk2026,
  title={EduRisk AI: Student Academic Risk Prediction},
  author={Khizar Akram and Safwan Marwat and Syed Mughees and Ifrahim Yousuf},
  year={2026},
  institution={Bahria University Karachi Campus},
  course={CSL 460 - Data Mining}
}
```

# ML Methodology

## Dataset

**Student Depression Dataset** from Kaggle (~27,000 records, 27 features).

## Feature Selection

11 features were selected based on domain relevance:

| Feature | Type | Reasoning |
|---------|------|-----------|
| Gender | Categorical | Demographic baseline |
| Age | Numerical | Developmental factor |
| Academic Pressure | Ordinal | Direct academic indicator |
| CGPA | Numerical | Academic performance |
| Study Satisfaction | Ordinal | Engagement indicator |
| Sleep Duration | Categorical | Lifestyle factor |
| Dietary Habits | Categorical | Lifestyle factor |
| Work/Study Hours | Numerical | Time allocation |
| Financial Stress | Ordinal | Environmental factor |
| Family History | Categorical | Genetic/environmental risk |
| Suicidal Thoughts | Categorical | Mental health severity |

## Preprocessing

1. **Missing Values**: Median (numerical), Mode (categorical)
2. **Encoding**: LabelEncoder for categorical features
3. **Scaling**: StandardScaler (zero mean, unit variance)

## Target Engineering

Composite scoring function combines multiple indicators into a 3-class risk level:

- **Risk factors**: Depression (+3), Academic Pressure ≥4 (+2), CGPA <2.0 (+2), Financial Stress ≥4 (+1), Suicidal Thoughts (+3), Low Sleep (+1)
- **Protective factors**: CGPA ≥3.0 (-2), Study Satisfaction ≥4 (-1)
- **Thresholds**: score ≤1 → Low, ≤4 → Medium, else → High

## Models

| Model | Type | Hyperparameters |
|-------|------|-----------------|
| Random Forest | Ensemble (bagging) | n_estimators, max_depth, min_samples_split |
| SVM (RBF) | Kernel method | kernel=rbf, probability=True |
| XGBoost | Ensemble (boosting) | n_estimators, max_depth, learning_rate |
| MLP | Neural network | (64, 32) architecture |

## Evaluation

- **Metrics**: Accuracy, ROC-AUC (OvR), 5-fold Cross-Validation
- **Visualizations**: Confusion matrices, ROC curves, model comparison charts
- **Explainability**: SHAP (TreeExplainer for tree models, KernelExplainer for others)

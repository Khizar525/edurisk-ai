<div align="center">

# EduRisk AI

### Predicting Academic Risk Before It Becomes a Crisis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-41B883.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-E0003D.svg?style=for-the-badge&logo=python&logoColor=white)](https://shap.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF5722.svg?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br>

An end-to-end machine learning system that predicts the **academic risk level** (Low / Medium / High) of university students using self-reported lifestyle, psychological, and academic indicators — with **SHAP explainability** and a **live Gradio dashboard**.

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Results](#-results) • [Notebooks](#-notebooks) • [Deployment](#-deployment)

</div>

---

## Overview

Every year, universities lose students to academic failure and mental health crises that could have been intercepted earlier. **EduRisk AI** uses machine learning to predict which students are at high academic risk based on survey data — and explains exactly why.

The system compares **four classifiers** — Random Forest, SVM, XGBoost, and MLP — with hyperparameter tuning, cross-validation, and SHAP-based explainability. A Gradio web interface provides real-time risk assessment with per-prediction feature contribution analysis.

### Key Features

- **Multi-Model Comparison**: Random Forest, SVM (RBF), XGBoost, MLP
- **Hyperparameter Tuning**: GridSearchCV with stratified cross-validation
- **SHAP Explainability**: Global feature importance + per-prediction explanations
- **Risk Engineering**: Composite scoring to generate 3-class academic risk levels
- **Live Dashboard**: Gradio web interface with real-time predictions
- **Prediction Logging**: Timestamped CSV logging of all predictions
- **Reproducible Pipeline**: Sklearn Pipeline + ColumnTransformer

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Khizar525/edurisk-ai.git
cd edurisk-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m app.main
```

The Gradio dashboard will launch at `http://localhost:7860` with a public shareable link.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EduRisk AI                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  Data     │──▶│ Preprocessing│──▶│  Feature Engineering   │  │
│  │  Ingestion│   │  Pipeline    │   │  (Risk Level Scoring)  │  │
│  └──────────┘   └──────────────┘   └───────────┬────────────┘  │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Model Training                         │   │
│  │  ┌─────────┐ ┌─────┐ ┌─────────┐ ┌─────┐               │   │
│  │  │   RF    │ │ SVM │ │ XGBoost │ │ MLP │  + GridSearch  │   │
│  │  └─────────┘ └─────┘ └─────────┘ └─────┘               │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                          │                                       │
│              ┌───────────┼───────────┐                          │
│              ▼           ▼           ▼                          │
│        ┌──────────┐ ┌─────────┐ ┌────────────┐                 │
│        │  SHAP    │ │  Gradio │ │ Prediction │                 │
│        │  (XAI)   │ │  (App)  │ │  Logger    │                 │
│        └──────────┘ └─────────┘ └────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Repository Structure

```
edurisk-ai/
├── app/                    # Gradio application
│   ├── main.py             # App entrypoint
│   ├── ui/                 # UI components and theme
│   └── assets/             # Static assets (icons, images)
├── src/                    # Core ML package
│   ├── config.py           # Central configuration
│   ├── data/               # Data loading and validation
│   ├── preprocessing/      # Cleaning, encoding, scaling
│   ├── features/           # Feature engineering
│   ├── training/           # Model training and tuning
│   ├── evaluation/         # Metrics, plots, reports
│   ├── explainability/     # SHAP utilities
│   ├── inference/          # Prediction service and logging
│   └── utils/              # Shared utilities
├── notebooks/              # Jupyter notebooks (exploration)
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── models/                 # Saved model artifacts
├── data/                   # Raw and processed data
└── docker/                 # Containerization
```

---

## Dataset

**Student Depression Dataset** — [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

| Property | Value |
|----------|-------|
| Records | ~27,000 |
| Features | 27 (11 selected for modeling) |
| Target | 3-class Risk Level (Low / Medium / High) |
| Source | Student self-report survey |

### Selected Features

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Student gender |
| Age | Numerical | Student age |
| Academic Pressure | Ordinal (1-5) | Self-reported academic pressure |
| CGPA | Numerical | Cumulative GPA |
| Study Satisfaction | Ordinal (1-5) | Satisfaction with study conditions |
| Sleep Duration | Categorical | Daily sleep hours |
| Dietary Habits | Categorical | General diet quality |
| Work/Study Hours | Numerical | Daily study/work hours |
| Financial Stress | Ordinal (1-5) | Self-reported financial stress |
| Family History | Categorical | Family history of mental illness |
| Suicidal Thoughts | Categorical | History of suicidal ideation |

---

## Methodology

### Pipeline

1. **Data Collection**: Kaggle API download + initial inspection
2. **EDA**: Correlation analysis, distributions, class balance
3. **Preprocessing**: Missing value imputation (median/mode), label encoding, standard scaling
4. **Feature Engineering**: Composite risk scoring → 3-class target variable
5. **Training**: 4 classifiers with GridSearchCV hyperparameter tuning
6. **Evaluation**: Accuracy, ROC-AUC, 5-fold CV, confusion matrices, ROC curves
7. **Explainability**: SHAP global + local feature importance
8. **Deployment**: Gradio web interface with prediction logging

### Models

| Model | Hyperparameters Tuned | Notes |
|-------|----------------------|-------|
| Random Forest | n_estimators, max_depth, min_samples_split | TreeExplainer for SHAP |
| SVM (RBF) | Kernel: RBF | Platt scaling for probabilities |
| XGBoost | n_estimators, max_depth, learning_rate | Gradient boosting |
| MLP | Architecture: (64, 32) | Neural network baseline |

---

## Results

> Results will be populated after training run. See `docs/results.md` for detailed metrics.

| Model | Accuracy | ROC-AUC | 5-Fold CV |
|-------|----------|---------|-----------|
| Random Forest | — | — | — |
| SVM | — | — | — |
| XGBoost | — | — | — |
| MLP | — | — | — |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Exploratory Data Analysis and visualizations |
| `02_Preprocessing.ipynb` | Data cleaning, encoding, and feature engineering |
| `03_Training.ipynb` | Model training and hyperparameter tuning |
| `04_Evaluation.ipynb` | Comprehensive model evaluation |
| `05_Explainability.ipynb` | SHAP analysis and interpretation |
| `06_Deployment.ipynb` | Gradio app demo and walkthrough |

---

## Deployment

### Local

```bash
python -m app.main
```

### Docker

```bash
cd docker
docker-compose up --build
```

### Gradio Share

The app supports `share=True` for temporary public URLs via ngrok.

---

## Documentation

- [Architecture Guide](docs/architecture.md)
- [ML Methodology](docs/methodology.md)
- [Results & Metrics](docs/results.md)

---

## Future Work

- [ ] Optuna-based hyperparameter optimization
- [ ] MLflow experiment tracking
- [ ] FastAPI REST API alongside Gradio
- [ ] PostgreSQL prediction logging
- [ ] CI/CD pipeline with GitHub Actions
- [ ] A/B testing framework for model comparison
- [ ] Student dashboard with historical trends
- [ ] Integration with university SIS (Student Information Systems)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Dataset**: [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) on Kaggle
- **Course**: CSL 460 — Data Mining, Bahria University Karachi Campus
- **Instructors**: Dr. Hussain (Course), Engr. Noor us Sabah (Lab)

---

<div align="center">

**Built with by the EduRisk AI Team**

[![Khizar](https://img.shields.io/badge/Khizar-Akram-blue?style=flat-square)](https://github.com/Khizar525)

</div>

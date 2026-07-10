<div align="center">

# EduRisk AI

### Production-Oriented Machine Learning for Academic Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-41B883.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-E0003D.svg?style=for-the-badge&logo=python&logoColor=white)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-000000.svg?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-62%20passing-brightgreen.svg?style=for-the-badge)](#testing)

<br>

**EduRisk AI is a production-oriented machine learning platform for predicting academic risk using explainable AI.** Built with Python, FastAPI, Next.js, SHAP, and Scikit-learn, it demonstrates the complete machine learning lifecycle — from data preprocessing and model training to real-time inference and interactive visualization.

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Screenshots](#-screenshots) • [Results](#-results) • [Deployment](#-deployment) • [Documentation](#-documentation)

</div>

---

## Why This Project?

Most machine learning repositories stop at training a model.

EduRisk AI was designed to demonstrate **production-oriented machine learning engineering** by combining:

- Reproducible preprocessing pipelines with no data leakage
- Explainable AI with SHAP (global + per-prediction)
- REST APIs with FastAPI
- Interactive web applications with Next.js and Tailwind CSS
- Modular software architecture across 8+ packages
- Automated testing (62 unit tests)
- Containerized deployment with Docker

The project emphasizes **engineering quality** as much as predictive performance.

---

## Tech Stack

```
Frontend          Backend           Machine Learning
─────────────     ─────────────     ─────────────────
Next.js           FastAPI           Scikit-learn
Tailwind CSS      Gradio            XGBoost
TypeScript        Uvicorn           Optuna
Framer Motion                       SHAP

Data              DevOps            Testing
─────────────     ─────────────     ─────────────────
Pandas            Docker            Pytest
NumPy             GitHub Actions    Coverage
Kaggle API
```

---

## Overview

Every year, universities lose students to academic failure and mental health crises that could have been intercepted earlier. **EduRisk AI** uses machine learning to predict which students are at high academic risk based on survey data — and explains exactly why.

The system compares **four classifiers** — Random Forest, SVM, XGBoost, and MLP — with hyperparameter tuning (GridSearchCV or Optuna), cross-validation, and SHAP-based explainability. The same prediction engine serves three interfaces: a Next.js web app, a FastAPI REST API, and a Gradio dashboard.

### Capabilities

**Machine Learning**
- Multi-model training and comparison (Random Forest, SVM, XGBoost, MLP)
- Hyperparameter optimization (GridSearchCV + Optuna Bayesian)
- SHAP explainability with waterfall plots and human-readable interpretations
- Composite risk scoring → 3-class academic risk levels
- Error analysis with misclassification patterns

**Engineering**
- FastAPI REST API with `/predict`, `/health`, `/model/info` endpoints
- Next.js frontend with dark-mode UI, animated gauges, and SHAP visualizations
- Gradio dashboard with real-time risk assessment
- Docker containerization
- Timestamped prediction logging with analytics

**MLOps**
- Reproducible pipeline (no data leakage, fixed random seeds)
- Centralized configuration management
- Saved model artifacts and preprocessing pipelines
- 62 unit tests across all modules

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

### Train Models

```bash
# Train all models and save best
python -m src.training.trainer
```

---

## Architecture

EduRisk AI separates training, inference, and presentation into independent layers, allowing the same prediction engine to serve both the REST API and web interfaces.

```mermaid
graph TB
    subgraph "Data Layer"
        A[Kaggle API] --> B[Raw CSV]
        B --> C[Validation]
    end

    subgraph "Processing"
        C --> D[Imputation]
        D --> E[Encoding]
        E --> F[Risk Engineering]
    end

    subgraph "Training"
        F --> G[Split]
        G --> H[Scale - Train Only]
        H --> I[4 Models]
        I --> J[Tuning - GridSearch/Optuna]
        J --> K[Best Model]
    end

    subgraph "Inference"
        K --> L[Predictor]
        L --> M[SHAP Explainer]
        L --> N[Logger]
    end

    subgraph "UI"
        L --> O[Gradio Dashboard]
        M --> P[Waterfall Plots]
        M --> Q[Interpretation]
        O --> R["Tabs: Predict, Analytics, About"]
    end

    style G fill:#ffcdd2
    style H fill:#c8e6c9
    style O fill:#fff3e0
```

### Data Flow (No Leakage)

```mermaid
flowchart LR
    A[Load] --> B[Impute]
    B --> C[Encode]
    C --> D[Risk Target]
    D --> E[Split 80/20]
    E --> F[Fit Scaler on Train]
    F --> G[Transform Train]
    F --> H[Transform Test]
    G --> I[Train]
    I --> J[Evaluate]

    style E fill:#ffcdd2
    style F fill:#c8e6c9
```

### Repository Structure

```
edurisk-ai/
├── app/                    # Gradio application + FastAPI
│   ├── main.py             # Gradio UI layout
│   └── api.py              # FastAPI REST API
├── frontend/               # Next.js + Tailwind frontend
│   ├── src/app/page.tsx    # Main page component
│   └── src/components/     # UI components (Gauge, SHAP, etc.)
├── src/                    # Core ML package
│   ├── config.py           # Central configuration
│   ├── data/               # Data loading and validation
│   ├── preprocessing/      # Cleaning, encoding, scaling
│   ├── features/           # Feature engineering
│   ├── training/           # Model training and tuning
│   ├── evaluation/         # Metrics, plots, error analysis
│   ├── explainability/     # SHAP utilities
│   ├── inference/          # Prediction service and logging
│   └── utils/              # Shared utilities
├── tests/                  # 62 unit tests
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentation (Mermaid diagrams)
├── assets/                 # Charts, screenshots, model figures
├── models/                 # Saved model artifacts
├── data/                   # Raw and processed data
└── docker/                 # Containerization
```

---

## Screenshots

### Dashboard — Empty State

![Dashboard](assets/images/screenshot-dashboard.png)

### Prediction — High Risk Result

![Prediction](assets/images/screenshot-prediction.png)

### SHAP Feature Contributions

![SHAP Waterfall](assets/images/screenshot-factors.png)

### FastAPI Swagger UI

![Swagger](assets/images/screenshot-swagger.png)

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

```mermaid
flowchart TD
    A[1. Data Collection] --> B[2. EDA]
    B --> C[3. Preprocessing]
    C --> D[4. Feature Engineering]
    D --> E[5. Model Training]
    E --> F[6. Evaluation]
    F --> G[7. SHAP Explainability]
    G --> H[8. Deployment]
```

### Models

| Model | Type | Tuning | SHAP |
|-------|------|--------|------|
| Random Forest | Ensemble (bagging) | GridSearchCV / Optuna | TreeExplainer |
| SVM (RBF) | Kernel method | GridSearchCV / Optuna | KernelExplainer |
| XGBoost | Ensemble (boosting) | GridSearchCV / Optuna | TreeExplainer |
| MLP | Neural network | GridSearchCV / Optuna | KernelExplainer |

### Evaluation

- **Metrics**: Accuracy, ROC-AUC (OvR), 5-fold CV, Precision, Recall, F1
- **Visualizations**: Confusion matrices, ROC curves, PR curves, calibration plots, learning curves, radar charts
- **Error Analysis**: Misclassification patterns, feature comparison, confusion pairs

---

## Results

> **Best Performing Model — Random Forest**
>
> - Accuracy: **85.58%**
> - ROC-AUC: **94.92%**
> - Cross-validation: **85.27 ± 0.39%**
> - Selected for deployment due to its balance of predictive performance and interpretability.

| Model | Accuracy | ROC-AUC | 3-Fold CV |
|-------|----------|---------|-----------|
| **Random Forest** | **85.58%** | **94.92%** | 85.27 ± 0.39% |
| XGBoost | 85.24% | 95.02% | 85.88 ± 0.27% |
| MLP | 85.18% | 94.69% | 84.66 ± 0.46% |
| SVM | 82.12% | 93.10% | 82.40 ± 0.19% |

![Model Comparison](assets/images/model-comparison.png)

### Per-Class Performance (Random Forest)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Low Risk | 0.87 | 0.88 | 0.88 |
| Medium Risk | 0.70 | 0.59 | 0.64 |
| High Risk | 0.91 | 0.97 | 0.94 |

![Confusion Matrix](assets/images/confusion-matrix.png)
![Per-Class Metrics](assets/images/per-class-metrics.png)

> See [docs/results.md](docs/results.md) for detailed analysis and visualizations.
> See [MODEL_CARD.md](MODEL_CARD.md) for model documentation.

---

## Deployment

### Local — Next.js Frontend (Recommended)

```bash
cd frontend
npm install
npm run dev
# Frontend: http://localhost:3000
```

### Local — FastAPI REST API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

### Local — Gradio Dashboard

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

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html
```

**62 tests** covering:
- Preprocessing (imputation, encoding, scaling)
- Training (model configs, Optuna, GridSearchCV)
- Evaluation (metrics, error analysis, plots)
- SHAP (helpers, local explanations, plots)
- Inference (validation, prediction, logging)
- API (FastAPI endpoints, request validation)

---

## Documentation

- [Architecture Guide](docs/architecture.md) — System design with Mermaid diagrams
- [ML Methodology](docs/methodology.md) — Pipeline, feature selection, model details
- [Results & Metrics](docs/results.md) — Performance analysis and error patterns
- [Model Card](MODEL_CARD.md) — Model details, performance, and usage

---

## Configuration

All settings are centralized in `src/config.py`:

```python
# Switch tuning strategy
TRAINING.use_optuna = False   # GridSearchCV (default)
TRAINING.use_optuna = True    # Optuna (Bayesian)

# Adjust trials (Optuna only)
TRAINING.optuna_n_trials = 50
```

---

## Future Work

- [x] Optuna-based hyperparameter optimization
- [x] CI/CD pipeline with GitHub Actions
- [x] Error analysis module
- [x] SHAP waterfall plots and human-readable interpretations
- [x] Modern dashboard with analytics and export
- [x] FastAPI REST API alongside Gradio
- [x] Next.js + Tailwind professional frontend with SHAP explainability
- [ ] MLflow experiment tracking
- [ ] Cloud deployment
- [ ] Continuous model monitoring
- [ ] Authentication and multi-user support
- [ ] Expanded dataset validation

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

### Contributors

| Name | Primary Responsibilities |
|------|--------------------------|
| **M. Khizar Akram (Team Lead)** | Project architecture, application development, deployment, integration, significant contributions to data preparation, preprocessing, and machine learning pipeline |
| **Safwan Marwat** | Data collection, exploration, and analysis |
| **Syed Mughees** | Preprocessing pipelines and feature engineering |
| **Ifrahim Yousuf** | Model training, evaluation, and experimentation |

[![Khizar](https://img.shields.io/badge/Khizar-Akram-blue?style=flat-square)](https://github.com/Khizar525)

</div>

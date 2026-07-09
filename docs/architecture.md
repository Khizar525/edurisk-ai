# Architecture

## System Overview

EduRisk AI follows a modular, production-oriented architecture that separates concerns across data, training, inference, and presentation layers.

## Component Diagram

```mermaid
graph TB
    subgraph "Data Layer"
        A[Kaggle API] --> B[Raw CSV]
        B --> C[Data Validation]
    end

    subgraph "Processing Layer"
        C --> D[Preprocessing Pipeline]
        D --> E[Feature Engineering]
        E --> F[Risk Level Target]
    end

    subgraph "Training Layer"
        F --> G[Model Training]
        G --> H[GridSearchCV Tuning]
        H --> I[Cross-Validation]
        I --> J[Model Selection]
    end

    subgraph "Inference Layer"
        J --> K[Best Model Artifact]
        K --> L[Prediction Service]
        L --> M[Prediction Logger]
    end

    subgraph "Presentation Layer"
        L --> N[Gradio Dashboard]
        L --> O[SHAP Explanations]
        N --> P[User Interface]
    end
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/data/` | Data loading, validation, Kaggle API integration |
| `src/preprocessing/` | Cleaning, encoding, scaling, sklearn Pipeline |
| `src/features/` | Risk level target engineering |
| `src/training/` | Model definitions, hyperparameter tuning, orchestration |
| `src/evaluation/` | Metrics, plots, comparison reports |
| `src/explainability/` | SHAP global and local explanations |
| `src/inference/` | Prediction service and logging |
| `src/utils/` | Shared helpers (logging, validation, I/O) |
| `app/` | Gradio web interface |

## Data Flow

1. **Ingestion**: Kaggle API → Raw CSV → Validation
2. **Processing**: Imputation → Label Encoding → Standard Scaling
3. **Training**: Stratified Split → GridSearchCV → Model Selection
4. **Inference**: User Input → Encoding → Scaling → Prediction → SHAP
5. **Logging**: Timestamped CSV of all predictions

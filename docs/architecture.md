# Architecture

## System Overview

EduRisk AI follows a modular, production-oriented architecture that separates concerns across data, training, inference, and presentation layers.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[Kaggle API] --> B[Raw CSV]
        B --> C[Data Validation]
    end

    subgraph "Processing Layer"
        C --> D[Imputation]
        D --> E[Label Encoding]
        E --> F[Risk Engineering]
        F --> G[Standard Scaling]
    end

    subgraph "Training Layer"
        G --> H[Train/Test Split]
        H --> I[Model Training]
        I --> J[Hyperparameter Tuning]
        J --> K[Cross-Validation]
        K --> L[Model Selection]
    end

    subgraph "Inference Layer"
        L --> M[Best Model Artifact]
        M --> N[Prediction Service]
        N --> O[Prediction Logger]
        N --> P[SHAP Explainer]
    end

    subgraph "Presentation Layer"
        N --> Q[Gradio Dashboard]
        P --> R[Waterfall Plots]
        P --> S[Interpretation]
        Q --> T[User Interface]
    end

    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style Q fill:#fff3e0
```

## Data Flow (Correct — No Leakage)

```mermaid
flowchart LR
    A[Load CSV] --> B[Impute]
    B --> C[Label Encode]
    C --> D[Engineer Risk Level]
    D --> E[Train/Test Split]
    E --> F[Fit Scaler on Train]
    F --> G[Transform Train]
    F --> H[Transform Test]
    G --> I[Train Models]
    I --> J[Evaluate on Test]

    style E fill:#ffcdd2
    style F fill:#c8e6c9
```

**Critical:** Scaler is fit on training data only. Test data is transformed using training statistics.

## Module Responsibilities

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| `src/config.py` | Central configuration | Paths, hyperparameters, feature lists |
| `src/data/` | Data loading, validation | `dataset.py` |
| `src/preprocessing/` | Cleaning, encoding, scaling | `pipeline.py` |
| `src/features/` | Risk level target engineering | `engineering.py` |
| `src/training/` | Model training and tuning | `models.py`, `tuner.py`, `trainer.py` |
| `src/evaluation/` | Metrics, plots, error analysis | `metrics.py`, `plots.py`, `reporter.py`, `error_analysis.py` |
| `src/explainability/` | SHAP explanations | `shap_utils.py` |
| `src/inference/` | Prediction service and logging | `predictor.py`, `logger.py` |
| `src/utils/` | Shared helpers | `logging.py`, `validators.py` |
| `app/` | Gradio web interface | `main.py` |

## Prediction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio UI
    participant P as Predictor
    participant S as SHAP Explainer
    participant L as Logger

    U->>G: Enter student profile
    G->>P: predict_risk(inputs)
    P->>P: Validate inputs
    P->>P: Encode categorical features
    P->>P: Scale features
    P->>P: Model predict + predict_proba
    P->>S: shap_values(input_scaled)
    S-->>P: SHAP values
    P->>P: Generate interpretation
    P-->>G: Result dict
    G->>L: log(inputs, result)
    G-->>U: Risk level, gauge, SHAP plot, interpretation
```

## Repository Structure

```
edurisk-ai/
├── app/                    # Gradio application (presentation layer)
│   └── main.py             # App entrypoint, UI layout, prediction function
├── src/                    # Core ML package (business logic)
│   ├── config.py           # Central configuration
│   ├── data/               # Data loading and validation
│   ├── preprocessing/      # Cleaning, encoding, scaling
│   ├── features/           # Feature engineering
│   ├── training/           # Model training and tuning
│   ├── evaluation/         # Metrics, plots, reports, error analysis
│   ├── explainability/     # SHAP utilities
│   ├── inference/          # Prediction service and logging
│   └── utils/              # Shared utilities
├── tests/                  # Unit tests (53 tests)
├── notebooks/              # Jupyter notebooks (exploration)
├── docs/                   # Documentation
├── models/                 # Saved model artifacts (.gitignored)
├── data/                   # Raw and processed data (.gitignored)
└── docker/                 # Containerization
```

## Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **No Data Leakage**: Scaler fits on training data only
3. **Configuration-Driven**: All paths, hyperparameters, and constants in `config.py`
4. **Reproducibility**: Fixed random seeds, deterministic pipelines
5. **Testability**: Each module is independently testable
6. **Graceful Degradation**: SHAP falls back gracefully if computation fails

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Client"
        A[Browser] --> B[Next.js Frontend :3000]
        A --> C[Gradio Dashboard :7860]
        A --> D[Swagger UI :8000/docs]
    end

    subgraph "Backend"
        B --> E[FastAPI REST API :8000]
        C --> F[Gradio Server :7860]
        E --> G[Prediction Service]
        F --> G
    end

    subgraph "ML Engine"
        G --> H[RiskPredictor]
        H --> I[Random Forest Model]
        H --> J[SHAP TreeExplainer]
        H --> K[Preprocessing Pipeline]
    end

    subgraph "Storage"
        I --> L[models/risk_model.pkl]
        K --> M[models/scaler.pkl]
        J --> N[models/shap_explainer.pkl]
        H --> O[predictions/predictions.csv]
    end

    subgraph "Deployment"
        P[Docker] --> Q[Container :7860]
        R[GitHub Actions] --> S[CI/CD Pipeline]
        S --> T[Tests + Lint + Build]
    end

    style A fill:#e3f2fd
    style G fill:#fff3e0
    style P fill:#e8f5e9
```

### Ports

| Service | Port | URL |
|---------|------|-----|
| Next.js Frontend | 3000 | http://localhost:3000 |
| FastAPI REST API | 8000 | http://localhost:8000 |
| Swagger Docs | 8000 | http://localhost:8000/docs |
| Gradio Dashboard | 7860 | http://localhost:7860 |

---

## Inference Flow (Request Lifecycle)

```mermaid
flowchart TD
    A[Client Request] --> B{Request Type}
    
    B -->|POST /predict| C[FastAPI Router]
    B -->|Gradio Input| D[Gradio Handler]
    
    C --> E[Pydantic Validation]
    D --> E
    
    E --> F[Input Preprocessing]
    F --> G[Categorical Encoding]
    G --> H[Feature Scaling]
    
    H --> I[RiskPredictor.predict]
    I --> J[Model predict_proba]
    J --> K[Argmax → Risk Level]
    
    I --> L[SHAP TreeExplainer]
    L --> M[Compute SHAP Values]
    M --> N[Sort by Magnitude]
    N --> O[Top Risk + Protective Factors]
    
    K --> P[Build Response]
    O --> P
    
    P --> Q[Prediction Logger]
    Q --> R[Append to CSV]
    
    P --> S[Return JSON]
    
    style A fill:#e3f2fd
    style I fill:#fff3e0
    style L fill:#fce4ec
    style S fill:#e8f5e9
```

### Response Structure

```json
{
  "prediction": 2,
  "risk_level": "High Risk",
  "confidence": "94.9%",
  "probabilities_raw": {"0": 0.0, "1": 0.05, "2": 0.95},
  "shap": {
    "sorted_features": ["Suicidal Thoughts", "Financial Stress", ...],
    "shap_values": [0.332, 0.117, ...],
    "top_risk": [...],
    "top_protective": [...]
  }
}
```

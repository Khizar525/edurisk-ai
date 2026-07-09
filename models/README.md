# Models

This directory contains trained model artifacts (`.pkl`, `.joblib`).

These files are **git-ignored** due to size. To generate them:

```bash
# Train all models and save best
python -m src.training.trainer
```

### Artifacts

| File | Description |
|------|-------------|
| `best_model.pkl` | Best-performing trained model |
| `scaler.pkl` | Fitted StandardScaler |
| `encoders.pkl` | Fitted LabelEncoders |
| `feature_names.pkl` | Ordered feature name list |

### Retraining

Models are retrained with the same random seed (`42`) for reproducibility.
Hyperparameters are defined in `src/config.py`.

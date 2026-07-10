"""Generate all evaluation figures for the EduRisk AI repository."""

import sys
sys.path.insert(0, r"D:\Semester Projects\edurisk-ai")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PATHS, TRAINING
from src.data.dataset import load_dataset
from src.preprocessing.pipeline import (
    impute_missing, label_encode_categoricals, scale_features_train_test
)
from src.features.engineering import engineer_risk_target, get_feature_target_split
from src.training.models import get_model_configs
from src.training.tuner import tune_grid_search
from src.evaluation.plots import (
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    plot_learning_curves, plot_calibration_curves, plot_model_comparison,
    plot_model_radar, plot_class_distribution,
)
from src.evaluation.error_analysis import analyze_errors
from src.explainability.shap_utils import compute_shap_values, plot_shap_summary_bar, plot_shap_beeswarm

OUT = Path(r"D:\Semester Projects\edurisk-ai\assets\results")
OUT.mkdir(parents=True, exist_ok=True)

# ── Step 1: Prepare data ──────────────────────────────────────
print("Loading and preparing data...")
raw_df = load_dataset()
df = impute_missing(raw_df)
df, encoders = label_encode_categoricals(df, fit=True)
df = engineer_risk_target(df, encoders=encoders, drop_depression=True)

X_df, y, feature_names = get_feature_target_split(df)

from sklearn.model_selection import train_test_split
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=TRAINING.test_size, random_state=TRAINING.random_state, stratify=y
)
X_train_raw = X_train_df.values.astype(np.float64)
X_test_raw = X_test_df.values.astype(np.float64)
X_train_scaled, X_test_scaled, scaler = scale_features_train_test(X_train_raw, X_test_raw)

# ── Step 2: Train models ─────────────────────────────────────
print("Training models...")
model_configs = get_model_configs()
models = {}

for name, config in model_configs.items():
    print(f"  Training {name}...")
    if name in ["Random Forest", "XGBoost"]:
        best_model, best_params, _ = tune_grid_search(
            model=config["model"], param_grid=config["param_grid"],
            X_train=X_train_scaled, y_train=y_train, model_name=name,
        )
    else:
        best_model = config["model"]
        best_model.fit(X_train_scaled, y_train)
        best_params = {}
    models[name] = {"model": best_model, "params": best_params}

# ── Step 3: Generate figures ──────────────────────────────────
print("\nGenerating figures...")

# Class distribution
plot_class_distribution(y, save_path=OUT / "class_distribution.png")
print("  [1/9] Class distribution")

# Confusion matrices
for name, info in models.items():
    y_pred = info["model"].predict(X_test_scaled)
    safe_name = name.replace(" ", "_")
    plot_confusion_matrix(y_test, y_pred, name, save_path=OUT / f"cm_{safe_name}.png")
print("  [2/9] Confusion matrices")

# ROC curves
plot_roc_curves(models, X_test_scaled, y_test, save_path=OUT / "roc_curves.png")
print("  [3/9] ROC curves")

# PR curves
plot_precision_recall_curves(models, X_test_scaled, y_test, save_path=OUT / "pr_curves.png")
print("  [4/9] PR curves")

# Calibration curves
plot_calibration_curves(models, X_test_scaled, y_test, save_path=OUT / "calibration_curves.png")
print("  [5/9] Calibration curves")

# Learning curves (use subset for speed)
from sklearn.model_selection import learning_curve as sk_learning_curve
print("  [6/9] Learning curves (this may take a minute)...")
fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
fig.suptitle("Learning Curves - Bias/Variance Diagnosis", fontsize=14, y=1.02)
for ax, (name, info) in zip(axes, models.items()):
    model = info["model"]
    train_sizes, train_scores, val_scores = sk_learning_curve(
        model, X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8), cv=3,
        scoring="accuracy", n_jobs=-1, random_state=42,
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#3498db")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="#e74c3c")
    ax.plot(train_sizes, train_mean, "o-", color="#3498db", lw=2, label="Training")
    ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", lw=2, label="Validation")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Training Samples")
    ax.set_xlim([train_sizes[0], train_sizes[-1]])
    ax.set_ylim([0.5, 1.02])
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig(OUT / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Model comparison bar chart
from src.evaluation.reporter import evaluate_model, compare_models
results = {}
for name, info in models.items():
    metrics = evaluate_model(info["model"], X_test_scaled, y_test, name)
    results[name] = metrics
comparison_df = compare_models(results)
plot_model_comparison(comparison_df, save_path=OUT / "model_comparison.png")
print("  [7/9] Model comparison")

# Radar chart
plot_model_radar(comparison_df, save_path=OUT / "model_radar.png")
print("  [8/9] Radar chart")

# SHAP (Random Forest - tree-based, fast)
print("  [9/9] SHAP analysis (Random Forest)...")
rf_model = models["Random Forest"]["model"]
shap_values, explainer, X_test_sub = compute_shap_values(
    rf_model, X_test_scaled, X_train_scaled, feature_names,
    "Random Forest", n_test_samples=500, n_background_samples=100,
)
plot_shap_summary_bar(shap_values, X_test_sub, feature_names, "Random Forest",
                       class_idx=2, save_path=OUT / "shap_bar_rf.png")
plot_shap_beeswarm(shap_values, X_test_sub, feature_names, "Random Forest",
                    class_idx=2, save_path=OUT / "shap_beeswarm_rf.png")

print(f"\nAll figures saved to: {OUT}")
print(f"Files: {[f.name for f in OUT.iterdir()]}")

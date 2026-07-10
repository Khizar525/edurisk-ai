"""
Generate publication-quality charts from actual trained model data.
No fake data. No AI-generated charts. Real results only.
"""

import sys
sys.path.insert(0, r"D:\Semester Projects\edurisk-ai")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from src.config import PATHS, TRAINING
from src.data.dataset import load_dataset
from src.preprocessing.pipeline import (
    impute_missing, label_encode_categoricals, scale_features_train_test
)
from src.features.engineering import engineer_risk_target, get_feature_target_split
from src.training.models import get_model_configs
from src.training.tuner import tune_grid_search
from src.evaluation.metrics import compute_metrics
from src.explainability.shap_utils import compute_shap_values, _extract_class_shap

OUT = Path(r"D:\Semester Projects\edurisk-ai\assets\images")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style Config ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#e0e0e0",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
})

COLORS = {
    "low": "#22c55e",
    "medium": "#f59e0b",
    "high": "#ef4444",
    "primary": "#14b8a6",
    "secondary": "#38bdf8",
    "accent": "#f97316",
    "bg": "#ffffff",
    "text": "#1f2937",
}


# ── Prepare Data ───────────────────────────────────────────────
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

# ── Train Models ───────────────────────────────────────────────
print("Training models...")
model_configs = get_model_configs()
models = {}
for name, config in model_configs.items():
    print(f"  {name}...")
    if name in ["Random Forest", "XGBoost"]:
        best_model, _, _ = tune_grid_search(
            model=config["model"], param_grid=config["param_grid"],
            X_train=X_train_scaled, y_train=y_train, model_name=name,
        )
    else:
        best_model = config["model"]
        best_model.fit(X_train_scaled, y_train)
    models[name] = best_model


# ═══════════════════════════════════════════════════════════════
# CHART 1: Model Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════
print("\n[1/6] Model Comparison...")

model_names = list(models.keys())
accuracies = []
roc_aucs = []

for name in model_names:
    model = models[name]
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    metrics = compute_metrics(y_test.values, y_pred, y_prob, name)
    accuracies.append(metrics["Accuracy"])
    roc_aucs.append(metrics.get("ROC-AUC", 0))

# Sort by accuracy
idx = np.argsort(accuracies)[::-1]
model_names = [model_names[i] for i in idx]
accuracies = [accuracies[i] for i in idx]
roc_aucs = [roc_aucs[i] for i in idx]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy",
               color=COLORS["primary"], edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, roc_aucs, width, label="ROC-AUC (OvR)",
               color=COLORS["secondary"], edgecolor="white", linewidth=0.5)

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylabel("Score")
ax.set_ylim(0.75, 1.0)
ax.set_title("Model Comparison — Accuracy vs ROC-AUC")
ax.legend(loc="lower right", framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

plt.tight_layout()
plt.savefig(OUT / "model-comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved model-comparison.png")


# ═══════════════════════════════════════════════════════════════
# CHART 2: Confusion Matrix (Best Model)
# ═══════════════════════════════════════════════════════════════
print("[2/6] Confusion Matrix...")

best_name = model_names[0]
best_model = models[best_name]
y_pred = best_model.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
classes = ["Low Risk", "Medium Risk", "High Risk"]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax, shrink=0.8)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel="True Label",
    xlabel="Predicted Label",
    title=f"Confusion Matrix — {best_name}",
)

plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
plt.setp(ax.get_yticklabels(), fontsize=10)

# Annotate cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold")

# Row/column totals
for i in range(cm.shape[0]):
    ax.text(cm.shape[1] + 0.1, i, f"n={cm[i].sum():,}",
            ha="left", va="center", fontsize=9, color="gray")
for j in range(cm.shape[1]):
    ax.text(j, -0.15, f"n={cm[:, j].sum():,}",
            ha="center", va="top", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(OUT / "confusion-matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved confusion-matrix.png")


# ═══════════════════════════════════════════════════════════════
# CHART 3: Class Distribution
# ═══════════════════════════════════════════════════════════════
print("[3/6] Class Distribution...")

from collections import Counter
counts = Counter(y)
labels = ["Low Risk", "Medium Risk", "High Risk"]
colors = [COLORS["low"], COLORS["medium"], COLORS["high"]]
values = [counts[i] for i in range(3)]
total = sum(values)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55, linewidth=1.5)

for bar, count in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.01,
        f"{count:,}\n({count/total*100:.1f}%)",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )

ax.set_ylabel("Number of Students", fontsize=12)
ax.set_title("Target Class Distribution — Risk Level")
ax.set_ylim(0, max(values) * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "class-distribution.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved class-distribution.png")


# ═══════════════════════════════════════════════════════════════
# CHART 4: Risk Level Breakdown (Donut)
# ═══════════════════════════════════════════════════════════════
print("[4/6] Risk Breakdown Donut...")

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))

wedges, texts, autotexts = ax.pie(
    values,
    labels=labels,
    colors=colors,
    autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100)):,})",
    startangle=90,
    pctdistance=0.72,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=3),
    textprops={"fontsize": 12},
)

for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight("bold")

ax.text(0, 0.05, f"{total:,}", ha="center", va="center", fontsize=22, fontweight="bold", color=COLORS["text"])
ax.text(0, -0.12, "students", ha="center", va="center", fontsize=12, color="gray")

ax.set_title("Risk Level Distribution", fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig(OUT / "risk-breakdown.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved risk-breakdown.png")


# ═══════════════════════════════════════════════════════════════
# CHART 5: SHAP Feature Importance
# ═══════════════════════════════════════════════════════════════
print("[5/6] SHAP Feature Importance...")

import shap
rf_model = models["Random Forest"]
explainer = shap.TreeExplainer(rf_model)
X_test_sub = X_test_scaled[:500]
shap_values = explainer.shap_values(X_test_sub)

# High Risk class (class 2)
shap_vals = _extract_class_shap(shap_values, 2)
mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

# Sort
sort_idx = np.argsort(mean_abs_shap)
sorted_feats = [feature_names[i] for i in sort_idx]
sorted_vals = mean_abs_shap[sort_idx]

# Color gradient
n = len(sorted_feats)
gradient_colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n))

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(range(n), sorted_vals, color=gradient_colors, edgecolor="white", height=0.7)

ax.set_yticks(range(n))
ax.set_yticklabels(sorted_feats, fontsize=10)
ax.set_xlabel("Mean |SHAP Value| (High Risk Class)")
ax.set_title("SHAP Feature Importance — Random Forest")

# Value labels
for i, (bar, val) in enumerate(zip(bars, sorted_vals)):
    ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9, color="gray")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "shap-importance.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved shap-importance.png")


# ═══════════════════════════════════════════════════════════════
# CHART 6: Per-Class F1 Scores
# ═══════════════════════════════════════════════════════════════
print("[6/6] Per-Class Metrics...")

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
metrics_to_plot = ["precision", "recall", "f1-score"]
metric_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

for ax, metric, color in zip(axes, metrics_to_plot, metric_colors):
    values = [report[cls][metric] for cls in classes]
    bars = ax.bar(classes, values, color=color, edgecolor="white", width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title(metric.capitalize(), fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Score")
fig.suptitle(f"Per-Class Metrics — {best_name}", fontsize=14, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig(OUT / "per-class-metrics.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved per-class-metrics.png")


# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"All charts saved to: {OUT}")
print(f"Files: {[f.name for f in sorted(OUT.iterdir()) if f.suffix == '.png']}")
print(f"{'='*50}")

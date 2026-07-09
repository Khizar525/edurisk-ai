"""
Evaluation visualization — confusion matrices, ROC curves, comparison charts.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import PATHS


def plot_confusion_matrix(
    y_true, y_pred, model_name: str, save_path: Optional[Path] = None
) -> None:
    """Plot and optionally save a confusion matrix."""
    labels = ["Low", "Medium", "High"]
    cm = np.array([
        [2190, 37, 116],
        [50, 314, 11],
        [158, 3, 721],
    ])  # Placeholder — real CM computed at runtime

    # Actually compute from data
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    models: dict, X_test, y_test, save_path: Optional[Path] = None
) -> None:
    """Plot ROC curves for all models (One-vs-Rest)."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    class_names = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    fig.suptitle("ROC Curves — One-vs-Rest (All Models)", fontsize=14)

    for ax, (name, info) in zip(axes, models.items()):
        model = info["model"]
        y_score = model.predict_proba(X_test)

        for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, cls_idx], y_score[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("False Positive Rate")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.legend(loc="lower right", fontsize=7)

    axes[0].set_ylabel("True Positive Rate")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison_df, save_path: Optional[Path] = None) -> None:
    """Plot bar chart comparing models across metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_df[["Accuracy", "ROC-AUC", "CV Mean"]].plot(
        kind="bar", ax=ax, colormap="Set2", edgecolor="white"
    )
    ax.set_title("Model Comparison — Accuracy, ROC-AUC, 5-Fold CV Mean", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(y, save_path: Optional[Path] = None) -> None:
    """Plot target class distribution."""
    from collections import Counter

    counts = Counter(y)
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    values = [counts.get(i, 0) for i in range(3)]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

    for bar, count in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count:,}\n({count/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Class Distribution — Risk Level (Target Variable)", fontsize=13)
    ax.set_ylabel("Number of Students")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

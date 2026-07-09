"""
SHAP explainability utilities — global and local explanations.
"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import shap

from src.config import PATHS, RISK_LABELS
from src.training.models import get_shap_compatible_model


def compute_shap_values(
    model,
    X_test: np.ndarray,
    X_background: np.ndarray,
    feature_names: List[str],
    model_name: str,
    n_test_samples: int = 500,
    n_background_samples: int = 100,
) -> tuple:
    """
    Compute SHAP values for a model.

    Returns:
        Tuple of (shap_values, explainer).
    """
    # Subsample for efficiency
    X_test_sub = X_test[:n_test_samples]
    background = shap.sample(X_background, min(n_background_samples, len(X_background)))

    if get_shap_compatible_model(model_name):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sub)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_sub[:100])
        X_test_sub = X_test_sub[:100]

    return shap_values, explainer, X_test_sub


def plot_shap_summary_bar(
    shap_values,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    class_idx: int = 2,
    save_path: Optional[Path] = None,
) -> None:
    """Plot SHAP summary bar chart for a specific class."""
    shap_vals = _extract_class_shap(shap_values, class_idx)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_vals,
        features=X_test[: shap_vals.shape[0]],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        color="#e74c3c",
    )
    plt.title(f"SHAP Feature Importance — {model_name} ({RISK_LABELS[class_idx]})", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_beeswarm(
    shap_values,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    class_idx: int = 2,
    save_path: Optional[Path] = None,
) -> None:
    """Plot SHAP beeswarm chart for a specific class."""
    shap_vals = _extract_class_shap(shap_values, class_idx)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_vals,
        features=X_test[: shap_vals.shape[0]],
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Beeswarm — {model_name} ({RISK_LABELS[class_idx]})", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_local_explanation(
    explainer,
    input_scaled: np.ndarray,
    feature_names: List[str],
    prediction: int,
) -> tuple:
    """
    Compute SHAP values for a single prediction.

    Returns:
        Tuple of (sorted_feature_names, shap_values, bar_colors).
    """
    shap_values = explainer.shap_values(input_scaled)

    sv_row = _extract_single_row_shap(shap_values, prediction)

    sorted_idx = np.argsort(np.abs(sv_row))
    sorted_vals = sv_row[sorted_idx]
    sorted_feats = [feature_names[i] for i in sorted_idx]
    bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in sorted_vals]

    return sorted_feats, sorted_vals, bar_colors


def _extract_class_shap(shap_values, class_idx: int) -> np.ndarray:
    """Extract SHAP values for a specific class from multi-class output."""
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    elif len(shap_values.shape) == 3:
        return shap_values[:, :, class_idx]
    else:
        return shap_values


def _extract_single_row_shap(shap_values, class_idx: int) -> np.ndarray:
    """Extract SHAP values for a single sample and class."""
    if isinstance(shap_values, list):
        return shap_values[class_idx][0]
    elif len(shap_values.shape) == 3:
        return shap_values[0, :, class_idx]
    else:
        return shap_values[0]

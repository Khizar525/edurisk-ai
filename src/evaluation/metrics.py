"""
Custom metrics and metric computation utilities.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None, model_name: str = "model") -> Dict:
    """
    Compute all evaluation metrics for a model.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional, for ROC-AUC).
        model_name: Name for logging.

    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
    }

    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = round(
                roc_auc_score(y_true, y_prob, multi_class="ovr"), 4
            )
        except ValueError:
            metrics["ROC-AUC"] = 0.0

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=["Low", "Medium", "High"],
        output_dict=True,
    )
    metrics["report"] = report

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics

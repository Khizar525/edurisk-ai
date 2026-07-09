"""
Evaluation reporting — metrics computation and comparison tables.
"""

from typing import Dict

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.evaluation.metrics import compute_metrics


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict:
    """
    Evaluate a single model and return metrics.

    Returns:
        Dictionary with Accuracy, ROC-AUC, and classification report.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob, model_name)

    print(f"\n  {model_name}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}  |  ROC-AUC: {metrics.get('ROC-AUC', 0):.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Low", "Medium", "High"],
        zero_division=0,
    ))

    return metrics


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from model results.

    Returns:
        DataFrame sorted by Accuracy (descending).
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Accuracy": metrics.get("Accuracy", 0),
            "ROC-AUC": metrics.get("ROC-AUC", 0),
            "CV Mean": metrics.get("CV Mean", 0),
            "CV Std": metrics.get("CV Std", 0),
        })

    df = pd.DataFrame(rows).set_index("Model")
    df = df.sort_values("Accuracy", ascending=False)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY (sorted by Accuracy)")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)

    return df

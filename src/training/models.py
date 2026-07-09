"""
Model definitions and factory methods.

All four classifiers used in the project.
"""

from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.config import TRAINING


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Return model configurations.

    Returns:
        Dictionary mapping model name to {model_class, param_grid, needs_grid}.
    """
    return {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=TRAINING.random_state),
            "param_grid": TRAINING.rf_param_grid,
            "needs_grid": True,
        },
        "SVM": {
            "model": SVC(
                kernel=TRAINING.svm_kernel,
                probability=TRAINING.svm_probability,
                random_state=TRAINING.random_state,
            ),
            "param_grid": {},
            "needs_grid": False,
        },
        "XGBoost": {
            "model": XGBClassifier(
                eval_metric="mlogloss",
                random_state=TRAINING.random_state,
            ),
            "param_grid": TRAINING.xgb_param_grid,
            "needs_grid": True,
        },
        "MLP": {
            "model": MLPClassifier(
                hidden_layer_sizes=TRAINING.mlp_hidden_layers,
                max_iter=TRAINING.mlp_max_iter,
                early_stopping=TRAINING.mlp_early_stopping,
                random_state=TRAINING.random_state,
            ),
            "param_grid": {},
            "needs_grid": False,
        },
    }


def get_shap_compatible_model(model_class: str) -> bool:
    """Check if a model is tree-based (compatible with TreeExplainer)."""
    return model_class in ["Random Forest", "XGBoost"]

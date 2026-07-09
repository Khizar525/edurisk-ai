"""
Hyperparameter tuning using GridSearchCV.
"""

from typing import Dict, Any, Tuple

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator

from src.config import TRAINING


def tune_model(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    model_name: str = "model",
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Tune a model using GridSearchCV.

    Args:
        model: Unfitted model instance.
        param_grid: Parameter grid for search.
        X_train: Training features.
        y_train: Training labels.
        model_name: Name for logging.

    Returns:
        Tuple of (best_model, best_params).
    """
    if not param_grid:
        # No tuning needed, just fit
        model.fit(X_train, y_train)
        print(f"  {model_name}: Fitted (no tuning)")
        return model, {}

    print(f"  {model_name}: Running GridSearchCV...")

    cv = StratifiedKFold(
        n_splits=min(TRAINING.cv_folds, min(
            len(y_train), len(set(y_train))
        )),
        shuffle=True,
        random_state=TRAINING.random_state,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=TRAINING.scoring,
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    print(f"  {model_name}: Best params = {grid_search.best_params_}")
    print(f"  {model_name}: Best CV score = {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

"""
Preprocessing pipeline — imputation, encoding, and scaling.

Data leakage is prevented by:
1. Fitting LabelEncoders on full data (deterministic, safe for risk engineering)
2. Fitting StandardScaler on TRAINING data only (after split)
"""

import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    PATHS, CATEGORICAL_COLS, NUMERICAL_COLS, COLS_TO_KEEP
)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values: median for numerical, mode for categorical.

    This is safe to run on the full dataset before splitting because
    imputation uses robust statistics (median/mode) and doesn't create
    information leakage for the downstream model training.
    """
    df = df.copy()
    for col in NUMERICAL_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_vals = df[col].mode()
            fill_val = mode_vals[0] if not mode_vals.empty else ""
            df[col] = df[col].fillna(fill_val)
    return df


def label_encode_categoricals(
    df: pd.DataFrame, fit: bool = True, encoders: Optional[dict] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Label encode categorical columns.

    Label encoding is a deterministic mapping (e.g., Male→0, Female→1).
    Fitting on full data is safe — it doesn't leak target information.
    This is needed before risk engineering (which checks encoded Sleep Duration).

    Args:
        df: Input DataFrame.
        fit: If True, fit new encoders. If False, use provided encoders.
        encoders: Pre-fitted encoders (required when fit=False).

    Returns:
        Tuple of (encoded DataFrame, encoder dict).
    """
    if fit and encoders is not None:
        raise ValueError("Cannot fit and provide encoders simultaneously")
    if not fit and encoders is None:
        raise ValueError("Must provide encoders when fit=False")

    result_encoders = encoders if not fit else {}
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            result_encoders[col] = le
        else:
            # Handle unseen categories gracefully
            known_classes = set(encoders[col].classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else encoders[col].classes_[0]
            )
            df[col] = encoders[col].transform(df[col])

    return df, result_encoders


def scale_features_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data only, transform both splits.

    This is the CORRECT way to scale — prevents data leakage.

    Args:
        X_train: Training features (n_train, n_features).
        X_test: Test features (n_test, n_features).

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train
    X_test_scaled = scaler.transform(X_test)        # only transform on test
    return X_train_scaled, X_test_scaled, scaler


def prepare_features_inference(
    X_raw: np.ndarray,
    encoders: dict,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Prepare features for inference using fitted artifacts.

    Args:
        X_raw: Raw feature array (1, n_features).
        encoders: Fitted LabelEncoders.
        scaler: Fitted StandardScaler.

    Returns:
        Scaled feature array ready for prediction.
    """
    return scaler.transform(X_raw)


def save_artifacts(
    model=None,
    scaler: StandardScaler = None,
    encoders: dict = None,
    feature_names: List[str] = None,
) -> None:
    """Save preprocessing artifacts to disk."""
    PATHS.models.mkdir(parents=True, exist_ok=True)

    if model is not None:
        with open(PATHS.best_model, "wb") as f:
            pickle.dump(model, f)

    if scaler is not None:
        with open(PATHS.scaler, "wb") as f:
            pickle.dump(scaler, f)

    if encoders is not None:
        with open(PATHS.encoders, "wb") as f:
            pickle.dump(encoders, f)

    if feature_names is not None:
        with open(PATHS.feature_names, "wb") as f:
            pickle.dump(feature_names, f)

    print(f"Artifacts saved to: {PATHS.models}")


def load_artifacts() -> dict:
    """Load all preprocessing artifacts from disk."""
    artifacts = {}

    for name, path in [
        ("model", PATHS.best_model),
        ("scaler", PATHS.scaler),
        ("encoders", PATHS.encoders),
        ("feature_names", PATHS.feature_names),
    ]:
        if path.exists():
            with open(path, "rb") as f:
                artifacts[name] = pickle.load(f)
        else:
            print(f"Warning: Artifact not found: {path}")

    return artifacts

"""
Preprocessing pipeline using sklearn Pipeline + ColumnTransformer.

Prevents data leakage by fitting only on training data.
"""

import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import (
    PATHS, CATEGORICAL_COLS, NUMERICAL_COLS, COLS_TO_KEEP
)


def build_preprocessing_pipeline() -> Tuple[ColumnTransformer, List[str]]:
    """
    Build a sklearn ColumnTransformer for preprocessing.

    Returns:
        Tuple of (preprocessor, feature_names_after_transform).
    """
    # Numerical pipeline: impute median → scale
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: impute mode → label encode (handled separately)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    feature_names = NUMERICAL_COLS + CATEGORICAL_COLS
    return preprocessor, feature_names


def label_encode_categoricals(
    df: pd.DataFrame, fit: bool = True, encoders: Optional[dict] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Label encode categorical columns.

    Args:
        df: Input DataFrame (modified in-place).
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
            df[col] = encoders[col].transform(df[col].astype(str))

    return df, result_encoders


def prepare_features(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: Optional[dict] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, StandardScaler]:
    """
    Full preprocessing pipeline: impute → encode → scale.

    Args:
        df: Raw DataFrame with COLS_TO_KEEP columns.
        fit: Whether to fit transformers.
        encoders: Pre-fitted LabelEncoders (required when fit=False).
        scaler: Pre-fitted StandardScaler (required when fit=False).

    Returns:
        Tuple of (X_scaled, y, encoders, scaler).
    """
    df = df.copy()

    # Impute missing values
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")

    # Label encode categoricals
    if fit:
        df, encoders = label_encode_categoricals(df, fit=True)
    else:
        df, _ = label_encode_categoricals(df, fit=False, encoders=encoders)

    # Separate features and target
    feature_cols = [c for c in df.columns if c != "Depression"]
    X = df[feature_cols].values
    y = df["Depression"].values if "Depression" in df.columns else None

    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, encoders, scaler


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

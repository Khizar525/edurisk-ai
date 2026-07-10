"""
Training orchestration — trains all models and selects the best.

Correct data flow (no leakage):
1. Load raw data
2. Impute missing values (full data — safe for risk engineering)
3. Label encode categoricals (full data — deterministic, needed for risk engineering)
4. Engineer Risk_Level target (uses Depression + encoded features)
5. Split into train/test (BEFORE scaling)
6. Fit StandardScaler on TRAINING data only
7. Transform both splits
8. Train models on scaled training data
9. Evaluate on scaled test data
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from src.config import PATHS, TRAINING, RISK_LABELS, COLS_TO_KEEP
from src.data.dataset import load_dataset
from src.preprocessing.pipeline import (
    impute_missing,
    label_encode_categoricals,
    scale_features_train_test,
    save_artifacts,
)
from src.features.engineering import engineer_risk_target, get_feature_target_split
from src.training.models import get_model_configs
from src.training.tuner import tune_model
from src.evaluation.reporter import evaluate_model, compare_models


def prepare_data() -> Tuple[pd.DataFrame, dict]:
    """
    Load and prepare data for training.

    Steps:
    1. Load raw CSV
    2. Select relevant columns
    3. Impute missing values (full data)
    4. Label encode categoricals (full data — needed for risk engineering)
    5. Engineer Risk_Level target
    6. Drop Depression column (replaced by Risk_Level)

    Returns:
        Tuple of (DataFrame with Risk_Level, fitted encoders).
    """
    print("Loading dataset...")
    raw_df = load_dataset()

    # Impute missing values (safe — uses median/mode)
    print("Imputing missing values...")
    df = impute_missing(raw_df)

    # Label encode categoricals (safe — deterministic mapping)
    # Needed before risk engineering because assign_risk_level checks
    # the encoded value of Sleep Duration
    print("Encoding categorical features...")
    df, encoders = label_encode_categoricals(df, fit=True)

    # Engineer Risk_Level target from Depression + other features
    print("Engineering Risk_Level target...")
    df = engineer_risk_target(df, encoders=encoders, drop_depression=True)

    return df, encoders


def train_all(
    df: pd.DataFrame = None,
    save: bool = True,
) -> Tuple[Dict[str, object], pd.DataFrame, dict]:
    """
    Train all four models, tune hyperparameters, evaluate, and select best.

    Args:
        df: Preprocessed DataFrame with Risk_Level column. If None, loads fresh.
        save: Whether to save the best model and artifacts.

    Returns:
        Tuple of (models_dict, comparison_df, best_model_info).
    """
    # ── Step 1: Prepare Data ───────────────────────────────────
    if df is None:
        df, encoders = prepare_data()
    else:
        encoders = {}

    # Get feature matrix and target vector
    X_df, y, feature_names = get_feature_target_split(df)
    print(f"\nFeatures: {feature_names}")
    print(f"Dataset shape: {X_df.shape}")

    # ── Step 2: Train/Test Split (BEFORE scaling) ─────────────
    print("\nSplitting data (80/20 stratified)...")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size=TRAINING.test_size,
        random_state=TRAINING.random_state,
        stratify=y,
    )

    X_train_raw = X_train_df.values.astype(np.float64)
    X_test_raw = X_test_df.values.astype(np.float64)

    print(f"  Train: {X_train_raw.shape[0]} samples")
    print(f"  Test:  {X_test_raw.shape[0]} samples")

    # ── Step 3: Scale Features (fit on TRAIN only) ────────────
    print("\nScaling features (fit on train only)...")
    X_train_scaled, X_test_scaled, scaler = scale_features_train_test(
        X_train_raw, X_test_raw
    )

    # Full scaled dataset for cross-validation
    X_all_scaled = np.vstack([X_train_scaled, X_test_scaled])
    y_all = np.concatenate([y_train, y_test])

    # ── Step 4: Train Models ───────────────────────────────────
    print("\nTraining models...")
    model_configs = get_model_configs()
    models = {}
    cv = StratifiedKFold(n_splits=TRAINING.cv_folds, shuffle=True, random_state=42)

    for name, config in model_configs.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        best_model, best_params = tune_model(
            model=config["model"],
            param_grid=config["param_grid"],
            X_train=X_train_scaled,
            y_train=y_train,
            model_name=name,
        )

        # Cross-validation on full scaled data
        cv_scores = cross_val_score(
            best_model, X_all_scaled, y_all, cv=cv, scoring=TRAINING.scoring
        )
        print(f"  {name}: 5-Fold CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        models[name] = {
            "model": best_model,
            "params": best_params,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    # ── Step 5: Evaluate All ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    results = {}
    for name, info in models.items():
        metrics = evaluate_model(info["model"], X_test_scaled, y_test, name)
        results[name] = {**metrics, "CV Mean": info["cv_mean"], "CV Std": info["cv_std"]}

    comparison_df = compare_models(results)

    # ── Step 6: Select Best ────────────────────────────────────
    best_name = comparison_df["Accuracy"].idxmax()
    best_model = models[best_name]["model"]

    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best_name}")
    print(f"  Accuracy: {comparison_df.loc[best_name, 'Accuracy']:.4f}")
    print(f"  ROC-AUC:  {comparison_df.loc[best_name, 'ROC-AUC']:.4f}")
    print(f"  5-Fold CV: {comparison_df.loc[best_name, 'CV Mean']:.4f} "
          f"± {comparison_df.loc[best_name, 'CV Std']:.4f}")
    print(f"{'='*60}")

    # ── Step 7: Save Artifacts ─────────────────────────────────
    if save:
        save_artifacts(
            model=best_model,
            scaler=scaler,
            encoders=encoders,
            feature_names=feature_names,
        )

    best_info = {
        "name": best_name,
        "accuracy": comparison_df.loc[best_name, "Accuracy"],
        "roc_auc": comparison_df.loc[best_name, "ROC-AUC"],
    }

    return models, comparison_df, best_info


if __name__ == "__main__":
    models, comparison, best = train_all()
    print("\n" + comparison.to_string())

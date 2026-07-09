"""
Training orchestration — trains all models and selects the best.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from src.config import PATHS, TRAINING, RISK_LABELS
from src.data.dataset import load_dataset
from src.preprocessing.pipeline import (
    prepare_features, save_artifacts, label_encode_categoricals
)
from src.features.engineering import engineer_risk_target, get_feature_target_split
from src.training.models import get_model_configs
from src.training.tuner import tune_model
from src.evaluation.reporter import evaluate_model, compare_models


def train_all(
    df: pd.DataFrame = None,
    save: bool = True,
) -> Tuple[Dict[str, object], pd.DataFrame, dict]:
    """
    Train all four models, tune hyperparameters, evaluate, and select best.

    Args:
        df: Preprocessed DataFrame with Risk_Level. If None, loads and processes.
        save: Whether to save the best model and artifacts.

    Returns:
        Tuple of (models_dict, comparison_df, best_model_info).
    """
    # ── Data Preparation ───────────────────────────────────────
    if df is None:
        print("Loading and preparing data...")
        raw_df = load_dataset()

        # Impute before engineering (same as original notebook)
        for col in raw_df.select_dtypes(include=np.number).columns:
            raw_df[col] = raw_df[col].fillna(raw_df[col].median())
        for col in raw_df.select_dtypes(include="object").columns:
            raw_df[col] = raw_df[col].fillna(raw_df[col].mode()[0])

        # Encode categoricals for risk engineering
        raw_df, encoders = label_encode_categoricals(raw_df, fit=True)

        # Engineer risk target
        df = engineer_risk_target(raw_df, encoders=encoders, drop_depression=True)

        # Get features
        X_df, y, feature_names = get_feature_target_split(df)

        # Scale features
        X_scaled, _, _, scaler = prepare_features(
            df, fit=True, encoders=encoders
        )
        # Re-build X with Risk_Level as target for prepare_features
        df_for_prep = df.copy()
        df_for_prep["Depression"] = y  # prepare_features expects this column
        X_scaled, y, encoders, scaler = prepare_features(df_for_prep, fit=True)
    else:
        X_df, y, feature_names = get_feature_target_split(df)
        X_scaled = X_df.values
        scaler = None
        encoders = None

    # ── Train/Test Split ───────────────────────────────────────
    print("\nSplitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TRAINING.test_size,
        random_state=TRAINING.random_state,
        stratify=y,
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # ── Train Models ───────────────────────────────────────────
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
            X_train=X_train,
            y_train=y_train,
            model_name=name,
        )

        # Cross-validation
        cv_scores = cross_val_score(
            best_model, X_scaled, y, cv=cv, scoring=TRAINING.scoring
        )
        print(f"  {name}: 5-Fold CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        models[name] = {
            "model": best_model,
            "params": best_params,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    # ── Evaluate All ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    results = {}
    for name, info in models.items():
        metrics = evaluate_model(info["model"], X_test, y_test, name)
        results[name] = {**metrics, "CV Mean": info["cv_mean"], "CV Std": info["cv_std"]}

    comparison_df = compare_models(results)

    # ── Select Best ────────────────────────────────────────────
    best_name = comparison_df["Accuracy"].idxmax()
    best_model = models[best_name]["model"]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Accuracy: {comparison_df.loc[best_name, 'Accuracy']:.4f}")
    print(f"   ROC-AUC:  {comparison_df.loc[best_name, 'ROC-AUC']:.4f}")

    # ── Save Artifacts ─────────────────────────────────────────
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

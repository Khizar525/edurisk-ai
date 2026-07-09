"""
Feature engineering — Risk Level target variable generation.

Preserves the original composite scoring logic from the notebook.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.config import RISK_THRESHOLDS, CATEGORICAL_COLS


def assign_risk_level(row: pd.Series, encoders: Optional[dict] = None) -> int:
    """
    Composite scoring function mapping raw indicators to 3-class risk level.

    Preserved from original notebook (Module 2, Syed Mughees).

    Args:
        row: A single row of the DataFrame.
        encoders: Fitted LabelEncoders (needed for Sleep Duration encoding).

    Returns:
        Risk level: 0 (Low), 1 (Medium), or 2 (High).
    """
    t = RISK_THRESHOLDS
    score = 0

    # Risk factors (positive contributors)
    if row.get("Depression") == 1:
        score += t["depression_weight"]
    if row.get("Academic Pressure", 0) >= 4:
        score += t["academic_pressure_weight"]
    if row.get("CGPA", 4.0) < 2.0:
        score += t["low_cgpa_weight"]
    if row.get("Financial Stress", 0) >= 4:
        score += t["financial_stress_weight"]
    if row.get("Have you ever had suicidal thoughts ?") == 1:
        score += t["suicidal_thoughts_weight"]

    # Sleep Duration check
    if encoders is not None and "Sleep Duration" in encoders:
        low_sleep_val = encoders["Sleep Duration"].transform(["Less than 5 hours"])[0]
        if row.get("Sleep Duration") == low_sleep_val:
            score += t["low_sleep_weight"]

    # Protective factors (negative contributors)
    if row.get("CGPA", 0) >= 3.0:
        score += t["high_cgpa_bonus"]
    if row.get("Study Satisfaction", 0) >= 4:
        score += t["high_satisfaction_bonus"]

    # Threshold mapping
    if score <= t["low_threshold"]:
        return 0   # Low Risk
    elif score <= t["medium_threshold"]:
        return 1   # Medium Risk
    else:
        return 2   # High Risk


def engineer_risk_target(
    df: pd.DataFrame, encoders: Optional[dict] = None, drop_depression: bool = True
) -> pd.DataFrame:
    """
    Add Risk_Level column and optionally drop original Depression column.

    Args:
        df: Input DataFrame.
        encoders: Fitted LabelEncoders.
        drop_depression: Whether to drop the Depression column after engineering.

    Returns:
        DataFrame with Risk_Level column added.
    """
    df = df.copy()
    df["Risk_Level"] = df.apply(lambda row: assign_risk_level(row, encoders), axis=1)

    if drop_depression and "Depression" in df.columns:
        df = df.drop(columns=["Depression"])

    # Print class distribution
    dist = df["Risk_Level"].value_counts().sort_index()
    total = len(df)
    print("Risk Level Distribution:")
    for level, count in dist.items():
        labels = {0: "Low", 1: "Medium", 2: "High"}
        print(f"  {labels[level]}: {count:,} ({count/total*100:.1f}%)")

    return df


def get_feature_target_split(
    df: pd.DataFrame, target_col: str = "Risk_Level"
) -> tuple:
    """
    Split DataFrame into features (X) and target (y).

    Returns:
        Tuple of (X DataFrame, y Series, feature_names list).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, X.columns.tolist()

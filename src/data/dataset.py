"""
Data loading and collection utilities.

Handles Kaggle API download, CSV loading, and initial validation.
"""

import os
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PATHS, KAGGLE_DATASET, COLS_TO_KEEP


def download_dataset(force: bool = False) -> Path:
    """
    Download the Student Depression Dataset from Kaggle.

    Args:
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded CSV file.
    """
    csv_path = PATHS.dataset_csv

    if csv_path.exists() and not force:
        print(f"Dataset already exists at: {csv_path}")
        return csv_path

    # Ensure kaggle.json credentials exist
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_json}. "
            "Download kaggle.json from https://www.kaggle.com/settings and place it there."
        )

    # Download dataset
    print(f"Downloading dataset: {KAGGLE_DATASET}...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -q --force")

    # Extract zip
    zip_files = list(Path(".").glob("*.zip"))
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(PATHS.data_raw)
        zip_file.unlink()

    print(f"Dataset downloaded to: {csv_path}")
    return csv_path


def load_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw dataset and select relevant columns.

    Args:
        csv_path: Path to CSV. Defaults to PATHS.dataset_csv.

    Returns:
        DataFrame with selected columns.

    Raises:
        FileNotFoundError: If CSV file does not exist.
    """
    path = csv_path or PATHS.dataset_csv

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Select columns
    available_cols = [c for c in COLS_TO_KEEP if c in df.columns]
    missing_cols = [c for c in COLS_TO_KEEP if c not in df.columns]

    if missing_cols:
        print(f"Warning: Missing columns in dataset: {missing_cols}")

    df = df[available_cols].copy()
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run basic validation checks on the dataset.

    Returns:
        Dictionary with validation results.
    """
    results = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }

    # Check for empty dataframe
    if results["rows"] == 0:
        raise ValueError("Dataset is empty")

    # Check for excessive missing data
    for col, count in results["missing_values"].items():
        pct = count / results["rows"] * 100
        if pct > 50:
            print(f"Warning: Column '{col}' has {pct:.1f}% missing values")

    return results


if __name__ == "__main__":
    download_dataset()
    df = load_dataset()
    report = validate_dataset(df)
    print(f"\nValidation Report:")
    print(f"  Rows: {report['rows']}")
    print(f"  Columns: {report['columns']}")
    print(f"  Duplicates: {report['duplicates']}")
    print(f"  Total missing: {sum(report['missing_values'].values())}")

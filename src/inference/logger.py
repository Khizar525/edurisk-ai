"""
Prediction logging — timestamped CSV logging of all predictions.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict

from src.config import PATHS


class PredictionLogger:
    """Logs predictions to a CSV file with timestamps."""

    COLUMNS = [
        "Timestamp", "Gender", "Age", "CGPA", "Academic Pressure",
        "Study Satisfaction", "Work/Study Hours", "Sleep Duration",
        "Dietary Habits", "Financial Stress", "Family History",
        "Suicidal Thoughts", "Risk Level", "Confidence",
    ]

    def __init__(self, log_path: Path = None):
        self.log_path = log_path or PATHS.prediction_log
        self._ensure_file()

    def _ensure_file(self):
        """Create log file with headers if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def log(self, inputs: Dict, result: Dict) -> None:
        """
        Log a prediction.

        Args:
            inputs: Dictionary of input features.
            result: Dictionary from RiskPredictor.predict().
        """
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inputs.get("gender", ""),
                inputs.get("age", ""),
                inputs.get("cgpa", ""),
                inputs.get("academic_pressure", ""),
                inputs.get("study_satisfaction", ""),
                inputs.get("work_study_hours", ""),
                inputs.get("sleep_duration", ""),
                inputs.get("dietary_habits", ""),
                inputs.get("financial_stress", ""),
                inputs.get("family_history", ""),
                inputs.get("suicidal_thoughts", ""),
                result.get("risk_level", ""),
                result.get("confidence", ""),
            ])

    def get_history(self, n: int = 100) -> list:
        """Read the last N predictions from the log."""
        if not self.log_path.exists():
            return []

        with open(self.log_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        return rows[-n:]

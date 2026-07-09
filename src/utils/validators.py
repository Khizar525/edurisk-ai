"""
Input validation utilities for the prediction service.
"""

from typing import Tuple


VALID_GENDERS = ["Male", "Female"]
VALID_SLEEP = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
VALID_DIET = ["Healthy", "Moderate", "Unhealthy"]
VALID_YES_NO = ["Yes", "No"]


def validate_prediction_inputs(
    gender: str,
    age: int,
    academic_pressure: int,
    cgpa: float,
    study_satisfaction: int,
    sleep_duration: str,
    dietary_habits: str,
    work_study_hours: int,
    financial_stress: int,
    family_history: str,
    suicidal_thoughts: str,
) -> Tuple[bool, str]:
    """
    Validate all prediction inputs.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if gender not in VALID_GENDERS:
        return False, f"Invalid gender: {gender}. Must be one of {VALID_GENDERS}"

    if not (15 <= age <= 40):
        return False, f"Invalid age: {age}. Must be between 15 and 40"

    if not (1 <= academic_pressure <= 5):
        return False, f"Invalid academic pressure: {academic_pressure}. Must be 1-5"

    if not (0.0 <= cgpa <= 4.0):
        return False, f"Invalid CGPA: {cgpa}. Must be 0.0-4.0"

    if not (1 <= study_satisfaction <= 5):
        return False, f"Invalid study satisfaction: {study_satisfaction}. Must be 1-5"

    if sleep_duration not in VALID_SLEEP:
        return False, f"Invalid sleep duration: {sleep_duration}. Must be one of {VALID_SLEEP}"

    if dietary_habits not in VALID_DIET:
        return False, f"Invalid dietary habits: {dietary_habits}. Must be one of {VALID_DIET}"

    if not (0 <= work_study_hours <= 24):
        return False, f"Invalid work/study hours: {work_study_hours}. Must be 0-24"

    if not (1 <= financial_stress <= 5):
        return False, f"Invalid financial stress: {financial_stress}. Must be 1-5"

    if family_history not in VALID_YES_NO:
        return False, f"Invalid family history: {family_history}. Must be Yes/No"

    if suicidal_thoughts not in VALID_YES_NO:
        return False, f"Invalid suicidal thoughts: {suicidal_thoughts}. Must be Yes/No"

    return True, ""

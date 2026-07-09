"""
Tests for inference module.
"""

import pytest

from src.utils.validators import validate_prediction_inputs


class TestInputValidation:
    """Tests for prediction input validation."""

    def test_valid_inputs_pass(self):
        is_valid, msg = validate_prediction_inputs(
            gender="Male", age=20, academic_pressure=3, cgpa=2.5,
            study_satisfaction=3, sleep_duration="7-8 hours",
            dietary_habits="Moderate", work_study_hours=6,
            financial_stress=2, family_history="No",
            suicidal_thoughts="No",
        )
        assert is_valid is True
        assert msg == ""

    def test_invalid_gender(self):
        is_valid, msg = validate_prediction_inputs(
            gender="Other", age=20, academic_pressure=3, cgpa=2.5,
            study_satisfaction=3, sleep_duration="7-8 hours",
            dietary_habits="Moderate", work_study_hours=6,
            financial_stress=2, family_history="No",
            suicidal_thoughts="No",
        )
        assert is_valid is False
        assert "gender" in msg.lower()

    def test_invalid_age(self):
        is_valid, msg = validate_prediction_inputs(
            gender="Male", age=50, academic_pressure=3, cgpa=2.5,
            study_satisfaction=3, sleep_duration="7-8 hours",
            dietary_habits="Moderate", work_study_hours=6,
            financial_stress=2, family_history="No",
            suicidal_thoughts="No",
        )
        assert is_valid is False
        assert "age" in msg.lower()

    def test_invalid_cgpa(self):
        is_valid, msg = validate_prediction_inputs(
            gender="Male", age=20, academic_pressure=3, cgpa=5.0,
            study_satisfaction=3, sleep_duration="7-8 hours",
            dietary_habits="Moderate", work_study_hours=6,
            financial_stress=2, family_history="No",
            suicidal_thoughts="No",
        )
        assert is_valid is False
        assert "cgpa" in msg.lower()

    def test_invalid_sleep_duration(self):
        is_valid, msg = validate_prediction_inputs(
            gender="Male", age=20, academic_pressure=3, cgpa=2.5,
            study_satisfaction=3, sleep_duration="9 hours",
            dietary_habits="Moderate", work_study_hours=6,
            financial_stress=2, family_history="No",
            suicidal_thoughts="No",
        )
        assert is_valid is False
        assert "sleep" in msg.lower()

    def test_invalid_academic_pressure(self):
        is_valid, msg = validate_prediction_inputs(
            gender="Male", age=20, academic_pressure=6, cgpa=2.5,
            study_satisfaction=3, sleep_duration="7-8 hours",
            dietary_habits="Moderate", work_study_hours=6,
            financial_stress=2, family_history="No",
            suicidal_thoughts="No",
        )
        assert is_valid is False
        assert "academic" in msg.lower()

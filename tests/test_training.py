"""
Tests for training module.
"""

import pytest
import numpy as np

from src.training.models import get_model_configs, get_shap_compatible_model


class TestModelConfigs:
    """Tests for model configuration factory."""

    def test_returns_four_models(self):
        configs = get_model_configs()
        assert len(configs) == 4
        assert "Random Forest" in configs
        assert "SVM" in configs
        assert "XGBoost" in configs
        assert "MLP" in configs

    def test_each_config_has_required_keys(self):
        configs = get_model_configs()
        for name, config in configs.items():
            assert "model" in config, f"{name} missing 'model'"
            assert "param_grid" in config, f"{name} missing 'param_grid'"
            assert "needs_grid" in config, f"{name} missing 'needs_grid'"

    def test_rf_and_xgb_need_grid(self):
        configs = get_model_configs()
        assert configs["Random Forest"]["needs_grid"] is True
        assert configs["XGBoost"]["needs_grid"] is True
        assert configs["SVM"]["needs_grid"] is False
        assert configs["MLP"]["needs_grid"] is False


class TestShapCompatibility:
    """Tests for SHAP model compatibility check."""

    def test_tree_models_are_compatible(self):
        assert get_shap_compatible_model("Random Forest") is True
        assert get_shap_compatible_model("XGBoost") is True

    def test_non_tree_models_are_not_compatible(self):
        assert get_shap_compatible_model("SVM") is False
        assert get_shap_compatible_model("MLP") is False

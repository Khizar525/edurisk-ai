"""
Tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.preprocessing.pipeline import (
    impute_missing,
    label_encode_categoricals,
    scale_features_train_test,
)
from src.config import CATEGORICAL_COLS, NUMERICAL_COLS


class TestImputation:
    """Tests for missing value imputation."""

    def test_no_missing_after_impute(self, sample_dataframe):
        df = impute_missing(sample_dataframe)
        assert df.isnull().sum().sum() == 0

    def test_preserves_shape(self, sample_dataframe):
        df = impute_missing(sample_dataframe)
        assert df.shape == sample_dataframe.shape

    def test_preserves_non_null_values(self, sample_dataframe):
        df = impute_missing(sample_dataframe)
        # Check that existing values are preserved (not replaced)
        for col in NUMERICAL_COLS:
            if col in sample_dataframe.columns:
                non_null = sample_dataframe[col].dropna()
                if len(non_null) > 0:
                    # At least some original values should be preserved
                    assert df[col].isin(non_null).any()


class TestLabelEncoding:
    """Tests for label encoding functionality."""

    def test_encode_returns_encoders(self, sample_dataframe):
        df, encoders = label_encode_categoricals(sample_dataframe, fit=True)
        assert isinstance(encoders, dict)
        assert all(isinstance(v, LabelEncoder) for v in encoders.values())

    def test_encode_produces_integers(self, sample_dataframe):
        df, _ = label_encode_categoricals(sample_dataframe, fit=True)
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                assert df[col].dtype in [np.int32, np.int64]

    def test_encode_with_pretrained_encoders(self, sample_dataframe):
        df1, encoders = label_encode_categoricals(sample_dataframe, fit=True)
        df2, _ = label_encode_categoricals(sample_dataframe, fit=False, encoders=encoders)
        pd.testing.assert_frame_equal(df1, df2)

    def test_fit_and_provide_encoders_raises(self, sample_dataframe):
        with pytest.raises(ValueError):
            label_encode_categoricals(sample_dataframe, fit=True, encoders={})

    def test_unseen_category_gets_default(self, sample_dataframe):
        """Unseen categories should be mapped to the first known class."""
        df, encoders = label_encode_categoricals(sample_dataframe, fit=True)
        # Add an unseen category
        df.loc[0, "Gender"] = "Other"
        df2, _ = label_encode_categoricals(df, fit=False, encoders=encoders)
        # Should not raise — unseen category mapped to first class
        assert df2.loc[0, "Gender"] == 0


class TestScaling:
    """Tests for train/test scaling."""

    def test_train_scaled_zero_mean(self, sample_features):
        X_train = sample_features[:80]
        X_test = sample_features[80:]
        X_train_s, X_test_s, scaler = scale_features_train_test(X_train, X_test)
        means = np.mean(X_train_s, axis=0)
        assert np.allclose(means, 0, atol=1e-10)

    def test_train_scaled_unit_variance(self, sample_features):
        X_train = sample_features[:80]
        X_test = sample_features[80:]
        X_train_s, X_test_s, scaler = scale_features_train_test(X_train, X_test)
        stds = np.std(X_train_s, axis=0)
        assert np.allclose(stds, 1, atol=1e-10)

    def test_test_uses_train_stats(self, sample_features):
        """Test set should be scaled using training statistics, not its own."""
        X_train = sample_features[:80]
        X_test = sample_features[80:]
        X_train_s, X_test_s, scaler = scale_features_train_test(X_train, X_test)

        # The scaler should have been fit on train
        np.testing.assert_array_equal(scaler.mean_, np.mean(X_train, axis=0))
        np.testing.assert_array_almost_equal(
            scaler.scale_, np.std(X_train, axis=0, ddof=0)
        )

    def test_test_not_necessarily_zero_mean(self, sample_features):
        """Test set mean after scaling should NOT necessarily be zero."""
        X_train = sample_features[:80]
        X_test = sample_features[80:]
        X_train_s, X_test_s, scaler = scale_features_train_test(X_train, X_test)
        test_means = np.mean(X_test_s, axis=0)
        # Test means should generally NOT be exactly zero
        assert not np.allclose(test_means, 0, atol=1e-10)

    def test_preserves_dimensions(self, sample_features):
        X_train = sample_features[:80]
        X_test = sample_features[80:]
        X_train_s, X_test_s, scaler = scale_features_train_test(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape

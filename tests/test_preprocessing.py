"""
Tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.preprocessing.pipeline import (
    label_encode_categoricals,
    prepare_features,
)
from src.config import CATEGORICAL_COLS


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


class TestPreparationPipeline:
    """Tests for the full preparation pipeline."""

    def test_returns_correct_shapes(self, sample_dataframe):
        X, y, encoders, scaler = prepare_features(sample_dataframe, fit=True)
        assert X.shape[0] == len(sample_dataframe)
        assert X.shape[1] == len(CATEGORICAL_COLS) + 6  # 6 numerical cols
        assert len(y) == len(sample_dataframe)

    def test_scaled_has_zero_mean(self, sample_dataframe):
        X, _, _, _ = prepare_features(sample_dataframe, fit=True)
        means = np.mean(X, axis=0)
        assert np.allclose(means, 0, atol=1e-10)

    def test_scaled_has_unit_variance(self, sample_dataframe):
        X, _, _, _ = prepare_features(sample_dataframe, fit=True)
        stds = np.std(X, axis=0)
        assert np.allclose(stds, 1, atol=1e-10)

    def test_reproducible_with_same_encoders(self, sample_dataframe):
        X1, y1, encoders, scaler = prepare_features(sample_dataframe, fit=True)
        X2, y2, _, _ = prepare_features(
            sample_dataframe, fit=False, encoders=encoders, scaler=scaler
        )
        np.testing.assert_array_equal(X1, X2)

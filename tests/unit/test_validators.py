# tests/unit/test_validators.py
"""
Unit tests for data validators.
"""

import numpy as np
import pandas as pd
import pytest

from shared.preprocessing.validators import DataValidator
from shared.constants import ValidationThresholds


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    def test_validate_column_names_success(self, validator):
        """Test successful column validation."""
        df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000000]
        })
        required = ['open', 'high', 'low', 'close', 'volume']

        result = validator.validate_column_names(df, required, 'TEST')
        assert result is True

    def test_validate_column_names_missing(self, validator):
        """Test column validation with missing columns."""
        df = pd.DataFrame({
            'open': [100],
            'close': [100.5]
        })
        required = ['open', 'high', 'low', 'close', 'volume']

        result = validator.validate_column_names(df, required, 'TEST')
        assert result is False

    def test_validate_raw_data_valid(self, validator, sample_ohlcv_data):
        """Test raw data validation with valid data."""
        result = validator.validate_raw_data(sample_ohlcv_data, 'TEST')

        assert not result.empty
        assert len(result) == len(sample_ohlcv_data)

    def test_validate_raw_data_invalid_prices(self, validator):
        """Test raw data validation with invalid prices."""
        df = pd.DataFrame({
            'open': [100, -10, 1000000],
            'high': [101, 100, 1000001],
            'low': [99, -20, 999999],
            'close': [100.5, 0, 500001],
            'volume': [1000000, 1000000, 1000000]
        })

        result = validator.validate_raw_data(df, 'TEST')

        # Invalid values should be set to NaN
        assert result['open'].isna().sum() > 0
        assert result['close'].isna().sum() > 0

    def test_validate_raw_data_invalid_volume(self, validator):
        """Test raw data validation with invalid volume."""
        df = pd.DataFrame({
            'open': [100, 100],
            'high': [101, 101],
            'low': [99, 99],
            'close': [100.5, 100.5],
            'volume': [0, 1e13]  # Zero and too large
        })

        result = validator.validate_raw_data(df, 'TEST')

        # Invalid volumes should be set to NaN
        assert result['volume'].isna().sum() > 0

    def test_validate_technical_indicators_valid(self, validator, sample_ohlcv_with_indicators):
        """Test technical indicator validation with valid data."""
        result = validator.validate_technical_indicators(sample_ohlcv_with_indicators)
        assert result is True

    def test_validate_technical_indicators_invalid_rsi(self, validator):
        """Test technical indicator validation with invalid RSI."""
        df = pd.DataFrame({
            'RSI': [50, 110, -10]  # 110 and -10 are out of bounds
        })

        result = validator.validate_technical_indicators(df)
        assert result is False

    def test_validate_technical_indicators_invalid_bb(self, validator):
        """Test technical indicator validation with invalid Bollinger Bands."""
        df = pd.DataFrame({
            'BB_Upper': [100, 90],
            'BB_Lower': [95, 100]  # Lower > Upper is invalid
        })

        result = validator.validate_technical_indicators(df)
        assert result is False

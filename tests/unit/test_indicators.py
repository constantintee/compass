# tests/unit/test_indicators.py
"""
Unit tests for technical indicators.
"""

import numpy as np
import pandas as pd
import pytest

from shared.technical_analysis.indicators import TechnicalIndicators
from shared.constants import ValidationThresholds


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""

    @pytest.fixture
    def indicators(self):
        """Create TechnicalIndicators instance."""
        return TechnicalIndicators()

    def test_calculate_support_resistance(self, indicators, sample_ohlcv_data):
        """Test support and resistance calculation."""
        result = indicators.calculate_support_resistance(sample_ohlcv_data)

        assert 'Support' in result.columns
        assert 'Resistance' in result.columns
        assert len(result) == len(sample_ohlcv_data)

        # Support should be less than or equal to Resistance
        valid_rows = result.dropna()
        assert (valid_rows['Support'] <= valid_rows['Resistance']).all()

    def test_calculate_fibonacci_levels(self, indicators, sample_ohlcv_data):
        """Test Fibonacci level calculation."""
        result = indicators.calculate_fibonacci_levels(sample_ohlcv_data)

        expected_columns = ['Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_0.786']
        for col in expected_columns:
            assert col in result.columns

        # Fibonacci levels should be ordered
        valid_rows = result.dropna()
        if len(valid_rows) > 0:
            assert (valid_rows['Fib_0.236'] <= valid_rows['Fib_0.786']).all()

    def test_calculate_on_balance_volume(self, indicators, sample_ohlcv_data):
        """Test OBV calculation."""
        result = indicators.calculate_on_balance_volume(sample_ohlcv_data)

        assert 'OBV' in result.columns
        assert len(result) == len(sample_ohlcv_data)

    def test_calculate_pivot_points(self, indicators, sample_ohlcv_data):
        """Test pivot point calculation."""
        result = indicators.calculate_pivot_points(sample_ohlcv_data)

        assert 'Pivot_Point' in result.columns
        assert 'R1' in result.columns
        assert 'S1' in result.columns

        # R1 should be greater than S1
        assert (result['R1'] > result['S1']).all()

    def test_calculate_peaks_troughs(self, indicators, sample_ohlcv_data):
        """Test peak and trough detection."""
        result = indicators.calculate_peaks_troughs(sample_ohlcv_data)

        assert 'Peak' in result.columns
        assert 'Trough' in result.columns

        # Should have some peaks and troughs detected
        assert result['Peak'].notna().sum() > 0
        assert result['Trough'].notna().sum() > 0

    def test_empty_dataframe(self, indicators):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        result = indicators.calculate_all_indicators(empty_df, 'TEST')

        assert result.empty

    def test_missing_columns(self, indicators):
        """Test handling of missing required columns."""
        incomplete_df = pd.DataFrame({'close': [100, 101, 102]})
        result = indicators.calculate_all_indicators(incomplete_df, 'TEST')

        assert result.empty


class TestRSIDivergence:
    """Tests for RSI divergence calculation."""

    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()

    def test_bullish_divergence(self, indicators):
        """Test detection of bullish divergence."""
        data = pd.DataFrame({
            'close': [100, 98, 96, 94, 92],  # Price going down
            'RSI': [30, 32, 34, 36, 38]  # RSI going up
        })

        result = indicators.calculate_rsi_divergence(data)

        assert 'RSI_Divergence' in result.columns
        # Should detect bullish divergence (value = 1)
        assert (result['RSI_Divergence'] == 1).any()

    def test_bearish_divergence(self, indicators):
        """Test detection of bearish divergence."""
        data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108],  # Price going up
            'RSI': [70, 68, 66, 64, 62]  # RSI going down
        })

        result = indicators.calculate_rsi_divergence(data)

        assert 'RSI_Divergence' in result.columns
        # Should detect bearish divergence (value = -1)
        assert (result['RSI_Divergence'] == -1).any()

    def test_no_rsi_column(self, indicators):
        """Test handling when RSI column is missing."""
        data = pd.DataFrame({'close': [100, 101, 102]})
        result = indicators.calculate_rsi_divergence(data)

        assert result.empty or 'RSI_Divergence' not in result.columns

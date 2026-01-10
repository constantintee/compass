# tests/conftest.py
"""
Shared pytest fixtures for Compass test suite.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(n_days) * 0.02
    close_prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.randn(n_days) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(n_days) * 0.01)),
        'low': close_prices * (1 - np.abs(np.random.randn(n_days) * 0.01)),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })

    return data


@pytest.fixture
def sample_ohlcv_with_indicators(sample_ohlcv_data):
    """Generate sample OHLCV data with technical indicators."""
    data = sample_ohlcv_data.copy()

    # Add mock indicators
    data['RSI'] = np.random.uniform(30, 70, len(data))
    data['MACD'] = np.random.uniform(-2, 2, len(data))
    data['MACD_Signal'] = np.random.uniform(-2, 2, len(data))
    data['BB_Upper'] = data['close'] * 1.02
    data['BB_Lower'] = data['close'] * 0.98
    data['Support'] = data['low'].rolling(20).min()
    data['Resistance'] = data['high'].rolling(20).max()

    return data


@pytest.fixture
def sample_config():
    """Generate sample configuration for testing."""
    return {
        'training': {
            'sequence_length': 60,
            'batch_size': 32,
            'epochs': 10,
            'cache_dir': '/tmp/compass_test_cache',
            'frequency': 'B'
        },
        'databasepsql': {
            'host': 'localhost',
            'port': '5432',
            'user': 'test',
            'password': 'test',
            'dbname': 'test_db'
        },
        'stocks': ['AAPL', 'GOOGL', 'MSFT']
    }


@pytest.fixture
def mock_db_connection(mocker):
    """Mock database connection for testing."""
    mock_conn = mocker.MagicMock()
    mock_cursor = mocker.MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn, mock_cursor

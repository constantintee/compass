# shared/__init__.py
"""
Shared module for Compass Stock Prediction System.

This module contains common code shared between the downloader, training, and webservice components.
"""

from .exceptions import (
    CompassError,
    DataFetchError,
    DataValidationError,
    ModelTrainingError,
    ModelLoadError,
    DatabaseError,
    ConfigurationError,
    PreprocessingError,
    TechnicalAnalysisError,
)

from .constants import (
    ValidationThresholds,
    TrainingConfig,
    RetryConfig,
    TechnicalIndicatorConfig,
)

__all__ = [
    # Exceptions
    'CompassError',
    'DataFetchError',
    'DataValidationError',
    'ModelTrainingError',
    'ModelLoadError',
    'DatabaseError',
    'ConfigurationError',
    'PreprocessingError',
    'TechnicalAnalysisError',
    # Constants
    'ValidationThresholds',
    'TrainingConfig',
    'RetryConfig',
    'TechnicalIndicatorConfig',
]

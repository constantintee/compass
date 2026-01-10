# shared/exceptions.py
"""
Custom exception hierarchy for Compass Stock Prediction System.

Provides specific exception types for different error scenarios,
enabling better error handling and debugging.
"""


class CompassError(Exception):
    """Base exception for all Compass application errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataFetchError(CompassError):
    """Raised when fetching stock data fails."""

    def __init__(self, ticker: str, source: str, message: str, details: dict = None):
        super().__init__(message, details)
        self.ticker = ticker
        self.source = source


class DataValidationError(CompassError):
    """Raised when data validation fails."""

    def __init__(self, ticker: str, message: str, invalid_columns: list = None, details: dict = None):
        super().__init__(message, details)
        self.ticker = ticker
        self.invalid_columns = invalid_columns or []


class ModelTrainingError(CompassError):
    """Raised when model training fails."""

    def __init__(self, model_name: str, message: str, epoch: int = None, details: dict = None):
        super().__init__(message, details)
        self.model_name = model_name
        self.epoch = epoch


class ModelLoadError(CompassError):
    """Raised when loading a model fails."""

    def __init__(self, model_path: str, message: str, details: dict = None):
        super().__init__(message, details)
        self.model_path = model_path


class DatabaseError(CompassError):
    """Raised when database operations fail."""

    def __init__(self, operation: str, message: str, query: str = None, details: dict = None):
        super().__init__(message, details)
        self.operation = operation
        self.query = query


class ConfigurationError(CompassError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, config_key: str, message: str, details: dict = None):
        super().__init__(message, details)
        self.config_key = config_key


class PreprocessingError(CompassError):
    """Raised when data preprocessing fails."""

    def __init__(self, ticker: str, step: str, message: str, details: dict = None):
        super().__init__(message, details)
        self.ticker = ticker
        self.step = step


class TechnicalAnalysisError(CompassError):
    """Raised when technical analysis calculations fail."""

    def __init__(self, ticker: str, indicator: str, message: str, details: dict = None):
        super().__init__(message, details)
        self.ticker = ticker
        self.indicator = indicator

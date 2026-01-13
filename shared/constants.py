# shared/constants.py
"""
Constants and configuration values for Compass Stock Prediction System.

Centralizes magic numbers and configuration values that were previously
scattered throughout the codebase.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ValidationThresholds:
    """Thresholds for data validation."""

    # Price validation
    MIN_STOCK_PRICE: float = 0.0
    MAX_STOCK_PRICE: float = 500_000.0

    # Volume validation
    MIN_VOLUME: int = 0
    MAX_VOLUME: float = 1e12

    # Data quality
    MIN_DATA_POINTS: int = 100
    MAX_MISSING_PERCENTAGE: float = 0.1  # 10%

    # RSI bounds
    RSI_MIN: float = 0.0
    RSI_MAX: float = 100.0

    # Forward fill limits
    PRICE_FFILL_LIMIT: int = 5
    VOLUME_FFILL_LIMIT: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    """Training-related configuration constants."""

    # Default sequence length for LSTM/Transformer
    DEFAULT_SEQUENCE_LENGTH: int = 60

    # Stuck training detection
    STUCK_THRESHOLD_EPOCHS: int = 60

    # Memory management
    MEMORY_CLEANUP_INTERVAL: int = 10

    # Default hyperparameters
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_EPOCHS: int = 30
    DEFAULT_LEARNING_RATE: float = 0.001

    # Validation split
    DEFAULT_VALIDATION_SPLIT: float = 0.2

    # Early stopping patience
    DEFAULT_PATIENCE: int = 10


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry logic."""

    MAX_RETRIES: int = 3
    BASE_DELAY: float = 1.0
    MAX_JITTER: float = 1.0
    BACKOFF_MULTIPLIER: int = 2

    # Database connection
    DB_MAX_RETRIES: int = 3
    DB_RETRY_DELAY: int = 5

    # API calls
    API_TIMEOUT: int = 10


@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    """Configuration for technical indicator calculations."""

    # EMA periods
    EMA_SHORT_PERIOD: int = 12
    EMA_LONG_PERIOD: int = 26

    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # RSI
    RSI_PERIOD: int = 14

    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD: int = 2

    # CCI and ATR
    CCI_PERIOD: int = 20
    ATR_PERIOD: int = 14

    # SuperTrend
    SUPERTREND_PERIOD: int = 10
    SUPERTREND_MULTIPLIER: int = 3

    # ZigZag
    ZIGZAG_SENSITIVITY: float = 0.05
    ZIGZAG_MIN_TREND_LENGTH: int = 5

    # Support/Resistance
    SUPPORT_RESISTANCE_WINDOW: int = 20

    # Fibonacci levels
    FIB_WINDOW: int = 14
    FIB_RETRACEMENT_LEVELS: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786)
    FIB_EXTENSION_LEVELS: Tuple[float, ...] = (1.618, 2.618, 3.618)

    # Peak/Trough detection
    PEAK_DISTANCE: int = 5


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration constants."""

    # Connection pool
    MIN_CONNECTIONS: int = 1
    MAX_CONNECTIONS: int = 20

    # Chunk sizes for batch operations
    INSERT_CHUNK_SIZE: int = 1000

    # TimescaleDB settings
    COMPRESSION_INTERVAL_DAYS: int = 7
    RETENTION_YEARS: int = 5


@dataclass(frozen=True)
class CacheConfig:
    """Cache configuration constants."""

    # Default cache size
    DEFAULT_CACHE_SIZE: int = 128

    # Cache timeouts (seconds)
    PREDICTION_CACHE_TIMEOUT: int = 900  # 15 minutes
    TOP_STOCKS_CACHE_TIMEOUT: int = 900  # 15 minutes
    TECHNICAL_INDICATORS_CACHE_TIMEOUT: int = 300  # 5 minutes


# Elliott Wave constants
class WaveDegree:
    """Elliott Wave degree classifications."""
    GRAND_SUPERCYCLE = "Grand Supercycle"
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"
    SUBMINUETTE = "Subminuette"


class WavePatternType:
    """Elliott Wave pattern types."""
    # Impulse Patterns
    IMPULSE_BULL = "Impulse (Bull)"
    IMPULSE_BEAR = "Impulse (Bear)"
    IMPULSE_LEADING_DIAGONAL = "Leading Diagonal"
    IMPULSE_ENDING_DIAGONAL = "Ending Diagonal"

    # Corrective Patterns
    CORRECTION_ZIGZAG = "Zigzag Correction"
    CORRECTION_FLAT = "Flat Correction"
    CORRECTION_TRIANGLE = "Triangle Correction"
    CORRECTION_DOUBLE_ZIGZAG = "Double Zigzag"
    CORRECTION_DOUBLE_THREE = "Double Three"
    CORRECTION_TRIPLE_THREE = "Triple Three"

    # Complex Patterns
    COMPLEX_WXY = "WXY Pattern"
    COMPLEX_COMBINATION = "Complex Combination"

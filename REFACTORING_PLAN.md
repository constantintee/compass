# Compass Stock Prediction System - Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan for the Compass stock prediction system. The codebase consists of ~9,500 lines of Python across three main services (downloader, training, webservice) with significant opportunities for improvement in code organization, maintainability, and reliability.

---

## Current Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Downloader │────▶│  Training   │────▶│ Webservice  │
│  (yfinance) │     │ (TensorFlow)│     │  (Django)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
              ┌────────────┴────────────┐
              │  PostgreSQL/TimescaleDB │
              │       + Redis           │
              └─────────────────────────┘
```

---

## Phase 1: Critical - Eliminate Code Duplication

### 1.1 Create Shared Module Structure

**Problem:** `technical_analysis.py` (2,032 lines) and `preprocessor.py` (593 lines) are duplicated in both `/downloader` and `/training` directories.

**Solution:** Create a shared module that both services import from.

```
/shared/
├── __init__.py
├── technical_analysis/
│   ├── __init__.py
│   ├── indicators.py      # RSI, MACD, Bollinger Bands (~600 lines)
│   ├── elliott_wave.py    # Elliott Wave analysis (~600 lines)
│   ├── cache.py           # TechnicalAnalysisCache (~100 lines)
│   └── orchestrator.py    # Main TechnicalAnalysis class (~200 lines)
├── preprocessing/
│   ├── __init__.py
│   ├── validators.py      # Data validation functions
│   ├── cleaners.py        # Missing value handling
│   └── transformers.py    # Data transformations
└── config/
    ├── __init__.py
    └── loader.py          # Centralized config management
```

**Impact:**
- Eliminates ~4,600 lines of duplicate code
- Single source of truth for bug fixes
- Easier maintenance and testing

**Files to modify:**
- Create `/shared/` directory structure
- Update `/downloader/downloader.py` imports
- Update `/training/training.py` imports
- Update Docker configurations to include shared module

---

### 1.2 Fix Deprecated Pandas Methods

**Problem:** Using deprecated `fillna(method='ffill')` and `fillna(method='bfill')` which will break in pandas 2.0+.

**Location:** `training/preprocessor.py` and `downloader/preprocessor.py` (lines 215-216, 282-283)

**Current Code:**
```python
df[col].fillna(method='ffill', inplace=True)
df[col].fillna(method='bfill', inplace=True)
```

**Refactored Code:**
```python
df[col] = df[col].ffill()
df[col] = df[col].bfill()
```

---

## Phase 2: High Priority - Code Quality

### 2.1 Implement Specific Exception Handling

**Problem:** Widespread use of generic `except Exception as e:` masks programming errors and makes debugging difficult.

**Files affected:**
- `downloader/downloader.py` (multiple locations)
- `training/training.py`
- `training/models.py`
- `training/ensemble.py`
- `webservice/predictor/services.py`

**Solution:** Create custom exception hierarchy:

```python
# shared/exceptions.py

class CompassError(Exception):
    """Base exception for Compass application"""
    pass

class DataFetchError(CompassError):
    """Raised when fetching stock data fails"""
    pass

class DataValidationError(CompassError):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(CompassError):
    """Raised when model training fails"""
    pass

class ModelLoadError(CompassError):
    """Raised when loading a model fails"""
    pass

class DatabaseError(CompassError):
    """Raised when database operations fail"""
    pass

class ConfigurationError(CompassError):
    """Raised when configuration is invalid"""
    pass
```

**Refactoring Example:**
```python
# Before
try:
    data = fetch_stock_data(ticker)
except Exception as e:
    logger.error(f"Error: {e}")
    return None

# After
try:
    data = fetch_stock_data(ticker)
except requests.RequestException as e:
    raise DataFetchError(f"Network error fetching {ticker}: {e}") from e
except ValueError as e:
    raise DataValidationError(f"Invalid data for {ticker}: {e}") from e
```

---

### 2.2 Break Up Large Files

**Problem:** `technical_analysis.py` is 2,032 lines - too large for effective maintenance.

**Proposed Split:**

| New File | Content | Lines (est.) |
|----------|---------|--------------|
| `indicators.py` | RSI, MACD, Bollinger Bands, SMA, EMA | ~600 |
| `elliott_wave.py` | Wave detection, pattern analysis | ~600 |
| `signals.py` | Buy/sell signal generation | ~300 |
| `cache.py` | Database caching layer | ~150 |
| `orchestrator.py` | Main class coordinating all components | ~200 |

---

### 2.3 Fix Variable Shadowing and Naming Issues

**Issues to fix:**

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `downloader/downloader.py` | 625 | `logging = setup_logging(config)` shadows module | Rename to `log_handler` |
| `webservice/predictor/templates/` | - | `error_massage.html` typo | Rename to `error_message.html` |
| `webservice/predictor/templates/` | - | `componets/` typo | Rename to `components/` |

---

### 2.4 Extract Magic Numbers to Configuration

**Problem:** Hardcoded values scattered throughout codebase reduce configurability.

**Current Magic Numbers:**
```python
# downloader.py
data[col] <= 0 or data[col] > 1000000  # Price validation threshold
volume > 1e12                           # Volume validation threshold

# models.py
threshold = 60                          # Stuck training detection

# Various files
sleep_time = delay + random.uniform(0, 1)  # Retry delay
```

**Solution:** Create constants module:

```python
# shared/constants.py

class ValidationThresholds:
    MAX_STOCK_PRICE = 1_000_000
    MAX_VOLUME = 1e12
    MIN_DATA_POINTS = 100

class TrainingConfig:
    STUCK_THRESHOLD_EPOCHS = 60
    MEMORY_CLEANUP_INTERVAL = 10

class RetryConfig:
    MAX_RETRIES = 3
    BASE_DELAY = 1.0
    MAX_JITTER = 1.0
    BACKOFF_MULTIPLIER = 2
```

---

## Phase 3: Medium Priority - Architecture Improvements

### 3.1 Decouple Webservice from Training Module

**Problem:** Django webservice imports directly from training module, creating tight coupling.

**Current (problematic):**
```python
# webservice/predictor/services.py
from training.technical_analysis import TechnicalAnalysis
from training.ensemble import EnsembleModel
```

**Proposed Solution:**

1. Create API abstraction layer:

```python
# webservice/predictor/ml_service.py

class MLService:
    """Abstraction layer for ML operations"""

    def __init__(self, model_path: str):
        self._model = None
        self._model_path = model_path

    def load_model(self):
        """Lazily load model with proper error handling"""
        if self._model is None:
            self._model = self._load_from_cache_or_file()
        return self._model

    def predict(self, ticker: str, features: pd.DataFrame) -> dict:
        """Generate prediction for a stock"""
        model = self.load_model()
        return model.predict(features)
```

2. Use dependency injection for services
3. Cache model in Redis to avoid file system coupling

---

### 3.2 Implement Repository Pattern for Data Access

**Current:** Direct database queries scattered throughout code.

**Proposed:**

```python
# shared/repositories/stock_repository.py

class StockRepository:
    def __init__(self, db_connection):
        self._conn = db_connection

    def get_historical_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch historical stock data"""
        pass

    def save_predictions(self, ticker: str, predictions: List[Prediction]) -> None:
        """Save model predictions"""
        pass

    def get_technical_indicators(self, ticker: str) -> dict:
        """Get cached technical indicators"""
        pass
```

---

### 3.3 Improve Memory Management

**Problem:** Inconsistent memory cleanup with manual `gc.collect()` calls.

**Solution:** Use context managers:

```python
# shared/memory.py

from contextlib import contextmanager
import gc
import tensorflow as tf

@contextmanager
def managed_tf_session():
    """Context manager for TensorFlow operations with automatic cleanup"""
    try:
        yield
    finally:
        tf.keras.backend.clear_session()
        gc.collect()

# Usage
with managed_tf_session():
    model = train_model(data)
    predictions = model.predict(test_data)
# Memory automatically cleaned up
```

---

## Phase 4: Lower Priority - Testing & Documentation

### 4.1 Add Comprehensive Testing

**Current:** No test files found.

**Proposed Test Structure:**

```
/tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_indicators.py         # RSI, MACD, etc.
│   ├── test_preprocessor.py       # Data cleaning
│   ├── test_validators.py         # Input validation
│   └── test_elliott_wave.py       # Wave detection
├── integration/
│   ├── test_data_pipeline.py      # End-to-end data flow
│   ├── test_model_training.py     # Training pipeline
│   └── test_predictions.py        # Prediction service
└── fixtures/
    ├── sample_stock_data.csv
    └── mock_responses.json
```

**Priority Test Areas:**
1. Technical indicator calculations (core business logic)
2. Data validation and preprocessing
3. Model loading and prediction
4. API endpoints

---

### 4.2 Add Structured Logging

**Current:** Inconsistent logging with missing context.

**Proposed:**

```python
# shared/logging_config.py

import structlog

def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

# Usage
logger = structlog.get_logger()
logger.info("fetching_stock_data", ticker=ticker, source="yfinance")
logger.error("fetch_failed", ticker=ticker, error=str(e), retry_count=attempt)
```

---

## Implementation Roadmap

### Phase 1: Foundation (Highest Priority)
| Task | Files | Complexity |
|------|-------|------------|
| Create `/shared/` module structure | New directory | Low |
| Move `technical_analysis.py` to shared | 2 files | Medium |
| Move `preprocessor.py` to shared | 2 files | Medium |
| Fix deprecated pandas methods | 2 files | Low |
| Update imports in downloader/training | 4 files | Low |

### Phase 2: Code Quality (High Priority)
| Task | Files | Complexity |
|------|-------|------------|
| Create exception hierarchy | New file | Low |
| Replace generic exception handlers | 5 files | Medium |
| Break up `technical_analysis.py` | 1 file → 5 files | High |
| Fix variable shadowing | 1 file | Low |
| Fix template naming | 2 files | Low |
| Extract magic numbers | 5 files | Low |

### Phase 3: Architecture (Medium Priority)
| Task | Files | Complexity |
|------|-------|------------|
| Create ML service abstraction | 2 files | Medium |
| Implement repository pattern | New files | Medium |
| Add memory management utilities | New file | Low |
| Decouple webservice from training | 3 files | Medium |

### Phase 4: Testing & Documentation (Lower Priority)
| Task | Files | Complexity |
|------|-------|------------|
| Set up pytest infrastructure | New files | Low |
| Add unit tests for indicators | New files | Medium |
| Add integration tests | New files | High |
| Configure structured logging | 3 files | Low |

---

## Proposed Final Directory Structure

```
/compass/
├── shared/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── constants.py
│   ├── logging_config.py
│   ├── memory.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── technical_analysis/
│   │   ├── __init__.py
│   │   ├── indicators.py
│   │   ├── elliott_wave.py
│   │   ├── signals.py
│   │   ├── cache.py
│   │   └── orchestrator.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── validators.py
│   │   ├── cleaners.py
│   │   └── transformers.py
│   └── repositories/
│       ├── __init__.py
│       └── stock_repository.py
├── downloader/
│   ├── downloader.py           # Simplified, uses shared modules
│   ├── validate_tickers.py
│   └── drop_table.py
├── training/
│   ├── training.py             # Simplified, uses shared modules
│   ├── models.py
│   ├── ensemble.py
│   ├── backtester.py
│   └── utils.py
├── webservice/
│   ├── stockpredictor/
│   │   ├── settings.py
│   │   ├── celery.py
│   │   └── urls.py
│   └── predictor/
│       ├── models.py
│       ├── views.py
│       ├── services.py
│       ├── ml_service.py       # New abstraction layer
│       └── templates/
│           ├── components/     # Fixed typo
│           └── error_message.html  # Fixed typo
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
└── data/
    └── config.yaml
```

---

## Metrics for Success

| Metric | Current | Target |
|--------|---------|--------|
| Duplicate code lines | ~4,600 | 0 |
| Generic exception handlers | 20+ | 0 |
| Max file size (lines) | 2,032 | <500 |
| Test coverage | 0% | 70%+ |
| Deprecated API usage | Yes | No |

---

## Risk Mitigation

1. **Breaking Changes:** Create comprehensive tests before refactoring
2. **Docker Compatibility:** Update all Dockerfiles and docker-compose.yml
3. **Import Paths:** Use search-and-replace carefully, test imports
4. **Database Migrations:** Plan for any schema changes
5. **Rollback Plan:** Use feature branches, merge incrementally

---

## Conclusion

This refactoring plan addresses the most critical issues first (code duplication, deprecated APIs) while building toward a more maintainable and testable architecture. The phased approach allows for incremental improvement without disrupting the existing functionality.

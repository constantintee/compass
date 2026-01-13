# webservice/stockpredictor/predictor/services.py
"""
Stock prediction services with security-focused input validation and error handling.
"""
import logging
import os
import re
import sys
import pandas as pd
from datetime import datetime, timedelta
from django.core.cache import cache
from django.utils import timezone
from django.conf import settings
from typing import List, Dict, Optional, Tuple

from .models import Stock, StockData, StockPrediction

# Import from shared module - add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from shared.technical_analysis import TechnicalAnalysis
from training.models import BaseModel, LSTMModel
from training.ensemble import EnsembleModel

logger = logging.getLogger('predictor')

# Security constants
MAX_TICKER_LENGTH = 15
TICKER_PATTERN = re.compile(r'^[A-Za-z0-9\-\.]+$')
MAX_LIMIT = 100
DEFAULT_LIMIT = 10


def validate_ticker(ticker: str) -> Tuple[bool, Optional[str], str]:
    """
    Validate a stock ticker symbol for security.

    Args:
        ticker: The ticker symbol to validate

    Returns:
        Tuple of (is_valid, error_message, sanitized_ticker)
    """
    if not ticker or not isinstance(ticker, str):
        return False, "Ticker must be a non-empty string", ""

    ticker = ticker.strip()

    if len(ticker) > MAX_TICKER_LENGTH:
        return False, f"Ticker must be {MAX_TICKER_LENGTH} characters or less", ""

    if not TICKER_PATTERN.match(ticker):
        return False, "Ticker contains invalid characters", ""

    return True, None, ticker.upper()


def sanitize_cache_key(key: str) -> str:
    """
    Sanitize a string for use as a cache key.

    Args:
        key: The key to sanitize

    Returns:
        Sanitized cache key safe for use with Django cache
    """
    # Remove any characters that could cause cache issues
    sanitized = re.sub(r'[^\w\-]', '_', key)

    # Limit length for cache backends
    max_key_length = 250
    if len(sanitized) > max_key_length:
        import hashlib
        hash_suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        sanitized = sanitized[:max_key_length - 17] + '_' + hash_suffix

    return sanitized


def validate_limit(limit: int, max_limit: int = MAX_LIMIT, default: int = DEFAULT_LIMIT) -> int:
    """
    Validate and bound a limit parameter.

    Args:
        limit: The limit value to validate
        max_limit: Maximum allowed limit
        default: Default value if invalid

    Returns:
        Validated limit value
    """
    try:
        limit = int(limit)
        if limit < 1:
            return default
        if limit > max_limit:
            return max_limit
        return limit
    except (ValueError, TypeError):
        return default


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe division with zero check.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division by zero

    Returns:
        Result of division or default if denominator is zero
    """
    if denominator == 0:
        return default
    return numerator / denominator


class MarketDataService:
    """Service for handling market data operations"""

    def __init__(self):
        try:
            from training.technical_analysis import TechnicalAnalysis
            self.technical_analysis = TechnicalAnalysis()
        except ImportError:
            logger.warning("TechnicalAnalysis not available")
            self.technical_analysis = None

    def get_stock_data(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Get historical stock data with technical indicators"""
        # Validate ticker
        is_valid, error_msg, sanitized_ticker = validate_ticker(ticker)
        if not is_valid:
            logger.warning(f"Invalid ticker in get_stock_data: {error_msg}")
            return None

        # Validate days parameter
        days = validate_limit(days, max_limit=365, default=60)

        try:
            end_date = timezone.now()
            start_date = end_date - timedelta(days=days + 10)  # Extra days for indicators

            stock_data = StockData.objects.filter(
                stock__ticker=sanitized_ticker,
                date__range=(start_date, end_date)
            ).order_by('date')

            if not stock_data.exists():
                logger.warning(f"No data found for ticker {sanitized_ticker}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_records(
                stock_data.values('date', 'open', 'high', 'low', 'close', 'volume')
            )

            # Calculate technical indicators if available
            if self.technical_analysis:
                df_with_indicators = self.technical_analysis.calculate_technical_indicators(
                    df, sanitized_ticker
                )
                return df_with_indicators.tail(days)

            return df.tail(days)

        except Exception as e:
            logger.error(f"Error getting stock data for {sanitized_ticker}: {str(e)}")
            return None

    def get_technical_indicators(self, ticker: str) -> Dict:
        """Get latest technical indicators for a stock"""
        # Validate ticker
        is_valid, error_msg, sanitized_ticker = validate_ticker(ticker)
        if not is_valid:
            logger.warning(f"Invalid ticker in get_technical_indicators: {error_msg}")
            return {}

        try:
            df = self.get_stock_data(sanitized_ticker, days=1)
            if df is None or df.empty:
                return {}

            # Get last row of indicators
            indicators = df.iloc[-1].to_dict()

            # Format indicators for display with safe defaults
            return {
                'RSI': round(float(indicators.get('RSI', 0) or 0), 2),
                'MACD': round(float(indicators.get('MACD', 0) or 0), 2),
                'Signal': round(float(indicators.get('MACD_Signal', 0) or 0), 2),
                'BB_Upper': round(float(indicators.get('BB_Upper', 0) or 0), 2),
                'BB_Lower': round(float(indicators.get('BB_Lower', 0) or 0), 2)
            }

        except Exception as e:
            logger.error(f"Error getting technical indicators for {sanitized_ticker}: {str(e)}")
            return {}


class PredictionService:
    """Service for handling predictions"""

    def __init__(self):
        self.market_data = MarketDataService()
        self.model = self._load_model()
        self.cache_timeout = 900  # 15 minutes

    def _load_model(self):
        """Load the prediction model safely"""
        try:
            # Use configurable model path from settings
            model_dir = getattr(settings, 'MODEL_DIR', '/app/data/models')
            model_path = os.path.join(str(model_dir), 'ensemble_model.h5')

            # Validate path exists and is within allowed directory
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}")
                return None

            # Verify path is within expected directory (prevent path traversal)
            model_dir_abs = os.path.abspath(str(model_dir))
            model_path_abs = os.path.abspath(model_path)
            if not model_path_abs.startswith(model_dir_abs):
                logger.error("Model path outside allowed directory")
                return None

            import tensorflow as tf
            from training.ensemble import EnsembleModel
            from training.models import SafeMSE, SafeMAE

            # Load the ensemble with custom objects
            custom_objects = {
                'SafeMSE': SafeMSE,
                'SafeMAE': SafeMAE,
            }
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.load_model(model_path)
            logger.info("Successfully loaded prediction model")
            return model

        except ImportError:
            logger.warning("EnsembleModel not available")
            return None
        except Exception as e:
            logger.error(f"Error loading prediction model: {str(e)}")
            return None

    def get_latest_prediction(self, ticker: str) -> Optional[StockPrediction]:
        """Get latest prediction for a stock"""
        # Validate ticker
        is_valid, error_msg, sanitized_ticker = validate_ticker(ticker)
        if not is_valid:
            logger.warning(f"Invalid ticker in get_latest_prediction: {error_msg}")
            return None

        try:
            # Use sanitized cache key
            cache_key = sanitize_cache_key(f"prediction_{sanitized_ticker}")

            # Try to get from cache first
            try:
                cached_prediction = cache.get(cache_key)
                if cached_prediction:
                    return cached_prediction
            except Exception as cache_error:
                logger.warning(f"Cache error: {cache_error}")

            # Get or create new prediction
            prediction = self.create_prediction(sanitized_ticker)
            if prediction:
                try:
                    cache.set(cache_key, prediction, self.cache_timeout)
                except Exception as cache_error:
                    logger.warning(f"Failed to cache prediction: {cache_error}")

            return prediction

        except Exception as e:
            logger.error(f"Error getting prediction for {sanitized_ticker}: {str(e)}")
            return None

    def create_prediction(self, ticker: str) -> Optional[StockPrediction]:
        """Create a new prediction for a stock"""
        # Validate ticker
        is_valid, error_msg, sanitized_ticker = validate_ticker(ticker)
        if not is_valid:
            logger.warning(f"Invalid ticker in create_prediction: {error_msg}")
            return None

        try:
            # Get stock data
            df = self.market_data.get_stock_data(sanitized_ticker)
            if df is None or df.empty:
                return None

            # Prepare features
            features = self._prepare_features(df)

            # Make prediction
            if self.model is None:
                logger.warning("Model not loaded, cannot create prediction")
                return None

            predicted_price, confidence = self.model.predict(features)

            # Validate prediction values
            if predicted_price is None or confidence is None:
                logger.warning(f"Invalid prediction values for {sanitized_ticker}")
                return None

            # Ensure confidence is within bounds (0-100)
            confidence_score = min(max(float(confidence) * 100, 0), 100)

            # Get or create stock
            stock, _ = Stock.objects.get_or_create(
                ticker=sanitized_ticker,
                defaults={'company_name': sanitized_ticker}
            )

            # Create prediction
            prediction = StockPrediction.objects.create(
                stock=stock,
                predicted_price=predicted_price,
                current_price=df['close'].iloc[-1],
                confidence_score=confidence_score,
                target_date=timezone.now() + timedelta(days=1)
            )

            return prediction

        except Exception as e:
            logger.error(f"Error creating prediction for {sanitized_ticker}: {str(e)}")
            return None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Feature preparation logic
            return df
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()


class TopStocksService:
    """Service for handling top stocks analysis"""

    def __init__(self):
        self.prediction_service = PredictionService()
        self.cache_timeout = 900  # 15 minutes

    def get_top_stocks(self, limit: int = 10, force_refresh: bool = False) -> List[Dict]:
        """Get top stocks based on prediction deltas"""
        # Validate and bound limit parameter
        limit = validate_limit(limit, max_limit=MAX_LIMIT, default=DEFAULT_LIMIT)

        try:
            # Use sanitized cache key
            cache_key = sanitize_cache_key("top_stocks")

            # Try to get from cache first
            if not force_refresh:
                try:
                    cached_stocks = cache.get(cache_key)
                    if cached_stocks:
                        return cached_stocks[:limit]
                except Exception as cache_error:
                    logger.warning(f"Cache error: {cache_error}")

            # Get all active stocks
            stocks = Stock.objects.all()
            predictions = []

            # Get predictions for each stock
            for stock in stocks:
                try:
                    prediction = self.prediction_service.get_latest_prediction(stock.ticker)
                    if prediction and prediction.current_price:
                        current_price = float(prediction.current_price)
                        predicted_price = float(prediction.predicted_price)
                        price_change = predicted_price - current_price

                        # Safe division for percentage calculation
                        price_change_percent = safe_division(
                            price_change, current_price, default=0.0
                        ) * 100

                        predictions.append({
                            'ticker': stock.ticker,
                            'current_price': current_price,
                            'predicted_price': predicted_price,
                            'price_change': price_change,
                            'price_change_percent': round(price_change_percent, 2),
                            'confidence_score': float(prediction.confidence_score)
                        })
                except Exception as stock_error:
                    logger.warning(f"Error processing stock {stock.ticker}: {stock_error}")
                    continue

            # Sort by absolute price change percentage
            predictions.sort(key=lambda x: abs(x['price_change_percent']), reverse=True)
            top_stocks = predictions[:limit]

            # Cache the results
            try:
                cache.set(cache_key, top_stocks, self.cache_timeout)
            except Exception as cache_error:
                logger.warning(f"Failed to cache top stocks: {cache_error}")

            return top_stocks

        except Exception as e:
            logger.error(f"Error getting top stocks: {str(e)}")
            return []

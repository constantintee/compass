# webservice/stockpredictor/predictor/services.py
import logging
import pandas as pd
from datetime import datetime, timedelta
from django.core.cache import cache
from django.utils import timezone
from django.db.models import F, ExpressionWrapper, FloatField
from typing import List, Dict, Optional

from .models import Stock, StockPrice, StockPrediction

# Import from shared module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from shared.technical_analysis import TechnicalAnalysis
from training.models import BaseModel, LSTMModel
from training.ensemble import EnsembleModel

logger = logging.getLogger('predictor')

class MarketDataService:
    """Service for handling market data operations"""
    
    def __init__(self):
        self.technical_analysis = TechnicalAnalysis()

    def get_stock_data(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Get historical stock data with technical indicators"""
        try:
            # Get raw data from database
            end_date = timezone.now()
            start_date = end_date - timedelta(days=days + 10)  # Extra days for indicators
            
            stock_prices = StockPrice.objects.filter(
                stock__ticker=ticker,
                date__range=(start_date, end_date)
            ).order_by('date')
            
            if not stock_prices.exists():
                logger.warning(f"No data found for ticker {ticker}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame.from_records(
                stock_prices.values('date', 'open', 'high', 'low', 'close', 'volume')
            )
            
            # Calculate technical indicators
            df_with_indicators = self.technical_analysis.calculate_technical_indicators(df, ticker)
            
            return df_with_indicators.tail(days)
            
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {str(e)}")
            return None

    def get_technical_indicators(self, ticker: str) -> Dict:
        """Get latest technical indicators for a stock"""
        try:
            df = self.get_stock_data(ticker, days=1)
            if df is None or df.empty:
                return {}
                
            # Get last row of indicators
            indicators = df.iloc[-1].to_dict()
            
            # Format indicators for display
            return {
                'RSI': round(indicators.get('RSI', 0), 2),
                'MACD': round(indicators.get('MACD', 0), 2),
                'Signal': round(indicators.get('MACD_Signal', 0), 2),
                'BB_Upper': round(indicators.get('BB_Upper', 0), 2),
                'BB_Lower': round(indicators.get('BB_Lower', 0), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting technical indicators for {ticker}: {str(e)}")
            return {}

class PredictionService:
    """Service for handling predictions"""
    
    def __init__(self):
        self.market_data = MarketDataService()
        self.model = self._load_model()
        self.cache_timeout = 900  # 15 minutes

    def _load_model(self) -> Optional[EnsembleModel]:
        """Load the prediction model"""
        try:
            model = EnsembleModel.load_from_path('/app/data/models/ensemble_model.h5')
            logger.info("Successfully loaded prediction model")
            return model
        except Exception as e:
            logger.error(f"Error loading prediction model: {str(e)}")
            return None

    def get_latest_prediction(self, ticker: str) -> Optional[StockPrediction]:
        """Get latest prediction for a stock"""
        try:
            # Try to get from cache first
            cache_key = f"prediction_{ticker}"
            cached_prediction = cache.get(cache_key)
            if cached_prediction:
                return cached_prediction
                
            # Get or create new prediction
            prediction = self.create_prediction(ticker)
            if prediction:
                cache.set(cache_key, prediction, self.cache_timeout)
                
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction for {ticker}: {str(e)}")
            return None

    def create_prediction(self, ticker: str) -> Optional[StockPrediction]:
        """Create a new prediction for a stock"""
        try:
            # Get stock data
            df = self.market_data.get_stock_data(ticker)
            if df is None or df.empty:
                return None
                
            # Prepare features
            features = self._prepare_features(df)
            
            # Make prediction
            if self.model is None:
                return None
                
            predicted_price, confidence = self.model.predict(features)
            
            # Get or create stock
            stock, _ = Stock.objects.get_or_create(ticker=ticker)
            
            # Create prediction
            prediction = StockPrediction.objects.create(
                stock=stock,
                predicted_price=predicted_price,
                current_price=df['close'].iloc[-1],
                confidence_score=confidence * 100,
                target_date=timezone.now() + timedelta(days=1)
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error creating prediction for {ticker}: {str(e)}")
            return None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Your feature preparation logic here
            # This should match the training data preparation
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
        try:
            # Try to get from cache first
            if not force_refresh:
                cached_stocks = cache.get('top_stocks')
                if cached_stocks:
                    return cached_stocks
                    
            # Get all active stocks
            stocks = Stock.objects.all()
            predictions = []
            
            # Get predictions for each stock
            for stock in stocks:
                prediction = self.prediction_service.get_latest_prediction(stock.ticker)
                if prediction:
                    predictions.append({
                        'ticker': stock.ticker,
                        'current_price': float(prediction.current_price),
                        'predicted_price': float(prediction.predicted_price),
                        'price_change': float(prediction.predicted_price - prediction.current_price),
                        'price_change_percent': (float(prediction.predicted_price - prediction.current_price) / 
                                               float(prediction.current_price) * 100),
                        'confidence_score': float(prediction.confidence_score)
                    })
            
            # Sort by absolute price change percentage
            predictions.sort(key=lambda x: abs(x['price_change_percent']), reverse=True)
            top_stocks = predictions[:limit]
            
            # Cache the results
            cache.set('top_stocks', top_stocks, self.cache_timeout)
            
            return top_stocks
            
        except Exception as e:
            logger.error(f"Error getting top stocks: {str(e)}")
            return []
# webservice/stockpredictor/predictor/views.py
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.core.cache import cache
from .models import Stock, StockData, StockPrediction
import pandas as pd

@require_http_methods(['GET'])
def get_stock_data(request, ticker):
    """Get stock data with technical indicators"""
    try:
        # Get from cache first
        cache_key = f"stock_data_{ticker}"
        data = cache.get(cache_key)
        
        if not data:
            # Get from database
            stock_data = StockData.objects.filter(
                stock__ticker=ticker
            ).order_by('-date')[:60]  # Last 60 days
            
            data = {
                'price_data': list(stock_data.values()),
                'technical_indicators': {
                    'rsi': stock_data.latest('date').rsi,
                    'macd': stock_data.latest('date').macd,
                    # ... other indicators
                }
            }
            
            # Cache for 5 minutes
            cache.set(cache_key, data, 300)
        
        return render(request, 'components/stock_card.html', {'data': data})
        
    except Exception as e:
        logger.error(f"Error getting stock data: {e}")
        return HttpResponse("Error loading stock data", status=500)
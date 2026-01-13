# webservice/stockpredictor/predictor/views.py
import logging
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.cache import cache
from django.db.models import Q
from .models import Stock, StockData, StockPrediction

logger = logging.getLogger(__name__)


@require_http_methods(['GET'])
def index(request):
    """Main index page"""
    try:
        top_stocks = Stock.objects.all()[:10]
        return render(request, 'index.html', {'stocks': top_stocks})
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return HttpResponse("Error loading page", status=500)


@require_http_methods(['GET'])
def get_prediction(request, ticker):
    """Get prediction for a specific ticker"""
    try:
        cache_key = f"prediction_{ticker}"
        prediction = cache.get(cache_key)

        if not prediction:
            stock = Stock.objects.get(ticker=ticker.upper())
            prediction = StockPrediction.objects.filter(
                stock=stock
            ).order_by('-prediction_date').first()

            if prediction:
                cache.set(cache_key, prediction, 300)

        return render(request, 'predictions.html', {
            'prediction': prediction,
            'ticker': ticker.upper()
        })
    except Stock.DoesNotExist:
        return HttpResponse(f"Stock {ticker} not found", status=404)
    except Exception as e:
        logger.error(f"Error getting prediction for {ticker}: {e}")
        return HttpResponse("Error loading prediction", status=500)


@require_http_methods(['POST'])
def refresh_prediction(request, ticker):
    """Trigger a prediction refresh for a ticker"""
    try:
        stock = Stock.objects.get(ticker=ticker.upper())
        # Clear the cache to force refresh
        cache.delete(f"prediction_{ticker}")
        cache.delete(f"stock_data_{ticker}")

        return HttpResponse("Prediction refresh triggered", status=200)
    except Stock.DoesNotExist:
        return HttpResponse(f"Stock {ticker} not found", status=404)
    except Exception as e:
        logger.error(f"Error refreshing prediction for {ticker}: {e}")
        return HttpResponse("Error refreshing prediction", status=500)


@require_http_methods(['GET'])
def get_prediction_graph(request, ticker):
    """Get prediction graph data for a ticker"""
    try:
        stock = Stock.objects.get(ticker=ticker.upper())
        predictions = StockPrediction.objects.filter(
            stock=stock
        ).order_by('-prediction_date')[:30]

        return render(request, 'components/prediction_graph.html', {
            'predictions': predictions,
            'ticker': ticker.upper()
        })
    except Stock.DoesNotExist:
        return HttpResponse(f"Stock {ticker} not found", status=404)
    except Exception as e:
        logger.error(f"Error getting prediction graph for {ticker}: {e}")
        return HttpResponse("Error loading graph", status=500)


@require_http_methods(['GET'])
def prediction_history(request, ticker):
    """Get prediction history for a ticker"""
    try:
        stock = Stock.objects.get(ticker=ticker.upper())
        predictions = StockPrediction.objects.filter(
            stock=stock
        ).order_by('-prediction_date')[:100]

        return render(request, 'components/prediction_chart.html', {
            'predictions': predictions,
            'ticker': ticker.upper()
        })
    except Stock.DoesNotExist:
        return HttpResponse(f"Stock {ticker} not found", status=404)
    except Exception as e:
        logger.error(f"Error getting prediction history for {ticker}: {e}")
        return HttpResponse("Error loading history", status=500)


@require_http_methods(['GET'])
def get_top_stocks(request):
    """Get top performing stocks"""
    try:
        cache_key = "top_stocks"
        stocks = cache.get(cache_key)

        if not stocks:
            stocks = Stock.objects.all().order_by('-market_cap')[:20]
            cache.set(cache_key, list(stocks), 600)

        return render(request, 'widgets/stock_summary.html', {'stocks': stocks})
    except Exception as e:
        logger.error(f"Error getting top stocks: {e}")
        return HttpResponse("Error loading top stocks", status=500)


@require_http_methods(['GET'])
def get_technical_indicators(request, ticker):
    """Get technical indicators for a ticker"""
    try:
        cache_key = f"indicators_{ticker}"
        data = cache.get(cache_key)

        if not data:
            stock_data = StockData.objects.filter(
                stock__ticker=ticker.upper()
            ).order_by('-date').first()

            if stock_data:
                data = {
                    'rsi': stock_data.rsi,
                    'macd': stock_data.macd,
                    'macd_signal': stock_data.macd_signal,
                    'macd_hist': stock_data.macd_hist,
                    'ema_12': stock_data.ema_12,
                    'ema_26': stock_data.ema_26,
                    'bb_upper': stock_data.bb_upper,
                    'bb_middle': stock_data.bb_middle,
                    'bb_lower': stock_data.bb_lower,
                    'support_level': stock_data.support_level,
                    'resistance_level': stock_data.resistance_level,
                }
                cache.set(cache_key, data, 300)

        return render(request, 'components/technical_indicators.html', {
            'indicators': data,
            'ticker': ticker.upper()
        })
    except Exception as e:
        logger.error(f"Error getting indicators for {ticker}: {e}")
        return HttpResponse("Error loading indicators", status=500)


@require_http_methods(['GET'])
def market_stats(request, ticker):
    """Get market statistics for a ticker"""
    try:
        stock = Stock.objects.get(ticker=ticker.upper())
        stock_data = StockData.objects.filter(
            stock=stock
        ).order_by('-date')[:60]

        if stock_data:
            latest = stock_data.first()
            stats = {
                'current_price': latest.close,
                'high': latest.high,
                'low': latest.low,
                'volume': latest.volume,
                'market_cap': stock.market_cap,
                'sector': stock.sector,
                'industry': stock.industry,
            }
        else:
            stats = {}

        return render(request, 'components/market_stats.html', {
            'stats': stats,
            'ticker': ticker.upper()
        })
    except Stock.DoesNotExist:
        return HttpResponse(f"Stock {ticker} not found", status=404)
    except Exception as e:
        logger.error(f"Error getting market stats for {ticker}: {e}")
        return HttpResponse("Error loading market stats", status=500)


@require_http_methods(['GET'])
def stock_search(request):
    """Search for stocks by ticker or company name"""
    try:
        query = request.GET.get('q', '').strip()

        if not query:
            return render(request, 'widgets/stock_summary.html', {'stocks': []})

        stocks = Stock.objects.filter(
            Q(ticker__icontains=query) | Q(company_name__icontains=query)
        )[:20]

        return render(request, 'widgets/stock_summary.html', {'stocks': stocks})
    except Exception as e:
        logger.error(f"Error searching stocks: {e}")
        return HttpResponse("Error searching stocks", status=500)


@require_http_methods(['GET'])
def get_stock_data(request, ticker):
    """Get stock data with technical indicators"""
    try:
        cache_key = f"stock_data_{ticker}"
        data = cache.get(cache_key)

        if not data:
            stock_data = StockData.objects.filter(
                stock__ticker=ticker
            ).order_by('-date')[:60]

            data = {
                'price_data': list(stock_data.values()),
                'technical_indicators': {
                    'rsi': stock_data.latest('date').rsi,
                    'macd': stock_data.latest('date').macd,
                }
            }

            cache.set(cache_key, data, 300)

        return render(request, 'components/stock_card.html', {'data': data, 'ticker': ticker})

    except Exception as e:
        logger.error(f"Error getting stock data: {e}")
        return HttpResponse("Error loading stock data", status=500)

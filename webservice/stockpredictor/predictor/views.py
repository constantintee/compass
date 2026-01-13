# webservice/stockpredictor/predictor/views.py
"""
Stock predictor views with security-focused input validation and error handling.
"""
import logging
from functools import wraps
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.cache import cache
from django.db.models import Q
from django.conf import settings
from .models import Stock, StockData, StockPrediction
from .validators import (
    validate_ticker,
    validate_search_query,
    sanitize_cache_key,
    get_safe_error_message,
)

logger = logging.getLogger(__name__)


def handle_view_exceptions(view_func):
    """
    Decorator to handle exceptions consistently across views.
    Logs errors and returns appropriate HTTP responses.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        try:
            return view_func(request, *args, **kwargs)
        except Stock.DoesNotExist:
            ticker = kwargs.get('ticker', 'Unknown')
            logger.warning(f"Stock not found: {ticker}")
            return HttpResponse(f"Stock not found", status=404)
        except StockData.DoesNotExist:
            ticker = kwargs.get('ticker', 'Unknown')
            logger.warning(f"Stock data not found for: {ticker}")
            return HttpResponse("Stock data not found", status=404)
        except Exception as e:
            logger.error(f"Error in {view_func.__name__}: {e}", exc_info=True)
            error_message = get_safe_error_message(e, include_details=settings.DEBUG)
            return HttpResponse(error_message, status=500)
    return wrapper


def validate_ticker_param(view_func):
    """
    Decorator to validate ticker parameter in views.
    Returns 400 Bad Request for invalid tickers.
    """
    @wraps(view_func)
    def wrapper(request, ticker, *args, **kwargs):
        is_valid, error_msg, sanitized_ticker = validate_ticker(ticker)
        if not is_valid:
            logger.warning(f"Invalid ticker parameter: {ticker} - {error_msg}")
            return HttpResponseBadRequest(error_msg)
        return view_func(request, sanitized_ticker, *args, **kwargs)
    return wrapper


@require_http_methods(['GET'])
@handle_view_exceptions
def index(request):
    """Main index page displaying top stocks."""
    top_stocks = Stock.objects.all()[:10]
    return render(request, 'index.html', {'stocks': top_stocks})


@require_http_methods(['GET'])
@validate_ticker_param
@handle_view_exceptions
def get_prediction(request, ticker):
    """
    Get prediction for a specific ticker.

    Args:
        request: HTTP request
        ticker: Stock ticker symbol (validated by decorator)

    Returns:
        Rendered prediction page or error response
    """
    cache_key = sanitize_cache_key(f"prediction_{ticker}")
    prediction = cache.get(cache_key)

    if not prediction:
        stock = Stock.objects.get(ticker=ticker)
        prediction = StockPrediction.objects.filter(
            stock=stock
        ).order_by('-prediction_date').first()

        if prediction:
            cache_timeout = getattr(settings, 'CACHE_TIMEOUT_PREDICTION', 300)
            cache.set(cache_key, prediction, cache_timeout)

    return render(request, 'predictions.html', {
        'prediction': prediction,
        'ticker': ticker
    })


@require_http_methods(['POST'])
@validate_ticker_param
@handle_view_exceptions
def refresh_prediction(request, ticker):
    """
    Trigger a prediction refresh for a ticker.
    Clears cached data to force a fresh lookup.

    Args:
        request: HTTP request
        ticker: Stock ticker symbol (validated by decorator)

    Returns:
        Success or error response
    """
    # Verify stock exists before clearing cache
    Stock.objects.get(ticker=ticker)

    # Clear relevant cache keys using sanitized keys
    cache.delete(sanitize_cache_key(f"prediction_{ticker}"))
    cache.delete(sanitize_cache_key(f"stock_data_{ticker}"))
    cache.delete(sanitize_cache_key(f"indicators_{ticker}"))

    logger.info(f"Prediction refresh triggered for {ticker}")
    return HttpResponse("Prediction refresh triggered", status=200)


@require_http_methods(['GET'])
@validate_ticker_param
@handle_view_exceptions
def get_prediction_graph(request, ticker):
    """Get prediction graph data for a ticker."""
    stock = Stock.objects.get(ticker=ticker)
    predictions = StockPrediction.objects.filter(
        stock=stock
    ).order_by('-prediction_date')[:30]

    return render(request, 'components/prediction_graph.html', {
        'predictions': predictions,
        'ticker': ticker
    })


@require_http_methods(['GET'])
@validate_ticker_param
@handle_view_exceptions
def prediction_history(request, ticker):
    """Get prediction history for a ticker."""
    stock = Stock.objects.get(ticker=ticker)
    predictions = StockPrediction.objects.filter(
        stock=stock
    ).order_by('-prediction_date')[:100]

    return render(request, 'components/prediction_chart.html', {
        'predictions': predictions,
        'ticker': ticker
    })


@require_http_methods(['GET'])
@handle_view_exceptions
def get_top_stocks(request):
    """Get top performing stocks with caching."""
    cache_key = sanitize_cache_key("top_stocks")
    stocks = cache.get(cache_key)

    if not stocks:
        stocks = Stock.objects.all().order_by('-market_cap')[:20]
        cache_timeout = getattr(settings, 'CACHE_TIMEOUT_TOP_STOCKS', 600)
        cache.set(cache_key, list(stocks), cache_timeout)

    return render(request, 'widgets/stock_summary.html', {'stocks': stocks})


@require_http_methods(['GET'])
@validate_ticker_param
@handle_view_exceptions
def get_technical_indicators(request, ticker):
    """Get technical indicators for a ticker with caching."""
    cache_key = sanitize_cache_key(f"indicators_{ticker}")
    data = cache.get(cache_key)

    if not data:
        stock_data = StockData.objects.filter(
            stock__ticker=ticker
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
            cache_timeout = getattr(settings, 'CACHE_TIMEOUT_INDICATORS', 300)
            cache.set(cache_key, data, cache_timeout)
        else:
            data = {}

    return render(request, 'components/technical_indicators.html', {
        'indicators': data,
        'ticker': ticker
    })


@require_http_methods(['GET'])
@validate_ticker_param
@handle_view_exceptions
def market_stats(request, ticker):
    """Get market statistics for a ticker."""
    stock = Stock.objects.get(ticker=ticker)
    stock_data = StockData.objects.filter(
        stock=stock
    ).order_by('-date')[:60]

    stats = {}
    if stock_data.exists():
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

    return render(request, 'components/market_stats.html', {
        'stats': stats,
        'ticker': ticker
    })


@require_http_methods(['GET'])
@handle_view_exceptions
def stock_search(request):
    """
    Search for stocks by ticker or company name.

    Security: Input is validated and sanitized to prevent injection attacks.
    """
    query = request.GET.get('q', '')

    # Validate and sanitize search query
    is_valid, error_msg, sanitized_query = validate_search_query(query)
    if not is_valid:
        logger.warning(f"Invalid search query: {error_msg}")
        return HttpResponseBadRequest(error_msg)

    if not sanitized_query:
        return render(request, 'widgets/stock_summary.html', {'stocks': []})

    # Use Django ORM for safe query execution (protected against SQL injection)
    stocks = Stock.objects.filter(
        Q(ticker__icontains=sanitized_query) | Q(company_name__icontains=sanitized_query)
    )[:20]

    return render(request, 'widgets/stock_summary.html', {'stocks': stocks})


@require_http_methods(['GET'])
@validate_ticker_param
@handle_view_exceptions
def get_stock_data(request, ticker):
    """Get stock data with technical indicators."""
    cache_key = sanitize_cache_key(f"stock_data_{ticker}")
    data = cache.get(cache_key)

    if not data:
        stock_data = StockData.objects.filter(
            stock__ticker=ticker
        ).order_by('-date')[:60]

        if not stock_data.exists():
            return render(request, 'components/stock_card.html', {
                'data': None,
                'ticker': ticker,
                'error': 'No data available'
            })

        # Get latest data safely
        latest = stock_data.first()

        data = {
            'price_data': list(stock_data.values()),
            'technical_indicators': {
                'rsi': latest.rsi if latest else None,
                'macd': latest.macd if latest else None,
            }
        }

        cache_timeout = getattr(settings, 'CACHE_TIMEOUT_PREDICTION', 300)
        cache.set(cache_key, data, cache_timeout)

    return render(request, 'components/stock_card.html', {'data': data, 'ticker': ticker})

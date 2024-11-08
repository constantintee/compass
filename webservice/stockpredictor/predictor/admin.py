# webservice/stockpredictor/predictor/admin.py
from django.contrib import admin
from .models import Stock, StockPrice, StockPrediction

@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'company_name', 'sector', 'market_cap', 'updated_at')
    search_fields = ('ticker', 'company_name')
    list_filter = ('sector', 'industry')

@admin.register(StockPrice)
class StockPriceAdmin(admin.ModelAdmin):
    list_display = ('stock', 'date', 'open', 'high', 'low', 'close', 'volume')
    search_fields = ('stock__ticker',)
    list_filter = ('date', 'stock')
    date_hierarchy = 'date'

@admin.register(StockPrediction)
class StockPredictionAdmin(admin.ModelAdmin):
    list_display = (
        'stock', 
        'prediction_date', 
        'target_date',
        'predicted_price', 
        'current_price', 
        'confidence_score'
    )
    search_fields = ('stock__ticker',)
    list_filter = ('prediction_date', 'target_date')
    date_hierarchy = 'prediction_date'

    def get_readonly_fields(self, request, obj=None):
        if obj:  # If editing an existing object
            return ('stock', 'prediction_date', 'predicted_price')
        return ()
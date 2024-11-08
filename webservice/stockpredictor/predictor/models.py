# webservice/stockpredictor/predictor/models.py
from django.db import models
from django.utils import timezone

class Stock(models.Model):
    ticker = models.CharField(max_length=10, unique=True)
    company_name = models.CharField(max_length=255)
    sector = models.CharField(max_length=100, blank=True)
    industry = models.CharField(max_length=100, blank=True)
    market_cap = models.BigIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['ticker']
        indexes = [models.Index(fields=['ticker'])]

    def __str__(self):
        return f"{self.ticker} - {self.company_name}"

class StockData(models.Model):
    """Combined model for price and technical indicators"""
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    date = models.DateTimeField(db_index=True)
    
    # Price data
    open = models.DecimalField(max_digits=10, decimal_places=2)
    high = models.DecimalField(max_digits=10, decimal_places=2)
    low = models.DecimalField(max_digits=10, decimal_places=2)
    close = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.BigIntegerField()
    
    # Technical Indicators
    rsi = models.FloatField(null=True)
    macd = models.FloatField(null=True)
    macd_signal = models.FloatField(null=True)
    macd_hist = models.FloatField(null=True)
    ema_12 = models.FloatField(null=True)
    ema_26 = models.FloatField(null=True)
    bb_upper = models.FloatField(null=True)
    bb_middle = models.FloatField(null=True)
    bb_lower = models.FloatField(null=True)
    
    # Elliott Wave data
    elliott_wave_pattern = models.CharField(max_length=50, null=True)
    elliott_wave_degree = models.CharField(max_length=50, null=True)
    elliott_wave_position = models.IntegerField(null=True)
    
    # Support/Resistance
    support_level = models.FloatField(null=True)
    resistance_level = models.FloatField(null=True)

    class Meta:
        unique_together = ['stock', 'date']
        indexes = [
            models.Index(fields=['stock', 'date']),
            models.Index(fields=['date']),
        ]

    def __str__(self):
        return f"{self.stock.ticker} - {self.date}"

class StockPrediction(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    prediction_date = models.DateTimeField(default=timezone.now)
    target_date = models.DateTimeField()
    predicted_price = models.DecimalField(max_digits=10, decimal_places=2)
    current_price = models.DecimalField(max_digits=10, decimal_places=2)
    confidence_score = models.FloatField()
    error_margin = models.FloatField(null=True)

    class Meta:
        ordering = ['-prediction_date']
        indexes = [
            models.Index(fields=['stock', 'prediction_date']),
            models.Index(fields=['target_date']),
        ]

    def __str__(self):
        return f"{self.stock.ticker} - {self.prediction_date}"
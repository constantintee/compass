# webservice/stockpredictor/predictor/models.py
"""
Stock predictor models with data integrity constraints and validation.
"""
from django.db import models
from django.utils import timezone
from django.core.validators import (
    MinValueValidator,
    MaxValueValidator,
    MinLengthValidator,
    RegexValidator,
)
from django.core.exceptions import ValidationError
from decimal import Decimal


# Custom validators
ticker_validator = RegexValidator(
    regex=r'^[A-Za-z0-9\-\.]+$',
    message='Ticker can only contain letters, numbers, hyphens, and dots.',
    code='invalid_ticker'
)


def validate_positive_price(value):
    """Validate that price is positive."""
    if value is not None and value < Decimal('0'):
        raise ValidationError(
            f'Price must be positive. Got: {value}',
            code='negative_price'
        )


def validate_confidence_score(value):
    """Validate that confidence score is between 0 and 100."""
    if value is not None and (value < 0 or value > 100):
        raise ValidationError(
            f'Confidence score must be between 0 and 100. Got: {value}',
            code='invalid_confidence'
        )


def validate_rsi(value):
    """Validate that RSI is between 0 and 100."""
    if value is not None and (value < 0 or value > 100):
        raise ValidationError(
            f'RSI must be between 0 and 100. Got: {value}',
            code='invalid_rsi'
        )


class Stock(models.Model):
    """
    Stock model representing a traded security.

    Attributes:
        ticker: Unique stock symbol (e.g., 'AAPL', 'MSFT')
        company_name: Full company name
        sector: Business sector classification
        industry: Industry classification
        market_cap: Market capitalization in USD
    """
    ticker = models.CharField(
        max_length=15,
        unique=True,
        validators=[
            MinLengthValidator(1),
            ticker_validator,
        ],
        db_index=True,
        help_text="Stock ticker symbol (e.g., AAPL)"
    )
    company_name = models.CharField(
        max_length=255,
        help_text="Full company name"
    )
    sector = models.CharField(
        max_length=100,
        blank=True,
        default='',
        help_text="Business sector"
    )
    industry = models.CharField(
        max_length=100,
        blank=True,
        default='',
        help_text="Industry classification"
    )
    market_cap = models.BigIntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Market capitalization in USD"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['ticker']
        indexes = [models.Index(fields=['ticker'])]
        verbose_name = 'Stock'
        verbose_name_plural = 'Stocks'

    def __str__(self):
        return f"{self.ticker} - {self.company_name}"

    def clean(self):
        """Normalize ticker to uppercase."""
        if self.ticker:
            self.ticker = self.ticker.upper()
        super().clean()

    def save(self, *args, **kwargs):
        """Ensure ticker is uppercase on save."""
        if self.ticker:
            self.ticker = self.ticker.upper()
        self.full_clean()
        super().save(*args, **kwargs)


class StockData(models.Model):
    """
    Combined model for price and technical indicators.

    Stores historical price data along with calculated technical indicators
    for machine learning features.
    """
    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name='stock_data'
    )
    date = models.DateTimeField(db_index=True)

    # Price data with validation
    open = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[validate_positive_price],
        help_text="Opening price"
    )
    high = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[validate_positive_price],
        help_text="Day high price"
    )
    low = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[validate_positive_price],
        help_text="Day low price"
    )
    close = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[validate_positive_price],
        help_text="Closing price"
    )
    volume = models.BigIntegerField(
        validators=[MinValueValidator(0)],
        help_text="Trading volume"
    )

    # Technical Indicators with validation
    rsi = models.FloatField(
        null=True,
        blank=True,
        validators=[validate_rsi],
        help_text="Relative Strength Index (0-100)"
    )
    macd = models.FloatField(
        null=True,
        blank=True,
        help_text="MACD value"
    )
    macd_signal = models.FloatField(
        null=True,
        blank=True,
        help_text="MACD signal line"
    )
    macd_hist = models.FloatField(
        null=True,
        blank=True,
        help_text="MACD histogram"
    )
    ema_12 = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="12-period EMA"
    )
    ema_26 = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="26-period EMA"
    )
    bb_upper = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Bollinger Band upper"
    )
    bb_middle = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Bollinger Band middle"
    )
    bb_lower = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Bollinger Band lower"
    )

    # Elliott Wave data
    elliott_wave_pattern = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        help_text="Elliott Wave pattern type"
    )
    elliott_wave_degree = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        help_text="Elliott Wave degree"
    )
    elliott_wave_position = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="Elliott Wave position (1-5)"
    )

    # Support/Resistance levels
    support_level = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Support price level"
    )
    resistance_level = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Resistance price level"
    )

    class Meta:
        unique_together = ['stock', 'date']
        indexes = [
            models.Index(fields=['stock', 'date']),
            models.Index(fields=['date']),
        ]
        ordering = ['-date']
        verbose_name = 'Stock Data'
        verbose_name_plural = 'Stock Data'

    def __str__(self):
        return f"{self.stock.ticker} - {self.date}"

    def clean(self):
        """Validate price relationships."""
        super().clean()

        # Validate high >= low
        if self.high is not None and self.low is not None:
            if self.high < self.low:
                raise ValidationError({
                    'high': 'High price cannot be less than low price.'
                })

        # Validate high >= open and high >= close
        if self.high is not None:
            if self.open is not None and self.high < self.open:
                raise ValidationError({
                    'high': 'High price cannot be less than open price.'
                })
            if self.close is not None and self.high < self.close:
                raise ValidationError({
                    'high': 'High price cannot be less than close price.'
                })

        # Validate low <= open and low <= close
        if self.low is not None:
            if self.open is not None and self.low > self.open:
                raise ValidationError({
                    'low': 'Low price cannot be greater than open price.'
                })
            if self.close is not None and self.low > self.close:
                raise ValidationError({
                    'low': 'Low price cannot be greater than close price.'
                })

        # Validate Bollinger Bands relationship
        if self.bb_upper is not None and self.bb_lower is not None:
            if self.bb_upper < self.bb_lower:
                raise ValidationError({
                    'bb_upper': 'Upper Bollinger Band cannot be less than lower band.'
                })


class StockPrediction(models.Model):
    """
    Model for storing stock price predictions.

    Attributes:
        stock: Foreign key to Stock
        prediction_date: When the prediction was made
        target_date: Date the prediction is for
        predicted_price: Predicted stock price
        current_price: Price at prediction time
        confidence_score: Model confidence (0-100)
        error_margin: Expected prediction error
    """
    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    prediction_date = models.DateTimeField(
        default=timezone.now,
        help_text="When the prediction was made"
    )
    target_date = models.DateTimeField(
        help_text="Date the prediction is for"
    )
    predicted_price = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[validate_positive_price],
        help_text="Predicted price"
    )
    current_price = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[validate_positive_price],
        help_text="Price when prediction was made"
    )
    confidence_score = models.FloatField(
        validators=[validate_confidence_score],
        help_text="Prediction confidence (0-100)"
    )
    error_margin = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Expected error percentage"
    )

    class Meta:
        ordering = ['-prediction_date']
        indexes = [
            models.Index(fields=['stock', 'prediction_date']),
            models.Index(fields=['target_date']),
        ]
        verbose_name = 'Stock Prediction'
        verbose_name_plural = 'Stock Predictions'

    def __str__(self):
        return f"{self.stock.ticker} - {self.prediction_date}"

    def clean(self):
        """Validate prediction dates."""
        super().clean()

        # Target date should be after or equal to prediction date
        if self.prediction_date and self.target_date:
            if self.target_date < self.prediction_date:
                raise ValidationError({
                    'target_date': 'Target date cannot be before prediction date.'
                })

    @property
    def price_change(self):
        """Calculate predicted price change."""
        if self.predicted_price and self.current_price:
            return self.predicted_price - self.current_price
        return None

    @property
    def price_change_percent(self):
        """Calculate predicted price change percentage."""
        if self.predicted_price and self.current_price and self.current_price != 0:
            return float((self.predicted_price - self.current_price) / self.current_price * 100)
        return None

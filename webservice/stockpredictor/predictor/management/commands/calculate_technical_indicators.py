# predictor/management/commands/calculate_technical_indicators.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from predictor.services import MarketDataService
from predictor.models import Stock
import logging

logger = logging.getLogger('predictor')

class Command(BaseCommand):
    help = 'Calculate technical indicators for stocks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--tickers',
            nargs='+',
            type=str,
            help='Specific tickers to calculate'
        )

    def handle(self, *args, **options):
        try:
            market_service = MarketDataService()
            tickers = options.get('tickers')

            if tickers:
                stocks = Stock.objects.filter(ticker__in=tickers)
            else:
                stocks = Stock.objects.all()

            total = stocks.count()
            self.stdout.write(f"Calculating indicators for {total} stocks...")

            for idx, stock in enumerate(stocks, 1):
                try:
                    market_service.calculate_technical_indicators(stock.ticker)
                    self.stdout.write(
                        f"[{idx}/{total}] Calculated indicators for {stock.ticker}"
                    )
                except Exception as e:
                    logger.error(f"Error calculating indicators for {stock.ticker}: {e}")
                    self.stdout.write(
                        self.style.ERROR(
                            f"[{idx}/{total}] Failed to calculate {stock.ticker}: {str(e)}"
                        )
                    )

            self.stdout.write(self.style.SUCCESS("Finished calculating indicators"))

        except Exception as e:
            logger.error(f"Error in calculate_technical_indicators command: {e}")
            self.stdout.write(self.style.ERROR(f"Command failed: {str(e)}"))
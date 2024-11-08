# predictor/management/commands/load_stock_data.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from predictor.services import MarketDataService
from predictor.models import Stock
import logging

logger = logging.getLogger('predictor')

class Command(BaseCommand):
    help = 'Load historical stock data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--tickers',
            nargs='+',
            type=str,
            help='Specific tickers to load'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=365,
            help='Number of days of historical data to load'
        )

    def handle(self, *args, **options):
        try:
            market_service = MarketDataService()
            tickers = options.get('tickers')
            days = options.get('days')

            if tickers:
                stocks = Stock.objects.filter(ticker__in=tickers)
            else:
                stocks = Stock.objects.all()

            total = stocks.count()
            self.stdout.write(f"Loading {days} days of data for {total} stocks...")

            for idx, stock in enumerate(stocks, 1):
                try:
                    market_service.load_historical_data(
                        stock.ticker,
                        days=days
                    )
                    self.stdout.write(
                        f"[{idx}/{total}] Loaded data for {stock.ticker}"
                    )
                except Exception as e:
                    logger.error(f"Error loading data for {stock.ticker}: {e}")
                    self.stdout.write(
                        self.style.ERROR(
                            f"[{idx}/{total}] Failed to load {stock.ticker}: {str(e)}"
                        )
                    )

            self.stdout.write(self.style.SUCCESS("Finished loading stock data"))

        except Exception as e:
            logger.error(f"Error in load_stock_data command: {e}")
            self.stdout.write(self.style.ERROR(f"Command failed: {str(e)}"))
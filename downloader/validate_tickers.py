import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('ticker_validation')

def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            return False
        return True
    except Exception:
        return False

def validate_tickers(tickers):
    valid_tickers = []
    invalid_tickers = []
    for ticker in tickers:
        if is_valid_ticker(ticker):
            valid_tickers.append(ticker)
            logger.info(f"Valid: {ticker}")
        else:
            invalid_tickers.append(ticker)
            logger.warning(f"Invalid: {ticker}")
    return valid_tickers, invalid_tickers

if __name__ == "__main__":
    # Example ticker list
    tickers = ['AAPL', 'MSFT', 'CSF0.DE', 'ABB', 'CHL', 'DISCA', 'PTR', 'TOT', 'RDS.A']
    valid, invalid = validate_tickers(tickers)
    
    logger.info(f"\nValid Tickers ({len(valid)}): {valid}")
    logger.info(f"Invalid Tickers ({len(invalid)}): {invalid}")

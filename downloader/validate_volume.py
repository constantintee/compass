import sqlite3
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('db_validation')

def validate_volume_column(db_path: str, ticker: str):
    try:
        conn = sqlite3.connect(db_path)
        query = f'SELECT volume FROM "{ticker.upper()}" LIMIT 5'
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df['volume'].dtype != 'int64':
            logger.error(f"{ticker}: 'volume' column is not of integer type. Current dtype: {df['volume'].dtype}")
        elif df['volume'].isnull().any():
            logger.error(f"{ticker}: 'volume' column contains NaN values.")
        else:
            logger.info(f"{ticker}: 'volume' column is valid and contains numeric values.")
    except Exception as e:
        logger.error(f"Error validating {ticker}: {e}")

# Example usage
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Add all 170 tickers here
db_path = '/usr/src/app/data/stocks.db'

for ticker in tickers:
    validate_volume_column(db_path, ticker)

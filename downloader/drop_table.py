import sqlite3
import logging
import yaml
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('db_cleanup')

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

def drop_all_ticker_tables(db_path: str, tickers: list):
    """Drop all ticker tables from the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for ticker in tickers:
            table_name = ticker.upper()  # Assuming table names are uppercase
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            logger.info(f"Dropped table for ticker: {ticker}")
        
        conn.commit()
        conn.close()
        logger.info("All ticker tables have been dropped successfully.")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        sys.exit(1)

def main():
    # Load configuration
    config_path = "data/config.yaml"
    config = load_config(config_path)
    
    db_path = config['database']['path']
    tickers = config['downloader']['tickers']
    
    drop_all_ticker_tables(db_path, tickers)

if __name__ == "__main__":
    main()

# downloader.py
import psycopg2
from psycopg2 import sql, pool
from psycopg2.extras import RealDictCursor
import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
#from datetime import datetime, timedelta
import logging
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from apscheduler.schedulers.blocking import BlockingScheduler
from threading import Lock
import random
from io import StringIO
import urllib.parse
import traceback
from dotenv import load_dotenv

from technical_analysis import TechnicalAnalysis, AdvancedElliottWaveAnalysis
from preprocessor import Preprocessor


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Override sensitive data with environment variables if present
        config['databasepsql']['host'] = os.getenv('DB_HOST', config['databasepsql']['host'])
        config['databasepsql']['port'] = os.getenv('DB_PORT', config['databasepsql']['port'])
        config['databasepsql']['user'] = os.getenv('DB_USER', config['databasepsql']['user'])
        config['databasepsql']['password'] = os.getenv('DB_PASSWORD', config['databasepsql']['password'])
        config['databasepsql']['dbname'] = os.getenv('DB_NAME', config['databasepsql']['dbname'])

        return config
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        sys.exit(1)

def setup_logging(log_config: dict):
    """Setup logging with rotating file handler and console output."""
    log_file = log_config.get('log_file', 'data/logs/data_download.log')
    max_bytes = log_config.get('max_bytes', 5242880)  # 5MB
    backup_count = log_config.get('backup_count', 5)
    log_level = log_config.get('level', 'INFO').upper()

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            print(f"Created log directory at {log_dir}.")
        except Exception as e:
            print(f"Failed to create log directory at {log_dir}: {e}")
            sys.exit(1)

    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def log_info(message: str):
    logging.info(message)

def log_warning(message: str):
    logging.warning(message)

def log_error(message: str):
    logging.error(message)
        

class StockDataPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.connection_pool = None
        self.db_lock = Lock()
        self.logger = logging.getLogger('data_logger')

        # Setup logging
#        self.logger = self.setup_logging()
        
        # Initialize database connection pool
        self.init_connection_pool()
        
        # Ensure database structure
        self.ensure_database_exists()

    def init_connection_pool(self):
        """Initialize PostgreSQL connection pool with TimescaleDB support"""
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.config['databasepsql']['host'],
                port=self.config['databasepsql']['port'],
                user=self.config['databasepsql']['user'],
                password=self.config['databasepsql']['password'],
                dbname=self.config['databasepsql']['dbname']
            )
            self.logger.info("Connection pool created successfully")
        except Exception as e:
            self.logger.error(f"Error creating connection pool: {e}")
            raise

    def ensure_database_exists(self):
        """Ensure TimescaleDB and necessary tables exist"""
        try:
            with self.db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                # Create stocks table if not exists
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stocks (
                        ticker VARCHAR(10) NOT NULL,
                        date TIMESTAMPTZ NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume BIGINT,
                        -- Technical indicators
                        rsi REAL,
                        macd REAL,
                        macd_signal REAL,
                        macd_hist REAL,
                        ema_12 REAL,
                        ema_26 REAL,
                        bb_upper REAL,
                        bb_middle REAL,
                        bb_lower REAL,
                        -- Elliott Wave
                        elliott_wave_pattern VARCHAR(50),
                        elliott_wave_degree VARCHAR(50),
                        elliott_wave_position INTEGER,
                        -- Support/Resistance
                        support_level REAL,
                        resistance_level REAL,
                        PRIMARY KEY (ticker, date)
                    );
                ''')

                # Convert to hypertable
                cursor.execute('''
                    SELECT create_hypertable('stocks', 'date', 
                        if_not_exists => TRUE, 
                        migrate_data => TRUE
                    );
                ''')

                # Enable compression
                cursor.execute('''
                    ALTER TABLE stocks SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'ticker'
                    );
                ''')

                # Add compression policy
                cursor.execute('''
                    SELECT add_compression_policy('stocks', 
                        INTERVAL '7 days',
                        if_not_exists => TRUE
                    );
                ''')

                conn.commit()
                self.logger.info("Database structure ensured")
                
        except Exception as e:
            self.logger.error(f"Error ensuring database structure: {e}")
            raise
        finally:
            if conn:
                self.return_connection(conn)

    def get_connection(self):
        """Get a connection from the pool with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                return self.connection_pool.getconn()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay * (attempt + 1))

    def fetch_and_process_data(self, ticker: str) -> bool:
        """Main processing pipeline for a single ticker"""
        training_config = self.config.get('training', {})

        # Initialize Preprocessor
        preprocessor = Preprocessor(
            sequence_length=training_config.get('sequence_length', 60),
            cache_dir=training_config.get('cache_dir', 'cache'),
            frequency=training_config.get('frequency', 'B'),
            logger=self.logger
        )

        # Directory setup
        individual_tfrecord_path = "data/memory/individual_tfrecords"
        os.makedirs(individual_tfrecord_path, exist_ok=True)
        
        try:
            # 1. Fetch raw data with retry mechanism
            raw_data = self.fetch_data_with_retry(ticker)
            if raw_data.empty:
                self.logger.warning(f"No data fetched for {ticker}")
                return False

            # 2. Validate raw data
            validated_data = self.validate_raw_data(raw_data, ticker)
            if validated_data.empty:
                return False

            # 3. Preprocess data
            stocks = ticker
            if stocks:
                Preprocessor.parallel_preprocess_stocks(
                    stocks, training_config, individual_tfrecord_path
                )

            # # 3. Preprocess data
            # preprocessed_data = self.preprocessor.preprocess_data(validated_data, ticker)
            # if preprocessed_data.empty:
            #     self.logger.warning(f"Preprocessing failed for {ticker}")
            #     return False

            # # 4. Calculate technical indicators
            # data_with_indicators = self.technical_analysis.calculate_technical_indicators(
            #     preprocessed_data, ticker
            # )
            # if not self.validate_technical_indicators(data_with_indicators):
            #     return False

            # # 5. Add Elliott Wave analysis
            # elliott_wave_df = self.elliott_wave.identify_elliott_waves(data_with_indicators)
            
            # # 6. Combine all data
            # final_data = pd.concat([data_with_indicators, elliott_wave_df], axis=1)
            
            # Data is now stored via the Preprocessor's TFRecord writing
            # The parallel_preprocess_stocks handles the complete pipeline
            return True

        except Exception as e:
            self.logger.error(f"Error processing {ticker}: {str(e)}")
            return False

    def fetch_data_with_retry(self, ticker: str) -> pd.DataFrame:
        """Fetch data with retry and backoff mechanism"""
        max_retries = self.config.get('max_retries', 3)
        backoff_factor = self.config.get('backoff_factor', 2)
        initial_delay = self.config.get('initial_delay', 1)
        
        for attempt in range(max_retries):
            try:
                # Try yfinance first
                data = self.fetch_data_yfinance(ticker, max_retries, backoff_factor, initial_delay)
                if not data.empty:
                    return data

                # Fallback to stooq
                data = self.fetch_data_stooq(ticker, max_retries, backoff_factor, initial_delay)
                if not data.empty:
                    return data

            except Exception as e:
                delay = initial_delay * (backoff_factor ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay)

        return pd.DataFrame()

    def validate_raw_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Comprehensive data validation"""
        try:
            if data.empty:
                self.logger.error(f"Empty data for {ticker}")
                return pd.DataFrame()

            # Normalize column names
            data.columns = data.columns.str.lower()
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing columns for {ticker}: {required_columns}")
                return pd.DataFrame()

            # Remove invalid prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                invalid_mask = (data[col] <= 0) | (data[col] > 1000000)
                if invalid_mask.any():
                    self.logger.warning(f"Removing {invalid_mask.sum()} invalid {col} values for {ticker}")
                    data.loc[invalid_mask, col] = np.nan

            # Handle invalid volume
            volume_mask = (data['volume'] <= 0) | (data['volume'] > 1e12)
            if volume_mask.any():
                self.logger.warning(f"Fixing {volume_mask.sum()} invalid volume values for {ticker}")
                data.loc[volume_mask, 'volume'] = 0

            # Handle missing values
            data = self.handle_missing_values(data)
            
            return data

        except Exception as e:
            self.logger.error(f"Error validating data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def validate_technical_indicators(self, data: pd.DataFrame) -> bool:
        """Validate calculated technical indicators"""
        try:
            # Check RSI bounds
            if 'RSI' in data.columns:
                invalid_rsi = (data['RSI'] < 0) | (data['RSI'] > 100)
                if invalid_rsi.any():
                    self.logger.error(f"Invalid RSI values detected")
                    return False

            # Check Bollinger Bands
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
                invalid_bb = data['BB_Upper'] < data['BB_Lower']
                if invalid_bb.any():
                    self.logger.error(f"Invalid Bollinger Bands detected")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating technical indicators: {str(e)}")
            return False

    def store_data(self, ticker: str, data: pd.DataFrame) -> bool:
        """Store processed data in TimescaleDB"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Store data in chunks
            chunk_size = 1000
            total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size else 0)

            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(data))
                chunk = data.iloc[start_idx:end_idx]

                # Prepare data for insertion
                values = []
                for idx, row in chunk.iterrows():
                    values.append((
                        ticker,
                        idx,  # date is index
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row.get('RSI', 0)),
                        float(row.get('MACD', 0)),
                        float(row.get('MACD_Signal', 0)),
                        float(row.get('MACD_Hist', 0)),
                        float(row.get('EMA_12', 0)),
                        float(row.get('EMA_26', 0)),
                        float(row.get('BB_Upper', 0)),
                        float(row.get('BB_Middle', 0)),
                        float(row.get('BB_Lower', 0)),
                        row.get('elliott_wave_pattern'),
                        row.get('elliott_wave_degree'),
                        int(row.get('elliott_wave_position', 0)),
                        float(row.get('support_level', 0)),
                        float(row.get('resistance_level', 0))
                    ))

                # Bulk insert
                cursor.executemany("""
                    INSERT INTO stocks (
                        ticker, date, open, high, low, close, volume,
                        rsi, macd, macd_signal, macd_hist,
                        ema_12, ema_26,
                        bb_upper, bb_middle, bb_lower,
                        elliott_wave_pattern, elliott_wave_degree, elliott_wave_position,
                        support_level, resistance_level
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    ON CONFLICT (ticker, date) 
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        rsi = EXCLUDED.rsi,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_hist = EXCLUDED.macd_hist,
                        ema_12 = EXCLUDED.ema_12,
                        ema_26 = EXCLUDED.ema_26,
                        bb_upper = EXCLUDED.bb_upper,
                        bb_middle = EXCLUDED.bb_middle,
                        bb_lower = EXCLUDED.bb_lower,
                        elliott_wave_pattern = EXCLUDED.elliott_wave_pattern,
                        elliott_wave_degree = EXCLUDED.elliott_wave_degree,
                        elliott_wave_position = EXCLUDED.elliott_wave_position,
                        support_level = EXCLUDED.support_level,
                        resistance_level = EXCLUDED.resistance_level
                """, values)

            conn.commit()
            self.logger.info(f"Successfully stored data for {ticker}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Error storing data for {ticker}: {str(e)}")
            return False

    def process_batch(self, tickers: list) -> None:
        """Process a batch of tickers in parallel"""
        try:
            with ThreadPoolExecutor(max_workers=self.config.get('thread_pool_size', 5)) as executor:
                futures = []
                for ticker in tickers:
                    futures.append(
                        executor.submit(self.fetch_and_process_data, ticker)
                    )

                # Use tqdm for progress tracking
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Tickers"):
                    try:
                        future.result()  # Will raise any exceptions that occurred
                    except Exception as e:
                        self.logger.error(f"Error in ticker processing: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")

    def fetch_data_yfinance(self, ticker: str, max_retries: int, backoff_factor: int, initial_delay: int) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with retry mechanism"""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching {ticker} from Yahoo Finance. Attempt {attempt + 1}/{max_retries}")
                stock = yf.Ticker(ticker)
                data = stock.history(period="max")

                if data.empty:
                    self.logger.warning(f"No data returned for {ticker} from Yahoo Finance")
                    return pd.DataFrame()

                # Ensure Volume is numeric
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).astype(int)

                # Log if any suspiciously high volume values
                if (data['Volume'] > 1e12).any():
                    self.logger.warning(f"Suspiciously high volume values found for {ticker}")

                self.logger.info(f"Successfully fetched data for {ticker} from Yahoo Finance")
                return data

            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker} from Yahoo Finance: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = delay + random.uniform(0, 1)
                    self.logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                else:
                    self.logger.error(f"All retry attempts failed for {ticker}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetch_data_stooq(self, ticker: str, max_retries: int, backoff_factor: int, initial_delay: int) -> pd.DataFrame:
        """Fetch data from Stooq with retry mechanism"""
        delay = initial_delay
        stooq_ticker = self.format_stooq_ticker(ticker)
        url = f'https://stooq.com/q/d/l/?s={stooq_ticker}&i=d'

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Fetching {ticker} from Stooq. Attempt {attempt + 1}/{max_retries}")
                response = requests.get(url, timeout=10)
                
                if response.status_code != 200:
                    self.logger.warning(f"Stooq returned status code {response.status_code} for {ticker}")
                    return pd.DataFrame()

                # Read CSV from response text
                data = pd.read_csv(StringIO(response.text))

                if data.empty:
                    self.logger.warning(f"Empty data received from Stooq for {ticker}")
                    return pd.DataFrame()

                # Format data
                data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data.dropna(subset=['date'], inplace=True)
                data.set_index('date', inplace=True)

                # Ensure Volume is numeric
                data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0).astype(int)

                self.logger.info(f"Successfully fetched data for {ticker} from Stooq")
                return data

            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker} from Stooq: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = delay + random.uniform(0, 1)
                    self.logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    delay *= backoff_factor
                else:
                    self.logger.error(f"All retry attempts failed for {ticker}")
                    return pd.DataFrame()

        return pd.DataFrame()

    @staticmethod
    def format_stooq_ticker(ticker: str) -> str:
        """Format ticker for Stooq API"""
        ticker = ticker.lower()
        if not ticker.endswith('.us'):
            ticker += '.us'
        return urllib.parse.quote(ticker)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Forward fill price data (limited to 5 days)
            price_columns = ['open', 'high', 'low', 'close']
            df[price_columns] = df[price_columns].fillna(method='ffill', limit=5)

            # Fill remaining gaps with linear interpolation
            df[price_columns] = df[price_columns].interpolate(method='linear', limit=5)

            # Handle volume separately
            df['volume'] = df['volume'].fillna(method='ffill').fillna(0)

            # Remove any remaining rows with NaN values
            df = df.dropna()

            return df

        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            return df

    def validate_db_entry(self, ticker: str) -> bool:
        """Validate database entries for a ticker"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM stocks WHERE ticker = %s;
            """, (ticker,))
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                self.logger.info(f"Validated data entry for {ticker}. Found {result[0]} rows.")
                return True
            else:
                self.logger.warning(f"No data found for {ticker} in database.")
                return False

        except Exception as e:
            self.logger.error(f"Error validating database entry for {ticker}: {str(e)}")
            return False
        finally:
            if conn:
                self.return_connection(conn)

    def return_connection(self, conn):
        """Return a connection to the pool"""
        self.connection_pool.putconn(conn)

def main():
    """Main execution function"""
    try:
        # Load configuration
        config_path = "data/config.yaml"
        config = load_config(config_path)

        setup_logging(config.get('logging', {}))


        # Initialize pipeline
        pipeline = StockDataPipeline(config)

        # Get ticker list
        tickers = config.get('stocks', [])
        if not tickers:
            pipeline.logger.error("No tickers found in configuration")
            return

        # Process tickers
        pipeline.process_batch(tickers)

        # Schedule regular updates if configured
        if config.get('enable_scheduler', False):
            scheduler = BlockingScheduler()
            scheduler.add_job(
                pipeline.process_batch,
                'interval',
                minutes=config.get('download_interval_minutes', 60),
                args=[tickers]
            )
            scheduler.start()

    except Exception as e:
        print(f"Fatal error in main execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

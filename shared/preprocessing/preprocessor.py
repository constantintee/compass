# shared/preprocessing/preprocessor.py
"""
Main Preprocessor class that orchestrates all preprocessing operations.
"""

import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import pool, OperationalError
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from ..constants import RetryConfig, DatabaseConfig
from ..exceptions import PreprocessingError, DatabaseError
from ..technical_analysis import TechnicalAnalysis
from .validators import DataValidator
from .cleaners import DataCleaner
from .transformers import DataTransformer

load_dotenv()


def preprocess_stock_worker(args):
    """Worker function for parallel preprocessing."""
    ticker, training_config, individual_tfrecord_path, sequence_length, cache_dir, frequency = args
    try:
        preprocessor = Preprocessor(
            sequence_length=sequence_length,
            cache_dir=cache_dir,
            frequency=frequency,
            logger=None
        )
        result = preprocessor.preprocess_stock(ticker, training_config, individual_tfrecord_path)
        return result
    finally:
        if 'preprocessor' in locals():
            preprocessor.close_connection_pool()


class Preprocessor:
    """
    Main preprocessor class that coordinates all preprocessing operations.

    Handles data fetching, validation, cleaning, transformation, and storage.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        cache_dir: str = 'cache',
        frequency: str = 'B',
        logger: Optional[logging.Logger] = None
    ):
        self.sequence_length = sequence_length
        self.cache_dir = cache_dir
        self.frequency = frequency

        # Setup logger
        self.logger = logger or logging.getLogger('training_logger')

        # Initialize components
        self.validator = DataValidator(self.logger)
        self.cleaner = DataCleaner(self.logger)
        self.transformer = DataTransformer(sequence_length, cache_dir, self.logger)

        # Database connection
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_name = os.getenv('DB_NAME')

        self.connection_pool = None
        self._create_connection_pool()

        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(
            f"[Preprocessor] Initialized with sequence_length={sequence_length}, "
            f"cache_dir={cache_dir}, frequency={frequency}"
        )

    def _create_connection_pool(
        self,
        max_retries: int = RetryConfig.DB_MAX_RETRIES,
        retry_delay: int = RetryConfig.DB_RETRY_DELAY
    ) -> None:
        """Create database connection pool with retry logic."""
        for attempt in range(max_retries):
            try:
                self.connection_pool = pool.SimpleConnectionPool(
                    minconn=DatabaseConfig.MIN_CONNECTIONS,
                    maxconn=DatabaseConfig.MAX_CONNECTIONS,
                    host=self.db_host,
                    port=self.db_port,
                    user=self.db_user,
                    password=self.db_password,
                    dbname=self.db_name,
                    sslmode='disable'
                )
                self.logger.info("Database connection pool created successfully.")
                return
            except OperationalError as e:
                self.logger.error(
                    f"Attempt {attempt + 1}/{max_retries} to create connection pool failed: {e}"
                )
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Failed to create connection pool after maximum retries.")
                    raise DatabaseError(
                        operation='create_connection_pool',
                        message="Failed to create connection pool"
                    )

    def close_connection_pool(self) -> None:
        """Close the database connection pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Database connection pool closed.")

    def fetch_data_from_db(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch stock data from database."""
        max_retries = RetryConfig.DB_MAX_RETRIES
        retry_delay = RetryConfig.DB_RETRY_DELAY

        for attempt in range(max_retries):
            connection = None
            try:
                connection = self.connection_pool.getconn()
                self.logger.debug(f"Got connection from pool for ticker {ticker}.")

                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = """
                        SELECT date, open, high, low, close, volume, ticker
                        FROM stocks
                        WHERE ticker = %s
                        ORDER BY date ASC
                    """
                    cursor.execute(query, (ticker,))
                    records = cursor.fetchall()

                    if not records:
                        self.logger.warning(
                            f"No data found in the database for ticker {ticker}."
                        )
                        return None

                    df = pd.DataFrame(records)
                    self.logger.debug(f"Data fetched from database for ticker {ticker}.")
                    return df

            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Database connection failed for ticker {ticker}. "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    self.logger.error(
                        f"Error fetching data from database for ticker {ticker}: {e}"
                    )
                    return None
            finally:
                if connection:
                    self.connection_pool.putconn(connection)
                    self.logger.debug(f"Returned connection to pool for ticker {ticker}.")

        return None

    def preprocess_stock(
        self,
        ticker: str,
        training_config: Dict,
        individual_tfrecord_path: str
    ) -> Optional[str]:
        """
        Main preprocessing pipeline for a single stock.

        Args:
            ticker: Stock ticker symbol
            training_config: Training configuration dictionary
            individual_tfrecord_path: Path to save TFRecord files

        Returns:
            Ticker symbol if successful, None otherwise
        """
        try:
            self.logger.info(f"[Preprocessing] Starting preprocessing for ticker {ticker}")

            # Step 1: Load Data
            data = self.fetch_data_from_db(ticker)
            if data is None or data.empty:
                self.logger.warning(
                    f"[Preprocessing] No data found for ticker {ticker}. Skipping."
                )
                return None

            # Ensure 'date' is a column and datetime type
            if 'date' not in data.columns:
                data = data.reset_index()

            if not np.issubdtype(data['date'].dtype, np.datetime64):
                data['date'] = pd.to_datetime(data['date'])

            # Step 2: Validate raw data
            data = self.validator.validate_raw_data(data, ticker)
            if data.empty:
                self.logger.warning(
                    f"[Preprocessing] Ticker {ticker}: Data is empty after validation."
                )
                return None

            # Step 3: Handle missing values
            data = self.cleaner.handle_missing_values(data, ticker)

            # Step 4: Validate cleaned data
            data = self.validator.validate_data(data, ticker)
            if data.empty:
                self.logger.warning(
                    f"[Preprocessing] Ticker {ticker}: Data is empty after handling missing values."
                )
                return None

            # Validate volume
            if data['volume'].isna().all():
                self.logger.error(
                    f"[Preprocessing] Ticker {ticker}: 'volume' column is entirely NaN."
                )
                return None

            # Step 5: Ensure continuous dates
            data = self.cleaner.ensure_continuous_dates(data, ticker, self.frequency)

            # Step 6: Preprocess data
            data = self.cleaner.preprocess_data(data, ticker)
            if data.empty:
                self.logger.warning(
                    f"[Preprocessing] Ticker {ticker}: Data is empty after preprocessing."
                )
                return None

            # Step 7: Calculate technical indicators
            technical_analysis = TechnicalAnalysis(self.logger)
            data = technical_analysis.calculate_technical_indicators(data, ticker)
            if data.empty:
                self.logger.warning(
                    f"[Preprocessing] Ticker {ticker}: Data is empty after calculating indicators."
                )
                return None

            # Step 8: Optimize memory usage
            data = self.transformer.optimize_memory_usage(data)

            # Step 9: Generate sequences
            X, y = self.transformer.sequence_generator(data, ticker)
            if X is None or y is None:
                self.logger.warning(
                    f"[Preprocessing] Ticker {ticker}: Sequence generation failed."
                )
                return None

            # Step 10: Save processed data
            checksum = self.transformer.compute_checksum(data)
            self.transformer.save_processed_data(ticker, X, y, checksum)

            # Step 11: Write TFRecord
            tfrecord_path = os.path.join(individual_tfrecord_path, f"{ticker}.tfrecord")
            self.transformer.write_tfrecord(X, y, tfrecord_path)

            self.logger.info(f"[Preprocessing] Completed preprocessing for ticker {ticker}.")
            return ticker

        except PreprocessingError:
            raise
        except Exception as e:
            self.logger.error(
                f"[Preprocessing] Error during preprocessing for ticker {ticker}: {e}"
            )
            self.logger.debug(traceback.format_exc())
            return None

    def parallel_preprocess_stocks(
        self,
        stocks: List[str],
        training_config: Dict,
        individual_tfrecord_path: str
    ) -> List[str]:
        """
        Preprocess multiple stocks in parallel.

        Args:
            stocks: List of stock ticker symbols
            training_config: Training configuration dictionary
            individual_tfrecord_path: Path to save TFRecord files

        Returns:
            List of successfully processed ticker symbols
        """
        processed_stocks = []
        pool_size = min(training_config.get('pool_size', 8), 4)
        self.logger.info(
            f"[Preprocessing] Starting parallel preprocessing with pool size: {pool_size}"
        )

        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            futures = {
                executor.submit(
                    preprocess_stock_worker,
                    (
                        ticker,
                        training_config,
                        individual_tfrecord_path,
                        self.sequence_length,
                        self.cache_dir,
                        self.frequency
                    )
                ): ticker
                for ticker in stocks
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_stocks.append(result)
                        self.logger.info(
                            f"[Preprocessing] Successfully processed ticker {ticker}"
                        )
                    else:
                        self.logger.warning(
                            f"[Preprocessing] Skipped ticker {ticker} due to processing issues."
                        )
                except Exception as e:
                    self.logger.error(
                        f"[Preprocessing] Exception occurred while processing ticker {ticker}: {e}"
                    )
                    self.logger.debug(traceback.format_exc())

        if not processed_stocks:
            self.logger.error("[Preprocessing] No stocks were successfully processed. Exiting.")
            sys.exit(1)

        return processed_stocks

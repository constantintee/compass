# preprocessor.py

import os
import sys
import time
import hashlib
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from psycopg2 import OperationalError
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import h5py
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import traceback
#from numba import jit
#import dask.dataframe as dd
#from dask.distributed import Client
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import from shared module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.technical_analysis import TechnicalAnalysis

load_dotenv()

# Configure logging at the beginning of your script
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO or WARNING in production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def preprocess_stock_worker(args):
    ticker, training_config, individual_tfrecord_path, sequence_length, cache_dir, frequency = args
    try:
        # Create a new Preprocessor instance for this process
        preprocessor = Preprocessor(
            sequence_length=sequence_length,
            cache_dir=cache_dir,
            frequency=frequency,
            logger=None  # Configure logging in each process if needed
        )
        
        # Process the stock
        result = preprocessor.preprocess_stock(ticker, training_config, individual_tfrecord_path)
        
        return result
    finally:
        # Ensure the connection pool is closed for this process
        if 'preprocessor' in locals():
            preprocessor.close_connection_pool()


class Preprocessor:
    def __init__(self, sequence_length=60, cache_dir='cache', frequency='B', logger=None):
        self.sequence_length = sequence_length
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.frequency = frequency
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Logger setup
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('training_logger')
            self.logger.setLevel(logging.DEBUG)  # Adjust as needed

        # Database connection details from .env
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_name = os.getenv('DB_NAME')

        # Initialize connection pool
        self.connection_pool = None
        self.create_connection_pool()

        self.logger.info(f"[Preprocessor] Initialized with sequence_length={self.sequence_length}, cache_dir={self.cache_dir}, frequency={self.frequency}")

    def create_connection_pool(self, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            try:
                self.connection_pool = pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=20,  # Adjust this number based on your needs
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
                self.logger.error(f"Attempt {attempt + 1}/{max_retries} to create connection pool failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Failed to create connection pool after maximum retries.")
                    raise

    def close_connection_pool(self):
        """
        Closes the database connection pool.
        """
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            self.logger.info("Database connection pool closed.")
    
    def compute_checksum(self, data: pd.DataFrame) -> str:
        return hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()

    def load_processed_data(self, ticker: str, checksum: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{checksum}.h5")
        if os.path.exists(cache_file):
            try:
                with h5py.File(cache_file, 'r') as f:
                    X = np.array(f['X'])
                    y = np.array(f['y'])
                self.logger.info(f"Loaded processed data for ticker {ticker} from cache.")
                return X, y
            except Exception as e:
                self.logger.error(f"Error loading cache for ticker {ticker}: {e}")
                return None, None
        return None, None

    def save_processed_data(self, ticker: str, X: np.ndarray, y: np.ndarray, checksum: str):
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{checksum}.h5")
        try:
            if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
                self.logger.error("X or y contains non-numeric data. Cannot save to HDF5.")
                return
            with h5py.File(cache_file, 'w') as f:
                f.create_dataset('X', data=X)
                f.create_dataset('y', data=y)
            self.logger.info(f"Processed data and checksum saved for ticker {ticker}.")
        except Exception as e:
            self.logger.error(f"Error saving cache for ticker {ticker}: {e}")

    def fetch_data_from_db(self, ticker: str) -> Optional[pd.DataFrame]:
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Get a connection from the pool
                connection = self.connection_pool.getconn()
                self.logger.debug(f"Got connection from pool for ticker {ticker}.")

                # Fetch data using a cursor
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
                        self.logger.warning(f"No data found in the database for ticker {ticker}.")
                        return None
                    df = pd.DataFrame(records)
                self.logger.debug(f"Data fetched from database for ticker {ticker}.")
                return df
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Database connection failed for ticker {ticker}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Error fetching data from database for ticker {ticker}: {e}")
                    self.logger.debug(traceback.format_exc())
                    return None
            finally:
                if 'connection' in locals():
                    self.connection_pool.putconn(connection)
                    self.logger.debug(f"Returned connection to pool for ticker {ticker}.")

    def preprocess_data(self, df, ticker):
        """
        Preprocesses the data for a given ticker.

        Parameters:
        - df (pd.DataFrame): The dataframe containing stock data.
        - ticker (str): The ticker symbol.

        Returns:
        - pd.DataFrame: The cleaned and preprocessed dataframe.
        """
        self.logger.info(f"[Preprocessing] Starting preprocessing for ticker {ticker}")

        # a. Validate 'open' column
        valid_open_range = (0, 500000)
        initial_count = len(df)
        df = df[(df['open'] > valid_open_range[0]) & (df['open'] < valid_open_range[1])]
        filtered_count = len(df)
        if filtered_count < initial_count:
            self.logger.error(f"Ticker {ticker}: {initial_count - filtered_count} entries in 'open' are outside the range {valid_open_range}. These entries have been removed.")

        # b. Handle NaNs in 'open' and 'volume'
        critical_columns = ['open', 'volume']
        for col in critical_columns:
            if df[col].isnull().any():
                self.logger.warning(f"[Preprocessing] Ticker {ticker}: Found NaNs in '{col}' after conversion.")
                if col == 'open':
                    # Attempt multiple imputation strategies
                    df[col].fillna(df[col].mean(), inplace=True)
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
                elif col == 'volume':
                    df[col].fillna(df[col].median(), inplace=True)
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)

                # Check if NaNs still exist
                remaining_nans = df[col].isnull().sum()
                if remaining_nans > 0:
                    if col == 'open':
                        # As 'open' is critical, remove rows if NaNs persist
                        df.dropna(subset=['open'], inplace=True)
                        self.logger.error(f"Ticker {ticker}: Data still contains {df['open'].isnull().sum()} NaN values in 'open' after all filling attempts.")
                    elif col == 'volume':
                        # Fill remaining NaNs with median or remove based on business logic
                        median_volume = df['volume'].median()
                        df['volume'].fillna(median_volume, inplace=True)
                        if df['volume'].isnull().any():
                            df.dropna(subset=['volume'], inplace=True)
                            self.logger.error(f"Ticker {ticker}: Data still contains NaN values in 'volume' after all filling attempts.")

        # d. Final Check for NaNs
        if df.isnull().any().any():
            nan_columns = df.columns[df.isnull().any()].tolist()
            self.logger.error(f"[Preprocessing] Ticker {ticker}: Columns with NaNs after preprocessing: {nan_columns}")
            # Depending on requirements, decide to drop or handle them
            df.dropna(inplace=True)

        self.logger.info(f"[Preprocessing] Completed preprocessing for ticker {ticker}. Cleaned data contains {len(df)} records.")
        return df

    def validate_column_names(self, df: pd.DataFrame, required_columns: List[str], ticker: str) -> bool:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Ticker {ticker}: Missing required columns: {', '.join(missing_columns)}.")
            return False
        else:
            self.logger.info(f"Ticker {ticker}: All required columns are present.")
            return True

    def validate_raw_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validate raw data by checking for outliers and invalid values in price columns.
        """
        try:
            price_columns = ['open', 'high', 'low', 'close']
            price_thresholds = (0, 500000)
            for column in price_columns:
                if column in df.columns:
                    invalid_mask = (df[column] <= price_thresholds[0]) | (df[column] >= price_thresholds[1])
                    num_invalid = invalid_mask.sum()
                    if num_invalid > 0:
                        self.logger.error(f"Ticker {ticker}: {num_invalid} entries in '{column}' are outside the range {price_thresholds}.")
                        df.loc[invalid_mask, column] = np.nan

            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['volume'] = df['volume'].replace(0, np.nan)

            return df
        except Exception as e:
            self.logger.error(f"Error validating raw data for ticker {ticker}: {e}")
            raise

    def handle_missing_values(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        try:
            # Fill NaNs using forward and backward filling first
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

            # Handle remaining NaNs for the 'volume' column separately
            if 'volume' in df.columns:
                # If volume still contains NaNs, fill with the median value, which is more robust against outliers
                if df['volume'].isna().sum() > 0:
                    median_volume = df['volume'].median()
                    if np.isnan(median_volume):
                        self.logger.warning(f"[Preprocessing] Ticker {ticker}: 'volume' column median is NaN. Attempting to fill with a default value.")
                        # Decide on a default value or strategy
                        default_volume = 1  # Example default value
                        df['volume'].fillna(default_volume, inplace=True)
                    else:
                        self.logger.warning(f"[Preprocessing] Ticker {ticker}: Filling remaining NaNs in 'volume' column with median value: {median_volume}.")
                        df['volume'].fillna(median_volume, inplace=True)

                    # Check if NaNs still exist
                    if df['volume'].isna().sum() > 0:
                        self.logger.warning(f"Ticker {ticker}: 'volume' still contains NaNs after filling. Dropping these rows.")
                        df.dropna(subset=['volume'], inplace=True)
                        self.logger.error(f"Ticker {ticker}: Data still contains {df['volume'].isna().sum()} NaN values in 'volume' after all filling attempts.")
            
            # Fill remaining NaNs for numeric columns with the column means
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # For any remaining NaNs, log and optionally drop rows
            if df.isnull().values.any():
                remaining_nans = df.isnull().sum().sum()
                self.logger.error(f"[Preprocessing] Ticker {ticker}: Data still contains {remaining_nans} NaN values after all filling attempts.")
                self.logger.error(f"[Preprocessing] Columns with NaNs: {df.columns[df.isna().any()].tolist()}")
                df.dropna(inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"[Preprocessing] Error handling missing values for ticker {ticker}: {e}")
            raise

    def validate_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validate the cleaned data to ensure no invalid values are present.
        """
        try:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            critical_columns = ['close', 'high', 'low', 'open', 'volume']
            for column in critical_columns:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    if df[column].isna().sum() > 0:
                        self.logger.warning(f"[Preprocessing] Ticker {ticker}: Found NaNs in '{column}' after conversion. Attempting to fill with column mean.")
                        df[column].fillna(df[column].mean(), inplace=True)

            df = self.handle_missing_values(df, ticker)

            # Final check to ensure no columns contain NaNs
            if df.isnull().values.any():
                remaining_nans = df.isnull().sum().sum()
                self.logger.error(f"[Preprocessing] Ticker {ticker}: Data still contains {remaining_nans} NaN values after all validations. Skipping preprocessing.")
                return pd.DataFrame()

            return df
        except Exception as e:
            self.logger.error(f"Error validating data for ticker {ticker}: {e}")
            raise

    def ensure_continuous_dates(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Ensures that all dates are present in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The original data fetched from the database.
        - ticker (str): The ticker symbol being processed.

        Returns:
        - pd.DataFrame: DataFrame with continuous dates and missing values handled.
        """
        try:
            # Create a complete date range and reindex the data
            full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq=self.frequency)
            df.set_index('date', inplace=True)
            df = df.reindex(full_date_range)
            df.index.name = 'date'  # Set the index name to 'date'

            missing_dates = df[df['open'].isna()].index
            num_missing = len(missing_dates)
            if num_missing > 0:
                self.logger.warning(f"[Preprocessing] {num_missing} missing dates found for ticker {ticker}. Filling missing data.")

                # Fill missing data using a combination of interpolation and forward/backward filling
                df.interpolate(method='linear', inplace=True, limit_direction='both')
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)

            # If there are still any NaN values, fill them with column means
            if df.isnull().values.any():
                remaining_nans = df.isnull().sum().sum()
                self.logger.warning(f"[Preprocessing] Ticker {ticker}: {remaining_nans} NaN values remaining after all filling attempts. Filling with column means.")
                df.fillna(df.mean(), inplace=True)

            # Reset the index and ensure 'date' is a column
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"[Preprocessing] Error ensuring continuous dates for ticker {ticker}: {e}")
            self.logger.debug(traceback.format_exc())
            return df

    def preprocess_stock_wrapper(self, ticker: str, config: dict, individual_tfrecord_path: str) -> Optional[str]:
        """
        Wrapper function for preprocess_stock to handle exceptions in multithreading.
        """
        try:
            return self.preprocess_stock(ticker, config, individual_tfrecord_path)
        except Exception as e:
            self.logger.error(f"Error processing ticker {ticker}: {e}")
            return None

    def write_individual_tfrecord(self, X: np.ndarray, y: np.ndarray, tfrecord_path: str):
        """
        Write sequences of features and targets to a TFRecord file.

        Parameters:
        - X (np.ndarray): Array of feature sequences.
        - y (np.ndarray): Array of target values.
        - tfrecord_path (str): Path to the TFRecord file.
        """
        try:
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for features_seq, target_value in zip(X, y):
                    # Ensure that `features_seq` and `target_value` are properly formatted (float)
                    feature = {
                        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features_seq.flatten())),
                        'target': tf.train.Feature(float_list=tf.train.FloatList(value=[target_value]))
                    }
                    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example_proto.SerializeToString())
            self.logger.info(f"TFRecord written to {tfrecord_path} successfully.")
        except Exception as e:
            self.logger.error(f"Error writing TFRecord for path {tfrecord_path}: {e}")

    def parallel_preprocess_stocks(self, stocks: List[str], training_config: dict, individual_tfrecord_path: str) -> List[str]:
        processed_stocks = []
        pool_size = min(training_config.get('pool_size', 8), 4)
        self.logger.info(f"[Preprocessing] Starting parallel preprocessing with process pool size: {pool_size}")

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
                        self.logger.info(f"[Preprocessing] Successfully processed ticker {ticker}")
                    else:
                        self.logger.warning(f"[Preprocessing] Skipped ticker {ticker} due to processing issues.")
                except Exception as e:
                    self.logger.error(f"[Preprocessing] Exception occurred while processing ticker {ticker}: {e}")
                    self.logger.debug(traceback.format_exc())

        if not processed_stocks:
            self.logger.error("[Preprocessing] No stocks were successfully processed. Exiting.")
            sys.exit(1)  # Exit if no stocks could be processed

        return processed_stocks

    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast numeric data types to reduce memory usage.

        Parameters:
        - df (pd.DataFrame): DataFrame to optimize.

        Returns:
        - pd.DataFrame: Optimized DataFrame.
        """
        try:
            for col in df.select_dtypes(include=['float']):
                df[col] = pd.to_numeric(df[col], downcast='float')
            for col in df.select_dtypes(include=['int']):
                df[col] = pd.to_numeric(df[col], downcast='unsigned')
            self.logger.debug("Optimized memory usage by downcasting numeric data types.")
            return df
        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {e}")
            return df

    def sequence_generator(self, data: pd.DataFrame, ticker: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            # Identify non-numeric columns
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
            # Exclude non-numeric columns from features
            feature_columns = [col for col in data.columns if col not in ['date', 'ticker', 'Elliott_Wave', 'Wave_Degree'] + non_numeric_columns]

            X = []
            y = []

            # Generate sequences
            for i in range(self.sequence_length, len(data)):
                X_sequence = data[feature_columns].iloc[i - self.sequence_length:i].values
                y_value = data['close'].iloc[i]
                X.append(X_sequence)
                y.append(y_value)

            X = np.array(X)
            y = np.array(y)

            # Scale features and target
            # Reshape X to 2D for scaling, then reshape back to original shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.feature_scaler.fit_transform(X_reshaped).reshape(X.shape)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

            self.logger.debug(f"Sequence generation and scaling completed for ticker {ticker}.")
            return X_scaled, y_scaled
        except Exception as e:
            self.logger.error(f"[Preprocessing] Error during sequence generation for ticker {ticker}: {e}")
            self.logger.debug(traceback.format_exc())
            return None, None

    def preprocess_stock(self, ticker: str, training_config: dict, individual_tfrecord_path: str) -> Optional[str]:
        try:
            self.logger.info(f"[Preprocessing] Starting preprocessing for ticker {ticker}")

            # Step 1: Load Data
            data = self.fetch_data_from_db(ticker)
            if data is None or data.empty:
                self.logger.warning(f"[Preprocessing] No data found for ticker {ticker}. Skipping.")
                return None

            # Ensure 'date' is a column and of datetime type
            if 'date' not in data.columns:
                data.reset_index(inplace=True)
                self.logger.debug("Reset index to make 'date' a column.")

            if not np.issubdtype(data['date'].dtype, np.datetime64):
                data['date'] = pd.to_datetime(data['date'])
                self.logger.debug("Converted 'date' column to datetime.")

            # Step 2: Data Cleaning and Preprocessing
            data = self.validate_raw_data(data, ticker)
            if data.empty:
                self.logger.warning(f"[Preprocessing] Ticker {ticker}: Data is empty after validation. Skipping.")
                return None

            data = self.handle_missing_values(data, ticker)
            data = self.validate_data(data, ticker)
            if data.empty:
                self.logger.warning(f"[Preprocessing] Ticker {ticker}: Data is empty after handling missing values. Skipping.")
                return None

            # Validate 'volume' after preprocessing
            if data['volume'].isna().all():
                self.logger.error(f"[Preprocessing] Ticker {ticker}: 'volume' column is entirely NaN after preprocessing. Skipping.")
                return None

            # Ensure continuous dates
            data = self.ensure_continuous_dates(data, ticker)

            # Step 3: Preprocess Data
            data = self.preprocess_data(data, ticker)
            if data.empty:
                self.logger.warning(f"[Preprocessing] Ticker {ticker}: Data is empty after preprocessing. Skipping.")
                return None

            # Step 4: Calculate Technical Indicators (including support/resistance)
            technical_analysis = TechnicalAnalysis()
            data = technical_analysis.calculate_technical_indicators(data, ticker)
            if data.empty:
                self.logger.warning(f"[Preprocessing] Ticker {ticker}: Data is empty after calculating technical indicators. Skipping.")
                return None

            # Step 5: Optimize Memory Usage
            data = self.optimize_memory_usage(data)

            # Step 6: Feature Scaling and Sequence Generation
            X, y = self.sequence_generator(data, ticker)
            if X is None or y is None:
                self.logger.warning(f"[Preprocessing] Ticker {ticker}: Sequence generation failed. Skipping.")
                return None

            # Step 7: Save Processed Data
            checksum = self.compute_checksum(data)
            self.save_processed_data(ticker, X, y, checksum)

            # Step 8: Write TFRecord
            tfrecord_path = os.path.join(individual_tfrecord_path, f"{ticker}.tfrecord")
            self.write_individual_tfrecord(X, y, tfrecord_path)

            self.logger.info(f"[Preprocessing] Completed preprocessing for ticker {ticker}.")
            return ticker

        except Exception as e:
            self.logger.error(f"[Preprocessing] Error during preprocessing for ticker {ticker}: {e}")
            self.logger.debug(traceback.format_exc())
            return None
        
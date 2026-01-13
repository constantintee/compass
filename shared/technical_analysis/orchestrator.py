# shared/technical_analysis/orchestrator.py
"""
Main TechnicalAnalysis orchestrator class.

This module provides the main entry point for technical analysis,
coordinating indicator calculations, Elliott Wave analysis, and database operations.
"""

import json
import logging
import os
import traceback
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import psycopg2
from psycopg2 import extras

from ..constants import DatabaseConfig
from ..exceptions import DatabaseError, TechnicalAnalysisError
from .cache import TechnicalAnalysisCache
from .indicators import TechnicalIndicators
from .elliott_wave import AdvancedElliottWaveAnalysis


class TechnicalAnalysis:
    """
    Main orchestrator for technical analysis operations.

    Coordinates indicator calculations, Elliott Wave analysis,
    database storage, and caching.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('training_logger')
        self.db_connection = None
        self.cache = TechnicalAnalysisCache()
        self.indicators = TechnicalIndicators(self.logger)
        self.elliott_wave_analyzer = AdvancedElliottWaveAnalysis(self.logger)
        self.current_version = 1

        self._setup_db_connection()

    def _setup_db_connection(self) -> None:
        """Setup database connection with credential validation."""
        # Validate required environment variables
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        db_name = os.getenv('DB_NAME')

        required_vars = {
            'DB_HOST': db_host,
            'DB_PORT': db_port,
            'DB_USER': db_user,
            'DB_PASSWORD': db_password,
            'DB_NAME': db_name
        }

        missing_vars = [name for name, value in required_vars.items() if not value]
        if missing_vars:
            self.logger.error(
                f"Missing required database environment variables: {', '.join(missing_vars)}"
            )
            self.db_connection = None
            return

        try:
            # Use SSL mode from environment, defaulting to 'prefer'
            sslmode = os.getenv('DB_SSLMODE', 'prefer')
            self.db_connection = psycopg2.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                dbname=db_name,
                sslmode=sslmode
            )
            self._get_current_version()
        except psycopg2.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            self.db_connection = None

    def _get_current_version(self) -> int:
        """Get current version of technical indicators."""
        if self.db_connection is None:
            return 1

        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT MAX(version) FROM technical_indicators;")
                result = cursor.fetchone()
                self.current_version = result[0] if result and result[0] else 1
                return self.current_version
        except psycopg2.Error as e:
            self.logger.error(f"Error getting current version: {e}")
            return 1

    def calculate_technical_indicators(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given data.

        This is the main entry point for technical analysis calculations.

        Args:
            data: DataFrame with OHLCV columns
            ticker: Stock ticker symbol

        Returns:
            DataFrame with original data plus all calculated indicators
        """
        try:
            if data.empty:
                self.logger.error("Empty dataframe provided")
                return pd.DataFrame()

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Required: {required_columns}")
                return pd.DataFrame()

            # Handle NaN values using modern pandas methods
            if data[required_columns].isna().any().any():
                self.logger.warning("Data contains NaN values. Attempting to handle them...")
                data = data.copy()  # Avoid SettingWithCopyWarning
                data = data.ffill().bfill()

            # Ensure data index is proper DatetimeIndex for Elliott Wave analysis
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data = data.set_index('date')
                    data.index = pd.to_datetime(data.index)

            # Calculate base indicators
            data_with_indicators = self.indicators.calculate_all_indicators(data, ticker)

            if data_with_indicators.empty:
                return data

            # Calculate Elliott Wave patterns
            elliott_wave_df = self.elliott_wave_analyzer.identify_elliott_waves(data_with_indicators)

            if not elliott_wave_df.empty:
                result = pd.concat([data_with_indicators, elliott_wave_df], axis=1)
            else:
                result = data_with_indicators

            return result

        except TechnicalAnalysisError:
            raise
        except Exception as e:
            self.logger.error(f"Error during technical indicator calculations for {ticker}: {e}")
            self.logger.debug(traceback.format_exc())
            return data

    def store_technical_indicators(self, data: pd.DataFrame, ticker: str, config: Dict) -> bool:
        """Store calculated technical indicators with versioning."""
        if self.db_connection is None:
            self.logger.error("No database connection available")
            return False

        try:
            config_hash = self.cache.get_config_hash(config)

            # Check if configuration has changed
            if config_hash != self.cache.config_hash:
                self.current_version += 1
                self.cache.config_hash = config_hash

            with self.db_connection.cursor() as cursor:
                # Store configuration
                cursor.execute("""
                    INSERT INTO indicator_configs (version, config_hash, config)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (version) DO NOTHING;
                """, (self.current_version, config_hash, json.dumps(config)))

                # Prepare data for batch insertion
                records = []
                metadata = {
                    'config_hash': config_hash,
                    'calculation_parameters': config
                }

                for index, row in data.iterrows():
                    record = (
                        ticker,
                        index,
                        self.current_version,
                        *self._prepare_record(row),
                        json.dumps(metadata)
                    )
                    records.append(record)

                # Use efficient batch insertion
                extras.execute_batch(cursor, self._get_insert_sql(), records)

                self.db_connection.commit()
                self.logger.info(f"Stored technical indicators for {ticker} (version {self.current_version})")
                return True

        except psycopg2.Error as e:
            self.logger.error(f"Error storing technical indicators for {ticker}: {e}")
            if self.db_connection:
                self.db_connection.rollback()
            return False

    def load_technical_indicators(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        version: Optional[int] = None
    ) -> pd.DataFrame:
        """Load technical indicators with caching."""
        if self.db_connection is None:
            self.logger.error("No database connection available")
            return pd.DataFrame()

        try:
            version = version or self.current_version
            date_range = (start_date, end_date)

            # Check cache first
            cached_data = self.cache.get_cached_data(ticker, date_range, str(version))
            if cached_data is not None:
                return cached_data

            # Load from database
            query = """
                SELECT * FROM technical_indicators
                WHERE ticker = %s
                AND date BETWEEN %s AND %s
                AND version = %s
                ORDER BY date;
            """

            data = pd.read_sql_query(
                query,
                self.db_connection,
                params=(ticker, start_date, end_date, version),
                index_col='date',
                parse_dates=['date']
            )

            # Cache the result
            self.cache.set_cached_data(ticker, date_range, str(version), data)

            return data

        except psycopg2.Error as e:
            self.logger.error(f"Error loading technical indicators for {ticker}: {e}")
            return pd.DataFrame()

    def cleanup_old_versions(self, keep_versions: int = 3) -> None:
        """Cleanup old indicator versions."""
        if self.db_connection is None:
            self.logger.error("No database connection available")
            return

        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM technical_indicators
                    WHERE version < (
                        SELECT MAX(version) - %s
                        FROM technical_indicators
                    );
                """, (keep_versions,))

                self.db_connection.commit()
                self.logger.info(f"Cleaned up old versions, keeping last {keep_versions} versions")

        except psycopg2.Error as e:
            self.logger.error(f"Error cleaning up old versions: {e}")
            if self.db_connection:
                self.db_connection.rollback()

    def _prepare_record(self, row: pd.Series) -> tuple:
        """Prepare a row for database insertion."""
        columns = [
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'CCI', 'ATR',
            'SuperTrend', 'SuperTrend_Direction', 'ZigZag', 'ZigZag_Highs', 'ZigZag_Lows',
            'Support', 'Resistance', 'Elliott_Wave', 'Wave_Degree', 'Wave_Type',
            'Wave_Number', 'Wave_Confidence'
        ]
        return tuple(row.get(col) for col in columns)

    def _get_insert_sql(self) -> str:
        """Get the SQL statement for inserting technical indicators."""
        return """
            INSERT INTO technical_indicators (
                ticker, date, version,
                ema_12, ema_26, macd, macd_signal, macd_histogram,
                rsi, bb_upper, bb_middle, bb_lower, cci, atr,
                supertrend, supertrend_direction, zigzag, zigzag_highs, zigzag_lows,
                support, resistance, elliott_wave, wave_degree, wave_type,
                wave_number, wave_confidence, metadata
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (ticker, date, version) DO UPDATE SET
                ema_12 = EXCLUDED.ema_12,
                ema_26 = EXCLUDED.ema_26,
                macd = EXCLUDED.macd,
                macd_signal = EXCLUDED.macd_signal,
                macd_histogram = EXCLUDED.macd_histogram,
                rsi = EXCLUDED.rsi,
                bb_upper = EXCLUDED.bb_upper,
                bb_middle = EXCLUDED.bb_middle,
                bb_lower = EXCLUDED.bb_lower,
                cci = EXCLUDED.cci,
                atr = EXCLUDED.atr,
                supertrend = EXCLUDED.supertrend,
                supertrend_direction = EXCLUDED.supertrend_direction,
                zigzag = EXCLUDED.zigzag,
                zigzag_highs = EXCLUDED.zigzag_highs,
                zigzag_lows = EXCLUDED.zigzag_lows,
                support = EXCLUDED.support,
                resistance = EXCLUDED.resistance,
                elliott_wave = EXCLUDED.elliott_wave,
                wave_degree = EXCLUDED.wave_degree,
                wave_type = EXCLUDED.wave_type,
                wave_number = EXCLUDED.wave_number,
                wave_confidence = EXCLUDED.wave_confidence,
                metadata = EXCLUDED.metadata;
        """

    def close(self) -> None:
        """Close database connection and clean up resources."""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
        self.cache.clear_cache()

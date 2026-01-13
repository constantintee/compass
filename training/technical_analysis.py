# technical_analysis.py

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from scipy import stats
from scipy.signal import find_peaks
import traceback

#from functools import lru_cache
import hashlib
import json

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from psycopg2 import OperationalError

from talipp.indicators import EMA, MACD, RSI, CCI, ATR
from talipp.indicators.SuperTrend import SuperTrend, Trend
from talipp.indicators.BB import BB as BollingerBands
from talipp.indicators.ZigZag import ZigZag, PivotType
from talipp.ohlcv import OHLCV, OHLCVFactory


class DBConnectionManager:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        if self._pool is None:
            self._pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=20,
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                dbname=os.getenv('DB_NAME')
            )

    def get_connection(self):
        return self._pool.getconn()

    def return_connection(self, conn):
        self._pool.putconn(conn)

    def close_all(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None


class TechnicalAnalysisCache:
    """Caching system for technical analysis calculations"""
    
    def __init__(self, cache_size: int = 128):
        self.cache = {}
        self.cache_size = cache_size
        self._config_hashes = {}

    def get_config_hash(self, config: Dict) -> str:
        """Generate a deterministic hash for a configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_cached_data(
        self, 
        ticker: str, 
        date_range: Tuple[datetime, datetime],
        config_hash: str
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available"""
        cache_key = (ticker, date_range, config_hash)
        return self.cache.get(cache_key)

    def set_cached_data(
        self,
        ticker: str,
        date_range: Tuple[datetime, datetime],
        config_hash: str,
        data: pd.DataFrame
    ):
        """Store data in cache with LRU eviction"""
        cache_key = (ticker, date_range, config_hash)
        
        # Implement simple LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[cache_key] = data

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self._config_hashes.clear()


class TechnicalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger('training_logger')
        self.db_connection = None
        self.cache = TechnicalAnalysisCache()
        self.setup_db_connection()
#        self.setup_technical_indicators_table()
        self.current_version = self._get_current_version()

    def setup_db_connection(self):
        """Setup database connection using existing pool"""
        try:
            self.db_connection = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                dbname=os.getenv('DB_NAME')
            )
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")

    def setup_technical_indicators_table(self):
        """Setup TimescaleDB hypertable for technical indicators"""
        try:
            with self.db_connection.cursor() as cursor:
                # First check if table exists
                cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'technical_indicators'
                );
                """)
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    # Create the base table first
                    cursor.execute("""
                    CREATE TABLE technical_indicators (
                        date TIMESTAMPTZ NOT NULL,  -- Changed from 'time' to 'date'
                        ticker TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        ema_12 DOUBLE PRECISION,
                        ema_26 DOUBLE PRECISION,
                        macd DOUBLE PRECISION,
                        macd_signal DOUBLE PRECISION,
                        macd_histogram DOUBLE PRECISION,
                        rsi DOUBLE PRECISION,
                        bb_upper DOUBLE PRECISION,
                        bb_middle DOUBLE PRECISION,
                        bb_lower DOUBLE PRECISION,
                        cci DOUBLE PRECISION,
                        atr DOUBLE PRECISION,
                        supertrend DOUBLE PRECISION,
                        supertrend_direction INTEGER,
                        zigzag DOUBLE PRECISION,
                        zigzag_highs DOUBLE PRECISION,
                        zigzag_lows DOUBLE PRECISION,
                        support DOUBLE PRECISION,
                        resistance DOUBLE PRECISION,
                        elliott_wave INTEGER,
                        wave_degree TEXT,
                        wave_type TEXT,
                        wave_number INTEGER,
                        wave_confidence DOUBLE PRECISION,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    """)

                    # Create hypertable
                    cursor.execute("""
                    SELECT create_hypertable('technical_indicators', 'date');
                    """)

                    self.db_connection.commit()

                    # Now create the materialized view
                    cursor.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS technical_indicators_daily
                    WITH (timescaledb.continuous) AS
                    SELECT time_bucket('1 day', date) AS bucket,  -- Changed from 'time' to 'date'
                        ticker,
                        version,
                        AVG(ema_12) as ema_12,
                        AVG(ema_26) as ema_26,
                        AVG(macd) as macd,
                        AVG(rsi) as rsi,
                        MAX(zigzag_highs) as zigzag_highs,
                        MIN(zigzag_lows) as zigzag_lows
                    FROM technical_indicators
                    GROUP BY bucket, ticker, version;
                    """)

                    # Add policies after view creation
                    cursor.execute("""
                    SELECT add_continuous_aggregate_policy('technical_indicators_daily',
                        start_offset => INTERVAL '3 days',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour');
                    """)

                    cursor.execute("""
                    ALTER TABLE technical_indicators SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'ticker,version',
                        timescaledb.compress_orderby = 'date DESC'
                    );
                    """)

                    cursor.execute("""
                    SELECT add_compression_policy('technical_indicators', 
                        compress_after => INTERVAL '7 days');
                    """)

                    cursor.execute("""
                    SELECT add_retention_policy('technical_indicators', 
                        drop_after => INTERVAL '5 years');
                    """)

                    # Create indexes
                    cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tech_ind_ticker_date 
                    ON technical_indicators (ticker, date DESC);
                    """)

                self.db_connection.commit()
                self.logger.info("[Technical Analysis] Technical indicators hypertable setup complete")

        except Exception as e:
            self.logger.error(f"Error setting up technical indicators table: {e}")
            self.db_connection.rollback()

    def _get_current_version(self) -> int:
        """Get current version of technical indicators"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                SELECT MAX(version) FROM technical_indicators;
                """)
                version = cursor.fetchone()[0]
                return version if version else 1
        except Exception as e:
            self.logger.error(f"Error getting current version: {e}")
            return 1

    def store_technical_indicators(self, data: pd.DataFrame, ticker: str, config: Dict) -> bool:
        """Store calculated technical indicators with versioning"""
        try:
            config_hash = self.cache.get_config_hash(config)
            
            # Check if configuration has changed
            if config_hash != self.cache._config_hash:
                self.current_version += 1
                self.cache._config_hash = config_hash

            with self.db_connection.cursor() as cursor:
                # Store configuration
                cursor.execute("""
                INSERT INTO indicator_configs (version, config_hash, config)
                VALUES (%s, %s, %s)
                ON CONFLICT (version) DO NOTHING;
                """, (self.current_version, config_hash, json.dumps(config)))

                # Prepare data for batch insertion using vectorized approach
                metadata = {
                    'config_hash': config_hash,
                    'calculation_parameters': config
                }
                metadata_json = json.dumps(metadata)

                # Define columns for extraction (matching _get_insert_sql column order)
                indicator_columns = [
                    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                    'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'CCI', 'ATR',
                    'SuperTrend', 'SuperTrend_Direction', 'ZigZag', 'ZigZag_Highs', 'ZigZag_Lows',
                    'Support', 'Resistance', 'Elliott_Wave', 'Wave_Degree', 'Wave_Type',
                    'Wave_Number', 'Wave_Confidence'
                ]

                # Use itertuples() which is ~100x faster than iterrows()
                records = [
                    (
                        ticker,
                        row.Index,
                        self.current_version,
                        *[getattr(row, col, None) if col in data.columns else None for col in indicator_columns],
                        metadata_json
                    )
                    for row in data.itertuples()
                ]

                # Use efficient batch insertion
                extras.execute_batch(cursor, self._get_insert_sql(), records)
                
                # Add compression policy for this chunk
                cursor.execute("""
                SELECT add_compression_policy('technical_indicators', 
                    INTERVAL '1 week',
                    if_not_exists => TRUE);
                """)

                self.db_connection.commit()
                self.logger.info(f"Stored technical indicators for {ticker} (version {self.current_version})")
                return True

        except Exception as e:
            self.logger.error(f"Error storing technical indicators for {ticker}: {e}")
            self.db_connection.rollback()
            return False

    def load_technical_indicators(self, ticker: str, start_date: datetime, end_date: datetime, version: Optional[int] = None) -> pd.DataFrame:
        """Load technical indicators with caching"""
        try:
            version = version or self.current_version
            date_range = (start_date, end_date)
            
            # Check cache first
            cached_data = self.cache.get_cached_data(ticker, date_range, version)
            if cached_data is not None:
                return cached_data

            # If not in cache, load from database
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
            self.cache.set_cached_data(ticker, date_range, version, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading technical indicators for {ticker}: {e}")
            return pd.DataFrame()

    def cleanup_old_versions(self, keep_versions: int = 3):
        """Cleanup old indicator versions"""
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
        except Exception as e:
            self.logger.error(f"Error cleaning up old versions: {e}")
            self.db_connection.rollback()

    def calculate_technical_indicators(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        try:
            # First check for empty data and required columns
            if data.empty:
                self.logger.error("Empty dataframe provided")
                return pd.DataFrame()

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Required: {required_columns}")
                return pd.DataFrame()

            # Handle NaN values
            if data[required_columns].isna().any().any():
                self.logger.warning("Data contains NaN values. Attempting to handle them...")
                data = data.fillna(method='ffill').fillna(method='bfill')            

            # Calculate support and resistance first
            support_resistance_df = self.calculate_support_resistance(data)

            # Initialize OHLCV objects using numpy arrays (faster than iterrows)
            opens = data['open'].values
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values

            ohlcv_data = [
                OHLCV(
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=float(volumes[i])
                )
                for i in range(len(data))
            ]
            
            # Create OHLCV objects using OHLCVFactory
            ohlcv = ohlcv_data  # Direct list of OHLCV objects

            # Initialize indicators
            ema_12 = EMA(12)
            ema_26 = EMA(26)
            macd = MACD(12, 26, 9)
            rsi = RSI(14)
            bb = BollingerBands(20, 2)
            cci = CCI(20)
            atr = ATR(14)
            supertrend = SuperTrend(10, 3)
            zigzag = ZigZag(
                sensitivity=0.05,    # 5% sensitivity for trend changes
                min_trend_length=5,  # Minimum 5 bars between pivots
                input_values=ohlcv,  # Added comma here
                input_modifier=None,
                input_sampling=None
            )

            # Initialize arrays to store indicator values with NaN
            ema_12_vals = np.full(len(data), np.nan)
            ema_26_vals = np.full(len(data), np.nan)
            macd_vals = np.full(len(data), np.nan)
            macd_signal_vals = np.full(len(data), np.nan)
            macd_hist_vals = np.full(len(data), np.nan)
            rsi_vals = np.full(len(data), np.nan)
            bb_upper_vals = np.full(len(data), np.nan)
            bb_middle_vals = np.full(len(data), np.nan)
            bb_lower_vals = np.full(len(data), np.nan)
            cci_vals = np.full(len(data), np.nan)
            atr_vals = np.full(len(data), np.nan)
            supertrend_vals = np.full(len(data), np.nan)
            supertrend_dir_vals = np.full(len(data), 0)
            zigzag_vals = np.full(len(data), np.nan)
            zigzag_highs = np.full(len(data), np.nan)
            zigzag_lows = np.full(len(data), np.nan)

            # Calculate indicators for each candle
            for i, candle in enumerate(ohlcv):
                # Add input values to indicators
                ema_12.add_input_value(candle.close)
                ema_26.add_input_value(candle.close)
                macd.add_input_value(candle.close)
                rsi.add_input_value(candle.close)
                bb.add_input_value(candle.close)
                cci.add_input_value(candle)
                atr.add_input_value(candle)
                supertrend.add_input_value(candle)

                # Get indicator values
                if len(ema_12.output_values) > 0:
                    ema_12_vals[i] = ema_12.output_values[-1]
                if len(ema_26.output_values) > 0:
                    ema_26_vals[i] = ema_26.output_values[-1]
                
                if len(macd.output_values) > 0 and macd.output_values[-1] is not None:
                    macd_vals[i] = macd.output_values[-1].macd
                    macd_signal_vals[i] = macd.output_values[-1].signal
                    macd_hist_vals[i] = macd.output_values[-1].histogram
                
                if len(rsi.output_values) > 0:
                    rsi_vals[i] = rsi.output_values[-1]
                
                # Handle Bollinger Bands output values
                if len(bb.output_values) > 0 and bb.output_values[-1] is not None:
                    bb_val = bb.output_values[-1]
                    bb_upper_vals[i] = bb_val.ub
                    bb_middle_vals[i] = bb_val.cb
                    bb_lower_vals[i] = bb_val.lb
                
                if len(cci.output_values) > 0:
                    cci_vals[i] = cci.output_values[-1]
                
                if len(atr.output_values) > 0:
                    atr_vals[i] = atr.output_values[-1]
                    
                if len(supertrend.output_values) > 0 and supertrend.output_values[-1] is not None:
                    # Access the correct attributes from SuperTrendVal
                    st_val = supertrend.output_values[-1]  # This is a SuperTrendVal object
                    supertrend_vals[i] = st_val.value  # Access the value attribute
                    # Convert trend to direction (-1 for DOWN, 1 for UP)
                    supertrend_dir_vals[i] = 1 if st_val.trend == Trend.UP else -1

            # Process ZigZag outputs after all inputs are processed
            for output in zigzag.output_values:
                if output is not None:
                    # Get the index by matching the OHLCV object
                    try:
                        idx = ohlcv.index(output.ohlcv)
                        if output.type == PivotType.HIGH:
                            zigzag_vals[idx] = output.ohlcv.high
                            zigzag_highs[idx] = output.ohlcv.high
                        elif output.type == PivotType.LOW:
                            zigzag_vals[idx] = output.ohlcv.low
                            zigzag_lows[idx] = output.ohlcv.low
                    except ValueError:
                        self.logger.warning(f"Could not find matching OHLCV object for ZigZag output")

            base_indicators_df = pd.DataFrame({
                'EMA_12': ema_12_vals,
                'EMA_26': ema_26_vals,
                'MACD': macd_vals,
                'MACD_Signal': macd_signal_vals,
                'MACD_Histogram': macd_hist_vals,
                'RSI': rsi_vals,
                'BB_Upper': bb_upper_vals,
                'BB_Middle': bb_middle_vals,
                'BB_Lower': bb_lower_vals,
                'CCI': cci_vals,
                'ATR': atr_vals,
                'SuperTrend': supertrend_vals,
                'SuperTrend_Direction': supertrend_dir_vals,
                'ZigZag': zigzag_vals,
                'ZigZag_Highs': zigzag_highs,
                'ZigZag_Lows': zigzag_lows
            }, index=data.index)

            # Combine data with base indicators and support/resistance
            data_with_indicators = pd.concat([
                data,
                base_indicators_df,
                support_resistance_df
            ], axis=1)

            # Calculate custom indicators
            rsi_divergence_df = self.calculate_rsi_divergence(data_with_indicators)
            fibonacci_levels_df = self.calculate_fibonacci_levels(data_with_indicators)
            fibonacci_ext_df = self.calculate_fibonacci_extensions(data_with_indicators)
            obv_df = self.calculate_on_balance_volume(data_with_indicators)
            peaks_troughs_df = self.calculate_peaks_troughs(data_with_indicators)
            pivot_points_df = self.calculate_pivot_points(data_with_indicators)
            elliott_wave_df = self.identify_elliott_waves(data_with_indicators)

            # Combine all indicators
            result = pd.concat([
                data_with_indicators,
                rsi_divergence_df,
                fibonacci_levels_df,
                fibonacci_ext_df,
                obv_df,
                peaks_troughs_df,
                pivot_points_df,
                elliott_wave_df
            ], axis=1)

#            store = self.store_technical_indicators(result, ticker, )

            return result

        except Exception as e:
            self.logger.error(f"Error during technical indicator calculations for {ticker}: {e}")
            self.logger.debug(traceback.format_exc())
            return data

    def calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        support_resistance = pd.DataFrame(index=df.index)
        support_resistance['Support'] = df['low'].rolling(window=20).min()
        support_resistance['Resistance'] = df['high'].rolling(window=20).max()
        return support_resistance

    def calculate_rsi_divergence(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        if 'RSI' not in data.columns:
            self.logger.warning("RSI column not found. Skipping RSI divergence calculation.")
            return df
                
        price_diff = data['close'].diff()
        rsi_diff = data['RSI'].diff()
        df['RSI_Divergence'] = np.select(
            [
                (price_diff > 0) & (rsi_diff < 0),
                (price_diff < 0) & (rsi_diff > 0)
            ],
            [-1, 1],
            default=0
        )
        return df

    def calculate_fibonacci_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        high = data['high'].rolling(window=14).max()
        low = data['low'].rolling(window=14).min()
        diff = high - low
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in levels:
            df[f'Fib_{level}'] = low + diff * level
        return df

    def calculate_fibonacci_extensions(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        high = data['high'].rolling(window=14).max()
        low = data['low'].rolling(window=14).min()
        diff = high - low
        levels = [1.618, 2.618, 3.618]
        for level in levels:
            df[f'FibExt_{level}'] = high + diff * (level - 1)
        return df

    def calculate_on_balance_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        df['OBV'] = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        return df

    def calculate_peaks_troughs(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        close_prices = data['close'].values
        peaks, _ = find_peaks(close_prices, distance=5)
        troughs, _ = find_peaks(-close_prices, distance=5)
        
        df['Peak'] = np.nan
        df['Trough'] = np.nan
        df.loc[peaks, 'Peak'] = close_prices[peaks]
        df.loc[troughs, 'Trough'] = close_prices[troughs]
        
        return df

    def calculate_pivot_points(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        df['Pivot_Point'] = (data['high'] + data['low'] + data['close']) / 3
        df['R1'] = 2 * df['Pivot_Point'] - data['low']
        df['S1'] = 2 * df['Pivot_Point'] - data['high']
        return df
    
    def identify_elliott_waves(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Initialize AdvancedElliottWaveAnalysis
            elliott_wave_analyzer = AdvancedElliottWaveAnalysis()
            
            # Perform Elliott Wave analysis
            elliott_wave_df = elliott_wave_analyzer.identify_elliott_waves(data)
            
            return elliott_wave_df
        except Exception as e:
            self.logger.error(f"Error identifying Elliott waves: {e}")
            self.logger.debug(traceback.format_exc())
            return pd.DataFrame(index=data.index)        


class TechnicalAnalysisCache:
    def __init__(self, cache_size=128):
        self.cache = {}
        self._config_hash = None

    def get_config_hash(self, config: Dict) -> str:
        """Generate hash for indicator configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @lru_cache(maxsize=128)
    def get_cached_data(self, ticker: str, date_range: Tuple[datetime, datetime], version: int) -> Optional[pd.DataFrame]:
        cache_key = (ticker, date_range, version)
        return self.cache.get(cache_key)

    def set_cached_data(self, ticker: str, date_range: Tuple[datetime, datetime], version: int, data: pd.DataFrame):
        cache_key = (ticker, date_range, version)
        self.cache[cache_key] = data

    def clear_cache(self):
        self.cache.clear()
        self.get_cached_data.cache_clear()

class WavePatternType:
    """Enum-like class for wave pattern types"""
    # Impulse Patterns
    IMPULSE_BULL = "Impulse (Bull)"
    IMPULSE_BEAR = "Impulse (Bear)"
    IMPULSE_LEADING_DIAGONAL = "Leading Diagonal"
    IMPULSE_ENDING_DIAGONAL = "Ending Diagonal"
    
    # Corrective Patterns
    CORRECTION_ZIGZAG = "Zigzag Correction"
    CORRECTION_FLAT = "Flat Correction"
    CORRECTION_TRIANGLE = "Triangle Correction"
    CORRECTION_DOUBLE_ZIGZAG = "Double Zigzag"
    CORRECTION_DOUBLE_THREE = "Double Three"
    CORRECTION_TRIPLE_THREE = "Triple Three"
    
    # Complex Patterns
    COMPLEX_WXY = "WXY Pattern"
    COMPLEX_COMBINATION = "Complex Combination"

class WaveDegree:
    """Enum-like class for wave degrees"""
    GRAND_SUPERCYCLE = "Grand Supercycle"
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"
    SUBMINUETTE = "Subminuette"

class WaveRules:
    """Class containing Elliott Wave rules and guidelines"""
    
    @staticmethod
    def check_impulse_rules(pattern: Dict) -> Tuple[bool, List[str]]:
        """
        Check if pattern follows impulse wave rules.
        Returns (is_valid, list_of_violations)
        """
        violations = []
        try:
            magnitudes = pattern['magnitudes']
            points = pattern['points']
            directions = pattern['directions']

            # Rule 1: Wave 2 never retraces more than 100% of Wave 1
            if magnitudes[1] > magnitudes[0]:
                violations.append("Wave 2 retraced more than 100% of Wave 1")
            
            # Rule 2: Wave 3 is never the shortest among waves 1, 3, and 5
            if magnitudes[2] <= min(magnitudes[0], magnitudes[4]):
                violations.append("Wave 3 is the shortest among waves 1, 3, and 5")
            
            # Rule 3: Wave 4 never overlaps with Wave 1
            wave1_high = max(points[0]['price'], points[1]['price'])
            wave4_low = min(points[3]['price'], points[4]['price'])
            if wave4_low <= wave1_high:
                violations.append("Wave 4 overlaps with Wave 1 territory")

            # Additional wave relationships
            # Wave 3 should be at least 1.618 times Wave 1
            if magnitudes[2] < 1.618 * magnitudes[0]:
                violations.append("Wave 3 is not extended enough (should be >= 1.618 * Wave 1)")

            # Wave 5 should be at least 0.618 times Wave 1
            if magnitudes[4] < 0.618 * magnitudes[0]:
                violations.append("Wave 5 is too short (should be >= 0.618 * Wave 1)")

            # Check alternation between waves 2 and 4
            wave2_time = (points[1]['date'] - points[0]['date']).days
            wave4_time = (points[3]['date'] - points[2]['date']).days
            if abs(wave2_time - wave4_time) / max(wave2_time, wave4_time) < 0.382:
                violations.append("Waves 2 and 4 lack alternation in time")

            return len(violations) == 0, violations

        except Exception as e:
            return False, [f"Error checking impulse rules: {str(e)}"]

    @staticmethod
    def check_diagonal_rules(pattern: Dict) -> Tuple[bool, List[str]]:
        """Check if pattern follows diagonal rules"""
        violations = []
        try:
            magnitudes = pattern['magnitudes']
            points = pattern['points']
            directions = pattern['directions']

            # Leading Diagonal Rules
            # 5-3-5-3-5 structure for leading diagonal
            wave_lengths = [len(w) for w in pattern.get('subwaves', [[]] * 5)]
            expected_lengths = [5, 3, 5, 3, 5]
            
            # Check wave structure
            if wave_lengths == expected_lengths:
                # Check converging channel
                wave1_width = abs(points[1]['price'] - points[0]['price'])
                wave5_width = abs(points[4]['price'] - points[3]['price'])
                if wave5_width >= wave1_width:
                    violations.append("Diagonal trendlines not converging")

                # Check wave relationships
                if magnitudes[2] >= magnitudes[0]:  # Wave 3 shorter than Wave 1
                    violations.append("Wave 3 not shorter than Wave 1 in diagonal")
                if magnitudes[4] >= magnitudes[2]:  # Wave 5 shorter than Wave 3
                    violations.append("Wave 5 not shorter than Wave 3 in diagonal")

            # Ending Diagonal Rules
            # 3-3-3-3-3 structure for ending diagonal
            elif all(length == 3 for length in wave_lengths):
                # Check converging channel
                if wave5_width >= wave1_width:
                    violations.append("Ending diagonal trendlines not converging")

                # Volume characteristics
                if pattern.get('volume_profile') != 'declining':
                    violations.append("Volume not declining in ending diagonal")

            else:
                violations.append("Wave structure doesn't match diagonal pattern")

            return len(violations) == 0, violations

        except Exception as e:
            return False, [f"Error checking diagonal rules: {str(e)}"]

    @staticmethod
    def check_corrective_rules(pattern: Dict) -> Tuple[bool, List[str]]:
        """Check if pattern follows corrective wave rules"""
        violations = []
        try:
            magnitudes = pattern['magnitudes']
            points = pattern['points']
            pattern_type = pattern.get('corrective_type', '')

            if pattern_type == 'Zigzag':
                # A-B-C pattern where B retraces 50-79% of A
                b_retracement = magnitudes[1] / magnitudes[0]
                if not (0.5 <= b_retracement <= 0.79):
                    violations.append("B wave retracement out of range for Zigzag")
                
                # C wave should be 61.8-161.8% of A wave
                c_extension = magnitudes[2] / magnitudes[0]
                if not (0.618 <= c_extension <= 1.618):
                    violations.append("C wave extension out of range for Zigzag")

            elif pattern_type == 'Flat':
                # A-B-C pattern where B retraces 90-105% of A
                b_retracement = magnitudes[1] / magnitudes[0]
                if not (0.9 <= b_retracement <= 1.05):
                    violations.append("B wave retracement out of range for Flat")
                
                # C wave should be 100-165% of A wave
                c_extension = magnitudes[2] / magnitudes[0]
                if not (1.0 <= c_extension <= 1.65):
                    violations.append("C wave extension out of range for Flat")

            elif pattern_type == 'Triangle':
                # Check converging trendlines
                if not pattern.get('converging_trendlines', False):
                    violations.append("Triangle trendlines not converging")
                
                # Check decreasing wave lengths
                for i in range(1, len(magnitudes)):
                    if magnitudes[i] >= magnitudes[i-1]:
                        violations.append(f"Wave {i+1} not shorter than previous wave in Triangle")

            return len(violations) == 0, violations

        except Exception as e:
            return False, [f"Error checking corrective rules: {str(e)}"]
        

class AdvancedElliottWaveAnalysis:
    def __init__(self):
        self.logger = logging.getLogger('training_logger')

    def identify_elliott_waves(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to identify Elliott Wave patterns using pre-calculated indicators.
        """
        try:
            self.logger.info("Identifying Elliott waves with advanced analysis...")
            
            required_columns = ['ZigZag', 'Support', 'Resistance']
            missing_columns = [col for col in required_columns if col not in data.columns]
                
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame(index=data.index)

            # Validate input data
            if not self._validate_input_data(data):
                return pd.DataFrame(index=data.index)

            # Get significant swing points from ZigZag
            swing_points = self._get_swing_points(data)
            if swing_points.empty:
                return pd.DataFrame(index=data.index)

            # Find all potential patterns
            wave_patterns = self._find_all_patterns(swing_points, data)
            
            # Create comprehensive Elliott Wave DataFrame
            elliott_wave_df = self.create_elliott_wave_dataframe(wave_patterns, data.index)
            
            return elliott_wave_df

        except Exception as e:
            self.logger.error(f"Error in Elliott Wave identification: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return pd.DataFrame(index=data.index)

    def extract_wave_patterns(self, swing_points: pd.DataFrame, support_resistance: pd.DataFrame) -> List[Dict]:
        """
        Extract potential Elliott Wave patterns from swing points.
        """
        wave_patterns = []

        # Pre-extract arrays for faster access
        zigzag_vals = swing_points['ZigZag'].values
        indices = swing_points.index

        for i in range(len(swing_points) - 4):
            window_indices = indices[i:i+5]
            window_zigzag = zigzag_vals[i:i+5]

            pattern = {
                'points': [],
                'directions': [],
                'magnitudes': [],
                'support_levels': [],
                'resistance_levels': []
            }

            prev_zigzag = None
            # Use index-based iteration instead of iterrows()
            for j in range(len(window_indices)):
                idx = window_indices[j]
                current_zigzag = window_zigzag[j]

                if prev_zigzag is not None:
                    direction = 1 if current_zigzag > prev_zigzag else -1
                    magnitude = abs(current_zigzag - prev_zigzag)

                    pattern['points'].append({
                        'date': idx,
                        'price': current_zigzag,
                        'type': 'HIGH' if direction > 0 else 'LOW'
                    })
                    pattern['directions'].append(direction)
                    pattern['magnitudes'].append(magnitude)

                    # Add support/resistance levels for validation
                    pattern['support_levels'].append(support_resistance.loc[idx, 'Support'])
                    pattern['resistance_levels'].append(support_resistance.loc[idx, 'Resistance'])

                prev_zigzag = current_zigzag

            if self.is_valid_wave_pattern(pattern):
                pattern['degree'] = self.determine_wave_degree(pattern)
                pattern['type'] = self.classify_wave_type(pattern)
                wave_patterns.append(pattern)

        return wave_patterns

    def is_valid_wave_pattern(self, pattern: Dict) -> bool:
        """Validate if a pattern follows Elliott Wave rules"""
        try:
            if not isinstance(pattern, dict):
                return False
                
            required_keys = ['directions', 'magnitudes', 'points']
            if not all(key in pattern for key in required_keys):
                return False
                
            if len(pattern['directions']) < 4 or len(pattern['magnitudes']) < 4:
                return False

            directions = pattern['directions']
            magnitudes = pattern['magnitudes']
            points = pattern['points']
            
            # Check wave sequence
            impulse_up = all(d == exp for d, exp in zip(directions, [1, -1, 1, -1, 1]))
            impulse_down = all(d == exp for d, exp in zip(directions, [-1, 1, -1, 1, -1]))
            
            if not (impulse_up or impulse_down):
                return False

            # Validate wave relationships
            try:
                # Wave magnitude relationships
                wave1_mag = magnitudes[0]
                wave2_mag = magnitudes[1]
                wave3_mag = magnitudes[2]
                wave4_mag = magnitudes[3]
                wave5_mag = magnitudes[4]
                
                # Rule 1: Wave 2 never retraces more than 100% of Wave 1
                if wave2_mag > wave1_mag:
                    return False
                
                # Rule 2: Wave 3 is never the shortest among waves 1, 3, and 5
                if wave3_mag <= min(wave1_mag, wave5_mag):
                    return False
                
                # Rule 3: Wave 4 never enters Wave 1's price territory
                if impulse_up and min(points[3]['price']) <= max(points[0]['price']):
                    return False
                if impulse_down and max(points[3]['price']) >= min(points[0]['price']):
                    return False
                
                # Fibonacci relationships
                if not (
                    0.382 <= wave2_mag/wave1_mag <= 0.618 and  # Wave 2 retracement
                    1.618 <= wave3_mag/wave1_mag <= 4.236 and  # Wave 3 extension
                    0.236 <= wave4_mag/wave3_mag <= 0.382 and  # Wave 4 retracement
                    0.618 <= wave5_mag/wave1_mag <= 1.618      # Wave 5 proportion
                ):
                    return False

                # Validate against support/resistance levels
                for point, support, resistance in zip(
                    pattern['points'],
                    pattern['support_levels'],
                    pattern['resistance_levels']
                ):
                    if point['type'] == 'HIGH' and point['price'] < support:
                        return False
                    if point['type'] == 'LOW' and point['price'] > resistance:
                        return False

                return True
                
            except Exception as e:
                self.logger.error(f"Error validating wave pattern: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Error validating wave pattern: {str(e)}")
        return False

    def classify_wave_type(self, pattern: Dict) -> str:
        """
        Classify the wave pattern type based on its characteristics.
        """
        directions = pattern['directions']
        magnitudes = pattern['magnitudes']
        
        # Determine if it's an impulse or corrective wave
        if all(d == exp for d, exp in zip(directions, [1, -1, 1, -1, 1])):
            wave_type = 'Impulse (Bull)'
        elif all(d == exp for d, exp in zip(directions, [-1, 1, -1, 1, -1])):
            wave_type = 'Impulse (Bear)'
        else:
            wave_type = 'Corrective'
        
        # Add extended wave classification if applicable
        if max(magnitudes) >= 1.618 * sum(magnitudes)/len(magnitudes):
            wave_type += ' (Extended)'
        
        return wave_type

    def determine_wave_degree(self, pattern: Dict) -> str:
        """
        Determine the degree of the wave pattern based on time span and price movement.
        """
        try:
            start_date = pd.to_datetime(pattern['points'][0]['date'])
            end_date = pd.to_datetime(pattern['points'][-1]['date'])
            time_span = (end_date - start_date).days
            
            # Calculate total price movement
            total_movement = sum(pattern.get('magnitudes', [0]))
            
            # Determine degree based on time span
            if time_span > 365 * 4:
                return WaveDegree.SUPERCYCLE
            elif time_span > 365:
                return WaveDegree.CYCLE
            elif time_span > 30:
                return WaveDegree.PRIMARY
            elif time_span > 7:
                return WaveDegree.INTERMEDIATE
            else:
                return WaveDegree.MINOR

        except Exception as e:
            self.logger.error(f"Error determining wave degree: {e}")
            return WaveDegree.MINOR  # Return MINOR as default

    def create_elliott_wave_dataframe(self, wave_patterns: List[Dict], index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create Elliott Wave DataFrame using efficient operations"""
        try:
            # Pre-allocate arrays for better performance
            n_rows = len(index)
            elliott_wave = np.zeros(n_rows)
            wave_numbers = np.zeros(n_rows)
            wave_confidence = np.zeros(n_rows)
            wave_types = np.full(n_rows, 'None', dtype='object')
            wave_degrees = np.full(n_rows, 'None', dtype='object')
            
            # Create date to index mapping for faster lookups
            date_to_idx = pd.Series(np.arange(len(index)), index=index)
            
            # Process patterns
            for pattern in wave_patterns:
                if not pattern.get('points'):
                    continue
                    
                # Convert dates to proper datetime objects if they aren't already
                try:
                    start_date = pd.to_datetime(pattern['points'][0]['date'])
                    end_date = pd.to_datetime(pattern['points'][-1]['date'])
                    
                    # Ensure dates are within the index range
                    if start_date not in date_to_idx.index or end_date not in date_to_idx.index:
                        self.logger.warning(f"Pattern dates {start_date} to {end_date} outside of data range")
                        continue
                    
                    start_idx = date_to_idx[start_date]
                    end_idx = date_to_idx[end_date]
                    
                    pattern_slice = slice(start_idx, end_idx + 1)
                    elliott_wave[pattern_slice] = 1
                    wave_types[pattern_slice] = pattern.get('type', 'Unknown')
                    wave_degrees[pattern_slice] = pattern.get('degree', 'Unknown')
                    wave_confidence[pattern_slice] = pattern.get('confidence', 0.0)
                    
                    # Set wave numbers
                    for i, point in enumerate(pattern['points'][:-1]):
                        next_point = pattern['points'][i + 1]
                        point_date = pd.to_datetime(point['date'])
                        next_date = pd.to_datetime(next_point['date'])
                        
                        if point_date in date_to_idx.index and next_date in date_to_idx.index:
                            wave_start = date_to_idx[point_date]
                            wave_end = date_to_idx[next_date]
                            wave_numbers[wave_start:wave_end] = i + 1
                    
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Error processing pattern dates: {e}")
                    continue
            
            # Create DataFrame efficiently
            df = pd.DataFrame({
                'Elliott_Wave': elliott_wave,
                'Wave_Degree': wave_degrees,
                'Wave_Type': wave_types,
                'Wave_Number': wave_numbers,
                'Wave_Confidence': wave_confidence
            }, index=index)
            
            # Fill any missing values
            df.fillna({'Elliott_Wave': 0, 'Wave_Degree': 'None', 
                    'Wave_Type': 'None', 'Wave_Number': 0, 
                    'Wave_Confidence': 0.0}, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating Elliott Wave DataFrame: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return pd.DataFrame(index=index)

    def check_momentum_alignment(self, pattern: Dict) -> float:
        """
        Check if momentum aligns with wave movement.
        Returns a score between 0 and 1.
        """
        try:
            momentum_data = pattern.get('momentum', [])
            directions = pattern.get('directions', [])
            
            if not momentum_data or not directions or len(momentum_data) != len(directions):
                self.logger.debug("Invalid momentum or direction data")
                return 0.5

            score = 1.0
            for i, (direction, momentum) in enumerate(zip(directions, momentum_data)):
                if momentum is None:
                    continue
                    
                # Calculate momentum alignment score
                if direction > 0:  # Upward wave
                    momentum_alignment = (momentum - 50) / 50  # Scale from -1 to 1
                else:  # Downward wave
                    momentum_alignment = (50 - momentum) / 50
                    
                # Weight later waves more heavily
                wave_weight = 1 + (i * 0.1)
                wave_score = (0.5 + 0.5 * momentum_alignment) * wave_weight
                score *= wave_score
                
                self.logger.debug(f"Wave {i} momentum analysis - alignment: {momentum_alignment:.2f}, weight: {wave_weight:.2f}, score: {wave_score:.2f}")

            return max(min(score, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"Error in momentum alignment analysis: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return 0.5

    def check_volume_confirmation(self, pattern: Dict) -> float:
        """
        Check if volume confirms wave movement.
        Returns a score between 0 and 1.
        """
        try:
            volumes = pattern.get('volume', [])
            directions = pattern.get('directions', [])
            
            if not volumes or not directions or len(volumes) != len(directions):
                self.logger.debug("Invalid volume or direction data")
                return 0.5

            score = 1.0
            prev_volume = volumes[0]

            # Analyze volume for each wave
            for i, (direction, volume) in enumerate(zip(directions, volumes[1:]), 1):
                if prev_volume <= 0:
                    self.logger.debug(f"Invalid volume detected at wave {i}")
                    continue

                volume_change = volume / prev_volume
                expected_volume_increase = i in [0, 2, 4]  # Waves 1, 3, 5 should have increasing volume
                
                if expected_volume_increase:
                    if direction > 0:  # Upward impulse wave
                        score *= min(volume_change, 2.0) / 2.0
                    else:  # Downward impulse wave
                        score *= min(volume_change, 2.0) / 2.0
                else:  # Corrective waves should have decreasing volume
                    score *= min(2.0 - volume_change, 2.0) / 2.0
                
                self.logger.debug(f"Wave {i} volume analysis - change: {volume_change:.2f}, score impact: {score:.2f}")
                prev_volume = volume

            return max(min(score, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"Error in volume confirmation analysis: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return 0.5

    def calculate_pattern_confidence(self, pattern: Dict) -> float:
        """Calculate pattern confidence with improved validation"""
        try:
            # Exit early if timing is invalid
            if not self._validate_pattern_timespan(pattern):
                return 0.0

            weights = {
                'fibonacci': 0.30,
                'time_symmetry': 0.20,
                'momentum': 0.20,
                'volume': 0.15,
                'support_resistance': 0.15
            }
            
            scores = {
                'fibonacci': self._calculate_fibonacci_score(pattern),
                'time_symmetry': self._calculate_time_symmetry_score(pattern),
                'momentum': self._calculate_momentum_score(pattern),
                'volume': self._calculate_volume_score(pattern),
                'support_resistance': self._calculate_support_resistance_score(pattern)
            }
            
            # Weight each component
            weighted_scores = {
                key: score * weights[key]
                for key, score in scores.items()
            }
            
            # Calculate total confidence
            confidence = sum(weighted_scores.values())
            
            # Add more strict validation
            if confidence > 0.0:
                # Verify minimal pattern requirements
                if len(pattern.get('points', [])) < 3:  # Need at least 3 points
                    return 0.0
                    
                # Verify price movement significance
                price_changes = [abs(b['price'] - a['price']) 
                            for a, b in zip(pattern['points'][:-1], pattern['points'][1:])]
                if not all(change > 0 for change in price_changes):
                    return 0.0
                    
                # Verify momentum alignment
                if scores['momentum'] < 0.1:  # Require some momentum confirmation
                    confidence *= 0.5
            
            return round(max(min(confidence, 1.0), 0.0), 3)

        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {e}")
            return 0.0

    def check_fibonacci_relationships(self, pattern: Dict) -> float:
        """
        Check how well the wave magnitudes align with Fibonacci ratios.
        Returns a score between 0 and 1.
        """
        try:
            magnitudes = pattern.get('magnitudes', [])
            if not magnitudes or len(magnitudes) < 5:
                self.logger.debug("Insufficient magnitudes for Fibonacci analysis")
                return 0.0

            # Check for zero values to avoid division by zero
            if 0 in magnitudes:
                self.logger.debug("Zero magnitude detected, skipping Fibonacci analysis")
                return 0.0

            score = 1.0
            ideal_ratios = {
                'wave2_1': 0.618,  # Wave 2 retracement of Wave 1
                'wave3_1': 1.618,  # Wave 3 extension of Wave 1
                'wave4_3': 0.382,  # Wave 4 retracement of Wave 3
                'wave5_1': 1.0     # Wave 5 in relation to Wave 1
            }
            
            actual_ratios = {
                'wave2_1': magnitudes[1] / magnitudes[0],
                'wave3_1': magnitudes[2] / magnitudes[0],
                'wave4_3': magnitudes[3] / magnitudes[2],
                'wave5_1': magnitudes[4] / magnitudes[0]
            }

            for key, ideal_ratio in ideal_ratios.items():
                actual_ratio = actual_ratios[key]
                deviation = abs(actual_ratio - ideal_ratio) / ideal_ratio
                score *= (1 - min(deviation, 1))
                self.logger.debug(f"Fibonacci ratio {key}: ideal={ideal_ratio:.3f}, actual={actual_ratio:.3f}, deviation={deviation:.3f}")

            return max(min(score, 1.0), 0.0)  # Ensure score is between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci analysis: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return 0.0

    def check_time_symmetry(self, pattern: Dict) -> float:
        """
        Check the time symmetry of the wave pattern.
        Returns a score between 0 and 1.
        """
        try:
            points = pattern.get('points', [])
            if len(points) < 5:
                self.logger.debug("Insufficient points for time symmetry analysis")
                return 0.5

            # Calculate time spans between points
            time_spans = []
            for i in range(len(points) - 1):
                # Convert dates to datetime if they're not already
                date1 = pd.to_datetime(points[i]['date'])
                date2 = pd.to_datetime(points[i + 1]['date'])
                span = (date2 - date1).days
                time_spans.append(span)
                
            if not all(span > 0 for span in time_spans):
                self.logger.debug("Invalid time spans detected")
                return 0.5

            symmetry_score = 1.0
            
            # Compare wave time ratios
            waves12 = time_spans[0] + time_spans[1]  # Time span of waves 1-2
            waves45 = time_spans[2] + time_spans[3]  # Time span of waves 4-5
            wave3 = time_spans[2]                    # Time span of wave 3
            
            # Calculate average time span of non-wave 3 components
            avg_other_waves = (waves12 + waves45) / 2 if waves12 + waves45 > 0 else 0
            
            if avg_other_waves > 0:
                wave3_ratio = wave3 / avg_other_waves
                ideal_ratio = 1.618  # Golden ratio
                ratio_deviation = abs(wave3_ratio - ideal_ratio) / ideal_ratio
                symmetry_score *= (1 - min(ratio_deviation, 1))
                
                self.logger.debug(f"Time symmetry analysis - Wave3 ratio: {wave3_ratio:.2f}, deviation: {ratio_deviation:.2f}")

            return max(min(symmetry_score, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"Error in time symmetry analysis: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return 0.5

    def check_support_resistance_alignment(
        self,
        points: List[Dict],
        support_levels: List[float],
        resistance_levels: List[float]
    ) -> float:
        """
        Check how well the wave points align with support and resistance levels.
        """
        score = 1.0
        for point, support, resistance in zip(points, support_levels, resistance_levels):
            if point['type'] == 'HIGH':
                deviation = abs(point['price'] - resistance) / resistance
            else:
                deviation = abs(point['price'] - support) / support
            
            score *= (1 - min(deviation, 1))
        
        return score

    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for Elliott Wave analysis"""
        try:
            required_columns = ['ZigZag', 'Support', 'Resistance', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            if data.empty:
                self.logger.error("Input data is empty")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating input data: {str(e)}")
            return False

    def _get_swing_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and validate swing points from ZigZag indicator using vectorized operations"""
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Use boolean indexing instead of .copy()
            mask = data['ZigZag'].notna()
            swing_points = data[mask].copy()
            
            if len(swing_points) < 5:
                self.logger.warning("Insufficient swing points for Elliott Wave analysis")
                return pd.DataFrame()
            
            # Vectorized calculations
            swing_points['swing_magnitude'] = np.abs(np.diff(swing_points['ZigZag'], prepend=swing_points['ZigZag'].iloc[0]))
            min_magnitude = swing_points['swing_magnitude'].mean() * 0.1
            
            # Use boolean indexing for filtering
            significant_swings = swing_points[swing_points['swing_magnitude'] > min_magnitude]
            
            return significant_swings
            
        except Exception as e:
            self.logger.error(f"Error processing swing points: {str(e)}")
            return pd.DataFrame()
    
    def _find_all_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find all possible Elliott Wave patterns"""
        all_patterns = []
        
        try:
            if len(swing_points) < 5:
                self.logger.warning("Insufficient swing points for pattern detection")
                return []

            # Add minimum size threshold
            min_movement = data['close'].std() * 0.5  # Half standard deviation
            
            # Look for different pattern types with size filtering
            impulse_patterns = self._find_impulse_patterns(swing_points[swing_points['swing_magnitude'] > min_movement], data)
            diagonal_patterns = self._find_diagonal_patterns(swing_points[swing_points['swing_magnitude'] > min_movement], data)
            corrective_patterns = self._find_corrective_patterns(swing_points[swing_points['swing_magnitude'] > min_movement], data)
            complex_patterns = self._find_complex_patterns(swing_points[swing_points['swing_magnitude'] > min_movement], data)
            
            all_patterns.extend(impulse_patterns)
            all_patterns.extend(diagonal_patterns)
            all_patterns.extend(corrective_patterns)
            all_patterns.extend(complex_patterns)
            
            # Remove overlapping patterns
            all_patterns = self._remove_overlapping_patterns(all_patterns)
            
            # Sort patterns by confidence score
            all_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return all_patterns
            
        except Exception as e:
            self.logger.error(f"Error finding wave patterns: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return []

    def _find_impulse_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find potential impulse wave patterns"""
        patterns = []
        
        try:
            for i in range(len(swing_points) - 4):
                window = swing_points.iloc[i:i+5]
                pattern = self._create_pattern_dict(window, data)
                
                is_valid, violations = WaveRules.check_impulse_rules(pattern)
                if is_valid:
                    pattern['type'] = WavePatternType.IMPULSE_BULL if pattern['trend'] == 'up' else WavePatternType.IMPULSE_BEAR
                    pattern['confidence'] = self.calculate_pattern_confidence(pattern)
                    patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding impulse patterns: {str(e)}")
            return []

    def _find_diagonal_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find potential diagonal patterns"""
        patterns = []
        
        try:
            for i in range(len(swing_points) - 4):
                window = swing_points.iloc[i:i+5]
                pattern = self._create_pattern_dict(window, data)
                
                is_valid, violations = WaveRules.check_diagonal_rules(pattern)
                if is_valid:
                    pattern['type'] = self._classify_diagonal_type(pattern)
                    pattern['confidence'] = self.calculate_pattern_confidence(pattern)
                    patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding diagonal patterns: {str(e)}")
            return []

    def _find_corrective_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find potential corrective patterns"""
        patterns = []
        
        try:
            # Find ABC corrections
            abc_patterns = self._find_abc_patterns(swing_points, data)
            patterns.extend(abc_patterns)
            
            # Find complex corrections
            complex_patterns = self._find_complex_corrections(swing_points, data)
            patterns.extend(complex_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding corrective patterns: {str(e)}")
            return []

    def _find_complex_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find potential complex corrective patterns"""
        patterns = []
        
        try:
            # Find WXY patterns
            wxy_patterns = self._find_wxy_patterns(swing_points, data)
            patterns.extend(wxy_patterns)
            
            # Find double and triple combinations
            combination_patterns = self._find_combinations(swing_points, data)
            patterns.extend(combination_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding complex patterns: {str(e)}")
            return []

    def _remove_overlapping_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Remove overlapping patterns, keeping the ones with higher confidence"""
        try:
            if not patterns:
                return []
                
            # Sort by confidence
            patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            non_overlapping = []
            used_dates = set()
            
            for pattern in patterns:
                start_date = pattern['points'][0]['date']
                end_date = pattern['points'][-1]['date']
                
                # Check if pattern overlaps with any existing pattern
                dates = set(pd.date_range(start_date, end_date, freq='D'))
                if not dates.intersection(used_dates):
                    non_overlapping.append(pattern)
                    used_dates.update(dates)
            
            return non_overlapping
            
        except Exception as e:
            self.logger.error(f"Error removing overlapping patterns: {str(e)}")
            return patterns

    def _create_pattern_dict(self, window: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """Create a pattern dictionary from a window of swing points"""
        try:
            if window.empty or len(window) < 2:
                return {}

            # Ensure dates are datetime objects
            window.index = pd.to_datetime(window.index)
            data.index = pd.to_datetime(data.index)

            pattern = {
                'points': [],
                'directions': [],
                'magnitudes': [],
                'trend': 'up' if window['ZigZag'].iloc[-1] > window['ZigZag'].iloc[0] else 'down',
                'momentum': [],
                'volume': []
            }

            prev_price = None
            prev_date = None

            # Use numpy arrays for faster iteration (instead of iterrows)
            zigzag_vals = window['ZigZag'].values
            high_vals = window['high'].values if 'high' in window.columns else zigzag_vals
            low_vals = window['low'].values if 'low' in window.columns else zigzag_vals
            indices = window.index

            for i in range(len(window)):
                current_price = zigzag_vals[i]
                current_date = indices[i]

                if prev_price is not None and prev_date is not None:
                    # Validate prices
                    if pd.isna(current_price) or pd.isna(prev_price):
                        continue

                    direction = 1 if current_price > prev_price else -1
                    magnitude = abs(current_price - prev_price)

                    # Get momentum and volume data for the period
                    period_mask = (data.index >= prev_date) & (data.index <= current_date)
                    period_data = data.loc[period_mask]

                    rsi = period_data['RSI'].mean() if 'RSI' in period_data else None
                    volume = period_data['volume'].mean() if not period_data['volume'].empty else None

                    # Only add valid data
                    if magnitude > 0:  # Avoid zero magnitude
                        pattern['directions'].append(direction)
                        pattern['magnitudes'].append(magnitude)
                        pattern['momentum'].append(rsi)
                        pattern['volume'].append(volume)

                pattern['points'].append({
                    'date': current_date,
                    'price': current_price,
                    'high': high_vals[i] if not pd.isna(high_vals[i]) else current_price,
                    'low': low_vals[i] if not pd.isna(low_vals[i]) else current_price
                })

                prev_price = current_price
                prev_date = current_date

            return pattern
        
        except Exception as e:
            self.logger.error(f"Error creating pattern dictionary: {str(e)}")
            return {}

    def _analyze_volume_profile(self, volumes: List[float]) -> str:
        """Analyze volume trend within pattern"""
        try:
            if not volumes:
                return 'unknown'
            
            # Calculate volume trend using linear regression
            x = np.arange(len(volumes))
            slope, _, _, _, _ = stats.linregress(x, volumes)
            
            # Classify volume trend
            if slope > 0.05:  # Significant increase
                return 'increasing'
            elif slope < -0.05:  # Significant decrease
                return 'declining'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {str(e)}")
            return 'unknown'

    def _identify_subwaves(self, window: pd.DataFrame) -> List[List[Dict]]:
        """Identify subwaves within the pattern using fractals"""
        try:
            subwaves = []
            min_points = 3  # Minimum points needed for a subwave

            # Pre-extract arrays for faster access
            zigzag_vals = window['ZigZag'].values
            zigzag_highs = window['ZigZag_Highs'].values if 'ZigZag_Highs' in window.columns else zigzag_vals
            indices = window.index

            for i in range(len(window) - min_points + 1):
                sub_window = window.iloc[i:i+min_points]

                # Check if this forms a valid subwave
                if self._is_valid_subwave(sub_window):
                    subwave = []
                    # Use index-based iteration instead of iterrows()
                    for j in range(min_points):
                        idx = indices[i + j]
                        zigzag = zigzag_vals[i + j]
                        zigzag_high = zigzag_highs[i + j]
                        subwave.append({
                            'date': idx,
                            'price': zigzag,
                            'type': 'HIGH' if zigzag_high == zigzag else 'LOW'
                        })
                    subwaves.append(subwave)

            return subwaves
        except Exception as e:
            self.logger.error(f"Error identifying subwaves: {str(e)}")
            return []

    def _is_valid_subwave(self, window: pd.DataFrame) -> bool:
        """Check if a window forms a valid subwave"""
        try:
            prices = window['ZigZag'].values
            
            # Check for alternating highs and lows
            diffs = np.diff(prices)
            if not all(diffs[i] * diffs[i+1] < 0 for i in range(len(diffs)-1)):
                return False
            
            # Check for minimum price movement
            total_movement = abs(prices[-1] - prices[0])
            avg_price = np.mean(prices)
            if total_movement < 0.01 * avg_price:  # Minimum 1% movement
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error checking subwave validity: {str(e)}")
            return False

    def _find_abc_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find ABC corrective patterns"""
        patterns = []
        try:
            for i in range(len(swing_points) - 2):
                window = swing_points.iloc[i:i+3]
                pattern = self._create_pattern_dict(window, data)
                
                # Check if it's a valid ABC pattern
                if self._is_valid_abc_pattern(pattern):
                    pattern['type'] = WavePatternType.CORRECTION_ZIGZAG
                    pattern['confidence'] = self.calculate_pattern_confidence(pattern)
                    patterns.append(pattern)
                    
            return patterns
        except Exception as e:
            self.logger.error(f"Error finding ABC patterns: {str(e)}")
            return []

    def _is_valid_abc_pattern(self, pattern: Dict) -> bool:
        """Validate ABC corrective pattern"""
        try:
            if len(pattern['points']) != 3:
                return False
                
            a_to_b = pattern['magnitudes'][0]
            b_to_c = pattern['magnitudes'][1]
            
            # B should retrace 50-79% of A
            b_retracement = a_to_b / pattern['points'][0]['price']
            if not (0.5 <= b_retracement <= 0.79):
                return False
                
            # C should extend 61.8-161.8% of A
            c_extension = b_to_c / pattern['points'][0]['price']
            if not (0.618 <= c_extension <= 1.618):
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating ABC pattern: {str(e)}")
            return False

    def _find_complex_corrections(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find complex corrective patterns"""
        patterns = []
        try:
            # Look for double and triple threes
            for i in range(len(swing_points) - 6):  # Need at least 7 points for a double three
                window = swing_points.iloc[i:i+7]
                pattern = self._create_pattern_dict(window, data)
                
                if self._is_valid_double_three(pattern):
                    pattern['type'] = WavePatternType.CORRECTION_DOUBLE_THREE
                    pattern['confidence'] = self.calculate_pattern_confidence(pattern)
                    patterns.append(pattern)
                    
            return patterns
        except Exception as e:
            self.logger.error(f"Error finding complex corrections: {str(e)}")
            return []

    def _find_wxy_patterns(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find WXY patterns"""
        patterns = []
        try:
            for i in range(len(swing_points) - 4):
                window = swing_points.iloc[i:i+5]
                pattern = self._create_pattern_dict(window, data)
                
                if self._is_valid_wxy_pattern(pattern):
                    pattern['type'] = WavePatternType.COMPLEX_WXY
                    pattern['confidence'] = self.calculate_pattern_confidence(pattern)
                    patterns.append(pattern)
                    
            return patterns
        except Exception as e:
            self.logger.error(f"Error finding WXY patterns: {str(e)}")
            return []

    def _classify_diagonal_type(self, pattern: Dict) -> str:
        """Classify diagonal as leading or ending"""
        try:
            # Check wave structure
            if len(pattern.get('subwaves', [])) != 5:
                return WavePatternType.IMPULSE_ENDING_DIAGONAL
                
            wave_structure = [len(wave) for wave in pattern['subwaves']]
            
            # Leading diagonal: 5-3-5-3-5
            if wave_structure == [5, 3, 5, 3, 5]:
                return WavePatternType.IMPULSE_LEADING_DIAGONAL
                
            # Ending diagonal: 3-3-3-3-3
            if all(count == 3 for count in wave_structure):
                return WavePatternType.IMPULSE_ENDING_DIAGONAL
                
            # Default to ending diagonal if structure is unclear
            return WavePatternType.IMPULSE_ENDING_DIAGONAL
            
        except Exception as e:
            self.logger.error(f"Error classifying diagonal type: {str(e)}")
            return WavePatternType.IMPULSE_ENDING_DIAGONAL

    def _is_valid_double_three(self, pattern: Dict) -> bool:
        """Validate double three corrective pattern"""
        try:
            if len(pattern['points']) < 7:
                return False
                
            # Check for 3-3-3 structure
            wave_counts = [len(wave) for wave in pattern.get('subwaves', [])]
            if not all(count == 3 for count in wave_counts[:3]):
                return False
                
            # Check price relationships
            magnitudes = pattern['magnitudes']
            if len(magnitudes) < 6:
                return False
                
            # First three should form a flat or zigzag
            if not (0.9 <= magnitudes[1]/magnitudes[0] <= 1.05):
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating double three pattern: {str(e)}")
            return False

    def _is_valid_wxy_pattern(self, pattern: Dict) -> bool:
        """Validate WXY pattern"""
        try:
            if len(pattern['points']) < 5:
                return False
                
            magnitudes = pattern['magnitudes']
            if len(magnitudes) < 4:
                return False
                
            # W should be larger than X
            if magnitudes[1] >= magnitudes[0]:
                return False
                
            # Y should be similar in size to W
            w_size = magnitudes[0]
            y_size = magnitudes[3]
            if not (0.618 <= y_size/w_size <= 1.618):
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating WXY pattern: {str(e)}")
            return False

    def _find_combinations(self, swing_points: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Find combination patterns in corrective waves"""
        patterns = []
        try:
            for i in range(len(swing_points) - 6):  # Need at least 7 points for a combination
                window = swing_points.iloc[i:i+7]
                pattern = self._create_pattern_dict(window, data)
                
                # Check for valid corrective combinations 
                if self._is_valid_combination(pattern):
                    pattern['type'] = WavePatternType.COMPLEX_COMBINATION
                    pattern['confidence'] = self.calculate_pattern_confidence(pattern)
                    patterns.append(pattern)
                    
            return patterns
        except Exception as e:
            self.logger.error(f"Error finding combination patterns: {str(e)}")
            return []

    def _is_valid_combination(self, pattern: Dict) -> bool:
        """Validate a corrective combination pattern"""
        try:
            if len(pattern['points']) < 7:
                return False
                
            magnitudes = pattern['magnitudes']
            if len(magnitudes) < 6:
                return False
                
            # Check pattern structure
            # First corrective wave should be within 61.8%-161.8% of second corrective wave
            ratio = magnitudes[2] / magnitudes[5]
            if not (0.618 <= ratio <= 1.618):
                return False
                
            # Check connecting wave (X wave)
            x_retracement = magnitudes[3] / magnitudes[2]
            if not (0.1 <= x_retracement <= 0.618):  # X wave retracement should be shallow
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating combination pattern: {str(e)}")
            return False

    def _calculate_fibonacci_score(self, pattern: Dict) -> float:
        """Calculate improved Fibonacci relationships score"""
        try:
            if len(pattern.get('magnitudes', [])) < 2:
                return 0.0
                
            fibonacci_ratios = {
                'retracement': [0.236, 0.382, 0.500, 0.618, 0.786],
                'extension': [1.618, 2.618, 4.236]
            }
            
            scores = []
            magnitudes = pattern['magnitudes']
            
            for i in range(len(magnitudes)-1):
                ratio = magnitudes[i+1] / magnitudes[i] if magnitudes[i] != 0 else 0
                
                # Find closest Fibonacci ratio
                closest_retracement = min(fibonacci_ratios['retracement'], 
                                    key=lambda x: abs(x - ratio))
                closest_extension = min(fibonacci_ratios['extension'], 
                                    key=lambda x: abs(x - ratio))
                                    
                # Calculate score based on deviation from closest ratio
                retracement_score = 1 - min(abs(ratio - closest_retracement), 1.0)
                extension_score = 1 - min(abs(ratio - closest_extension), 1.0)
                
                scores.append(max(retracement_score, extension_score))
                
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci score: {e}")
            return 0.0

    def _calculate_time_symmetry_score(self, pattern: Dict) -> float:
        """Calculate improved time symmetry score"""
        try:
            if not pattern.get('points'):
                return 0.0
                
            dates = [pd.to_datetime(point['date']) for point in pattern['points']]
            time_spans = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            if not time_spans:
                return 0.0
                
            # Calculate time ratios between subsequent waves
            ratios = []
            for i in range(len(time_spans)-1):
                if time_spans[i] > 0:
                    ratio = time_spans[i+1] / time_spans[i]
                    ratios.append(min(ratio, 1/ratio) if ratio > 0 else 0)
                    
            # Score based on consistency of time ratios
            return sum(ratios) / len(ratios) if ratios else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating time symmetry score: {e}")
            return 0.0

    def _calculate_support_resistance_score(self, pattern: Dict, support_levels: List[float], resistance_levels: List[float]) -> float:
        """Calculate improved support/resistance alignment score"""
        try:
            if not pattern.get('points'):
                return 0.0
                
            scores = []
            for point in pattern['points']:
                price = point['price']
                
                # Find closest support/resistance level
                closest_support = min(support_levels, key=lambda x: abs(x - price))
                closest_resistance = min(resistance_levels, key=lambda x: abs(x - price))
                
                # Calculate alignment score
                support_score = 1 - min(abs(price - closest_support) / price, 1.0)
                resistance_score = 1 - min(abs(price - closest_resistance) / price, 1.0)
                scores.append(max(support_score, resistance_score))
                
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance score: {e}")
            return 0.0

    def _validate_pattern_dates(self, pattern: Dict) -> bool:
        """Validate that all dates in a pattern are proper datetime objects"""
        try:
            if not pattern.get('points'):
                return False
                
            for point in pattern['points']:
                if 'date' not in point:
                    return False
                    
                # Convert to datetime if not already
                if not isinstance(point['date'], pd.Timestamp):
                    point['date'] = pd.to_datetime(point['date'])
                    
            return True
        except Exception as e:
            self.logger.error(f"Error validating pattern dates: {str(e)}")
            return False

    def _is_valid_pattern_time_sequence(self, pattern: Dict) -> bool:
        """Check if the pattern's dates are in correct sequential order"""
        try:
            if not self._validate_pattern_dates(pattern):
                return False
                
            dates = [point['date'] for point in pattern['points']]
            return all(dates[i] < dates[i+1] for i in range(len(dates)-1))
        except Exception as e:
            self.logger.error(f"Error validating pattern time sequence: {str(e)}")
            return False

    def _get_pattern_timespan(self, pattern: Dict) -> pd.Timedelta:
        """Get the total timespan of a pattern"""
        try:
            if not self._validate_pattern_dates(pattern):
                return pd.Timedelta(0)
                
            start_date = pattern['points'][0]['date']
            end_date = pattern['points'][-1]['date']
            return end_date - start_date
        except Exception as e:
            self.logger.error(f"Error getting pattern timespan: {str(e)}")
            return pd.Timedelta(0)

    def _validate_pattern_timespan(self, pattern: Dict) -> bool:
        """Validate the time sequence of pattern points"""
        try:
            if not pattern.get('points'):
                return False
                
            dates = [pd.to_datetime(point['date']) for point in pattern['points']]
            
            # Ensure dates are sequential
            for i in range(len(dates)-1):
                if dates[i] >= dates[i+1]:
                    return False
                    
            # Check minimum timespan requirements
            total_span = (dates[-1] - dates[0]).days
            if total_span < 5:  # Minimum 5 days for a valid pattern
                return False
                
            return True
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error validating pattern timespan: {e}")
            return False

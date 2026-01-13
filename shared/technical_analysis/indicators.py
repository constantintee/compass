# shared/technical_analysis/indicators.py
"""
Technical indicator calculations.

This module provides calculations for various technical indicators:
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- CCI (Commodity Channel Index)
- ATR (Average True Range)
- SuperTrend
- ZigZag
- Support/Resistance levels
- Fibonacci levels
- OBV (On Balance Volume)
- Pivot Points
"""

import logging
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from talipp.indicators import EMA, MACD, RSI, CCI, ATR
from talipp.indicators.SuperTrend import SuperTrend, Trend
from talipp.indicators.BB import BB as BollingerBands
from talipp.indicators.ZigZag import ZigZag, PivotType
from talipp.ohlcv import OHLCV

from ..constants import TechnicalIndicatorConfig
from ..exceptions import TechnicalAnalysisError


class TechnicalIndicators:
    """Calculator for technical indicators."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('technical_indicators')
        self.config = TechnicalIndicatorConfig()

    def calculate_all_indicators(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given data.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
            ticker: Stock ticker symbol

        Returns:
            DataFrame with original data plus calculated indicators
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

            # Validate minimum data length for indicator calculations
            min_required_length = max(
                self.config.EMA_LONG_PERIOD,
                self.config.BB_PERIOD,
                self.config.SUPPORT_RESISTANCE_WINDOW,
                self.config.RSI_PERIOD
            ) + 10  # Add buffer for warm-up period

            if len(data) < min_required_length:
                self.logger.warning(
                    f"Insufficient data points ({len(data)}) for ticker {ticker}. "
                    f"Minimum required: {min_required_length}"
                )

            # Calculate support and resistance first
            support_resistance_df = self.calculate_support_resistance(data)

            # Calculate base indicators using talipp
            base_indicators_df = self._calculate_base_indicators(data)

            # Combine data with base indicators and support/resistance
            data_with_indicators = pd.concat([
                data,
                base_indicators_df,
                support_resistance_df
            ], axis=1)

            # Calculate additional indicators
            rsi_divergence_df = self.calculate_rsi_divergence(data_with_indicators)
            fibonacci_levels_df = self.calculate_fibonacci_levels(data_with_indicators)
            fibonacci_ext_df = self.calculate_fibonacci_extensions(data_with_indicators)
            obv_df = self.calculate_on_balance_volume(data_with_indicators)
            peaks_troughs_df = self.calculate_peaks_troughs(data_with_indicators)
            pivot_points_df = self.calculate_pivot_points(data_with_indicators)

            # Combine all indicators
            result = pd.concat([
                data_with_indicators,
                rsi_divergence_df,
                fibonacci_levels_df,
                fibonacci_ext_df,
                obv_df,
                peaks_troughs_df,
                pivot_points_df
            ], axis=1)

            return result

        except Exception as e:
            self.logger.error(f"Error calculating indicators for {ticker}: {e}")
            self.logger.debug(traceback.format_exc())
            raise TechnicalAnalysisError(
                ticker=ticker,
                indicator="all",
                message=f"Failed to calculate indicators: {e}"
            )

    def _calculate_base_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate base technical indicators using talipp."""
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

        # Initialize indicators
        ema_12 = EMA(self.config.EMA_SHORT_PERIOD)
        ema_26 = EMA(self.config.EMA_LONG_PERIOD)
        macd = MACD(self.config.MACD_FAST, self.config.MACD_SLOW, self.config.MACD_SIGNAL)
        rsi = RSI(self.config.RSI_PERIOD)
        bb = BollingerBands(self.config.BB_PERIOD, self.config.BB_STD)
        cci = CCI(self.config.CCI_PERIOD)
        atr = ATR(self.config.ATR_PERIOD)
        supertrend = SuperTrend(self.config.SUPERTREND_PERIOD, self.config.SUPERTREND_MULTIPLIER)
        zigzag = ZigZag(
            sensitivity=self.config.ZIGZAG_SENSITIVITY,
            min_trend_length=self.config.ZIGZAG_MIN_TREND_LENGTH,
            input_values=ohlcv_data,
            input_modifier=None,
            input_sampling=None
        )

        # Initialize arrays
        n = len(data)
        ema_12_vals = np.full(n, np.nan)
        ema_26_vals = np.full(n, np.nan)
        macd_vals = np.full(n, np.nan)
        macd_signal_vals = np.full(n, np.nan)
        macd_hist_vals = np.full(n, np.nan)
        rsi_vals = np.full(n, np.nan)
        bb_upper_vals = np.full(n, np.nan)
        bb_middle_vals = np.full(n, np.nan)
        bb_lower_vals = np.full(n, np.nan)
        cci_vals = np.full(n, np.nan)
        atr_vals = np.full(n, np.nan)
        supertrend_vals = np.full(n, np.nan)
        supertrend_dir_vals = np.full(n, 0)
        zigzag_vals = np.full(n, np.nan)
        zigzag_highs = np.full(n, np.nan)
        zigzag_lows = np.full(n, np.nan)

        # Calculate indicators for each candle
        for i, candle in enumerate(ohlcv_data):
            ema_12.add_input_value(candle.close)
            ema_26.add_input_value(candle.close)
            macd.add_input_value(candle.close)
            rsi.add_input_value(candle.close)
            bb.add_input_value(candle.close)
            cci.add_input_value(candle)
            atr.add_input_value(candle)
            supertrend.add_input_value(candle)

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
                st_val = supertrend.output_values[-1]
                supertrend_vals[i] = st_val.value
                supertrend_dir_vals[i] = 1 if st_val.trend == Trend.UP else -1

        # Process ZigZag outputs
        for output in zigzag.output_values:
            if output is not None:
                try:
                    idx = ohlcv_data.index(output.ohlcv)
                    if output.type == PivotType.HIGH:
                        zigzag_vals[idx] = output.ohlcv.high
                        zigzag_highs[idx] = output.ohlcv.high
                    elif output.type == PivotType.LOW:
                        zigzag_vals[idx] = output.ohlcv.low
                        zigzag_lows[idx] = output.ohlcv.low
                except ValueError:
                    self.logger.warning("Could not find matching OHLCV object for ZigZag output")

        return pd.DataFrame({
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

    def calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels."""
        window = self.config.SUPPORT_RESISTANCE_WINDOW
        return pd.DataFrame({
            'Support': df['low'].rolling(window=window).min(),
            'Resistance': df['high'].rolling(window=window).max()
        }, index=df.index)

    def calculate_rsi_divergence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI divergence signals."""
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
        """Calculate Fibonacci retracement levels."""
        df = pd.DataFrame(index=data.index)
        window = self.config.FIB_WINDOW
        high = data['high'].rolling(window=window).max()
        low = data['low'].rolling(window=window).min()
        diff = high - low

        for level in self.config.FIB_RETRACEMENT_LEVELS:
            df[f'Fib_{level}'] = low + diff * level
        return df

    def calculate_fibonacci_extensions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci extension levels."""
        df = pd.DataFrame(index=data.index)
        window = self.config.FIB_WINDOW
        high = data['high'].rolling(window=window).max()
        low = data['low'].rolling(window=window).min()
        diff = high - low

        for level in self.config.FIB_EXTENSION_LEVELS:
            df[f'FibExt_{level}'] = high + diff * (level - 1)
        return df

    def calculate_on_balance_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate On Balance Volume (OBV)."""
        return pd.DataFrame({
            'OBV': (np.sign(data['close'].diff()) * data['volume']).cumsum()
        }, index=data.index)

    def calculate_peaks_troughs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price peaks and troughs."""
        df = pd.DataFrame(index=data.index)
        close_prices = data['close'].values
        distance = self.config.PEAK_DISTANCE

        peaks, _ = find_peaks(close_prices, distance=distance)
        troughs, _ = find_peaks(-close_prices, distance=distance)

        df['Peak'] = np.nan
        df['Trough'] = np.nan
        df.iloc[peaks, df.columns.get_loc('Peak')] = close_prices[peaks]
        df.iloc[troughs, df.columns.get_loc('Trough')] = close_prices[troughs]

        return df

    def calculate_pivot_points(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate pivot points."""
        df = pd.DataFrame(index=data.index)
        df['Pivot_Point'] = (data['high'] + data['low'] + data['close']) / 3
        df['R1'] = 2 * df['Pivot_Point'] - data['low']
        df['S1'] = 2 * df['Pivot_Point'] - data['high']
        return df

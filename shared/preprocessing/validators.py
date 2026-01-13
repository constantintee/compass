# shared/preprocessing/validators.py
"""
Data validation functionality for preprocessing.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ..constants import ValidationThresholds
from ..exceptions import DataValidationError


class DataValidator:
    """Validates stock data for quality and completeness."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('data_validator')
        self.thresholds = ValidationThresholds()

    def validate_column_names(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        ticker: str
    ) -> bool:
        """Validate that all required columns are present."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(
                f"Ticker {ticker}: Missing required columns: {', '.join(missing_columns)}."
            )
            return False
        self.logger.info(f"Ticker {ticker}: All required columns are present.")
        return True

    def validate_raw_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validate raw data by checking for outliers and invalid values.

        Args:
            df: DataFrame to validate
            ticker: Stock ticker symbol

        Returns:
            DataFrame with invalid values set to NaN
        """
        try:
            price_columns = ['open', 'high', 'low', 'close']

            for column in price_columns:
                if column in df.columns:
                    invalid_mask = (
                        (df[column] <= self.thresholds.MIN_STOCK_PRICE) |
                        (df[column] >= self.thresholds.MAX_STOCK_PRICE)
                    )
                    num_invalid = invalid_mask.sum()
                    if num_invalid > 0:
                        self.logger.error(
                            f"Ticker {ticker}: {num_invalid} entries in '{column}' "
                            f"are outside the range ({self.thresholds.MIN_STOCK_PRICE}, "
                            f"{self.thresholds.MAX_STOCK_PRICE})."
                        )
                        df.loc[invalid_mask, column] = np.nan

            # Validate volume
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['volume'] = df['volume'].replace(0, np.nan)

                volume_invalid = (
                    (df['volume'] < self.thresholds.MIN_VOLUME) |
                    (df['volume'] > self.thresholds.MAX_VOLUME)
                )
                if volume_invalid.any():
                    self.logger.warning(
                        f"Ticker {ticker}: {volume_invalid.sum()} invalid volume values found."
                    )
                    df.loc[volume_invalid, 'volume'] = np.nan

            return df

        except Exception as e:
            self.logger.error(f"Error validating raw data for ticker {ticker}: {e}")
            raise DataValidationError(
                ticker=ticker,
                message=f"Raw data validation failed: {e}",
                invalid_columns=price_columns
            )

    def validate_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validate the cleaned data to ensure no invalid values are present.

        Args:
            df: DataFrame to validate
            ticker: Stock ticker symbol

        Returns:
            Validated DataFrame or empty DataFrame if validation fails
        """
        try:
            # Replace infinities with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            critical_columns = ['close', 'high', 'low', 'open', 'volume']
            for column in critical_columns:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    if df[column].isna().sum() > 0:
                        self.logger.warning(
                            f"Ticker {ticker}: Found NaNs in '{column}' after conversion. "
                            "Attempting to fill with column mean."
                        )
                        df[column] = df[column].fillna(df[column].mean())

            # Final check for NaN values
            if df.isnull().values.any():
                remaining_nans = df.isnull().sum().sum()
                self.logger.error(
                    f"Ticker {ticker}: Data still contains {remaining_nans} NaN values "
                    "after all validations."
                )

            return df

        except Exception as e:
            self.logger.error(f"Error validating data for ticker {ticker}: {e}")
            raise DataValidationError(
                ticker=ticker,
                message=f"Data validation failed: {e}"
            )

    def validate_technical_indicators(self, data: pd.DataFrame) -> bool:
        """Validate calculated technical indicators."""
        try:
            # Check RSI bounds
            if 'RSI' in data.columns:
                invalid_rsi = (
                    (data['RSI'] < self.thresholds.RSI_MIN) |
                    (data['RSI'] > self.thresholds.RSI_MAX)
                )
                if invalid_rsi.any():
                    self.logger.error("Invalid RSI values detected")
                    return False

            # Check Bollinger Bands
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
                invalid_bb = data['BB_Upper'] < data['BB_Lower']
                if invalid_bb.any():
                    self.logger.error("Invalid Bollinger Bands detected")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating technical indicators: {str(e)}")
            return False

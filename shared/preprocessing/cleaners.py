# shared/preprocessing/cleaners.py
"""
Data cleaning functionality for preprocessing.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..constants import ValidationThresholds
from ..exceptions import PreprocessingError


class DataCleaner:
    """Handles missing values and data cleaning operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('data_cleaner')
        self.thresholds = ValidationThresholds()

    def handle_missing_values(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Uses forward fill, backward fill, and interpolation strategies.

        Args:
            df: DataFrame with potential missing values
            ticker: Stock ticker symbol

        Returns:
            DataFrame with missing values handled
        """
        try:
            # Use modern pandas methods (ffill/bfill instead of deprecated fillna with method)
            df = df.ffill()
            df = df.bfill()

            # Handle volume separately
            if 'volume' in df.columns:
                if df['volume'].isna().sum() > 0:
                    median_volume = df['volume'].median()
                    if pd.isna(median_volume):
                        self.logger.warning(
                            f"Ticker {ticker}: 'volume' column median is NaN. "
                            "Using default value of 1."
                        )
                        df['volume'] = df['volume'].fillna(1)
                    else:
                        self.logger.warning(
                            f"Ticker {ticker}: Filling remaining NaNs in 'volume' "
                            f"with median value: {median_volume}."
                        )
                        df['volume'] = df['volume'].fillna(median_volume)

                    # Drop rows if volume still has NaNs
                    if df['volume'].isna().sum() > 0:
                        self.logger.warning(
                            f"Ticker {ticker}: 'volume' still contains NaNs. Dropping these rows."
                        )
                        df = df.dropna(subset=['volume'])

            # Fill remaining NaNs for numeric columns with column means
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())

            # Log if any NaNs remain
            if df.isnull().values.any():
                remaining_nans = df.isnull().sum().sum()
                self.logger.error(
                    f"Ticker {ticker}: Data still contains {remaining_nans} NaN values "
                    "after all filling attempts."
                )
                nan_columns = df.columns[df.isna().any()].tolist()
                self.logger.error(f"Columns with NaNs: {nan_columns}")
                df = df.dropna()

            return df

        except Exception as e:
            self.logger.error(f"Error handling missing values for ticker {ticker}: {e}")
            raise PreprocessingError(
                ticker=ticker,
                step='handle_missing_values',
                message=f"Failed to handle missing values: {e}"
            )

    def preprocess_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Preprocess the data for a given ticker.

        Args:
            df: DataFrame containing stock data
            ticker: Stock ticker symbol

        Returns:
            Cleaned and preprocessed DataFrame
        """
        self.logger.info(f"[Preprocessing] Starting preprocessing for ticker {ticker}")

        # Validate 'open' column
        valid_open_range = (
            self.thresholds.MIN_STOCK_PRICE,
            self.thresholds.MAX_STOCK_PRICE
        )
        initial_count = len(df)
        df = df[
            (df['open'] > valid_open_range[0]) &
            (df['open'] < valid_open_range[1])
        ]
        filtered_count = len(df)

        if filtered_count < initial_count:
            self.logger.error(
                f"Ticker {ticker}: {initial_count - filtered_count} entries in 'open' "
                f"are outside the range {valid_open_range}. These entries have been removed."
            )

        # Handle NaNs in critical columns
        critical_columns = ['open', 'volume']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                self.logger.warning(
                    f"Ticker {ticker}: Found NaNs in '{col}' after conversion."
                )
                if col == 'open':
                    df[col] = df[col].fillna(df[col].mean())
                    df[col] = df[col].ffill()
                    df[col] = df[col].bfill()
                elif col == 'volume':
                    df[col] = df[col].fillna(df[col].median())
                    df[col] = df[col].ffill()
                    df[col] = df[col].bfill()

                # Check if NaNs still exist
                remaining_nans = df[col].isnull().sum()
                if remaining_nans > 0:
                    if col == 'open':
                        df = df.dropna(subset=['open'])
                        self.logger.error(
                            f"Ticker {ticker}: Data still contains NaN values in 'open' "
                            "after all filling attempts."
                        )
                    elif col == 'volume':
                        median_volume = df['volume'].median()
                        df['volume'] = df['volume'].fillna(median_volume)
                        if df['volume'].isnull().any():
                            df = df.dropna(subset=['volume'])

        # Final check for NaNs
        if df.isnull().any().any():
            nan_columns = df.columns[df.isnull().any()].tolist()
            self.logger.error(
                f"Ticker {ticker}: Columns with NaNs after preprocessing: {nan_columns}"
            )
            df = df.dropna()

        self.logger.info(
            f"[Preprocessing] Completed preprocessing for ticker {ticker}. "
            f"Cleaned data contains {len(df)} records."
        )
        return df

    def ensure_continuous_dates(
        self,
        df: pd.DataFrame,
        ticker: str,
        frequency: str = 'B'
    ) -> pd.DataFrame:
        """
        Ensure all dates are present in the DataFrame.

        Args:
            df: Original data with potential date gaps
            ticker: Stock ticker symbol
            frequency: Date frequency ('B' for business days, 'D' for daily)

        Returns:
            DataFrame with continuous dates and missing values handled
        """
        try:
            # Create a complete date range and reindex the data
            full_date_range = pd.date_range(
                start=df['date'].min(),
                end=df['date'].max(),
                freq=frequency
            )
            df = df.set_index('date')
            df = df.reindex(full_date_range)
            df.index.name = 'date'

            missing_dates = df[df['open'].isna()].index
            num_missing = len(missing_dates)

            if num_missing > 0:
                self.logger.warning(
                    f"{num_missing} missing dates found for ticker {ticker}. "
                    "Filling missing data."
                )

                # Fill missing data using interpolation and fill methods
                df = df.interpolate(method='linear', limit_direction='both')
                df = df.ffill()
                df = df.bfill()

            # If there are still NaN values, fill with column means
            if df.isnull().values.any():
                remaining_nans = df.isnull().sum().sum()
                self.logger.warning(
                    f"Ticker {ticker}: {remaining_nans} NaN values remaining. "
                    "Filling with column means."
                )
                df = df.fillna(df.mean())

            # Reset the index
            df = df.reset_index()
            return df

        except Exception as e:
            self.logger.error(
                f"Error ensuring continuous dates for ticker {ticker}: {e}"
            )
            raise PreprocessingError(
                ticker=ticker,
                step='ensure_continuous_dates',
                message=f"Failed to ensure continuous dates: {e}"
            )

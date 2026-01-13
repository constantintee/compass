# shared/preprocessing/transformers.py
"""
Data transformation functionality for preprocessing.
"""

import hashlib
import logging
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from ..exceptions import PreprocessingError


class DataTransformer:
    """Handles data transformations for machine learning."""

    def __init__(
        self,
        sequence_length: int = 60,
        cache_dir: str = 'cache',
        logger: Optional[logging.Logger] = None
    ):
        self.sequence_length = sequence_length
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger('data_transformer')
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        os.makedirs(self.cache_dir, exist_ok=True)

    def compute_checksum(self, data: pd.DataFrame) -> str:
        """Compute checksum for caching purposes."""
        return hashlib.md5(
            pd.util.hash_pandas_object(data, index=True).values
        ).hexdigest()

    def load_processed_data(
        self,
        ticker: str,
        checksum: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load processed data from cache."""
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

    def save_processed_data(
        self,
        ticker: str,
        X: np.ndarray,
        y: np.ndarray,
        checksum: str
    ) -> None:
        """Save processed data to cache."""
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

    def sequence_generator(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate sequences for time series prediction.

        Args:
            data: DataFrame with features
            ticker: Stock ticker symbol

        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            # Identify non-numeric columns
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

            # Exclude non-numeric columns from features
            exclude_cols = ['date', 'ticker', 'Elliott_Wave', 'Wave_Degree'] + non_numeric_columns
            feature_columns = [col for col in data.columns if col not in exclude_cols]

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
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.feature_scaler.fit_transform(X_reshaped).reshape(X.shape)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

            self.logger.debug(
                f"Sequence generation and scaling completed for ticker {ticker}."
            )
            return X_scaled, y_scaled

        except Exception as e:
            self.logger.error(
                f"Error during sequence generation for ticker {ticker}: {e}"
            )
            raise PreprocessingError(
                ticker=ticker,
                step='sequence_generator',
                message=f"Sequence generation failed: {e}"
            )

    def write_tfrecord(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tfrecord_path: str
    ) -> None:
        """
        Write sequences to a TFRecord file.

        Args:
            X: Feature sequences array
            y: Target values array
            tfrecord_path: Path to save the TFRecord file
        """
        try:
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for features_seq, target_value in zip(X, y):
                    feature = {
                        'features': tf.train.Feature(
                            float_list=tf.train.FloatList(value=features_seq.flatten())
                        ),
                        'target': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[target_value])
                        )
                    }
                    example_proto = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(example_proto.SerializeToString())
            self.logger.info(f"TFRecord written to {tfrecord_path} successfully.")
        except Exception as e:
            self.logger.error(f"Error writing TFRecord for path {tfrecord_path}: {e}")
            raise

    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast numeric data types to reduce memory usage.

        Args:
            df: DataFrame to optimize

        Returns:
            Optimized DataFrame
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

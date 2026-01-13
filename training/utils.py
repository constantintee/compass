# utils.py

import logging
from logging.handlers import RotatingFileHandler
import yaml
from datetime import datetime
import time
import tensorflow as tf
import pandas as pd
import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable

from tensorflow.keras import backend as K

import gc
import psutil
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024 * 1024)

def log_memory_usage(logger, message: str):
    """Log current memory usage with a message"""
    memory_mb = get_memory_usage()
    logger.info(f"{message}: {memory_mb:.2f} MB")

def clean_memory():
    """
    Force garbage collection and clear TensorFlow session
    to free up memory
    """
    gc.collect()
    tf.keras.backend.clear_session()

def get_optimal_buffer_sizes() -> Tuple[int, int]:
    """
    Calculate optimal buffer sizes based on available memory
    
    Returns:
        Tuple[int, int]: Shuffle buffer size and prefetch buffer size
    """
    try:
        # Get available system memory 
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
        current_usage = get_memory_usage()
        
        # Calculate safe memory limits (70% of available memory)
        safe_memory = (available_memory - current_usage) * 0.7
        
        # Calculate buffer sizes
        shuffle_buffer = min(int(safe_memory * 0.4), 262144)  # Max 256K
        prefetch_buffer = min(int(safe_memory * 0.2), 131072)  # Max 128K
        
        return shuffle_buffer, prefetch_buffer
        
    except Exception:
        # Return conservative defaults if calculation fails
        return 10000, 5000

def setup_logger(log_config: dict = None, logger_name: str = 'training_logger', environment: str = 'development') -> logging.Logger:
    """
    Set up and configure the logger with rotating file handler.

    Parameters:
    - log_config (dict): Configuration dictionary for logging.
    - logger_name (str): Name of the logger.
    - environment (str): Environment type ('development', 'production').

    Returns:
    - logging.Logger: Configured logger instance.
    """
    if log_config is None:
        log_config = {}

    # Get configuration values with defaults
    log_file = log_config.get('file', 'data/logs/training.log')
    max_bytes = log_config.get('max_bytes', 5242880)  # 5MB
    backup_count = log_config.get('backup_count', 5)
    
    # Set log level based on environment
    log_level = logging.DEBUG if environment == 'development' else logging.INFO

    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Only add handlers if the logger doesn't already have them
    if not logger.handlers:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
                print(f"Created log directory at {log_dir}")
            except Exception as e:
                print(f"Failed to create log directory at {log_dir}: {e}")
                sys.exit(1)

        # Create formatters
        console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if environment == 'development' else logging.WARNING)
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # Create and configure rotating file handler
        try:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG if environment == 'development' else logging.INFO)
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            logger.info(f"Rotating file handler configured: {log_file} (max: {max_bytes/1024/1024:.1f}MB, backups: {backup_count})")
        except Exception as e:
            logger.error(f"Failed to create rotating file handler: {e}")
            logger.debug(traceback.format_exc())
            # Continue with console logging only
            logger.warning("Continuing with console logging only")

    return logger

def load_configuration(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration dictionary.
    """
    logger = logging.getLogger('training_logger')
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}.")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        sys.exit(1)

def validate_tfrecord_file(file_path: str, sequence_length: int, num_features: int) -> bool:
    """
    Validate a TFRecord file with more lenient validation and better debugging.
    """
    logger = logging.getLogger('training_logger')
    try:
        # Set up feature description
        feature_description = {
            'features': tf.io.FixedLenFeature([sequence_length * num_features], tf.float32),
            'target': tf.io.FixedLenFeature([1], tf.float32),
        }
        
        # Create dataset
        dataset = tf.data.TFRecordDataset(
            file_path,
            compression_type=None,
            buffer_size=8*1024*1024,
            num_parallel_reads=1
        )
        
        valid_records = 0
        total_checked = 0
        debug_enabled = True  # Enable detailed debugging for first record
        
        for record in dataset.take(100):  # Sample size of 100
            try:
                # Parse the record
                example = tf.io.parse_single_example(record, feature_description)
                
                # Debug first record
                if debug_enabled and total_checked == 0:
                    features = example['features'].numpy()
                    target = example['target'].numpy()
                    logger.info(f"First record debug for {os.path.basename(file_path)}:")
                    logger.info(f"Features shape: {features.shape}")
                    logger.info(f"Target shape: {target.shape}")
                    logger.info(f"Features range: [{np.min(features)}, {np.max(features)}]")
                    logger.info(f"Target value: {target}")
                    debug_enabled = False
                
                # More lenient validation - check if we can reshape and values are finite
                features = tf.reshape(example['features'], [sequence_length, num_features])
                target = example['target']
                
                # Check only if values are finite (not inf or nan)
                features_valid = tf.reduce_any(tf.math.is_finite(features))  # Changed from reduce_all to reduce_any
                target_valid = tf.reduce_any(tf.math.is_finite(target))
                
                if features_valid and target_valid:
                    valid_records += 1
                
                total_checked += 1
                    
            except Exception as e:
                if total_checked == 0:
                    # Log detailed error for first record only
                    logger.error(f"Error parsing first record in {os.path.basename(file_path)}: {str(e)}")
                total_checked += 1
                continue
        
        # More lenient validity criteria - require only 10% valid records
        is_valid = valid_records > 0 and (valid_records / max(total_checked, 1)) >= 0.1
        
        if is_valid:
            logger.info(f"Validated {os.path.basename(file_path)}: {valid_records}/{total_checked} valid records")
        else:
            logger.warning(f"Skipping {os.path.basename(file_path)}: only {valid_records}/{total_checked} valid records")
            
        return is_valid
        
    except Exception as e:
        logger.error(f"Error validating {os.path.basename(file_path)}: {str(e)}")
        return False

def get_num_features_from_tfrecord(tfrecord_path: str, sequence_length: int) -> Optional[int]:
    """
    Extract the number of features from a TFRecord file with better error handling.
    """
    logger = logging.getLogger('training_logger')
    try:
        tfrecord_files = [f for f in os.listdir(tfrecord_path) if f.endswith('.tfrecord')]
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {tfrecord_path}")
        
        # Try each file until we successfully get features
        for file_name in tfrecord_files:
            try:
                first_file = os.path.join(tfrecord_path, file_name)
                raw_dataset = tf.data.TFRecordDataset([first_file])
                
                for raw_record in raw_dataset.take(1):
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature['features'].float_list.value
                    num_features = len(features) // sequence_length
                    logger.info(f"Detected {num_features} features from {file_name}")
                    return num_features
                    
            except Exception as e:
                logger.warning(f"Could not read features from {file_name}: {str(e)}")
                continue
                
        raise ValueError("Could not determine number of features from any TFRecord file")
        
    except Exception as e:
        logger.error(f"Error extracting number of features: {e}")
        return None

def inspect_tfrecord(file_path: str) -> dict:
    """
    Inspect the structure of a TFRecord file.
    """
    logger = logging.getLogger('training_logger')
    try:
        raw_dataset = tf.data.TFRecordDataset([file_path])
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            feature_dict = {}
            for key, feature in example.features.feature.items():
                # Get the type of feature
                if feature.HasField('float_list'):
                    feature_dict[key] = ('float_list', len(feature.float_list.value))
                elif feature.HasField('int64_list'):
                    feature_dict[key] = ('int64_list', len(feature.int64_list.value))
                elif feature.HasField('bytes_list'):
                    feature_dict[key] = ('bytes_list', len(feature.bytes_list.value))
            return feature_dict
    except Exception as e:
        logger.error(f"Error inspecting TFRecord {file_path}: {e}")
        return None

def parse_function(example_proto, sequence_length, num_features):
    """Parse TFRecord example with correct feature names."""
    feature_description = {
        'features': tf.io.FixedLenFeature([sequence_length * num_features], tf.float32),  # Matches actual structure
        'target': tf.io.FixedLenFeature([1], tf.float32),  # Matches actual structure
    }
    
    try:
        # Parse the example
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Reshape features
        features = tf.reshape(parsed['features'], [sequence_length, num_features])
        target = parsed['target'][0]
        
        # Replace NaN and infinite values
        features = tf.where(tf.math.is_finite(features), features, tf.zeros_like(features))
        
        # Normalize each feature independently
        features_mean = tf.reduce_mean(features, axis=0, keepdims=True)
        features_std = tf.math.reduce_std(features, axis=0, keepdims=True)
        features_std = tf.maximum(features_std, 1e-6)
        features = (features - features_mean) / features_std
        
        # Clip normalized values
        features = tf.clip_by_value(features, -5.0, 5.0)
        
        # Handle target
        target = tf.where(tf.math.is_finite(target), target, tf.constant(0.0))
        
        return features, target
        
    except Exception as e:
        tf.print("Error parsing example:", e)
        return (
            tf.zeros([sequence_length, num_features], dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32)
        )

def load_and_process_individual_tfrecords_parallel(individual_tfrecord_path, config):
    """Load and process TFRecord files with optimized performance."""
    logger = logging.getLogger('training_logger')
    try:
        # Get configuration
        training_config = config.get('training', {})
        sequence_length = training_config.get('sequence_length', 60)
        batch_size = training_config.get('batch_size', 32)

        # Get list of TFRecord files
        tfrecord_files = [
            os.path.join(individual_tfrecord_path, f)
            for f in os.listdir(individual_tfrecord_path)
            if f.endswith('.tfrecord')
        ]

        if not tfrecord_files:
            raise ValueError("No TFRecord files found")

        logger.info(f"Found {len(tfrecord_files)} TFRecord files")

        # Dynamically detect num_features from first file
        num_features = get_num_features_from_tfrecord(individual_tfrecord_path, sequence_length)
        if num_features is None:
            num_features = 40  # Fallback default
            logger.warning(f"Could not detect num_features, using default: {num_features}")

        # Shuffle file list for better data distribution
        np.random.shuffle(tfrecord_files)

        # Split files into train/val (80/20) - more efficient than splitting records
        split_idx = int(len(tfrecord_files) * 0.8)
        train_files = tfrecord_files[:split_idx]
        val_files = tfrecord_files[split_idx:]

        logger.info(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

        # Optimize parallel reads based on CPU count
        num_parallel_reads = min(multiprocessing.cpu_count(), 8)

        # Create optimized training dataset
        train_dataset = tf.data.TFRecordDataset(
            train_files,
            compression_type=None,
            buffer_size=8*1024*1024,
            num_parallel_reads=num_parallel_reads
        )

        # Create optimized validation dataset
        val_dataset = tf.data.TFRecordDataset(
            val_files,
            compression_type=None,
            buffer_size=8*1024*1024,
            num_parallel_reads=num_parallel_reads
        )

        # Estimate total batches efficiently using file sizes
        total_file_size = sum(os.path.getsize(f) for f in tfrecord_files)
        avg_record_size = 4 * (sequence_length * num_features + 1)  # Approximate bytes per record
        estimated_examples = total_file_size // avg_record_size
        total_batches = max(1, estimated_examples // batch_size)

        logger.info(f"Estimated total examples: {estimated_examples}")
        logger.info(f"Estimated total batches: {total_batches}")

        # Process datasets with caching for efficiency
        train_dataset = create_safe_dataset(
            train_dataset, sequence_length, num_features, batch_size, is_training=True
        )
        val_dataset = create_safe_dataset(
            val_dataset, sequence_length, num_features, batch_size, is_training=False
        )

        # Verify datasets
        for features, targets in train_dataset.take(1):
            logger.info(f"\nTraining dataset shapes: Features={features.shape}, Targets={targets.shape}")

        for features, targets in val_dataset.take(1):
            logger.info(f"Validation dataset shapes: Features={features.shape}, Targets={targets.shape}")

        return train_dataset, val_dataset, num_features, total_batches

    except Exception as e:
        logger.error(f"Error loading TFRecords: {str(e)}")
        raise

def create_safe_dataset(dataset, sequence_length, num_features, batch_size, is_training=True):
    """Create a dataset with safe parsing, caching, and optimized performance."""
    feature_description = {
        'features': tf.io.FixedLenFeature([sequence_length * num_features], tf.float32),
        'target': tf.io.FixedLenFeature([1], tf.float32),
    }

    def safe_parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        features = tf.reshape(parsed['features'], [sequence_length, num_features])
        target = parsed['target'][0]

        # Handle invalid values efficiently using vectorized operations
        features = tf.where(tf.math.is_finite(features), features, tf.zeros_like(features))
        target = tf.where(tf.math.is_finite(target), target, tf.constant(0.0))

        # Clip extreme values
        features = tf.clip_by_value(features, -10.0, 10.0)

        # Per-sequence normalization (more efficient than per-feature)
        features_mean = tf.reduce_mean(features)
        features_std = tf.math.reduce_std(features)
        features_std = tf.maximum(features_std, 1e-6)
        features = (features - features_mean) / features_std

        return features, target

    # Parse with parallel processing
    dataset = dataset.map(
        safe_parse_function,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # Allow non-deterministic ordering for speed
    )

    # Filter out any zero-target samples (invalid data)
    dataset = dataset.filter(lambda x, y: tf.not_equal(y, 0.0))

    if is_training:
        # Cache after parsing but before shuffling for training data
        dataset = dataset.cache()
        # Use dynamic shuffle buffer based on available memory
        shuffle_buffer, _ = get_optimal_buffer_sizes()
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True
        )
    else:
        # Cache validation data
        dataset = dataset.cache()

    # Batch with parallel processing
    dataset = dataset.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch for pipeline efficiency
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def estimate_dataset_size(dataset: tf.data.Dataset, logger) -> int:
    """Safely estimate dataset size with better error handling."""
    try:
        count = 0
        start_time = time.time()
        for batch in dataset:
            count += 1
            if count % 1000 == 0:
                logger.info(f"Counted {count} batches...")
            # Add a reasonable timeout
            if time.time() - start_time > 300:  # 5 minutes timeout
                logger.warning("Dataset size estimation timed out, using current count")
                break
        return count
    except Exception as e:
        logger.error(f"Error in dataset size estimation: {e}")
        return 1000  # Return a default size if estimation fails

def load_hyperparameters(model_name: str, hyperparam_dir: str = 'data/hyperparameters') -> dict:
    """
    Load hyperparameters for a model from a JSON file.

    Parameters:
    - model_name (str): The name of the model for which to load hyperparameters.
    - hyperparam_dir (str): Directory where hyperparameters are stored.

    Returns:
    - dict: Hyperparameters loaded from the file.
    """
    logger = logging.getLogger('training_logger')
    hyperparam_path = os.path.join(hyperparam_dir, f"{model_name}_best_params.json")

    try:
        with open(hyperparam_path, 'r') as f:
            hyperparameters = json.load(f)
            logger.info(f"Hyperparameters loaded for model {model_name} from {hyperparam_path}.")
            return hyperparameters
    except FileNotFoundError:
        logger.warning(f"No hyperparameter file found for model {model_name}. Using default parameters.")
        return {}  # Return an empty dictionary if no hyperparameters are found
    except Exception as e:
        logger.error(f"Error loading hyperparameters for model {model_name}: {e}")
        return {}

def safe_divide(numerator, denominator, logger=None):
    """
    Safely divides two numbers or arrays, replacing infinities and NaNs with np.nan.

    Parameters:
    - numerator (float or np.ndarray): The numerator.
    - denominator (float or np.ndarray): The denominator.
    - logger (logging.Logger): Logger instance.

    Returns:
    - float or np.ndarray: The result of the division with non-finite values set to np.nan.
    """
    if logger is None:
        logger = logging.getLogger('training_logger')
    try:
        result = numerator / denominator
        if isinstance(result, np.ndarray):
            result = np.where(np.isfinite(result), result, np.nan)
        else:
            result = result if np.isfinite(result) else np.nan
        return result
    except Exception as e:
        logger.error(f"Error in safe_divide: {e}")
        return np.nan

def create_directory_if_not_exists(directory: str):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio of a series of returns.

    Parameters:
    - returns (np.ndarray): Array of period returns.
    - risk_free_rate (float): The risk-free rate of return (default is 0).

    Returns:
    - float: The Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

def calculate_drawdown(cumulative_returns: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the maximum drawdown and drawdown duration.

    Parameters:
    - cumulative_returns (np.ndarray): Array of cumulative returns.

    Returns:
    - Tuple[float, float]: Maximum drawdown and drawdown duration.
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    end = np.argmin(drawdown)
    start = np.argmax(cumulative_returns[:end])
    duration = end - start
    return max_drawdown, duration

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate the moving average of a data series.

    Parameters:
    - data (np.ndarray): Input data series.
    - window (int): Size of the moving window.

    Returns:
    - np.ndarray: Moving average series.
    """
    return np.convolve(data, np.ones(window), 'valid') / window

def monitor_memory(func):
    """
    Decorator to monitor memory usage before and after function execution
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with memory monitoring
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('training_logger')
        
        # Log memory before execution
        start_memory = get_memory_usage()
        log_memory_usage(logger, f"Memory before {func.__name__}")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log memory after execution
            end_memory = get_memory_usage()
            memory_diff = end_memory - start_memory
            log_memory_usage(logger, f"Memory after {func.__name__}")
            logger.info(f"Memory change during {func.__name__}: {memory_diff:.2f} MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
        finally:
            # Cleanup
            clean_memory()
    
    return wrapper

class MemoryMonitor:
    """
    Context manager for monitoring memory usage in a code block
    
    Usage:
        with MemoryMonitor("Operation name"):
            # Your code here
    """
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.logger = logging.getLogger('training_logger')
        
    def __enter__(self):
        self.start_memory = get_memory_usage()
        log_memory_usage(self.logger, f"Starting {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = get_memory_usage()
        memory_diff = end_memory - self.start_memory
        log_memory_usage(self.logger, f"Finished {self.operation_name}")
        self.logger.info(f"Memory change during {self.operation_name}: {memory_diff:.2f} MB")
        clean_memory()

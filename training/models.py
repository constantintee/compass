# models.py

import logging
import os
import sys
import time
import datetime
from datetime import datetime, timedelta
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import gc
import psutil
import traceback
import numpy as np

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, Flatten, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization, Bidirectional, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.mixed_precision import set_global_policy
# Set the global mixed precision policy to 'mixed_float16'
set_global_policy('mixed_float16')

from typing import List, Dict, Any, Optional, Tuple, Callable

from utils import MemoryMonitor, monitor_memory, log_memory_usage, clean_memory, get_memory_usage, load_hyperparameters


# Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('training_logger')


class VerboseTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.epoch_times = []
        self.best_val_loss = float('inf')
        self.start_time = None
        self.last_batch_time = None
        self.stuck_threshold = 60  # seconds before considering training stuck

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.last_batch_time = time.time()
        self.logger.info("Training started")
        self.logger.info("-"*50)

    def on_epoch_begin(self, epoch, logs=None):
        try:
            self.epoch_start = time.time()
            self.last_batch_time = time.time()
            self.logger.info(f"Starting epoch {epoch+1}/{self.params.get('epochs', '?')}")
        except Exception as e:
            self.logger.error(f"Error in on_epoch_begin: {str(e)}")

    def on_batch_begin(self, batch, logs=None):
        try:
            self.last_batch_time = time.time()
        except Exception as e:
            self.logger.error(f"Error in on_batch_begin: {str(e)}")

    def on_batch_end(self, batch, logs=None):
        try:
            current_time = time.time()
            time_since_last_batch = current_time - self.last_batch_time
            
            # Check if training might be stuck
            if time_since_last_batch > self.stuck_threshold:
                self.logger.warning(f"Training might be stuck! {time_since_last_batch:.1f}s since last batch")
            
            if batch % 50 == 0:  # Log every 50 batches
                loss = logs.get('loss', 0.0)
                mae = logs.get('safe_mae', 0.0)
                
                # Ensure numeric values
                loss = float(loss) if isinstance(loss, (int, float)) else 0.0
                mae = float(mae) if isinstance(mae, (int, float)) else 0.0
                
                self.logger.info(
                    f"  Batch {batch}/{self.params.get('steps', 'None')}: "
                    f"loss: {loss:.4f}, "
                    f"mae: {mae:.4f}, "
                    f"batch_time: {time_since_last_batch:.2f}s"
                )
            
            self.last_batch_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error in on_batch_end: {str(e)}")
            self.logger.error(f"Current batch state - batch: {batch}, logs: {logs}")

    def on_epoch_end(self, epoch, logs=None):
        try:
            if logs is None:
                logs = {}
                
            epoch_time = time.time() - self.epoch_start
            self.epoch_times.append(epoch_time)
            
            # Calculate ETA
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = self.params.get('epochs', 0) - (epoch + 1)
            eta_seconds = int(avg_epoch_time * remaining_epochs)
            eta = str(timedelta(seconds=eta_seconds))
            
            # Get metrics with safe conversion to float
            loss = float(logs.get('loss', 0.0)) if isinstance(logs.get('loss'), (int, float)) else 0.0
            mae = float(logs.get('safe_mae', 0.0)) if isinstance(logs.get('safe_mae'), (int, float)) else 0.0
            val_loss = float(logs.get('val_loss', float('inf'))) if isinstance(logs.get('val_loss'), (int, float)) else float('inf')
            val_mae = float(logs.get('val_safe_mae', 0.0)) if isinstance(logs.get('val_safe_mae'), (int, float)) else 0.0
            
            # Track best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Detailed epoch summary
            self.logger.info(
                f"\nEpoch {epoch+1}/{self.params.get('epochs', '?')} Summary:")
            self.logger.info(f"Time taken: {epoch_time:.1f}s")
            self.logger.info(
                f"Training metrics:\n"
                f"  - loss: {loss:.4f}\n"
                f"  - mae: {mae:.4f}"
            )
            
            # Only log validation metrics if they exist
            if val_loss != float('inf'):
                self.logger.info(
                    f"Validation metrics:\n"
                    f"  - val_loss: {val_loss:.4f}\n"
                    f"  - val_mae: {val_mae:.4f}"
                    f"{' (Best)' if is_best else ''}"
                )
            
            self.logger.info(f"ETA: {eta}")
            
            # Memory tracking
            gc.collect()
            tf.keras.backend.clear_session()
            
            self.logger.info("-"*50)
            
            # Force flush of logging
            sys.stdout.flush()
            sys.stderr.flush()
            
        except Exception as e:
            self.logger.error(f"Error in on_epoch_end: {str(e)}")
            self.logger.error("Attempting to continue training...")

    def on_train_end(self, logs=None):
        try:
            total_time = time.time() - self.start_time
            total_time_str = str(timedelta(seconds=int(total_time)))
            
            self.logger.info("="*50)
            self.logger.info("Training completed")
            self.logger.info(f"Total time: {total_time_str}")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            self.logger.info("Final memory cleanup...")
            self.logger.info("="*50)
        except Exception as e:
            self.logger.error(f"Error in on_train_end: {str(e)}")

class SafeValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger, val_dataset, model_save_path):
        super().__init__()
        self.logger = logger
        self.val_dataset = val_dataset
        self.model_save_path = model_save_path
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        try:
            val_loss = logs.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save(self.model_save_path)
                self.logger.info(f"Model saved to {self.model_save_path}")
        except Exception as e:
            self.logger.error(f"Error in SafeValidationCallback: {e}")

class DatasetValidator:
    """Utility class for validating TFRecord files."""
    
    def __init__(self, sequence_length: int, num_features: int):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.logger = logging.getLogger('training_logger')
        
    def validate_tfrecord(self, file_path: str) -> bool:
        """Validate a single TFRecord file."""
        try:
            # Set up feature description
            feature_description = {
                'features': tf.io.FixedLenFeature([self.sequence_length * self.num_features], tf.float32),
                'target': tf.io.FixedLenFeature([1], tf.float32),
            }
            
            dataset = tf.data.TFRecordDataset(
                file_path,
                compression_type=None,
                buffer_size=8*1024*1024,  # 8MB buffer
                num_parallel_reads=1
            )
            
            # Check first few records
            valid_records = 0
            corrupted_records = 0
            
            for record in dataset.take(2000):  # Check first 100 records
                try:
                    example = tf.io.parse_single_example(record, feature_description)
                    features = tf.reshape(example['features'], 
                                       [self.sequence_length, self.num_features])
                    
                    # Validate shapes and values
                    if (features.shape == (self.sequence_length, self.num_features) and
                        tf.reduce_all(tf.math.is_finite(features)) and
                        tf.reduce_all(tf.math.is_finite(example['target']))):
                        valid_records += 1
                    else:
                        corrupted_records += 1
                        
                except Exception:
                    corrupted_records += 1
                    continue
                    
            # File is valid if it has enough valid records
            total_checked = valid_records + corrupted_records
            if total_checked == 0:
                return False
                
            valid_ratio = valid_records / total_checked
            is_valid = valid_ratio >= 0.8  # At least 80% valid records
            
            if is_valid:
                self.logger.info(f"Validated {file_path}: {valid_records}/{total_checked} valid records")
            else:
                self.logger.warning(f"Invalid file {file_path}: {valid_records}/{total_checked} valid records")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error validating {file_path}: {str(e)}")
            return False
            
    def validate_files_parallel(self, file_paths: List[str], num_workers: int = None) -> List[str]:
        """Validate multiple TFRecord files in parallel."""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count() - 1, 8)
            
        valid_files = []
        total_files = len(file_paths)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(self.validate_tfrecord, f): f 
                for f in file_paths
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    if future.result():
                        valid_files.append(file_path)
                except Exception as e:
                    self.logger.error(f"Validation failed for {file_path}: {str(e)}")
                    
                completed += 1
                if completed % 10 == 0:
                    self.logger.info(f"Validated {completed}/{total_files} files")
                    
        self.logger.info(f"Found {len(valid_files)} valid files out of {total_files}")
        return valid_files

class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
    """Enhanced Early Stopping callback with better logging."""
    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.best_epoch = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None and self.monitor_op(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.logger.info(
                    f"\nEarly stopping triggered at epoch {epoch+1}:\n"
                    f"  Best {self.monitor}: {self.best:.6f} (epoch {self.best_epoch+1})\n"
                    f"  No improvement for {self.patience} epochs"
                )
                self.model.stop_training = True

class PerformanceMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger, batch_size, total_batches):
        super().__init__()
        self.logger = logger
        self.batch_size = batch_size
        self.total_batches = total_batches
        self.batch_times = []
        self.epoch_times = []
        self.start_time = None
        self.batch_start = None
        self.training_start = None
        self.samples_per_second = []
        
    def on_train_begin(self, logs=None):
        self.training_start = time.time()
        self.logger.info("Training started - Performance Monitoring Enabled")
        self.logger.info(f"Total batches per epoch: {self.total_batches}")
        self.logger.info(f"Batch size: {self.batch_size}")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.batch_times = []
        gc.collect()  # Clean memory between epochs
        
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()
        
    def on_train_batch_end(self, batch, logs=None):
        if self.batch_start is None:
            return
            
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
        
        # Calculate samples per second
        samples_per_second = self.batch_size / batch_time
        self.samples_per_second.append(samples_per_second)
        
        if batch % 50 == 0:
            avg_samples_per_second = np.mean(self.samples_per_second[-50:]) if self.samples_per_second else 0
            avg_batch_time = np.mean(self.batch_times[-50:]) if self.batch_times else 0
            
            self.logger.info(
                f"Batch {batch}/{self.total_batches} - "
                f"Loss: {logs.get('loss', 0):.4f}, "
                f"MAE: {logs.get('safe_mae', 0):.4f}, "
                f"Samples/sec: {avg_samples_per_second:.1f}, "
                f"Batch time: {avg_batch_time*1000:.1f}ms"
            )
            
            # Monitor for potential issues
            if avg_batch_time > 1.0:  # Warning if batch takes more than 1 second
                self.logger.warning(f"Slow batch processing detected: {avg_batch_time:.2f}s per batch")
            
            if np.std(self.batch_times[-50:]) > avg_batch_time:
                self.logger.warning("High variance in batch processing times detected")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        # Calculate metrics
        avg_batch_time = np.mean(self.batch_times)
        avg_samples_per_second = np.mean(self.samples_per_second)
        
        # Memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.logger.info(
            f"\nEpoch {epoch+1} Performance Metrics:"
            f"\n  Time: {epoch_time:.1f}s"
            f"\n  Avg batch time: {avg_batch_time*1000:.1f}ms"
            f"\n  Avg samples/second: {avg_samples_per_second:.1f}"
            f"\n  Memory usage: {memory_usage:.1f}MB"
            f"\n  Training metrics:"
            f"\n    Loss: {logs.get('loss', 0):.4f}"
            f"\n    MAE: {logs.get('safe_mae', 0):.4f}"
            f"\n    Val Loss: {logs.get('val_loss', 0):.4f}"
            f"\n    Val MAE: {logs.get('val_safe_mae', 0):.4f}"
        )
        
    def on_train_end(self, logs=None):
        total_time = time.time() - self.training_start
        avg_epoch_time = np.mean(self.epoch_times)
        avg_samples_per_second = np.mean(self.samples_per_second)
        
        self.logger.info(
            f"\nTraining Performance Summary:"
            f"\n  Total time: {total_time:.1f}s"
            f"\n  Average epoch time: {avg_epoch_time:.1f}s"
            f"\n  Average samples/second: {avg_samples_per_second:.1f}"
            f"\n  Final metrics:"
            f"\n    Loss: {logs.get('loss', 0):.4f}"
            f"\n    MAE: {logs.get('safe_mae', 0):.4f}"
        )

class SafeMSE(tf.keras.losses.Loss):
    def __init__(self, name='safe_mse', reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self._name = name

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_pred = tf.clip_by_value(y_pred, -10.0, 10.0)
        diff = y_true - y_pred
        diff = tf.clip_by_value(diff, -10.0, 10.0)
        squared_diff = tf.square(diff)
        
        valid_mask = tf.math.is_finite(squared_diff)
        safe_diff = tf.where(valid_mask, squared_diff, tf.zeros_like(squared_diff))
        return tf.reduce_mean(safe_diff) + 1e-7

class SafeMAE(tf.keras.metrics.Metric):
    def __init__(self, name='safe_mae', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, **kwargs)
        self._name = name
        self.reduction = reduction
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_pred = tf.clip_by_value(y_pred, -10.0, 10.0)
        diff = y_true - y_pred
        abs_diff = tf.abs(tf.clip_by_value(diff, -10.0, 10.0))
        
        valid_mask = tf.math.is_finite(abs_diff)
        safe_diff = tf.where(valid_mask, abs_diff, tf.zeros_like(abs_diff))
        
        if sample_weight is not None:
            safe_diff = safe_diff * tf.cast(sample_weight, tf.float32)
            
        self.total.assign_add(tf.reduce_sum(safe_diff))
        self.count.assign_add(tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.float32)), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'reduction': self.reduction}    

class BaseModel:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = None
        self.num_features = None
        self.logger = logging.getLogger('training_logger')
        
        # Enable XLA compilation
        tf.config.optimizer.set_jit(True)
        
        # Configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Use memory limit only, don't set memory growth
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]
                )
                self.logger.info("GPU memory limit set to 6GB")
                
                # Set up distribution strategy
                self.strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
                self.logger.info("Using GPU with OneDeviceStrategy")
            except RuntimeError as e:
                self.logger.error(f"Error configuring GPU: {e}")
                self.strategy = tf.distribute.get_strategy()
        else:
            self.logger.info("No GPU found, using default strategy")
            self.strategy = tf.distribute.get_strategy()

    def build_model(self, num_features: int) -> tf.keras.Model:
        """
        Abstract method to be implemented by child classes
        """
        raise NotImplementedError("Subclasses must implement build_model")

    def _ensure_correct_shape(self, dataset, name="dataset"):
        """Verify and fix dataset shape."""
        try:
            for features, target in dataset.take(1):
                self.logger.info(f"Checking {name} dataset shape - Features: {features.shape}, Target: {target.shape}")
                
                # Double-batched: (batch1, batch2, seq, features)
                if len(features.shape) == 4:
                    self.logger.info(f"Unbatching {name} dataset")
                    return dataset.unbatch()
                    
                # Verify expected shape: (batch, seq, features)
                elif len(features.shape) == 3:
                    if features.shape[1] != self.sequence_length:
                        raise ValueError(f"Incorrect sequence length in {name} dataset")
                    return dataset
                    
                else:
                    raise ValueError(f"Unexpected shape in {name} dataset: {features.shape}")
                    
        except Exception as e:
            self.logger.error(f"Error in _ensure_correct_shape for {name} dataset: {e}")
            raise

    def train_model(self, X, y, X_val, y_val, epochs: int = 30, batch_size: int = 16, model_save_path: str = 'model.h5', total_batches: int = None):
        """Train model with robust validation handling."""
        try:
            # Initial setup
            K.clear_session()
            
            self.logger.info("="*80)
            self.logger.info(f"Starting training for {self.__class__.__name__}")
            self.logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}")
            
            # Get features if not set
            if self.num_features is None:
                for features, _ in X.take(1):
                    self.num_features = features.shape[-1]
                    break
                self.logger.info(f"Set num_features to {self.num_features}")
            
            # Build model if needed
            if self.model is None:
                self.build_model(self.num_features)
                self.logger.info(f"Built model with {self.num_features} features")
            
            # Configure optimizer
            optimizer = Adam(
                learning_rate=1e-4,
                clipnorm=1.0,
                clipvalue=0.5,
                epsilon=1e-7,
                amsgrad=True
            )
            
            if tf.config.list_physical_devices('GPU'):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                self.logger.info("Mixed precision training enabled")
            
            # Get batch count from dataset cardinality or use provided total_batches
            try:
                cardinality = X.cardinality().numpy()
                if cardinality == tf.data.INFINITE_CARDINALITY or cardinality == tf.data.UNKNOWN_CARDINALITY:
                    # Use total_batches if provided, otherwise estimate
                    estimated_batches = total_batches if total_batches else 1000
                else:
                    estimated_batches = cardinality
            except Exception:
                estimated_batches = total_batches if total_batches else 1000

            # Configure callbacks
            callbacks = [
                CustomEarlyStopping(
                    logger=self.logger,
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    mode='min',
                    min_delta=1e-3,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=model_save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    save_weights_only=False,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_delta=1e-3,
                    cooldown=1,
                    min_lr=1e-6,
                    verbose=1
                ),
                PerformanceMonitorCallback(
                    self.logger,
                    batch_size,
                    estimated_batches
                )
            ]
            
            # Compile model
            self.model.compile(
                optimizer=optimizer,
                loss=SafeMSE(),
                metrics=[SafeMAE()]
            )
            
            # Train with validation settings
            try:
                self.logger.info("\nStarting training...")
                history = self.model.fit(
                    X,
                    epochs=epochs,
                    validation_data=X_val,
                    validation_freq=2,  # Validate every 2 epochs
                    callbacks=callbacks,
                    verbose=0
                )
                return history
                
            except tf.errors.DataLossError as e:
                self.logger.warning(f"Data loss during training: {str(e)}")
                # Try training without validation
                self.logger.info("Retrying training without validation...")
                history = self.model.fit(
                    X,
                    epochs=epochs,
                    callbacks=[cb for cb in callbacks if not isinstance(cb, EarlyStopping)],
                    verbose=0
                )
                return history
                
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise
            
        finally:
            gc.collect()
            K.clear_session()

    def predict(self, X):
        """Make predictions with error handling"""
        if self.model is None:
            raise ValueError("Model is not built or loaded")
        try:
            predictions = self.model.predict(
                X,
                batch_size=16,
                workers=1,
                use_multiprocessing=False,
                verbose=0
            )
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return None
        

class LSTMModel(BaseModel):
    def __init__(self, sequence_length: int = 60, units: int = 128, 
                 dropout_rate: float = 0.2, l2_reg: float = 0.001):
        super().__init__(sequence_length)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.logger = logging.getLogger('training_logger')

    def build_model(self, num_features: int) -> tf.keras.Model:
        with self.strategy.scope():
            inputs = Input(shape=(self.sequence_length, num_features), name='input_layer')
            
            # Initial normalization
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = BatchNormalization(name='input_norm')(x)
            
            # Bidirectional LSTM layers
            x = Bidirectional(LSTM(
                units=self.units*2,
                return_sequences=True,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg/2),
                name='lstm_1'
            ))(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(self.dropout_rate, name='drop_1')(x)
            
            x = Bidirectional(LSTM(
                units=self.units,
                return_sequences=True,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg/2),
                name='lstm_2'
            ))(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(self.dropout_rate, name='drop_2')(x)
            
            x = Bidirectional(LSTM(
                units=self.units//2,
                return_sequences=False,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg/2),
                name='lstm_3'
            ))(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(self.dropout_rate, name='drop_3')(x)
            
            # Dense layers with skip connections
            dense_1 = Dense(self.units//2, activation='relu', kernel_regularizer=l2(self.l2_reg), name='dense_1')(x)
            dense_1 = LayerNormalization(epsilon=1e-6)(dense_1)
            dense_1 = Dropout(self.dropout_rate/2)(dense_1)
            
            dense_2 = Dense(self.units//4, activation='relu', kernel_regularizer=l2(self.l2_reg), name='dense_2')(dense_1)
            dense_2 = LayerNormalization(epsilon=1e-6)(dense_2)
            dense_2 = Dropout(self.dropout_rate/2)(dense_2)
            
            # Output layer
            outputs = Dense(1, name='output')(dense_2)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs, name='lstm_model')
            
            self.model = model
            self.logger.info(f"LSTM model built with {self.units} base units")
            self.logger.info(f"Model summary:\n{model.summary()}")
            
            return model


class TransformerModel(BaseModel):
    def __init__(self, sequence_length: int = 60, num_heads: int = 4, 
                 ff_dim: int = 128, num_transformer_blocks: int = 2, 
                 dropout_rate: float = 0.1):
        super().__init__(sequence_length)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.logger = logging.getLogger('training_logger')

    def build_model(self, num_features: int) -> tf.keras.Model:
        with self.strategy.scope():
            inputs = Input(shape=(self.sequence_length, num_features))
            
            # Initial normalization and projection
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = Dense(self.ff_dim)(x)
            
            # Positional encoding
            positions = tf.range(self.sequence_length, dtype=tf.float32)[:, tf.newaxis]
            indices = tf.range(self.ff_dim, dtype=tf.float32)[tf.newaxis, :]
            
            # Ensure consistent dtype for division
            angle_rates = 1.0 / (10000.0 ** (2 * (indices // 2) / tf.cast(self.ff_dim, tf.float32)))
            angle_rads = positions * angle_rates

            # Apply sin to even indices
            sines = tf.math.sin(angle_rads[:, 0::2])
            # Apply cos to odd indices
            cosines = tf.math.cos(angle_rads[:, 1::2])

            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = tf.expand_dims(pos_encoding, 0)
            
            # Add positional encoding
            x = x + tf.cast(pos_encoding, x.dtype)
            x = Dropout(self.dropout_rate)(x)
            
            # Transformer blocks
            for _ in range(self.num_transformer_blocks):
                x = self._transformer_block(x)
            
            # Global pooling and dense layers
            x = GlobalAveragePooling1D()(x)
            x = Dense(self.ff_dim // 2, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(self.ff_dim // 4, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dense(1)(x)
            
            model = Model(inputs=inputs, outputs=x)
            self.model = model
            
            self.logger.info(f"Transformer model built with {self.num_heads} heads and {self.ff_dim} FF dimension")
            self.logger.info(f"Model summary:\n{model.summary()}")
            
            return model

    def _transformer_block(self, x):
        """Transformer block with layer normalization and residual connections."""
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.ff_dim // self.num_heads,
            dropout=self.dropout_rate
        )(x, x)
        
        # Add & Normalize
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ffn = Sequential([
            Dense(self.ff_dim * 2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.ff_dim)
        ])
        
        ffn_output = ffn(x)
        
        # Add & Normalize
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)

    def _add_positional_encoding(self, x):
        """Add positional encoding to the input"""
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        input_dim = tf.shape(x)[2]
        
        # Create positional encoding
        positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, input_dim, 2, dtype=tf.float32) * 
            -(tf.math.log(10000.0) / input_dim)
        )
        
        # Calculate sine and cosine encodings
        pos_encoding = tf.zeros((seq_length, input_dim))
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.reshape(tf.range(0, seq_length), (-1, 1)),
            tf.sin(positions * div_term)
        )
        if input_dim % 2 == 0:
            pos_encoding = tf.tensor_scatter_nd_update(
                pos_encoding,
                tf.reshape(tf.range(1, seq_length), (-1, 1)),
                tf.cos(positions[:-1] * div_term)
            )
        
        # Expand to batch size
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])
        
        return x + tf.cast(pos_encoding, x.dtype)

    def _positional_encoding(self, sequence_length, d_model):
        positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        indices = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (indices // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates

        # Apply sin to even indices
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        return tf.cast(pos_encoding, dtype=tf.float32)


class DenseModel(BaseModel):
    def __init__(self, sequence_length: int = 60, layers: List[Dict[str, Any]] = None):
        super().__init__(sequence_length)
        self.layers_config = layers or [
            {'units': 128, 'activation': 'relu'},
            {'units': 64, 'activation': 'relu'}
        ]
        self.dropout_rate = 0.2
        self.l2_reg = 0.001
        self.logger = logging.getLogger('training_logger')

    def build_model(self, num_features: int) -> tf.keras.Model:
        with self.strategy.scope():
            model = Sequential()
            # Input layer with mixed precision
            model.add(Input(shape=(self.sequence_length, num_features), dtype='float16'))
            model.add(Flatten())
            model.add(BatchNormalization())

            # Add layers from config
            for layer in self.layers_config:
                units = layer['units']
                activation = layer['activation']
                model.add(Dense(
                    units=units,
                    kernel_regularizer=l2(self.l2_reg)
                ))
                model.add(BatchNormalization())
                model.add(Activation(activation))
                model.add(Dropout(self.dropout_rate))

            # Add final layers
            model.add(Dense(32, kernel_regularizer=l2(self.l2_reg)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout_rate / 2))
            model.add(Dense(1, activation='linear'))

            self.model = model
            self.logger.info(f"Dense model built with {len(self.layers_config)} hidden layers")
            return model


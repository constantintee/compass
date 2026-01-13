# models.py
# Optimized for maximum prediction accuracy

import logging
import os
import sys
import time
import datetime
import math
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
    Input, Dense, Dropout, LSTM, GRU, Flatten, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, GlobalMaxPooling1D,
    BatchNormalization, Bidirectional, Activation, Add, Concatenate,
    Conv1D, MaxPooling1D, AveragePooling1D, SpatialDropout1D,
    Attention, AdditiveAttention, Layer, Multiply, Permute, RepeatVector,
    TimeDistributed, Lambda
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard,
    LearningRateScheduler, Callback
)
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras import backend as K
from tensorflow.keras.mixed_precision import set_global_policy
# Set the global mixed precision policy to 'mixed_float16'
set_global_policy('mixed_float16')

from typing import List, Dict, Any, Optional, Tuple, Callable

from utils import MemoryMonitor, monitor_memory, log_memory_usage, clean_memory, get_memory_usage, load_hyperparameters


# Configure logging if not already configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('training_logger')


# =============================================================================
# ADVANCED LOSS FUNCTIONS FOR BETTER PREDICTIONS
# =============================================================================

class HuberLoss(tf.keras.losses.Loss):
    """Huber loss - more robust to outliers than MSE."""
    def __init__(self, delta=1.0, name='huber_loss', reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.delta = delta
        self._name = name

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, -10.0, 10.0)

        error = y_true - y_pred
        abs_error = tf.abs(error)

        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic

        loss = 0.5 * tf.square(quadratic) + self.delta * linear

        valid_mask = tf.math.is_finite(loss)
        safe_loss = tf.where(valid_mask, loss, tf.zeros_like(loss))
        return tf.reduce_mean(safe_loss) + 1e-7


class DirectionalLoss(tf.keras.losses.Loss):
    """Loss that penalizes wrong directional predictions."""
    def __init__(self, direction_weight=0.3, name='directional_loss', reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.direction_weight = direction_weight
        self._name = name

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, -10.0, 10.0)

        # MSE component
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Directional component - penalize wrong direction predictions
        true_direction = tf.sign(y_true[1:] - y_true[:-1])
        pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])

        direction_match = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)
        direction_loss = 1.0 - tf.reduce_mean(direction_match)

        total_loss = (1.0 - self.direction_weight) * mse + self.direction_weight * direction_loss
        return total_loss + 1e-7


class CombinedLoss(tf.keras.losses.Loss):
    """Combined loss: Huber + MSE + Directional penalty for best predictions."""
    def __init__(self, huber_weight=0.4, mse_weight=0.4, direction_weight=0.2,
                 delta=1.0, name='combined_loss', reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.huber_weight = huber_weight
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.delta = delta
        self._name = name

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, -10.0, 10.0)

        # Huber component
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber = tf.reduce_mean(0.5 * tf.square(quadratic) + self.delta * linear)

        # MSE component
        mse = tf.reduce_mean(tf.square(error))

        # Directional component
        if tf.shape(y_true)[0] > 1:
            true_direction = tf.sign(y_true[1:] - y_true[:-1])
            pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])
            direction_match = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)
            direction_loss = 1.0 - tf.reduce_mean(direction_match)
        else:
            direction_loss = 0.0

        # Combined
        total_loss = (self.huber_weight * huber +
                     self.mse_weight * mse +
                     self.direction_weight * direction_loss)

        valid_mask = tf.math.is_finite(total_loss)
        safe_loss = tf.where(valid_mask, total_loss, tf.constant(0.1))
        return safe_loss + 1e-7


# =============================================================================
# CUSTOM ATTENTION LAYERS
# =============================================================================

class TemporalAttention(Layer):
    """Self-attention layer for temporal sequences with learnable query/key/value."""
    def __init__(self, units, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W_q = self.add_weight(name='W_q', shape=(feature_dim, self.units),
                                   initializer='glorot_uniform', trainable=True)
        self.W_k = self.add_weight(name='W_k', shape=(feature_dim, self.units),
                                   initializer='glorot_uniform', trainable=True)
        self.W_v = self.add_weight(name='W_v', shape=(feature_dim, self.units),
                                   initializer='glorot_uniform', trainable=True)
        self.W_o = self.add_weight(name='W_o', shape=(self.units, feature_dim),
                                   initializer='glorot_uniform', trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        # Compute Q, K, V
        Q = tf.matmul(x, self.W_q)
        K = tf.matmul(x, self.W_k)
        V = tf.matmul(x, self.W_v)

        # Scaled dot-product attention
        d_k = tf.cast(self.units, tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention to values
        context = tf.matmul(attention_weights, V)
        output = tf.matmul(context, self.W_o)

        return output

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        config.update({'units': self.units})
        return config


class SqueezeExcitation(Layer):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, reduction_ratio=4, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
        super(SqueezeExcitation, self).build(input_shape)

    def call(self, x):
        # Squeeze: global average pooling
        squeeze = tf.reduce_mean(x, axis=1, keepdims=True)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        # Scale
        return x * excitation

    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine decay learning rate schedule for better convergence."""
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps,
                 min_learning_rate=1e-7, warmup_target=None):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_learning_rate = min_learning_rate
        self.warmup_target = warmup_target or initial_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Warmup phase
        warmup_lr = self.initial_learning_rate + (self.warmup_target - self.initial_learning_rate) * (step / warmup_steps)

        # Cosine decay phase
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(math.pi * decay_step / decay_steps))
        decay_lr = self.min_learning_rate + (self.warmup_target - self.min_learning_rate) * cosine_decay

        # Return warmup LR if in warmup phase, else decay LR
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'min_learning_rate': self.min_learning_rate,
            'warmup_target': self.warmup_target
        }


class CosineAnnealingWarmRestarts(Callback):
    """Cosine annealing with warm restarts for escaping local minima."""
    def __init__(self, initial_lr=1e-3, T_0=10, T_mult=2, eta_min=1e-6, logger=None):
        super(CosineAnnealingWarmRestarts, self).__init__()
        self.initial_lr = initial_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.logger = logger
        self.T_cur = 0
        self.T_i = T_0

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate current learning rate
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

        K.set_value(self.model.optimizer.learning_rate, lr)

        if self.logger:
            self.logger.info(f"Epoch {epoch+1}: LR set to {lr:.6f} (T_cur={self.T_cur}, T_i={self.T_i})")

        # Update counters
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult


# =============================================================================
# STOCHASTIC WEIGHT AVERAGING CALLBACK
# =============================================================================

class StochasticWeightAveraging(Callback):
    """SWA for better generalization - averages weights from multiple epochs."""
    def __init__(self, start_epoch=10, swa_lr=1e-4, logger=None):
        super(StochasticWeightAveraging, self).__init__()
        self.start_epoch = start_epoch
        self.swa_lr = swa_lr
        self.logger = logger
        self.swa_weights = None
        self.swa_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            # Update SWA weights
            current_weights = self.model.get_weights()

            if self.swa_weights is None:
                self.swa_weights = [np.copy(w) for w in current_weights]
            else:
                for i, w in enumerate(current_weights):
                    self.swa_weights[i] = (self.swa_weights[i] * self.swa_count + w) / (self.swa_count + 1)

            self.swa_count += 1

            if self.logger:
                self.logger.info(f"SWA: Updated averaged weights (count={self.swa_count})")

    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            self.model.set_weights(self.swa_weights)
            if self.logger:
                self.logger.info(f"SWA: Applied averaged weights from {self.swa_count} epochs")


# =============================================================================
# DATA AUGMENTATION FOR TIME SERIES
# =============================================================================

class TimeSeriesAugmentation:
    """Data augmentation techniques for time series data."""

    @staticmethod
    def jitter(x, sigma=0.03):
        """Add random noise to the sequence."""
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=sigma)
        return x + noise

    @staticmethod
    def scaling(x, sigma=0.1):
        """Scale the sequence by a random factor."""
        factor = tf.random.normal([1], mean=1.0, stddev=sigma)
        return x * factor

    @staticmethod
    def magnitude_warp(x, sigma=0.2, knot=4):
        """Warp the magnitude of the sequence."""
        orig_shape = tf.shape(x)
        seq_len = orig_shape[0]

        # Create random warping curve
        warp_steps = tf.linspace(0.0, 1.0, knot + 2)
        warp_values = tf.random.normal([knot + 2], mean=1.0, stddev=sigma)

        # Interpolate to full sequence length
        indices = tf.linspace(0.0, tf.cast(knot + 1, tf.float32), seq_len)
        indices_floor = tf.cast(tf.floor(indices), tf.int32)
        indices_ceil = tf.minimum(indices_floor + 1, knot + 1)

        weights = indices - tf.cast(indices_floor, tf.float32)

        warp_curve = tf.gather(warp_values, indices_floor) * (1 - weights) + \
                     tf.gather(warp_values, indices_ceil) * weights

        return x * tf.expand_dims(warp_curve, -1)

    @staticmethod
    def window_slice(x, reduce_ratio=0.9):
        """Randomly slice a window from the sequence."""
        seq_len = tf.shape(x)[0]
        target_len = tf.cast(tf.cast(seq_len, tf.float32) * reduce_ratio, tf.int32)

        start = tf.random.uniform([], 0, seq_len - target_len, dtype=tf.int32)
        return x[start:start + target_len]

    @staticmethod
    def augment_batch(x, y, augment_prob=0.5):
        """Apply random augmentations to a batch."""
        if tf.random.uniform([]) < augment_prob:
            aug_type = tf.random.uniform([], 0, 3, dtype=tf.int32)

            if aug_type == 0:
                x = TimeSeriesAugmentation.jitter(x)
            elif aug_type == 1:
                x = TimeSeriesAugmentation.scaling(x)
            else:
                x = TimeSeriesAugmentation.magnitude_warp(x)

        return x, y


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

    def train_model(self, X, y, X_val, y_val, epochs: int = 50, batch_size: int = 32, model_save_path: str = 'model.h5', total_batches: int = None):
        """
        Enhanced training with:
        - Combined loss function (Huber + MSE + Directional)
        - Learning rate warmup with cosine decay
        - Stochastic Weight Averaging (SWA)
        - Cosine annealing with warm restarts
        """
        try:
            # Initial setup
            K.clear_session()

            self.logger.info("="*80)
            self.logger.info(f"Starting OPTIMIZED training for {self.__class__.__name__}")
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

            # Get batch count from dataset cardinality or use provided total_batches
            try:
                cardinality = X.cardinality().numpy()
                if cardinality == tf.data.INFINITE_CARDINALITY or cardinality == tf.data.UNKNOWN_CARDINALITY:
                    estimated_batches = total_batches if total_batches else 1000
                else:
                    estimated_batches = cardinality
            except Exception:
                estimated_batches = total_batches if total_batches else 1000

            # Calculate warmup and decay steps
            warmup_epochs = 5
            warmup_steps = warmup_epochs * estimated_batches
            total_steps = epochs * estimated_batches
            decay_steps = total_steps - warmup_steps

            # Configure learning rate schedule with warmup + cosine decay
            lr_schedule = WarmupCosineDecay(
                initial_learning_rate=1e-6,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                min_learning_rate=1e-7,
                warmup_target=3e-4  # Peak learning rate after warmup
            )

            # Configure optimizer with weight decay (AdamW)
            optimizer = AdamW(
                learning_rate=lr_schedule,
                weight_decay=0.01,
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

            # Configure callbacks with advanced features
            callbacks = [
                # Early stopping with longer patience for better convergence
                CustomEarlyStopping(
                    logger=self.logger,
                    monitor='val_loss',
                    patience=10,  # Increased patience
                    restore_best_weights=True,
                    mode='min',
                    min_delta=1e-4,
                    verbose=1
                ),
                # Save best model
                ModelCheckpoint(
                    filepath=model_save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    save_weights_only=False,
                    verbose=1
                ),
                # Cosine annealing with warm restarts for escaping local minima
                CosineAnnealingWarmRestarts(
                    initial_lr=3e-4,
                    T_0=10,
                    T_mult=2,
                    eta_min=1e-6,
                    logger=self.logger
                ),
                # Stochastic Weight Averaging for better generalization
                StochasticWeightAveraging(
                    start_epoch=max(15, epochs // 3),  # Start SWA after 1/3 of training
                    swa_lr=1e-4,
                    logger=self.logger
                ),
                # Performance monitoring
                PerformanceMonitorCallback(
                    self.logger,
                    batch_size,
                    estimated_batches
                )
            ]

            # Use Combined Loss for best predictions (Huber + MSE + Directional)
            loss_function = CombinedLoss(
                huber_weight=0.4,
                mse_weight=0.4,
                direction_weight=0.2,
                delta=1.0
            )

            # Compile model with combined loss
            self.model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=[SafeMAE()]
            )

            self.logger.info(f"Using CombinedLoss (Huber: 0.4, MSE: 0.4, Directional: 0.2)")
            self.logger.info(f"Learning rate schedule: Warmup ({warmup_epochs} epochs) + Cosine Decay")
            self.logger.info(f"SWA enabled starting at epoch {max(15, epochs // 3)}")

            # Train with validation settings
            try:
                self.logger.info("\nStarting optimized training...")
                history = self.model.fit(
                    X,
                    epochs=epochs,
                    validation_data=X_val,
                    validation_freq=1,  # Validate every epoch for better monitoring
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
                    callbacks=[cb for cb in callbacks if not isinstance(cb, (EarlyStopping, CustomEarlyStopping))],
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
    """Enhanced LSTM with multi-head attention and residual connections for superior predictions."""

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

            # Initial normalization and feature projection
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = BatchNormalization(name='input_norm')(x)

            # Project features to consistent dimension
            x = Dense(self.units, kernel_regularizer=l2(self.l2_reg))(x)
            x = LayerNormalization(epsilon=1e-6)(x)

            # ==== FIRST LSTM BLOCK WITH ATTENTION ====
            # Bidirectional LSTM layer 1
            lstm_1 = Bidirectional(LSTM(
                units=self.units,
                return_sequences=True,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg/2),
                recurrent_dropout=0.1,
                name='lstm_1'
            ))(x)
            lstm_1 = LayerNormalization(epsilon=1e-6)(lstm_1)
            lstm_1 = Dropout(self.dropout_rate)(lstm_1)

            # Multi-head self-attention on LSTM output
            attention_1 = MultiHeadAttention(
                num_heads=4,
                key_dim=self.units // 4,
                dropout=self.dropout_rate,
                name='mha_1'
            )(lstm_1, lstm_1)
            attention_1 = LayerNormalization(epsilon=1e-6)(attention_1)

            # Residual connection
            x = Add()([lstm_1, attention_1])
            x = Dropout(self.dropout_rate)(x)

            # ==== SECOND LSTM BLOCK WITH ATTENTION ====
            lstm_2 = Bidirectional(LSTM(
                units=self.units // 2,
                return_sequences=True,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg/2),
                recurrent_dropout=0.1,
                name='lstm_2'
            ))(x)
            lstm_2 = LayerNormalization(epsilon=1e-6)(lstm_2)
            lstm_2 = Dropout(self.dropout_rate)(lstm_2)

            # Multi-head self-attention
            attention_2 = MultiHeadAttention(
                num_heads=4,
                key_dim=self.units // 8,
                dropout=self.dropout_rate,
                name='mha_2'
            )(lstm_2, lstm_2)
            attention_2 = LayerNormalization(epsilon=1e-6)(attention_2)

            # Residual connection
            x = Add()([lstm_2, attention_2])

            # ==== TEMPORAL ATTENTION FOR FINAL SEQUENCE ====
            # Apply temporal attention to focus on important timesteps
            x = TemporalAttention(units=self.units // 2, name='temporal_attention')(x)
            x = LayerNormalization(epsilon=1e-6)(x)

            # ==== FINAL LSTM LAYER ====
            lstm_3 = Bidirectional(LSTM(
                units=self.units // 4,
                return_sequences=False,
                kernel_regularizer=l2(self.l2_reg),
                recurrent_regularizer=l2(self.l2_reg/2),
                name='lstm_3'
            ))(x)
            lstm_3 = LayerNormalization(epsilon=1e-6)(lstm_3)
            lstm_3 = Dropout(self.dropout_rate)(lstm_3)

            # ==== DENSE HEAD WITH RESIDUAL ====
            dense_1 = Dense(self.units // 2, kernel_regularizer=l2(self.l2_reg))(lstm_3)
            dense_1 = BatchNormalization()(dense_1)
            dense_1 = tf.keras.layers.LeakyReLU(alpha=0.1)(dense_1)
            dense_1 = Dropout(self.dropout_rate / 2)(dense_1)

            dense_2 = Dense(self.units // 4, kernel_regularizer=l2(self.l2_reg))(dense_1)
            dense_2 = BatchNormalization()(dense_2)
            dense_2 = tf.keras.layers.LeakyReLU(alpha=0.1)(dense_2)
            dense_2 = Dropout(self.dropout_rate / 2)(dense_2)

            # Skip connection from first dense to output
            skip_dense = Dense(self.units // 4, kernel_regularizer=l2(self.l2_reg))(dense_1)
            dense_combined = Add()([dense_2, skip_dense])
            dense_combined = LayerNormalization(epsilon=1e-6)(dense_combined)

            # Output layer
            outputs = Dense(1, name='output')(dense_combined)

            # Create model
            model = Model(inputs=inputs, outputs=outputs, name='attention_lstm_model')

            self.model = model
            self.logger.info(f"Enhanced Attention-LSTM model built with {self.units} base units")
            self.logger.info(f"Model parameters: {model.count_params():,}")

            return model


class TransformerModel(BaseModel):
    """Enhanced Transformer with relative positional encoding and improved attention."""

    def __init__(self, sequence_length: int = 60, num_heads: int = 8,
                 ff_dim: int = 256, num_transformer_blocks: int = 4,
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

            # Initial normalization and projection to model dimension
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = Dense(self.ff_dim, kernel_regularizer=l2(0.001))(x)
            x = LayerNormalization(epsilon=1e-6)(x)

            # Learnable positional encoding
            pos_embedding = self.add_weight(
                name='pos_embedding',
                shape=(1, self.sequence_length, self.ff_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            x = x + pos_embedding

            # Add sinusoidal positional encoding (combined with learnable)
            positions = tf.range(self.sequence_length, dtype=tf.float32)[:, tf.newaxis]
            indices = tf.range(self.ff_dim, dtype=tf.float32)[tf.newaxis, :]
            angle_rates = 1.0 / (10000.0 ** (2 * (indices // 2) / tf.cast(self.ff_dim, tf.float32)))
            angle_rads = positions * angle_rates

            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = tf.expand_dims(pos_encoding, 0) * 0.1  # Scale down sinusoidal

            x = x + tf.cast(pos_encoding, x.dtype)
            x = Dropout(self.dropout_rate)(x)

            # Pre-transformer squeeze-excitation for channel attention
            x = SqueezeExcitation(reduction_ratio=4)(x)

            # Transformer blocks with progressive widening
            for i in range(self.num_transformer_blocks):
                x = self._transformer_block(x, block_idx=i)

            # Multi-scale pooling for better representation
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)

            # Last timestep (most recent information)
            last_timestep = x[:, -1, :]

            # Combine different representations
            x = Concatenate()([avg_pool, max_pool, last_timestep])
            x = LayerNormalization(epsilon=1e-6)(x)

            # Dense head with residual
            dense_1 = Dense(self.ff_dim, kernel_regularizer=l2(0.001))(x)
            dense_1 = BatchNormalization()(dense_1)
            dense_1 = tf.keras.layers.LeakyReLU(alpha=0.1)(dense_1)
            dense_1 = Dropout(self.dropout_rate)(dense_1)

            dense_2 = Dense(self.ff_dim // 2, kernel_regularizer=l2(0.001))(dense_1)
            dense_2 = BatchNormalization()(dense_2)
            dense_2 = tf.keras.layers.LeakyReLU(alpha=0.1)(dense_2)
            dense_2 = Dropout(self.dropout_rate)(dense_2)

            dense_3 = Dense(self.ff_dim // 4, kernel_regularizer=l2(0.001))(dense_2)
            dense_3 = BatchNormalization()(dense_3)
            dense_3 = tf.keras.layers.LeakyReLU(alpha=0.1)(dense_3)

            # Output layer
            outputs = Dense(1, name='output')(dense_3)

            model = Model(inputs=inputs, outputs=outputs, name='enhanced_transformer_model')
            self.model = model

            self.logger.info(f"Enhanced Transformer built with {self.num_heads} heads, {self.ff_dim} FF dim, {self.num_transformer_blocks} blocks")
            self.logger.info(f"Model parameters: {model.count_params():,}")

            return model

    def _transformer_block(self, x, block_idx=0):
        """Enhanced Transformer block with pre-norm and gated feed-forward."""
        # Pre-norm architecture (more stable training)
        normed = LayerNormalization(epsilon=1e-6)(x)

        # Multi-head attention with relative position bias
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.ff_dim // self.num_heads,
            value_dim=self.ff_dim // self.num_heads,
            dropout=self.dropout_rate,
            name=f'mha_block_{block_idx}'
        )(normed, normed)

        # Residual connection
        x = Add()([x, Dropout(self.dropout_rate)(attn_output)])

        # Pre-norm for feed-forward
        normed = LayerNormalization(epsilon=1e-6)(x)

        # Gated feed-forward network (GLU variant)
        ff_gate = Dense(self.ff_dim * 4, activation='sigmoid', kernel_regularizer=l2(0.001))(normed)
        ff_value = Dense(self.ff_dim * 4, kernel_regularizer=l2(0.001))(normed)
        ff_value = tf.keras.layers.LeakyReLU(alpha=0.1)(ff_value)
        ff_gated = Multiply()([ff_gate, ff_value])
        ff_gated = Dropout(self.dropout_rate)(ff_gated)
        ffn_output = Dense(self.ff_dim, kernel_regularizer=l2(0.001))(ff_gated)

        # Residual connection
        return Add()([x, Dropout(self.dropout_rate)(ffn_output)])

    def add_weight(self, name, shape, initializer, trainable):
        """Helper to add learnable weights."""
        return tf.Variable(
            initial_value=tf.keras.initializers.get(initializer)(shape),
            trainable=trainable,
            name=name
        )


class DenseModel(BaseModel):
    """CNN-based model with multi-scale convolutions for capturing temporal patterns."""

    def __init__(self, sequence_length: int = 60, layers: List[Dict[str, Any]] = None):
        super().__init__(sequence_length)
        self.layers_config = layers or [
            {'units': 512, 'activation': 'relu'},
            {'units': 256, 'activation': 'relu'},
            {'units': 128, 'activation': 'relu'},
            {'units': 64, 'activation': 'relu'}
        ]
        self.dropout_rate = 0.2
        self.l2_reg = 0.001
        self.logger = logging.getLogger('training_logger')

    def build_model(self, num_features: int) -> tf.keras.Model:
        with self.strategy.scope():
            inputs = Input(shape=(self.sequence_length, num_features), name='input_layer')

            # Initial normalization
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = BatchNormalization()(x)

            # ==== MULTI-SCALE CNN BLOCK ====
            # Different kernel sizes capture patterns at different time scales

            # Short-term patterns (3 timesteps)
            conv_3 = Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            conv_3 = BatchNormalization()(conv_3)
            conv_3 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv_3)

            # Medium-term patterns (7 timesteps)
            conv_7 = Conv1D(64, kernel_size=7, padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            conv_7 = BatchNormalization()(conv_7)
            conv_7 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv_7)

            # Longer-term patterns (15 timesteps)
            conv_15 = Conv1D(64, kernel_size=15, padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            conv_15 = BatchNormalization()(conv_15)
            conv_15 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv_15)

            # Concatenate multi-scale features
            x = Concatenate()([conv_3, conv_7, conv_15])
            x = SpatialDropout1D(self.dropout_rate)(x)

            # ==== DEEPER CNN LAYERS ====
            x = Conv1D(128, kernel_size=5, padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = SpatialDropout1D(self.dropout_rate)(x)

            x = Conv1D(256, kernel_size=3, padding='same', kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = SpatialDropout1D(self.dropout_rate)(x)

            # Squeeze-Excitation block for channel attention
            x = SqueezeExcitation(reduction_ratio=4)(x)

            # ==== GRU LAYER FOR SEQUENTIAL PATTERNS ====
            x = Bidirectional(GRU(64, return_sequences=False, kernel_regularizer=l2(self.l2_reg)))(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(self.dropout_rate)(x)

            # ==== DENSE HEAD ====
            for i, layer in enumerate(self.layers_config):
                units = layer['units']
                x = Dense(units, kernel_regularizer=l2(self.l2_reg))(x)
                x = BatchNormalization()(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
                x = Dropout(self.dropout_rate)(x)

            # Final dense layer before output
            x = Dense(32, kernel_regularizer=l2(self.l2_reg))(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            x = Dropout(self.dropout_rate / 2)(x)

            # Output layer
            outputs = Dense(1, activation='linear', name='output')(x)

            model = Model(inputs=inputs, outputs=outputs, name='cnn_hybrid_model')
            self.model = model
            self.logger.info(f"CNN-Hybrid model built with multi-scale convolutions")
            self.logger.info(f"Model parameters: {model.count_params():,}")

            return model


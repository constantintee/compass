# ensemble.py
# Optimized ensemble with uncertainty estimation for best predictions

import os
import json
import logging
import gc
import psutil
import traceback
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, Flatten, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization,
    Bidirectional, Activation, Add, Concatenate, GaussianNoise,
    GaussianDropout, Layer)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2, l1_l2

from models import BaseModel, LSTMModel, TransformerModel, DenseModel
from utils import MemoryMonitor, monitor_memory, log_memory_usage, clean_memory, get_memory_usage
from models import SafeMSE, SafeMAE, PerformanceMonitorCallback, CombinedLoss, HuberLoss


# =============================================================================
# MONTE CARLO DROPOUT LAYER FOR UNCERTAINTY ESTIMATION
# =============================================================================

class MCDropout(Layer):
    """Monte Carlo Dropout - applies dropout during inference for uncertainty estimation."""
    def __init__(self, rate=0.1, **kwargs):
        super(MCDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        # Always apply dropout, even during inference
        return tf.nn.dropout(inputs, rate=self.rate)

    def get_config(self):
        config = super(MCDropout, self).get_config()
        config.update({'rate': self.rate})
        return config


class EnsembleModel:
    def __init__(self, base_models: List[BaseModel], config: dict):
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        self.base_models = base_models
        self.config = config
        self.meta_model = None
        #self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_save_dir = 'data/models'
        self.hyperparam_save_dir = 'data/hyperparameters'
        self.logger = logging.getLogger('training_logger')
        self.run_number = 0
        self.ensemble_models = []

        self.meta_mean = None
        self.meta_std = None

        # Initialize strategy
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                self.strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
                self.logger.info("Using GPU with OneDeviceStrategy")
            except RuntimeError as e:
                self.logger.error(f"Error configuring GPU: {e}")
                self.strategy = tf.distribute.get_strategy()
        else:
            self.logger.info("No GPU found, using default strategy")
            self.strategy = tf.distribute.get_strategy()

        # Try to load recent models
        self.load_recent_models()

    @monitor_memory
    def plot_training_history(self, history: tf.keras.callbacks.History, model_index: int):
        """Plot training history with graceful handling of missing validation data."""
        try:
            plt.figure(figsize=(12, 5))
            
            # Create two subplots
            plt.subplot(1, 2, 1)
            # Plot training loss
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Model {model_index} Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MAE
            plt.subplot(1, 2, 2)
            plt.plot(history.history['safe_mae'], label='Training MAE')
            if 'val_safe_mae' in history.history:
                plt.plot(history.history['val_safe_mae'], label='Validation MAE')
            plt.title(f'Model {model_index} MAE')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.model_save_dir, f'training_history_model_{model_index}.png')
            plt.savefig(plot_path)
            plt.close()
            
            # Log training summary
            final_loss = history.history['loss'][-1]
            final_mae = history.history['safe_mae'][-1]
            best_loss = min(history.history['loss'])
            best_mae = min(history.history['safe_mae'])
            
            self.logger.info(f"\nTraining Summary for Model {model_index}:")
            self.logger.info(f"  Best Loss: {best_loss:.4f}")
            self.logger.info(f"  Best MAE: {best_mae:.4f}")
            self.logger.info(f"  Final Loss: {final_loss:.4f}")
            self.logger.info(f"  Final MAE: {final_mae:.4f}")
            
            if 'val_loss' in history.history:
                best_val_loss = min(history.history['val_loss'])
                best_val_mae = min(history.history['val_safe_mae'])
                self.logger.info(f"  Best Validation Loss: {best_val_loss:.4f}")
                self.logger.info(f"  Best Validation MAE: {best_val_mae:.4f}")
            else:
                self.logger.info("  Note: Training completed without validation data")
            
            self.logger.info(f"Training history plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting training history for model {model_index}: {str(e)}")
            # Continue execution even if plotting fails
            pass

    def load_recent_models(self):
        """Load most recent model for each run up to configured model count"""
        try:
            if not os.path.exists(self.model_save_dir):
                self.logger.info("No existing model directory found")
                return

            current_time = datetime.now()
            model_files = []
            # Get number of models from training config
            training_config = self.config.get('training', {})
            max_models = int(training_config.get('num_models', 3))
            self.logger.info(f"Loading up to {max_models} most recent models")
            
            latest_models = {}  # Dictionary to store latest model for each run

            # Get all model files and their timestamps
            for filename in os.listdir(self.model_save_dir):
                if filename.endswith('.h5') and filename.startswith('ensemble_model_run_'):
                    file_path = os.path.join(self.model_save_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    time_diff = current_time - file_time
                    
                    # Check if file is less than 24 hours old
                    if time_diff.total_seconds() < 24 * 3600:
                        # Extract run number from filename
                        try:
                            run_num = int(filename.split('_')[3])
                            # Update latest model for this run
                            if run_num not in latest_models or file_time > latest_models[run_num][1]:
                                latest_models[run_num] = (file_path, file_time)
                                self.logger.debug(f"Found model for run {run_num}: {filename}")
                        except (IndexError, ValueError) as e:
                            self.logger.warning(f"Skipping file with invalid format: {filename} - {str(e)}")
                            continue

            if not latest_models:
                self.logger.info("No recent models found to load")
                return

            # Sort runs by number and take only up to max_models
            sorted_runs = sorted(latest_models.keys(), reverse=True)[:max_models]
            model_files = [latest_models[run] for run in sorted_runs]

            self.logger.info(f"Found latest models for runs: {sorted_runs}")
            
            # Create instances of custom objects
            safe_mse = SafeMSE()
            safe_mae = SafeMAE()
            
            # Define custom objects
            custom_objects = {
                'SafeMSE': SafeMSE,
                'safe_mae': safe_mae,
                'SafeMAE': SafeMAE,
                'safe_mse': safe_mse,
                'safe_mse_loss': safe_mse  # Add this if you named it differently in compile
            }
            
            # Load models
            loaded_models = []
            for file_path, timestamp in model_files:
                try:
                    with tf.keras.utils.custom_object_scope(custom_objects):
                        model = tf.keras.models.load_model(file_path)
                        loaded_models.append(model)
                        self.logger.info(f"Loaded latest model: {os.path.basename(file_path)} (saved {timestamp})")
                except Exception as e:
                    self.logger.error(f"Error loading model from {file_path}: {str(e)}")
                    continue

            if loaded_models:
                self.ensemble_models = loaded_models
                self.logger.info(f"Successfully loaded {len(loaded_models)} latest models")
                
                # Try to load meta-model if it exists
                meta_model_path = os.path.join(self.model_save_dir, 'meta_model.h5')
                if os.path.exists(meta_model_path):
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(meta_model_path))
                        time_diff = current_time - file_time
                        
                        if time_diff.total_seconds() < 24 * 3600:
                            with tf.keras.utils.custom_object_scope(custom_objects):
                                self.meta_model = tf.keras.models.load_model(meta_model_path)
                                self.logger.info(f"Loaded meta-model (saved {file_time})")
                                
                                # Load scaling parameters if available
                                scaling_path = os.path.join(self.model_save_dir, 'meta_scaling.npz')
                                if os.path.exists(scaling_path):
                                    try:
                                        scaling_data = np.load(scaling_path)
                                        self.meta_mean = tf.constant(scaling_data['mean'])
                                        self.meta_std = tf.constant(scaling_data['std'])
                                        self.logger.info("Loaded meta scaling parameters")
                                    except Exception as e:
                                        self.logger.warning(f"Error loading scaling parameters: {e}")
                                        self.meta_mean = None
                                        self.meta_std = None
                    except Exception as e:
                        self.logger.error(f"Error loading meta-model: {str(e)}")
                        self.meta_model = None
                        self.meta_mean = None
                        self.meta_std = None

        except Exception as e:
            self.logger.error(f"Error loading recent models: {str(e)}")
            self.logger.debug(traceback.format_exc())

    def train_single_model(self, model: BaseModel, X, y, X_val, y_val, epochs: int, batch_size: int, total_batches: int = None):
        """Modified to check if we already have models loaded"""
        try:
            if self.ensemble_models:
                self.logger.info(f"Using {len(self.ensemble_models)} previously loaded models")
                return None
                
            self.logger.info("Memory before train_single_model: %.2f MB", 
                            psutil.Process().memory_info().rss / 1024 / 1024)
            
            self.run_number += 1
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f'ensemble_model_run_{self.run_number:03d}_{timestamp}.h5'
            unique_model_path = os.path.join(self.model_save_dir, model_filename)

            # Train model with total_batches
            history = model.train_model(
                X=X,
                y=None,
                X_val=X_val,
                y_val=None,
                epochs=epochs,
                batch_size=batch_size,
                model_save_path=unique_model_path,
                total_batches=total_batches
            )

            if history is not None:
                self.ensemble_models.append(model.model)
                try:
                    self.save_hyperparameters(model, run_number=self.run_number, timestamp=timestamp)
                except Exception as e:
                    self.logger.warning(f"Failed to save hyperparameters: {str(e)}")
                    
                self.logger.info(f"Base model saved to {unique_model_path}")

            return history

        except Exception as e:
            self.logger.error(f"Error in train_single_model: {str(e)}")
            raise

        finally:
            clean_memory()

    def prepare_meta_features(self, dataset):
        """Optimized meta feature preparation using batch processing."""
        try:
            all_predictions = [[] for _ in self.ensemble_models]
            actual_targets = []
            batch_count = 0

            # Process dataset in a single pass - no counting needed
            for X_batch, y_batch in dataset:
                batch_count += 1

                # Get predictions from all models for this batch
                for idx, model in enumerate(self.ensemble_models):
                    try:
                        pred = model.predict(X_batch, verbose=0)
                        all_predictions[idx].extend(pred.flatten())
                    except Exception as e:
                        self.logger.error(f"Error in model {idx} prediction: {str(e)}")
                        # Pad with zeros to maintain alignment
                        all_predictions[idx].extend([0.0] * len(y_batch))

                # Collect targets
                actual_targets.extend(y_batch.numpy().flatten())

                # Log progress every 50 batches
                if batch_count % 50 == 0:
                    self.logger.info(f"Processed {batch_count} batches for meta features")

            if batch_count == 0:
                self.logger.error("Empty dataset provided")
                return np.array([]), np.array([])

            # Convert to numpy arrays and transpose
            meta_features = np.array(all_predictions).T
            actual_targets = np.array(actual_targets)

            self.logger.info(f"Final meta features shape: {meta_features.shape}")
            self.logger.info(f"Total batches processed: {batch_count}")

            return meta_features, actual_targets

        except Exception as e:
            self.logger.error(f"Error preparing meta features: {e}")
            return np.array([]), np.array([])

    def train_meta_model(self, X, y, X_val, y_val, config):
        """
        Train advanced meta-model with:
        - Deep architecture with residual connections
        - Monte Carlo Dropout for uncertainty estimation
        - Combined loss function
        - Cross-validation stacking
        """
        try:
            # Get training configuration
            training_config = config.get('training', {})
            batch_size = training_config.get('batch_size', 32)
            epochs = training_config.get('meta_epochs', 100)  # More epochs for meta-model
            learning_rate = training_config.get('meta_learning_rate', 0.001)

            self.logger.info(f"Training ADVANCED meta-model with batch_size={batch_size}, epochs={epochs}")

            # Prepare meta features
            self.logger.info("Preparing meta features for training data...")
            meta_train, y_train = self.prepare_meta_features(X)

            if meta_train.size == 0:
                self.logger.error("No training meta features generated")
                return

            # Replace any NaN/Inf values
            meta_train = tf.where(tf.math.is_finite(meta_train), meta_train, tf.zeros_like(meta_train))
            y_train = tf.where(tf.math.is_finite(y_train), y_train, tf.zeros_like(y_train))

            # Add interaction features between base model predictions
            n_models = len(self.ensemble_models)
            interaction_features = []
            has_interaction_features = False
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    # Difference between models
                    diff = meta_train[:, i] - meta_train[:, j]
                    interaction_features.append(tf.expand_dims(diff, -1))
                    # Ratio between models (with safety)
                    ratio = meta_train[:, i] / (tf.abs(meta_train[:, j]) + 1e-7)
                    interaction_features.append(tf.expand_dims(ratio, -1))

            if interaction_features:
                interaction_features = tf.concat(interaction_features, axis=-1)
                meta_train_enhanced = tf.concat([meta_train, interaction_features], axis=-1)
                has_interaction_features = True
            else:
                meta_train_enhanced = meta_train

            # Normalize features using tf operations
            meta_mean = tf.reduce_mean(meta_train_enhanced, axis=0)
            meta_std = tf.math.reduce_std(meta_train_enhanced, axis=0)
            meta_std = tf.where(meta_std == 0, tf.ones_like(meta_std), meta_std)
            meta_train_scaled = (meta_train_enhanced - meta_mean) / (meta_std + 1e-7)

            self.logger.info(f"Enhanced meta features shape: {meta_train_scaled.shape}")

            self.logger.info("Preparing meta features for validation data...")
            meta_val, y_val = self.prepare_meta_features(X_val)

            if meta_val.size == 0:
                self.logger.warning("No validation meta features generated, will train without validation")
                meta_val_scaled, y_val_scaled = None, None
            else:
                meta_val = tf.where(tf.math.is_finite(meta_val), meta_val, tf.zeros_like(meta_val))
                y_val = tf.where(tf.math.is_finite(y_val), y_val, tf.zeros_like(y_val))

                # Add interaction features for validation (only if training had them)
                if has_interaction_features:
                    val_interaction_features = []
                    for i in range(n_models):
                        for j in range(i + 1, n_models):
                            diff = meta_val[:, i] - meta_val[:, j]
                            val_interaction_features.append(tf.expand_dims(diff, -1))
                            ratio = meta_val[:, i] / (tf.abs(meta_val[:, j]) + 1e-7)
                            val_interaction_features.append(tf.expand_dims(ratio, -1))
                    val_interaction_features = tf.concat(val_interaction_features, axis=-1)
                    meta_val_enhanced = tf.concat([meta_val, val_interaction_features], axis=-1)
                else:
                    meta_val_enhanced = meta_val

                meta_val_scaled = (meta_val_enhanced - meta_mean) / (meta_std + 1e-7)
                y_val_scaled = y_val

            try:
                with self.strategy.scope():
                    # Build advanced meta-model architecture
                    input_dim = meta_train_scaled.shape[1]
                    inputs = tf.keras.Input(shape=(input_dim,))

                    # Input normalization with Gaussian noise for robustness
                    x = BatchNormalization()(inputs)
                    x = GaussianNoise(0.01)(x)

                    # First residual block
                    x1 = Dense(128, kernel_initializer='he_normal',
                               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
                    x1 = BatchNormalization()(x1)
                    x1 = tf.keras.layers.LeakyReLU(alpha=0.1)(x1)
                    x1 = MCDropout(rate=0.15)(x1)  # MC Dropout for uncertainty

                    # Second block
                    x2 = Dense(64, kernel_initializer='he_normal',
                               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x1)
                    x2 = BatchNormalization()(x2)
                    x2 = tf.keras.layers.LeakyReLU(alpha=0.1)(x2)
                    x2 = MCDropout(rate=0.15)(x2)

                    # Third block with residual from first
                    x3 = Dense(64, kernel_initializer='he_normal',
                               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x2)
                    x3 = BatchNormalization()(x3)

                    # Skip connection from x1 (project to same dimension)
                    x1_proj = Dense(64, kernel_initializer='he_normal')(x1)
                    x3 = Add()([x3, x1_proj])
                    x3 = tf.keras.layers.LeakyReLU(alpha=0.1)(x3)
                    x3 = MCDropout(rate=0.1)(x3)

                    # Fourth block
                    x4 = Dense(32, kernel_initializer='he_normal',
                               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x3)
                    x4 = BatchNormalization()(x4)
                    x4 = tf.keras.layers.LeakyReLU(alpha=0.1)(x4)
                    x4 = MCDropout(rate=0.1)(x4)

                    # Final block before output
                    x5 = Dense(16, kernel_initializer='he_normal',
                               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x4)
                    x5 = BatchNormalization()(x5)
                    x5 = tf.keras.layers.LeakyReLU(alpha=0.1)(x5)

                    # Output layer
                    outputs = Dense(1, kernel_initializer='glorot_normal',
                                    activation='linear')(x5)

                    self.meta_model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    self.logger.info(f"Advanced meta-model built with {self.meta_model.count_params():,} parameters")

                    # Use AdamW optimizer with weight decay
                    optimizer = AdamW(
                        learning_rate=learning_rate,
                        weight_decay=0.01,
                        clipnorm=1.0,
                        clipvalue=0.5,
                        epsilon=1e-7
                    )

                    # Use combined loss for meta-model too
                    loss_fn = CombinedLoss(
                        huber_weight=0.5,
                        mse_weight=0.3,
                        direction_weight=0.2,
                        delta=1.0
                    )

                    self.meta_model.compile(
                        optimizer=optimizer,
                        loss=loss_fn,
                        metrics=[SafeMAE()]
                    )

                    # Calculate total batches from meta features shape
                    total_meta_batches = max(1, len(meta_train_scaled) // batch_size)

                    callbacks = [
                        EarlyStopping(
                            monitor='val_loss' if meta_val is not None else 'loss',
                            patience=15,  # Increased patience for meta-model
                            restore_best_weights=True,
                            verbose=1
                        ),
                        ReduceLROnPlateau(
                            monitor='val_loss' if meta_val is not None else 'loss',
                            factor=0.5,
                            patience=5,
                            min_lr=1e-7,
                            verbose=1
                        ),
                        ModelCheckpoint(
                            filepath=os.path.join(self.model_save_dir, 'meta_model.h5'),
                            monitor='val_loss' if meta_val is not None else 'loss',
                            save_best_only=True,
                            verbose=1
                        ),
                        PerformanceMonitorCallback(
                            self.logger,
                            batch_size,
                            total_meta_batches
                        )
                    ]

                    self.logger.info("Starting advanced meta-model training with CombinedLoss...")
                    history = self.meta_model.fit(
                        meta_train_scaled, y_train,
                        validation_data=(meta_val_scaled, y_val) if meta_val is not None else None,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )

                    self.logger.info("Meta-model training completed successfully")
                    # Save scaling parameters
                    self.meta_mean = meta_mean
                    self.meta_std = meta_std
                    return history

            except Exception as e:
                self.logger.error(f"Error training meta-model: {e}")
                self.logger.debug(traceback.format_exc())
                raise

        except Exception as e:
            self.logger.error(f"Error in train_meta_model: {e}")
            self.logger.debug(traceback.format_exc())
            raise

        finally:
            clean_memory()

    def _add_interaction_features(self, meta_features):
        """Add interaction features between base model predictions."""
        n_models = len(self.ensemble_models)
        interaction_features = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Difference between models
                diff = meta_features[:, i] - meta_features[:, j]
                interaction_features.append(tf.expand_dims(diff, -1))
                # Ratio between models (with safety)
                ratio = meta_features[:, i] / (tf.abs(meta_features[:, j]) + 1e-7)
                interaction_features.append(tf.expand_dims(ratio, -1))

        if interaction_features:
            interaction_features = tf.concat(interaction_features, axis=-1)
            return tf.concat([meta_features, interaction_features], axis=-1)
        return meta_features

    def predict(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with the ensemble."""
        try:
            meta_features = []
            actual_targets = []

            # Handle both dataset and direct feature input
            if isinstance(dataset, np.ndarray):
                # Direct feature input
                batch_predictions = []
                for model in self.ensemble_models:
                    preds = model.predict(dataset)
                    batch_predictions.append(preds.flatten())
                meta_features = np.array(batch_predictions).T

            else:
                # Dataset input
                for data in dataset:
                    # Check if the data is a tuple (features, targets) or just features
                    if isinstance(data, tuple) and len(data) == 2:
                        X_batch, y_batch = data
                        actual_targets.extend(y_batch.numpy())
                    else:
                        X_batch = data

                    batch_predictions = []
                    for model in self.ensemble_models:
                        preds = model.predict(X_batch)
                        batch_predictions.append(preds.flatten())
                    batch_meta_features = np.array(batch_predictions).T
                    meta_features.append(batch_meta_features)

                meta_features = np.vstack(meta_features)

            if self.meta_model is None:
                raise ValueError("Meta-model is not trained.")

            # Replace NaN values
            meta_features = tf.where(tf.math.is_finite(meta_features),
                                     meta_features, tf.zeros_like(meta_features))

            # Add interaction features
            meta_features_enhanced = self._add_interaction_features(meta_features)

            # Scale using saved parameters if available
            if self.meta_mean is not None and self.meta_std is not None:
                meta_features_scaled = (meta_features_enhanced - self.meta_mean) / (self.meta_std + 1e-7)
            else:
                # If scaling parameters aren't available, just use the raw features
                self.logger.warning("Meta scaling parameters not found, using unscaled features")
                meta_features_scaled = meta_features_enhanced

            predictions = self.meta_model.predict(meta_features_scaled)

            if len(actual_targets) == 0:
                actual_targets = np.zeros_like(predictions)  # Placeholder when no targets available

            return predictions.flatten(), np.array(actual_targets)
        except Exception as e:
            self.logger.error(f"Error during ensemble prediction: {e}")
            self.logger.debug(traceback.format_exc())
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    def predict_with_uncertainty(self, dataset, n_samples: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using Monte Carlo Dropout.

        Args:
            dataset: Input data
            n_samples: Number of forward passes for MC Dropout

        Returns:
            mean_predictions: Mean of predictions
            std_predictions: Standard deviation (uncertainty)
            actual_targets: True values
        """
        try:
            meta_features = []
            actual_targets = []

            # Handle both dataset and direct feature input
            if isinstance(dataset, np.ndarray):
                batch_predictions = []
                for model in self.ensemble_models:
                    preds = model.predict(dataset)
                    batch_predictions.append(preds.flatten())
                meta_features = np.array(batch_predictions).T
            else:
                for data in dataset:
                    if isinstance(data, tuple) and len(data) == 2:
                        X_batch, y_batch = data
                        actual_targets.extend(y_batch.numpy())
                    else:
                        X_batch = data

                    batch_predictions = []
                    for model in self.ensemble_models:
                        preds = model.predict(X_batch)
                        batch_predictions.append(preds.flatten())
                    batch_meta_features = np.array(batch_predictions).T
                    meta_features.append(batch_meta_features)

                meta_features = np.vstack(meta_features)

            if self.meta_model is None:
                raise ValueError("Meta-model is not trained.")

            # Replace NaN values
            meta_features = tf.where(tf.math.is_finite(meta_features),
                                     meta_features, tf.zeros_like(meta_features))

            # Add interaction features
            meta_features_enhanced = self._add_interaction_features(meta_features)

            # Scale using saved parameters
            if self.meta_mean is not None and self.meta_std is not None:
                meta_features_scaled = (meta_features_enhanced - self.meta_mean) / (self.meta_std + 1e-7)
            else:
                meta_features_scaled = meta_features_enhanced

            # Monte Carlo Dropout: Run multiple forward passes
            all_predictions = []
            for _ in range(n_samples):
                # MC Dropout layers are active during prediction
                preds = self.meta_model(meta_features_scaled, training=True)  # training=True enables dropout
                all_predictions.append(preds.numpy().flatten())

            all_predictions = np.array(all_predictions)

            # Calculate mean and standard deviation
            mean_predictions = np.mean(all_predictions, axis=0)
            std_predictions = np.std(all_predictions, axis=0)

            if len(actual_targets) == 0:
                actual_targets = np.zeros_like(mean_predictions)

            self.logger.info(f"Uncertainty estimation: mean std = {np.mean(std_predictions):.4f}")

            return mean_predictions, std_predictions, np.array(actual_targets)

        except Exception as e:
            self.logger.error(f"Error during uncertainty prediction: {e}")
            self.logger.debug(traceback.format_exc())
            return np.array([]), np.array([]), np.array([])

    def evaluate_meta_model(self, dataset) -> Dict[str, float]:
        try:
            predictions, y_true = self.predict(dataset)
            if len(predictions) == 0:
                self.logger.error("No predictions to evaluate.")
                return {}

            try:
                # Convert to float32 tensors
                predictions = tf.cast(predictions, tf.float32)
                y_true = tf.cast(y_true, tf.float32)

                # Basic metrics
                mae = tf.reduce_mean(tf.abs(predictions - y_true))

                # Safe MAPE calculation
                mask = tf.math.not_equal(y_true, 0)
                y_true_safe = tf.boolean_mask(y_true, mask)
                pred_safe = tf.boolean_mask(predictions, mask)
                mape = tf.reduce_mean(tf.abs((y_true_safe - pred_safe) / y_true_safe)) * 100

                # Directional accuracy with type casting
                pred_diff = predictions[1:] - predictions[:-1]
                true_diff = y_true[1:] - y_true[:-1]
                
                # Ensure signs are float32
                pred_sign = tf.cast(tf.math.sign(pred_diff), tf.float32)
                true_sign = tf.cast(tf.math.sign(true_diff), tf.float32)
                
                # Calculate directional accuracy
                matches = tf.cast(tf.equal(pred_sign, true_sign), tf.float32)
                directional_accuracy = tf.reduce_mean(matches) * 100

                metrics = {
                    'MAE': float(mae.numpy()),
                    'MAPE': float(mape.numpy()),
                    'Directional Accuracy': float(directional_accuracy.numpy())
                }

                # Log all raw values for debugging
                self.logger.debug(f"Raw predictions shape: {predictions.shape}")
                self.logger.debug(f"Raw y_true shape: {y_true.shape}")
                self.logger.debug(f"Raw MAE value: {mae}")
                self.logger.debug(f"Raw MAPE value: {mape}")
                self.logger.debug(f"Raw Directional Accuracy value: {directional_accuracy}")

                self.logger.info(f"Ensemble Evaluation Metrics: {metrics}")
                return metrics

            except Exception as e:
                self.logger.error(f"Error calculating metrics: {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                # Fallback to numpy calculations if TensorFlow calculations fail
                try:
                    self.logger.info("Attempting fallback to numpy calculations")
                    mae = np.mean(np.abs(predictions - y_true))
                    
                    # Safe MAPE calculation
                    mask = y_true != 0
                    mape = np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100
                    
                    # Directional accuracy
                    pred_diff = np.diff(predictions)
                    true_diff = np.diff(y_true)
                    directional_accuracy = np.mean(np.sign(pred_diff) == np.sign(true_diff)) * 100
                    
                    metrics = {
                        'MAE': float(mae),
                        'MAPE': float(mape),
                        'Directional Accuracy': float(directional_accuracy)
                    }
                    
                    self.logger.info(f"Ensemble Evaluation Metrics (numpy fallback): {metrics}")
                    return metrics
                    
                except Exception as e2:
                    self.logger.error(f"Fallback calculation also failed: {str(e2)}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error during ensemble model evaluation: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return {}

    def save_hyperparameters(self, model: BaseModel, run_number: int, timestamp: str):
        try:
            hyperparams = {
                "model_type": model.__class__.__name__,
                "sequence_length": model.sequence_length,
            }

            # Add specific hyperparameters based on model type
            if isinstance(model, LSTMModel):
                hyperparams.update({
                    "units": model.units,
                    "dropout_rate": model.dropout_rate,
                    "l2_reg": model.l2_reg
                })
            elif isinstance(model, TransformerModel):
                hyperparams.update({
                    "num_heads": model.num_heads,
                    "ff_dim": model.ff_dim,
                    "num_transformer_blocks": model.num_transformer_blocks,
                    "dropout_rate": model.dropout_rate
                })
            elif isinstance(model, DenseModel):
                hyperparams.update({
                    "layers": model.layers_config
                })

            # Save hyperparameters to a JSON file
            hyperparam_filename = f'hyperparams_run_{run_number:03d}_{timestamp}.json'
            hyperparam_path = os.path.join(self.hyperparam_save_dir, hyperparam_filename)

            with open(hyperparam_path, 'w') as f:
                json.dump(hyperparams, f, indent=4)

            self.logger.info(f"Hyperparameters saved to {hyperparam_path}.")
        except Exception as e:
            self.logger.error(f"Error saving hyperparameters for model run {run_number}: {e}")

    def save_ensemble_models(self):
        try:
            os.makedirs(self.model_save_dir, exist_ok=True)
            for idx, model in enumerate(self.ensemble_models, start=1):
                model_path = os.path.join(self.model_save_dir, f'ensemble_model_{idx}.h5')
                model.save(model_path)
                self.logger.info(f"Ensemble model {idx} saved to {model_path}.")
            
            # Save meta-model and scaling parameters
            if self.meta_model:
                meta_model_path = os.path.join(self.model_save_dir, 'meta_model.h5')
                self.meta_model.save(meta_model_path)
                self.logger.info(f"Meta-model saved to {meta_model_path}.")
                
                # Save scaling parameters
                if self.meta_mean is not None and self.meta_std is not None:
                    scaling_path = os.path.join(self.model_save_dir, 'meta_scaling.npz')
                    np.savez(scaling_path, 
                            mean=self.meta_mean.numpy(), 
                            std=self.meta_std.numpy())
                    self.logger.info(f"Meta scaling parameters saved to {scaling_path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving ensemble models: {e}")
            self.logger.debug(traceback.format_exc())
            raise

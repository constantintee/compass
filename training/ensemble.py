# ensemble.py

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, Flatten, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization, Bidirectional, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from models import BaseModel, LSTMModel, TransformerModel, DenseModel
from utils import MemoryMonitor, monitor_memory, log_memory_usage, clean_memory, get_memory_usage
from models import SafeMSE, SafeMAE, PerformanceMonitorCallback


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
        """Memory-efficient meta feature preparation with dataset length handling"""
        try:
            meta_features = []
            actual_targets = []
            total_examples = 0
                
            # First pass to count examples
            for _ in dataset:
                total_examples += 1
                
            if total_examples == 0:
                self.logger.error("Empty dataset provided")
                return np.array([]), np.array([])
                
            self.logger.info(f"Processing {total_examples} examples for meta features")
                
            # Reset dataset iterator
            dataset = dataset.repeat(1)
            
            # Process in smaller chunks
            chunk_size = min(100, total_examples)
            for chunk_start in range(0, total_examples, chunk_size):
                chunk_data = list(dataset.skip(chunk_start).take(chunk_size))
                if not chunk_data:
                    break
                    
                chunk_predictions = []
                for model in self.ensemble_models:
                    preds = []
                    for X_batch, _ in chunk_data:
                        try:
                            pred = model.predict(X_batch, verbose=0)
                            preds.extend(pred.flatten())
                        except Exception as e:
                            self.logger.error(f"Error in model prediction: {str(e)}")
                            continue
                    chunk_predictions.append(preds)
                
                if chunk_predictions:
                    # Transpose predictions and add to meta features
                    chunk_meta = np.array(chunk_predictions).T
                    meta_features.append(chunk_meta)
                    
                    # Extract targets
                    chunk_targets = np.concatenate([y.numpy() for _, y in chunk_data])
                    actual_targets.append(chunk_targets)
                    
                    # Log progress
                    self.logger.info(f"Processed {len(meta_features) * chunk_size} examples")
                    
                # Cleanup
                gc.collect()
                
            if not meta_features:
                self.logger.error("No valid predictions generated")
                return np.array([]), np.array([])
                
            # Combine results
            meta_features = np.vstack(meta_features)
            actual_targets = np.concatenate(actual_targets)
                
            self.logger.info(f"Final meta features shape: {meta_features.shape}")
            return meta_features, actual_targets
                
        except Exception as e:
            self.logger.error(f"Error preparing meta features: {e}")
            return np.array([]), np.array([])
            
        finally:
            gc.collect()

    def train_meta_model(self, X, y, X_val, y_val, config):
        """Train meta-model using configuration settings with robust NaN handling"""
        try:
            # Get training configuration
            training_config = config.get('training', {})
            batch_size = training_config.get('batch_size', 32)
            epochs = training_config.get('epochs', 30)
            learning_rate = training_config.get('learning_rate', 0.001)
            
            self.logger.info(f"Training meta-model with batch_size={batch_size}, epochs={epochs}")
            
            # Prepare meta features
            self.logger.info("Preparing meta features for training data...")
            meta_train, y_train = self.prepare_meta_features(X)
            
            if meta_train.size == 0:
                self.logger.error("No training meta features generated")
                return
                
            # Replace any NaN/Inf values
            meta_train = tf.where(tf.math.is_finite(meta_train), meta_train, tf.zeros_like(meta_train))
            y_train = tf.where(tf.math.is_finite(y_train), y_train, tf.zeros_like(y_train))
                
            # Normalize features using tf operations
            meta_mean = tf.reduce_mean(meta_train, axis=0)
            meta_std = tf.math.reduce_std(meta_train, axis=0)
            meta_std = tf.where(meta_std == 0, tf.ones_like(meta_std), meta_std)
            meta_train_scaled = (meta_train - meta_mean) / (meta_std + 1e-7)
            
            self.logger.info("Preparing meta features for validation data...")
            meta_val, y_val = self.prepare_meta_features(X_val)
            
            if meta_val.size == 0:
                self.logger.warning("No validation meta features generated, will train without validation")
                meta_val_scaled, y_val_scaled = None, None
            else:
                meta_val = tf.where(tf.math.is_finite(meta_val), meta_val, tf.zeros_like(meta_val))
                y_val = tf.where(tf.math.is_finite(y_val), y_val, tf.zeros_like(y_val))
                meta_val_scaled = (meta_val - meta_mean) / (meta_std + 1e-7)
                y_val_scaled = y_val

            n_models = len(self.ensemble_models)
            if meta_train_scaled.shape[1] != n_models:
                self.logger.error(f"Meta-features shape mismatch. Expected {n_models} features, got {meta_train_scaled.shape[1]}.")
                return

            try:
                with self.strategy.scope():
                    # Add input batch normalization
                    inputs = tf.keras.Input(shape=(meta_train_scaled.shape[1],))
                    x = BatchNormalization()(inputs)
                    
                    # First dense block with residual connection
                    x1 = Dense(64, kernel_initializer='glorot_normal', 
                            kernel_regularizer=l2(0.01))(x)
                    x1 = BatchNormalization()(x1)
                    x1 = tf.keras.layers.LeakyReLU(alpha=0.1)(x1)
                    x1 = Dropout(0.2)(x1)
                    
                    # Second dense block with residual connection
                    x2 = Dense(32, kernel_initializer='glorot_normal',
                            kernel_regularizer=l2(0.01))(x1)
                    x2 = BatchNormalization()(x2)
                    x2 = tf.keras.layers.LeakyReLU(alpha=0.1)(x2)
                    x2 = Dropout(0.2)(x2)
                    
                    # Output layer
                    outputs = Dense(1, kernel_initializer='glorot_normal',
                                activation='linear')(x2)
                    
                    self.meta_model = tf.keras.Model(inputs=inputs, outputs=outputs)

                    # Custom loss function to handle NaN
                    def safe_mse(y_true, y_pred):
                        mask = tf.math.is_finite(y_true)
                        y_true_safe = tf.where(mask, y_true, tf.zeros_like(y_true))
                        y_pred_safe = tf.where(mask, y_pred, tf.zeros_like(y_pred))
                        
                        squared_error = tf.square(y_true_safe - y_pred_safe)
                        loss = tf.reduce_mean(squared_error)
                        return loss

                    optimizer = Adam(
                        learning_rate=learning_rate,
                        clipnorm=1.0,
                        clipvalue=0.5,
                        epsilon=1e-7
                    )

                    self.meta_model.compile(
                        optimizer=optimizer,
                        loss=SafeMSE(),
                        metrics=[SafeMAE()]
                    )

                    callbacks = [
                        EarlyStopping(
                            monitor='val_loss' if meta_val is not None else 'loss',
                            patience=5,
                            restore_best_weights=True,
                            verbose=1
                        ),
                        ReduceLROnPlateau(
                            monitor='val_loss' if meta_val is not None else 'loss',
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6,
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
                            sum(1 for _ in X)  # Count batches in training dataset
                        )
                    ]

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

    def predict(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
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
                
            # Scale using saved parameters if available
            if self.meta_mean is not None and self.meta_std is not None:
                meta_features_scaled = (meta_features - self.meta_mean) / (self.meta_std + 1e-7)
            else:
                # If scaling parameters aren't available, just use the raw features
                self.logger.warning("Meta scaling parameters not found, using unscaled features")
                meta_features_scaled = meta_features
                
            predictions = self.meta_model.predict(meta_features_scaled)
            
            if len(actual_targets) == 0:
                actual_targets = np.zeros_like(predictions)  # Placeholder when no targets available
                
            return predictions.flatten(), np.array(actual_targets)
        except Exception as e:
            self.logger.error(f"Error during ensemble prediction: {e}")
            self.logger.debug(traceback.format_exc())
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

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

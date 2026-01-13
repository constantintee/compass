# File: training.py
# Optimized training script for best predictions

import os
import sys
import gc
import psutil
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Dict, Any, Optional, Tuple, Callable
from psycopg2 import OperationalError

# Import classes and functions from other modules
# Add shared module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.preprocessing import Preprocessor
from models import (BaseModel, LSTMModel, TransformerModel, DenseModel,
                    DatasetValidator, TimeSeriesAugmentation)
from ensemble import EnsembleModel
from backtester import Backtester
from utils import (setup_logger, load_configuration,
                   load_and_process_individual_tfrecords_parallel,
                   estimate_dataset_size, log_memory_usage, get_memory_usage,
                   clean_memory, MemoryMonitor, monitor_memory)


# Set GPU memory growth before any TensorFlow operations
#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.set_logical_device_configuration(
#                gpu,
#                [tf.config.LogicalDeviceConfiguration(memory_limit=7018)]
#            )
#        print("Memory limit set to 7680 MiB for each GPU.")
#    except RuntimeError as e:
#        print(f"Error setting memory limit: {e}")


def main():
    # Load configuration
    config_path = "data/config.yaml"
    config = load_configuration(config_path)

    # Setup logger with rotating file handler
    log_config = config.get('training_logging', {})
    environment = log_config.get('environment', 'development')
    logger = setup_logger(
        log_config=log_config,
        logger_name='training_logger',
        environment=environment
    )

    # Access training configuration
    training_config = config.get('training', {})
    models_config = config.get('models', [])

    # Directory setup
    individual_tfrecord_path = "data/memory/individual_tfrecords"
    os.makedirs(individual_tfrecord_path, exist_ok=True)

    try:
        # Initialize Preprocessor
        preprocessor = Preprocessor(
            sequence_length=training_config.get('sequence_length', 60),
            cache_dir=training_config.get('cache_dir', 'cache'),
            frequency=training_config.get('frequency', 'B'),
            logger=logger
        )

        #  # Preprocess stocks if needed
        #  stocks = config.get('stocks')
        #  if stocks:
        #      preprocessor.parallel_preprocess_stocks(
        #          stocks, training_config, individual_tfrecord_path
        #      )
        
        # Load and process TFRecords
        logger.info("[Dataset Loading] Starting dataset loading...")
        log_memory_usage(logger, "Memory usage before loading")
        
        try:
            logger.info("[Dataset Loading] Starting dataset loading...")
            log_memory_usage(logger, "Memory usage before loading")
            
            result = load_and_process_individual_tfrecords_parallel(
                individual_tfrecord_path=individual_tfrecord_path,
                config=config
            )
            
            train_dataset, val_dataset, num_features, total_batches = result
            
            if total_batches == 0:
                logger.error("[Dataset Loading] No valid data loaded. Exiting.")
                sys.exit(1)
            
            train_size = int(0.8 * total_batches)
            logger.info(f"[Dataset Loading] Created train/val splits")
            logger.info(f"[Dataset Loading] Training batches: {train_size}")
            logger.info(f"[Dataset Loading] Validation batches: {total_batches - train_size}")
            
            log_memory_usage(logger, "Memory usage after dataset preparation")
            clean_memory()
                    
            # Initialize EnsembleModel with optimized configuration
            try:
                logger.info("="*80)
                logger.info("[Ensemble Initialization] Initializing OPTIMIZED EnsembleModel...")
                logger.info("="*80)

                base_models = []
                for model_conf in models_config:
                    model_type = model_conf.get('type')

                    if model_type == 'LSTMModel':
                        model = LSTMModel(
                            sequence_length=model_conf.get('sequence_length', 60),
                            units=model_conf.get('units', 192),  # Increased capacity
                            dropout_rate=model_conf.get('dropout_rate', 0.25),
                            l2_reg=model_conf.get('l2_reg', 0.0005)
                        )
                        base_models.append(model)
                        logger.info(f"[Ensemble] Initialized Enhanced Attention-LSTM Model (units={model_conf.get('units', 192)})")

                    elif model_type == 'TransformerModel':
                        model = TransformerModel(
                            sequence_length=model_conf.get('sequence_length', 60),
                            num_heads=model_conf.get('num_heads', 8),
                            ff_dim=model_conf.get('ff_dim', 384),  # Increased capacity
                            num_transformer_blocks=model_conf.get('num_transformer_blocks', 6),  # Deeper
                            dropout_rate=model_conf.get('dropout_rate', 0.2)
                        )
                        base_models.append(model)
                        logger.info(f"[Ensemble] Initialized Enhanced Transformer Model (blocks={model_conf.get('num_transformer_blocks', 6)}, ff_dim={model_conf.get('ff_dim', 384)})")

                    elif model_type == 'DenseModel':
                        model = DenseModel(
                            sequence_length=model_conf.get('sequence_length', 60),
                            layers=model_conf.get('layers', [
                                {'units': 512, 'activation': 'relu'},
                                {'units': 256, 'activation': 'relu'},
                                {'units': 128, 'activation': 'relu'},
                                {'units': 64, 'activation': 'relu'}
                            ])
                        )
                        base_models.append(model)
                        logger.info(f"[Ensemble] Initialized CNN-Hybrid Model with multi-scale convolutions")

                # Log optimization features
                logger.info("-"*80)
                logger.info("Optimization Features Enabled:")
                logger.info("  - Advanced Loss: Combined (Huber + MSE + Directional)")
                logger.info("  - Learning Rate: Warmup + Cosine Decay")
                logger.info("  - Regularization: AdamW with weight decay")
                logger.info("  - Generalization: Stochastic Weight Averaging (SWA)")
                logger.info("  - Attention: Multi-head self-attention in LSTM")
                logger.info("  - Meta-model: Deep network with MC Dropout for uncertainty")
                logger.info("-"*80)

                ensemble = EnsembleModel(base_models=base_models, config=config)
                logger.info(f"[Ensemble Initialization] EnsembleModel initialized with {len(base_models)} optimized models.")

            except Exception as e:
                logger.error(f"[Ensemble Initialization] Error initializing EnsembleModel: {e}")
                logger.debug(traceback.format_exc())
                sys.exit(1)

            # Training Loop
            try:
                target_mae = training_config.get('target_mae', 0.12)
                max_iterations = training_config.get('max_iterations', 10)
                best_mae = float('inf')
                iteration = 0

                # Skip base model training if we already have loaded models
                if not ensemble.ensemble_models:
                    while best_mae > target_mae and iteration < max_iterations:
                        iteration += 1
                        logger.info(f"Starting training iteration {iteration}.")

                        # Train each base model
                        for idx, model in enumerate(ensemble.base_models, start=1):
                            logger.info(
                                f"Training base model {idx}/{len(ensemble.base_models)}: {model.__class__.__name__}"
                            )
                            ensemble.train_single_model(
                                model=model,
                                X=train_dataset,
                                y=None,
                                X_val=val_dataset,
                                y_val=None,
                                epochs=training_config.get('epochs', 30),
                                batch_size=training_config.get('batch_size', 16),
                                total_batches=total_batches
                            )
                            clean_memory()
                else:
                    logger.info(f"Using {len(ensemble.ensemble_models)} previously trained models")

                # Train the meta-model if needed
                if ensemble.meta_model is None:
                    logger.info("Training the meta-model.")
                    ensemble.train_meta_model(
                        X=train_dataset,
                        y=None,
                        X_val=val_dataset,
                        y_val=None,
                        config=config
                        # epochs=training_config.get('epochs', 30),
                        # batch_size=training_config.get('batch_size', 64),
                    )
                else:
                    logger.info("Using previously trained meta-model")
                    
                    # Evaluate the ensemble on validation data
                    logger.info("Evaluating the ensemble's meta-model on validation data.")
                    metrics = ensemble.evaluate_meta_model(val_dataset)
                    logger.info(f"Iteration {iteration} - Validation Metrics: {metrics}")

                    current_mae = metrics.get('MAE', float('inf'))
                    if current_mae < best_mae:
                        best_mae = current_mae
                        logger.info(f"New best MAE achieved: {best_mae}")

                if best_mae <= target_mae:
                    logger.info(f"Training completed successfully with best validation MAE: {best_mae}")
                    logger.info("Starting backtesting for the best model.")

                    # Extract features from validation dataset
                    val_features = []
                    val_targets = []
                    
                    try:
                        # Create feature identifiers
                        for batch_idx, (features, targets) in enumerate(val_dataset):
                            # Convert to numpy and store
                            batch_features = features.numpy()
                            batch_targets = targets.numpy()
                            
                            # Log shapes for debugging
                            logger.debug(f"Batch {batch_idx} shapes - Features: {batch_features.shape}, Targets: {batch_targets.shape}")
                            
                            for i in range(len(batch_features)):
                                val_features.append(batch_features[i])
                                val_targets.append(batch_targets[i])

                        # Convert lists to numpy arrays
                        val_features = np.array(val_features)
                        val_targets = np.array(val_targets)

                        logger.debug(f"Final shapes - Features: {val_features.shape}, Targets: {val_targets.shape}")

                        # Create DataFrame with feature identifiers instead of the actual features
                        df_data = {
                            'features': [f"feature_sequence_{i}" for i in range(len(val_features))],
                            'actual': val_targets
                        }
                        
                        backtester = Backtester(
                            data=pd.DataFrame(df_data),
                            model=None,
                            evaluation_metrics=[mean_absolute_error, mean_squared_error],
                            ensemble=ensemble
                        )

                        # Store the actual features in the backtester for use during prediction
                        backtester.feature_data = val_features

                        results = backtester.run_backtest()
                        if results:
                            backtesting_results_path = config.get(
                                'backtesting_results_path', 'data/backtesting_results'
                            )
                            os.makedirs(backtesting_results_path, exist_ok=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            result_file = os.path.join(
                                backtesting_results_path, f"backtest_results_{timestamp}.csv"
                            )
                            backtester.save_results(results, 'Combined', file_path=result_file)
                            logger.info(f"Backtesting completed. Results saved at {result_file}.")
                            
                            # Create a DatetimeIndex for plotting
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=len(val_targets))
                            date_index = pd.date_range(start=start_date, end=end_date, periods=len(val_targets))
                            
                            # Set the index for the DataFrame
                            backtester.data.index = date_index
                            
                            # Get predictions from results
                            predictions = np.array(results.get('predictions', []))
                            if len(predictions) == 0:  # If predictions not in results, get them from ensemble
                                predictions, _ = ensemble.predict(val_features)
                                
                            backtester.plot_results(predictions, val_targets, 'Combined')
                        else:
                            logger.warning("Backtesting did not produce any results.")

                        # Save ensemble models
                        logger.info("Saving ensemble models.")
                        ensemble.save_ensemble_models()
                        logger.info("Ensemble models saved successfully.")

                    except Exception as e:
                        logger.error(f"[Ensemble Initialization] Error initializing EnsembleModel: {str(e)}")
                        logger.debug(traceback.format_exc())
                else:
                    logger.warning("Target MAE not reached. Training iterations exceeded maximum attempts.")

            except Exception as e:
                logger.error(f"[Ensemble Initialization] Error initializing EnsembleModel: {e}")
                logger.debug(traceback.format_exc())
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"[Training] An unexpected error occurred: {e}")
        logger.debug(traceback.format_exc())
        return
    finally:
        if 'preprocessor' in locals():
            preprocessor.close_connection_pool()

if __name__ == "__main__":
    main()
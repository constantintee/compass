# backtester.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import logging
from models import BaseModel
from ensemble import EnsembleModel
import traceback

class Backtester:
    def __init__(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        evaluation_metrics: List[Callable[[np.ndarray, np.ndarray], float]],
        ensemble: Optional[EnsembleModel] = None
    ):
        self.data = data
        self.model = model
        self.evaluation_metrics = evaluation_metrics
        self.ensemble = ensemble
        self.logger = logging.getLogger('training_logger')
        self.run_number = 0  # Initialize run number to track each run

    def run_backtest(self) -> Optional[Dict[str, float]]:
        try:
            self.logger.info("Starting the backtesting process...")

            if self.data is None or self.data.empty:
                raise ValueError("No data provided for backtesting.")

            if 'features' not in self.data or 'actual' not in self.data:
                raise KeyError("Data must contain 'features' and 'actual' columns.")

            if self.model is None and self.ensemble is None:
                raise ValueError("No model provided for backtesting.")

            if not self.evaluation_metrics:
                raise ValueError("No evaluation metrics provided.")

            # Increment the run number for this backtest
            self.run_number += 1

            # Get actual values
            y_true = self.data['actual'].values

            # Generate predictions using the ensemble if provided, otherwise use the base model
            if self.ensemble:
                if not hasattr(self, 'feature_data'):
                    raise ValueError("Feature data not provided to backtester")
                    
                predictions, _ = self.ensemble.predict(self.feature_data)
                self.logger.info("Predictions generated successfully using the ensemble model.")
            else:
                predictions = self.model.predict(self.feature_data)
                self.logger.info("Predictions generated successfully using the base model.")

            # Ensure predictions and y_true have the same shape
            predictions = predictions.reshape(-1)
            y_true = y_true.reshape(-1)

            # Log shapes for debugging
            self.logger.debug(f"predictions shape: {predictions.shape}")
            self.logger.debug(f"y_true shape: {y_true.shape}")

            # Store predictions for later use
            results = {'predictions': predictions}

            # Evaluate the predictions
            results = {}
            for metric in self.evaluation_metrics:
                if not callable(metric):
                    raise ValueError(f"Evaluation metric '{metric}' is not callable.")
                metric_value = metric(y_true, predictions)
                results[metric.__name__] = metric_value
                self.logger.info(f"Evaluated {metric.__name__}: {metric_value}")

            # Additional metrics
            results['RMSE'] = np.sqrt(mean_squared_error(y_true, predictions))
            results['MAE'] = mean_absolute_error(y_true, predictions)
            results['R2'] = r2_score(y_true, predictions)
            
            # Calculate directional accuracy
            direction_true = np.sign(y_true[1:] - y_true[:-1])
            direction_pred = np.sign(predictions[1:] - predictions[:-1])
            directional_accuracy = np.mean(direction_true == direction_pred)
            results['Directional_Accuracy'] = directional_accuracy

            self.logger.info(f"Backtesting results: {results}")
            return results

        except Exception as e:
            self.logger.error(f"{type(e).__name__} occurred during backtesting: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def plot_results(self, predictions: np.ndarray, actuals: np.ndarray, ticker: str, save_plot: bool = True):
        """
        Plot the results of the backtesting for visualization.

        Parameters:
        - predictions (np.ndarray): The predicted values from the model.
        - actuals (np.ndarray): The actual target values.
        - ticker (str): The stock ticker symbol.
        - save_plot (bool): Whether to save the plot as a file.
        """
        try:
            if not isinstance(self.data.index, pd.DatetimeIndex):
                raise ValueError("Data index must be a DatetimeIndex for plotting.")

            plt.figure(figsize=(14, 7))
            plt.plot(self.data.index, actuals, label="Actual Prices", color="blue")
            plt.plot(self.data.index, predictions, label="Predicted Prices", color="red", linestyle='--')
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.title(f"Backtesting Results for {ticker}: Actual vs Predicted Prices")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_plot:
                plot_dir = 'data/backtesting'
                os.makedirs(plot_dir, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                run_number_str = f"run_{self.run_number:03d}"
                plot_path = os.path.join(plot_dir, f'{ticker}_backtest_results_{run_number_str}_{timestamp}.png')
                plt.savefig(plot_path)
                plt.close()
                self.logger.info(f"Backtesting results plot saved to {plot_path}.")
            else:
                plt.show()
                self.logger.info("Backtesting results plotted successfully.")

        except Exception as e:
            self.logger.error(f"Error while plotting backtesting results: {e}")

    def save_results(self, results: Dict[str, float], ticker: str, file_path: str = 'backtest_results.csv'):
        """
        Save the backtesting results to a CSV file.

        Parameters:
        - results (Dict[str, float]): The results dictionary containing evaluation metrics.
        - ticker (str): The stock ticker symbol.
        - file_path (str): The file path to save the results CSV.
        """
        try:
            results_df = pd.DataFrame([results])
            results_df['ticker'] = ticker
            results_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            results_df['run_number'] = self.run_number

            if os.path.exists(file_path):
                results_df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                results_df.to_csv(file_path, index=False)
            self.logger.info(f"Backtesting results saved successfully to {file_path}.")
        except Exception as e:
            self.logger.error(f"Error while saving backtesting results: {e}")

    def calculate_trading_metrics(self, predictions: np.ndarray, actuals: np.ndarray, initial_balance: float = 10000.0, transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate trading metrics based on the backtesting results.

        Parameters:
        - predictions (np.ndarray): The predicted values from the model.
        - actuals (np.ndarray): The actual target values.
        - initial_balance (float): The initial balance for the simulated trading.
        - transaction_cost (float): The transaction cost as a percentage of the trade value.

        Returns:
        - Dict[str, float]: A dictionary containing various trading metrics.
        """
        try:
            balance = initial_balance
            position = 0
            trades = 0
            profitable_trades = 0

            for i in range(1, len(predictions)):
                if predictions[i] > actuals[i-1]:  # Buy signal
                    if position == 0:
                        shares = balance / actuals[i]
                        cost = shares * actuals[i] * (1 + transaction_cost)
                        if cost <= balance:
                            position = shares
                            balance -= cost
                            trades += 1
                elif predictions[i] < actuals[i-1]:  # Sell signal
                    if position > 0:
                        sale = position * actuals[i] * (1 - transaction_cost)
                        balance += sale
                        if sale > cost:
                            profitable_trades += 1
                        position = 0
                        trades += 1

            # Close any open position at the end
            if position > 0:
                balance += position * actuals[-1] * (1 - transaction_cost)

            total_return = (balance - initial_balance) / initial_balance
            sharpe_ratio = np.sqrt(252) * total_return / (np.std(np.diff(actuals) / actuals[:-1]) * np.sqrt(252))

            metrics = {
                'Total_Return': total_return,
                'Sharpe_Ratio': sharpe_ratio,
                'Total_Trades': trades,
                'Profitable_Trades': profitable_trades,
                'Win_Rate': profitable_trades / trades if trades > 0 else 0
            }

            self.logger.info("Trading metrics calculated successfully.")
            return metrics

        except Exception as e:
            self.logger.error(f"Error while calculating trading metrics: {e}")
            return {}

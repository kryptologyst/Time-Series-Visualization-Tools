"""
Time Series Forecasting Module

This module provides various forecasting methods including ARIMA, Prophet,
and deep learning approaches for time series prediction.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError:
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

try:
    from prophet import Prophet
except ImportError:
    print("Warning: Prophet not available. Install with: pip install prophet")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """ARIMA-based forecasting implementation."""
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        """
        Initialize ARIMA forecaster.
        
        Args:
            max_p: Maximum autoregressive order
            max_d: Maximum differencing order
            max_q: Maximum moving average order
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.fitted_model = None
    
    def check_stationarity(self, series: pd.Series) -> Dict[str, any]:
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Make time series stationary through differencing.
        
        Args:
            series: Time series data
            max_diff: Maximum number of differences to apply
            
        Returns:
            Tuple of (stationary_series, number_of_differences)
        """
        current_series = series.copy()
        diff_count = 0
        
        for i in range(max_diff):
            stationarity_result = self.check_stationarity(current_series)
            if stationarity_result['is_stationary']:
                break
            
            current_series = current_series.diff().dropna()
            diff_count += 1
        
        return current_series, diff_count
    
    def find_best_arima_params(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Find best ARIMA parameters using grid search.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (p, d, q) parameters
        """
        best_aic = float('inf')
        best_params = (0, 0, 0)
        
        # Make series stationary
        stationary_series, d = self.make_stationary(series)
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                
                except:
                    continue
        
        return best_params
    
    def fit(self, series: pd.Series) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to the time series.
        
        Args:
            series: Time series data
            
        Returns:
            Self for method chaining
        """
        p, d, q = self.find_best_arima_params(series)
        logger.info(f"Best ARIMA parameters: p={p}, d={d}, q={q}")
        
        self.model = ARIMA(series, order=(p, d, q))
        self.fitted_model = self.model.fit()
        
        return self
    
    def forecast(self, steps: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts using the fitted ARIMA model.
        
        Args:
            steps: Number of steps to forecast ahead
            
        Returns:
            Tuple of (forecast, confidence_lower, confidence_upper)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
        
        return (
            forecast_result,
            conf_int.iloc[:, 0].values,
            conf_int.iloc[:, 1].values
        )
    
    def plot_diagnostics(self) -> None:
        """Plot ARIMA model diagnostics."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        self.fitted_model.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()
        plt.show()


class ProphetForecaster:
    """Prophet-based forecasting implementation."""
    
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        """
        Initialize Prophet forecaster.
        
        Args:
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model = None
        self.fitted_model = None
    
    def prepare_data(self, series: pd.Series) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).
        
        Args:
            series: Time series data
            
        Returns:
            DataFrame formatted for Prophet
        """
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        return df
    
    def fit(self, series: pd.Series) -> 'ProphetForecaster':
        """
        Fit Prophet model to the time series.
        
        Args:
            series: Time series data
            
        Returns:
            Self for method chaining
        """
        df = self.prepare_data(series)
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality
        )
        
        self.fitted_model = self.model.fit(df)
        
        return self
    
    def forecast(self, periods: int = 30) -> pd.DataFrame:
        """
        Generate forecasts using the fitted Prophet model.
        
        Args:
            periods: Number of periods to forecast ahead
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        future = self.fitted_model.make_future_dataframe(periods=periods)
        forecast = self.fitted_model.predict(future)
        
        return forecast
    
    def plot_components(self) -> None:
        """Plot Prophet model components."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting components")
        
        self.fitted_model.plot_components(self.fitted_model.predict())
        plt.show()


class LSTMForecaster:
    """LSTM-based forecasting implementation using PyTorch."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_size: int = 1) -> nn.Module:
        """
        Build LSTM model architecture.
        
        Args:
            input_size: Size of input features
            
        Returns:
            PyTorch LSTM model
        """
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout
                )
                self.fc = nn.Linear(hidden_size, 1)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        return LSTMModel(input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def fit(self, series: pd.Series) -> 'LSTMForecaster':
        """
        Fit LSTM model to the time series.
        
        Args:
            series: Time series data
            
        Returns:
            Self for method chaining
        """
        # Prepare data
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = self.create_sequences(scaled_data.flatten())
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build and train model
        self.model = self.build_model().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f'Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}')
        
        return self
    
    def forecast(self, series: pd.Series, steps: int = 30) -> np.ndarray:
        """
        Generate forecasts using the fitted LSTM model.
        
        Args:
            series: Time series data
            steps: Number of steps to forecast ahead
            
        Returns:
            Array of forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        self.model.eval()
        forecasts = []
        
        # Use last sequence_length points as starting point
        last_sequence = series.values[-self.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        current_sequence = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_sequence)
                forecasts.append(pred.item())
                
                # Update sequence for next prediction
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    pred.unsqueeze(0).unsqueeze(0)
                ], dim=1)
        
        # Inverse transform forecasts
        forecasts_scaled = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts_scaled).flatten()
        
        return forecasts


class ForecastingPipeline:
    """Pipeline for comparing multiple forecasting methods."""
    
    def __init__(self):
        """Initialize the forecasting pipeline."""
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model) -> None:
        """
        Add a forecasting model to the pipeline.
        
        Args:
            name: Name of the model
            model: Forecasting model instance
        """
        self.models[name] = model
    
    def fit_all(self, series: pd.Series) -> None:
        """
        Fit all models in the pipeline.
        
        Args:
            series: Time series data
        """
        for name, model in self.models.items():
            logger.info(f"Fitting {name} model...")
            try:
                model.fit(series)
                logger.info(f"{name} model fitted successfully")
            except Exception as e:
                logger.error(f"Error fitting {name} model: {e}")
    
    def forecast_all(self, steps: int = 30) -> Dict[str, np.ndarray]:
        """
        Generate forecasts from all models.
        
        Args:
            steps: Number of steps to forecast ahead
            
        Returns:
            Dictionary with forecasts from each model
        """
        forecasts = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'forecast'):
                    forecast = model.forecast(steps)
                    if isinstance(forecast, tuple):
                        forecasts[name] = forecast[0]  # Take only the forecast values
                    else:
                        forecasts[name] = forecast
                logger.info(f"{name} forecast generated successfully")
            except Exception as e:
                logger.error(f"Error forecasting with {name}: {e}")
        
        return forecasts
    
    def evaluate_forecasts(
        self,
        actual: np.ndarray,
        forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecast accuracy using multiple metrics.
        
        Args:
            actual: Actual values
            forecasts: Dictionary of forecasts
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        metrics = {}
        
        for name, forecast in forecasts.items():
            if len(forecast) == len(actual):
                mse = mean_squared_error(actual, forecast)
                mae = mean_absolute_error(actual, forecast)
                rmse = np.sqrt(mse)
                
                metrics[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse
                }
        
        return metrics
    
    def plot_forecasts(
        self,
        series: pd.Series,
        forecasts: Dict[str, np.ndarray],
        title: str = "Forecast Comparison"
    ) -> None:
        """
        Plot forecasts from all models.
        
        Args:
            series: Original time series
            forecasts: Dictionary of forecasts
            title: Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Plot original series
        plt.plot(series.index, series.values, label='Actual', color='black', linewidth=2)
        
        # Plot forecasts
        colors = plt.cm.Set1(np.linspace(0, 1, len(forecasts)))
        for i, (name, forecast) in enumerate(forecasts.items()):
            future_dates = pd.date_range(
                start=series.index[-1],
                periods=len(forecast) + 1,
                freq=series.index.freq
            )[1:]
            plt.plot(future_dates, forecast, label=f'{name} Forecast', 
                    color=colors[i], linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    """Main function to demonstrate forecasting capabilities."""
    logger.info("Starting Forecasting Demo")
    
    # Generate sample data
    from core import TimeSeriesGenerator
    generator = TimeSeriesGenerator()
    series = generator.generate_univariate_series(n_points=500)
    
    # Split data
    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]
    
    # Initialize forecasting pipeline
    pipeline = ForecastingPipeline()
    
    # Add models
    pipeline.add_model('ARIMA', ARIMAForecaster())
    pipeline.add_model('Prophet', ProphetForecaster())
    pipeline.add_model('LSTM', LSTMForecaster(epochs=50))  # Reduced epochs for demo
    
    # Fit all models
    pipeline.fit_all(train_series)
    
    # Generate forecasts
    forecasts = pipeline.forecast_all(steps=len(test_series))
    
    # Evaluate forecasts
    metrics = pipeline.evaluate_forecasts(test_series.values, forecasts)
    
    # Print results
    for model_name, model_metrics in metrics.items():
        logger.info(f"{model_name} Metrics:")
        for metric_name, value in model_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # Plot forecasts
    pipeline.plot_forecasts(train_series, forecasts)
    
    logger.info("Forecasting Demo completed")


if __name__ == "__main__":
    main()

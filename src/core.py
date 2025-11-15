"""
Time Series Analysis Core Module

This module provides the core functionality for time series analysis including
data generation, preprocessing, visualization, and modeling.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesGenerator:
    """Generate synthetic time series data for testing and demonstration."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the generator with a random seed."""
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_multivariate_series(
        self,
        n_points: int = 1000,
        start_date: str = "2020-01-01",
        freq: str = "D",
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate multivariate time series with trends, seasonality, and noise.
        
        Args:
            n_points: Number of data points to generate
            start_date: Start date for the time series
            freq: Frequency of the time series
            noise_level: Level of noise to add
            
        Returns:
            DataFrame with multivariate time series data
        """
        dates = pd.date_range(start_date, periods=n_points, freq=freq)
        
        # Generate different types of time series
        temperature = (
            20 + 5 * np.sin(np.linspace(0, 3 * np.pi, n_points)) +
            noise_level * np.random.randn(n_points)
        )
        
        humidity = (
            60 + 10 * np.cos(np.linspace(0, 2 * np.pi, n_points)) +
            noise_level * np.random.randn(n_points)
        )
        
        pressure = (
            1013 + 2 * np.random.randn(n_points) +
            0.5 * np.sin(np.linspace(0, 4 * np.pi, n_points))
        )
        
        # Add some trend
        trend = np.linspace(0, 2, n_points)
        temperature += trend
        humidity -= trend * 0.5
        
        return pd.DataFrame({
            'Date': dates,
            'Temperature': temperature,
            'Humidity': humidity,
            'Pressure': pressure
        })
    
    def generate_univariate_series(
        self,
        n_points: int = 1000,
        start_date: str = "2020-01-01",
        freq: str = "D",
        trend: float = 0.01,
        seasonality: float = 1.0,
        noise_level: float = 0.1
    ) -> pd.Series:
        """
        Generate univariate time series with trend, seasonality, and noise.
        
        Args:
            n_points: Number of data points to generate
            start_date: Start date for the time series
            freq: Frequency of the time series
            trend: Linear trend coefficient
            seasonality: Amplitude of seasonal component
            noise_level: Level of noise to add
            
        Returns:
            Series with univariate time series data
        """
        dates = pd.date_range(start_date, periods=n_points, freq=freq)
        
        # Generate components
        trend_component = trend * np.arange(n_points)
        seasonal_component = seasonality * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
        noise_component = noise_level * np.random.randn(n_points)
        
        # Combine components
        values = trend_component + seasonal_component + noise_component
        
        return pd.Series(values, index=dates, name='value')


class TimeSeriesVisualizer:
    """Handle visualization of time series data."""
    
    def __init__(self, style: str = "seaborn-v0_8", figure_size: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer with style preferences."""
        plt.style.use(style)
        self.figure_size = figure_size
        self.color_palette = sns.color_palette("husl", 8)
    
    def plot_multivariate_lines(
        self,
        data: pd.DataFrame,
        title: str = "Multivariate Time Series",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create line plots for multivariate time series.
        
        Args:
            data: DataFrame with time series data
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        for i, column in enumerate(data.columns[1:]):  # Skip 'Date' column
            plt.plot(data['Date'], data[column], 
                    label=column, color=self.color_palette[i])
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create correlation heatmap for time series variables.
        
        Args:
            data: DataFrame with time series data
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate correlation matrix (exclude Date column)
        corr_matrix = data.drop('Date', axis=1).corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
                   square=True, fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_multivariate(
        self,
        data: pd.DataFrame,
        title: str = "Interactive Multivariate Time Series"
    ) -> None:
        """
        Create interactive Plotly visualization for multivariate time series.
        
        Args:
            data: DataFrame with time series data
            title: Plot title
        """
        fig = px.line(data, x='Date', y=data.columns[1:], title=title)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified'
        )
        fig.show()
    
    def plot_decomposition(
        self,
        series: pd.Series,
        model: str = "additive",
        title: str = "Time Series Decomposition"
    ) -> None:
        """
        Plot time series decomposition (trend, seasonal, residual).
        
        Args:
            series: Time series data
            model: Decomposition model ('additive' or 'multiplicative')
            title: Plot title
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(series, model=model)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class TimeSeriesAnalyzer:
    """Core time series analysis functionality."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.generator = TimeSeriesGenerator()
        self.visualizer = TimeSeriesVisualizer()
    
    def analyze_multivariate_data(
        self,
        data: Optional[pd.DataFrame] = None,
        n_points: int = 1000
    ) -> Dict[str, any]:
        """
        Perform comprehensive analysis of multivariate time series data.
        
        Args:
            data: Optional DataFrame with time series data
            n_points: Number of points to generate if data is None
            
        Returns:
            Dictionary containing analysis results
        """
        if data is None:
            data = self.generator.generate_multivariate_series(n_points)
        
        # Basic statistics
        stats = data.describe()
        
        # Correlation analysis
        correlations = data.drop('Date', axis=1).corr()
        
        # Missing values
        missing_values = data.isnull().sum()
        
        # Data types
        data_types = data.dtypes
        
        return {
            'data': data,
            'statistics': stats,
            'correlations': correlations,
            'missing_values': missing_values,
            'data_types': data_types
        }
    
    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Dict[str, List[int]]:
        """
        Detect outliers in time series data.
        
        Args:
            data: DataFrame with time series data
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier indices for each column
        """
        outliers = {}
        
        for column in data.columns[1:]:  # Skip 'Date' column
            values = data[column].dropna()
            
            if method == "iqr":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (values < lower_bound) | (values > upper_bound)
            
            elif method == "zscore":
                z_scores = np.abs((values - values.mean()) / values.std())
                outlier_mask = z_scores > threshold
            
            outliers[column] = values[outlier_mask].index.tolist()
        
        return outliers


def main():
    """Main function to demonstrate the time series analysis capabilities."""
    logger.info("Starting Time Series Analysis Demo")
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer()
    
    # Generate and analyze data
    results = analyzer.analyze_multivariate_data(n_points=500)
    data = results['data']
    
    # Create visualizations
    analyzer.visualizer.plot_multivariate_lines(data)
    analyzer.visualizer.plot_correlation_heatmap(data)
    analyzer.visualizer.plot_interactive_multivariate(data)
    
    # Detect outliers
    outliers = analyzer.detect_outliers(data)
    logger.info(f"Outliers detected: {outliers}")
    
    logger.info("Time Series Analysis Demo completed")


if __name__ == "__main__":
    main()

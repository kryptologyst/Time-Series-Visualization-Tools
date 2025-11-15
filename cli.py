#!/usr/bin/env python3
"""
Command Line Interface for Time Series Analysis

This script provides a command-line interface for running time series analysis
without the need for a web browser or Jupyter notebook.
"""

import argparse
import sys
import os
import logging
from typing import Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core import TimeSeriesGenerator, TimeSeriesAnalyzer
from forecasting import ForecastingPipeline, ARIMAForecaster, ProphetForecaster, LSTMForecaster
from anomaly_detection import (
    AnomalyDetectionPipeline, 
    StatisticalAnomalyDetector, 
    IsolationForestDetector, 
    AutoencoderDetector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_data(n_points: int, output_file: Optional[str] = None) -> None:
    """Generate synthetic time series data."""
    logger.info(f"Generating {n_points} data points...")
    
    generator = TimeSeriesGenerator(random_state=42)
    data = generator.generate_multivariate_series(n_points=n_points)
    
    if output_file:
        data.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
    else:
        print("\nGenerated Data Preview:")
        print(data.head())
        print(f"\nData shape: {data.shape}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")


def run_forecasting(n_points: int, models: list, steps: int = 30) -> None:
    """Run forecasting analysis."""
    logger.info("Running forecasting analysis...")
    
    # Generate data
    generator = TimeSeriesGenerator(random_state=42)
    series = generator.generate_univariate_series(n_points=n_points)
    
    # Split data
    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]
    
    # Initialize pipeline
    pipeline = ForecastingPipeline()
    
    # Add models
    if 'arima' in models:
        pipeline.add_model('ARIMA', ARIMAForecaster())
    if 'prophet' in models:
        pipeline.add_model('Prophet', ProphetForecaster())
    if 'lstm' in models:
        pipeline.add_model('LSTM', LSTMForecaster(epochs=50))
    
    # Fit and forecast
    pipeline.fit_all(train_series)
    forecasts = pipeline.forecast_all(steps=steps)
    
    # Evaluate if we have test data
    if len(test_series) >= steps:
        test_subset = test_series[:steps]
        metrics = pipeline.evaluate_forecasts(test_subset.values, forecasts)
        
        print("\nForecasting Results:")
        print("-" * 50)
        for model_name, model_metrics in metrics.items():
            print(f"{model_name}:")
            for metric_name, value in model_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            print()
    
    print(f"Forecasts generated for {steps} steps using {len(models)} models")


def run_anomaly_detection(n_points: int, methods: list, contamination: float = 0.1) -> None:
    """Run anomaly detection analysis."""
    logger.info("Running anomaly detection analysis...")
    
    # Generate data with anomalies
    generator = TimeSeriesGenerator(random_state=42)
    series = generator.generate_univariate_series(n_points=n_points)
    
    # Add artificial anomalies
    import numpy as np
    np.random.seed(42)
    anomaly_indices = np.random.choice(len(series), size=int(len(series) * contamination), replace=False)
    series.iloc[anomaly_indices] += np.random.normal(0, 3, len(anomaly_indices))
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline()
    
    # Add detectors
    if 'statistical' in methods:
        pipeline.add_detector('Statistical (Z-Score)', StatisticalAnomalyDetector('zscore'))
    if 'iqr' in methods:
        pipeline.add_detector('Statistical (IQR)', StatisticalAnomalyDetector('iqr'))
    if 'isolation' in methods:
        pipeline.add_detector('Isolation Forest', IsolationForestDetector(contamination=contamination))
    if 'autoencoder' in methods:
        pipeline.add_detector('Autoencoder', AutoencoderDetector(epochs=50))
    
    # Fit and detect
    pipeline.fit_all(series)
    results = pipeline.detect_all(series)
    
    # Generate report
    report = pipeline.generate_report(series, results)
    
    print("\nAnomaly Detection Results:")
    print("-" * 50)
    print(report.to_string(index=False))
    
    print(f"\nAnomaly detection completed using {len(methods)} methods")


def run_comprehensive_analysis(n_points: int) -> None:
    """Run comprehensive analysis combining all methods."""
    logger.info("Running comprehensive analysis...")
    
    # Generate data
    generator = TimeSeriesGenerator(random_state=42)
    multivariate_data = generator.generate_multivariate_series(n_points=n_points)
    univariate_series = generator.generate_univariate_series(n_points=n_points)
    
    # Statistical analysis
    analyzer = TimeSeriesAnalyzer()
    analysis_results = analyzer.analyze_multivariate_data(multivariate_data)
    
    print("\nComprehensive Analysis Results:")
    print("=" * 60)
    
    # Data overview
    print(f"Data Points: {len(multivariate_data)}")
    print(f"Variables: {len(multivariate_data.columns) - 1}")
    print(f"Date Range: {multivariate_data['Date'].min()} to {multivariate_data['Date'].max()}")
    print(f"Missing Values: {analysis_results['missing_values'].sum()}")
    
    # Basic statistics
    print("\nStatistical Summary:")
    print(analysis_results['statistics'].round(4))
    
    # Forecasting
    print("\nForecasting Analysis:")
    forecasting_pipeline = ForecastingPipeline()
    forecasting_pipeline.add_model('ARIMA', ARIMAForecaster())
    forecasting_pipeline.add_model('Prophet', ProphetForecaster())
    
    train_size = int(len(univariate_series) * 0.8)
    train_series = univariate_series[:train_size]
    test_series = univariate_series[train_size:]
    
    forecasting_pipeline.fit_all(train_series)
    forecasts = forecasting_pipeline.forecast_all(steps=len(test_series))
    
    metrics = forecasting_pipeline.evaluate_forecasts(test_series.values, forecasts)
    for model_name, model_metrics in metrics.items():
        print(f"  {model_name} RMSE: {model_metrics['RMSE']:.4f}")
    
    # Anomaly detection
    print("\nAnomaly Detection:")
    anomaly_detector = IsolationForestDetector(contamination=0.1)
    anomaly_detector.fit(univariate_series)
    scores, anomalies = anomaly_detector.detect(univariate_series)
    
    print(f"  Anomalies detected: {anomalies.sum()}")
    print(f"  Anomaly rate: {anomalies.sum()/len(univariate_series)*100:.1f}%")
    
    print("\nComprehensive analysis completed successfully!")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Time Series Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate --points 1000 --output data.csv
  %(prog)s forecast --points 500 --models arima prophet --steps 30
  %(prog)s anomalies --points 500 --methods statistical isolation
  %(prog)s comprehensive --points 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic time series data')
    generate_parser.add_argument('--points', type=int, default=1000, help='Number of data points')
    generate_parser.add_argument('--output', type=str, help='Output CSV file path')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Run forecasting analysis')
    forecast_parser.add_argument('--points', type=int, default=1000, help='Number of data points')
    forecast_parser.add_argument('--models', nargs='+', choices=['arima', 'prophet', 'lstm'], 
                                default=['arima', 'prophet'], help='Forecasting models to use')
    forecast_parser.add_argument('--steps', type=int, default=30, help='Number of forecast steps')
    
    # Anomaly detection command
    anomaly_parser = subparsers.add_parser('anomalies', help='Run anomaly detection analysis')
    anomaly_parser.add_argument('--points', type=int, default=1000, help='Number of data points')
    anomaly_parser.add_argument('--methods', nargs='+', 
                               choices=['statistical', 'iqr', 'isolation', 'autoencoder'],
                               default=['statistical', 'isolation'], help='Detection methods to use')
    anomaly_parser.add_argument('--contamination', type=float, default=0.1, 
                               help='Expected anomaly rate')
    
    # Comprehensive analysis command
    comprehensive_parser = subparsers.add_parser('comprehensive', help='Run comprehensive analysis')
    comprehensive_parser.add_argument('--points', type=int, default=1000, help='Number of data points')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'generate':
            generate_data(args.points, args.output)
        
        elif args.command == 'forecast':
            run_forecasting(args.points, args.models, args.steps)
        
        elif args.command == 'anomalies':
            run_anomaly_detection(args.points, args.methods, args.contamination)
        
        elif args.command == 'comprehensive':
            run_comprehensive_analysis(args.points)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

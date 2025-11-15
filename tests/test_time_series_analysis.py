"""
Unit Tests for Time Series Analysis Project

This module contains comprehensive unit tests for all components of the
time series analysis project.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import TimeSeriesGenerator, TimeSeriesVisualizer, TimeSeriesAnalyzer
from forecasting import ARIMAForecaster, ProphetForecaster, LSTMForecaster, ForecastingPipeline
from anomaly_detection import (
    StatisticalAnomalyDetector, 
    IsolationForestDetector, 
    AutoencoderDetector,
    AnomalyDetectionPipeline
)


class TestTimeSeriesGenerator(unittest.TestCase):
    """Test cases for TimeSeriesGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TimeSeriesGenerator(random_state=42)
    
    def test_generate_multivariate_series(self):
        """Test multivariate time series generation."""
        data = self.generator.generate_multivariate_series(n_points=100)
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertIn('Date', data.columns)
        self.assertIn('Temperature', data.columns)
        self.assertIn('Humidity', data.columns)
        self.assertIn('Pressure', data.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['Date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['Temperature']))
    
    def test_generate_univariate_series(self):
        """Test univariate time series generation."""
        series = self.generator.generate_univariate_series(n_points=100)
        
        # Check data structure
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), 100)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(series.index))
        self.assertTrue(pd.api.types.is_numeric_dtype(series))
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        data1 = self.generator.generate_multivariate_series(n_points=50)
        data2 = self.generator.generate_multivariate_series(n_points=50)
        
        # Should be identical with same random state
        np.testing.assert_array_almost_equal(data1.values, data2.values)


class TestTimeSeriesVisualizer(unittest.TestCase):
    """Test cases for TimeSeriesVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = TimeSeriesVisualizer()
        self.generator = TimeSeriesGenerator(random_state=42)
        self.data = self.generator.generate_multivariate_series(n_points=100)
    
    def test_plot_multivariate_lines(self):
        """Test multivariate line plotting."""
        # This test mainly checks that the method doesn't raise an exception
        try:
            self.visualizer.plot_multivariate_lines(self.data)
        except Exception as e:
            self.fail(f"plot_multivariate_lines raised an exception: {e}")
    
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting."""
        try:
            self.visualizer.plot_correlation_heatmap(self.data)
        except Exception as e:
            self.fail(f"plot_correlation_heatmap raised an exception: {e}")
    
    def test_plot_interactive_multivariate(self):
        """Test interactive multivariate plotting."""
        try:
            self.visualizer.plot_interactive_multivariate(self.data)
        except Exception as e:
            self.fail(f"plot_interactive_multivariate raised an exception: {e}")


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """Test cases for TimeSeriesAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TimeSeriesAnalyzer()
        self.generator = TimeSeriesGenerator(random_state=42)
        self.data = self.generator.generate_multivariate_series(n_points=100)
    
    def test_analyze_multivariate_data(self):
        """Test multivariate data analysis."""
        results = self.analyzer.analyze_multivariate_data(self.data)
        
        # Check results structure
        self.assertIn('data', results)
        self.assertIn('statistics', results)
        self.assertIn('correlations', results)
        self.assertIn('missing_values', results)
        self.assertIn('data_types', results)
        
        # Check data types
        self.assertIsInstance(results['data'], pd.DataFrame)
        self.assertIsInstance(results['statistics'], pd.DataFrame)
        self.assertIsInstance(results['correlations'], pd.DataFrame)
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        outliers = self.analyzer.detect_outliers(self.data, method='iqr')
        
        # Check structure
        self.assertIsInstance(outliers, dict)
        self.assertIn('Temperature', outliers)
        self.assertIn('Humidity', outliers)
        self.assertIn('Pressure', outliers)
        
        # Check that outlier indices are valid
        for column, indices in outliers.items():
            self.assertIsInstance(indices, list)
            for idx in indices:
                self.assertLess(idx, len(self.data))


class TestARIMAForecaster(unittest.TestCase):
    """Test cases for ARIMAForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = ARIMAForecaster()
        self.generator = TimeSeriesGenerator(random_state=42)
        self.series = self.generator.generate_univariate_series(n_points=200)
    
    def test_check_stationarity(self):
        """Test stationarity checking."""
        result = self.forecaster.check_stationarity(self.series)
        
        # Check result structure
        self.assertIn('adf_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('critical_values', result)
        self.assertIn('is_stationary', result)
        
        # Check data types
        self.assertIsInstance(result['adf_statistic'], float)
        self.assertIsInstance(result['p_value'], float)
        self.assertIsInstance(result['is_stationary'], bool)
    
    def test_make_stationary(self):
        """Test making series stationary."""
        stationary_series, diff_count = self.forecaster.make_stationary(self.series)
        
        # Check structure
        self.assertIsInstance(stationary_series, pd.Series)
        self.assertIsInstance(diff_count, int)
        self.assertGreaterEqual(diff_count, 0)
        self.assertLessEqual(diff_count, 2)
    
    def test_fit_and_forecast(self):
        """Test model fitting and forecasting."""
        # Fit model
        self.forecaster.fit(self.series)
        
        # Check that model is fitted
        self.assertIsNotNone(self.forecaster.fitted_model)
        
        # Generate forecast
        forecast, lower, upper = self.forecaster.forecast(steps=10)
        
        # Check forecast structure
        self.assertEqual(len(forecast), 10)
        self.assertEqual(len(lower), 10)
        self.assertEqual(len(upper), 10)
        
        # Check that confidence intervals make sense
        self.assertTrue(np.all(lower <= forecast))
        self.assertTrue(np.all(forecast <= upper))


class TestProphetForecaster(unittest.TestCase):
    """Test cases for ProphetForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = ProphetForecaster()
        self.generator = TimeSeriesGenerator(random_state=42)
        self.series = self.generator.generate_univariate_series(n_points=200)
    
    def test_prepare_data(self):
        """Test data preparation for Prophet."""
        df = self.forecaster.prepare_data(self.series)
        
        # Check structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('ds', df.columns)
        self.assertIn('y', df.columns)
        self.assertEqual(len(df), len(self.series))
    
    def test_fit_and_forecast(self):
        """Test model fitting and forecasting."""
        # Fit model
        self.forecaster.fit(self.series)
        
        # Check that model is fitted
        self.assertIsNotNone(self.forecaster.fitted_model)
        
        # Generate forecast
        forecast_df = self.forecaster.forecast(periods=10)
        
        # Check forecast structure
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertIn('yhat', forecast_df.columns)
        self.assertIn('yhat_lower', forecast_df.columns)
        self.assertIn('yhat_upper', forecast_df.columns)


class TestStatisticalAnomalyDetector(unittest.TestCase):
    """Test cases for StatisticalAnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = StatisticalAnomalyDetector(method='zscore', threshold=3.0)
        self.generator = TimeSeriesGenerator(random_state=42)
        self.series = self.generator.generate_univariate_series(n_points=200)
    
    def test_fit(self):
        """Test model fitting."""
        self.detector.fit(self.series)
        
        # Check that parameters are fitted
        self.assertIsNotNone(self.detector.fitted_params)
        self.assertIn('mean', self.detector.fitted_params)
        self.assertIn('std', self.detector.fitted_params)
    
    def test_detect(self):
        """Test anomaly detection."""
        self.detector.fit(self.series)
        scores, anomalies = self.detector.detect(self.series)
        
        # Check structure
        self.assertEqual(len(scores), len(self.series))
        self.assertEqual(len(anomalies), len(self.series))
        
        # Check data types
        self.assertTrue(np.issubdtype(scores.dtype, np.floating))
        self.assertTrue(np.issubdtype(anomalies.dtype, np.bool_))
        
        # Check that scores are non-negative
        self.assertTrue(np.all(scores >= 0))


class TestIsolationForestDetector(unittest.TestCase):
    """Test cases for IsolationForestDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = IsolationForestDetector(contamination=0.1)
        self.generator = TimeSeriesGenerator(random_state=42)
        self.series = self.generator.generate_univariate_series(n_points=200)
    
    def test_fit_and_detect(self):
        """Test model fitting and anomaly detection."""
        # Fit model
        self.detector.fit(self.series)
        
        # Detect anomalies
        scores, anomalies = self.detector.detect(self.series)
        
        # Check structure
        self.assertEqual(len(scores), len(self.series))
        self.assertEqual(len(anomalies), len(self.series))
        
        # Check data types
        self.assertTrue(np.issubdtype(scores.dtype, np.floating))
        self.assertTrue(np.issubdtype(anomalies.dtype, np.bool_))
    
    def test_create_features(self):
        """Test feature creation."""
        features = self.detector._create_features(self.series)
        
        # Check structure
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.series))
        self.assertGreater(features.shape[1], 0)


class TestForecastingPipeline(unittest.TestCase):
    """Test cases for ForecastingPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = ForecastingPipeline()
        self.generator = TimeSeriesGenerator(random_state=42)
        self.series = self.generator.generate_univariate_series(n_points=200)
    
    def test_add_model(self):
        """Test adding models to pipeline."""
        arima_model = ARIMAForecaster()
        self.pipeline.add_model('ARIMA', arima_model)
        
        self.assertIn('ARIMA', self.pipeline.models)
        self.assertEqual(self.pipeline.models['ARIMA'], arima_model)
    
    def test_fit_all(self):
        """Test fitting all models."""
        self.pipeline.add_model('ARIMA', ARIMAForecaster())
        
        # Should not raise an exception
        try:
            self.pipeline.fit_all(self.series)
        except Exception as e:
            self.fail(f"fit_all raised an exception: {e}")
    
    def test_forecast_all(self):
        """Test forecasting with all models."""
        self.pipeline.add_model('ARIMA', ARIMAForecaster())
        self.pipeline.fit_all(self.series)
        
        forecasts = self.pipeline.forecast_all(steps=10)
        
        # Check structure
        self.assertIsInstance(forecasts, dict)
        self.assertIn('ARIMA', forecasts)
        self.assertEqual(len(forecasts['ARIMA']), 10)


class TestAnomalyDetectionPipeline(unittest.TestCase):
    """Test cases for AnomalyDetectionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = AnomalyDetectionPipeline()
        self.generator = TimeSeriesGenerator(random_state=42)
        self.series = self.generator.generate_univariate_series(n_points=200)
    
    def test_add_detector(self):
        """Test adding detectors to pipeline."""
        detector = StatisticalAnomalyDetector()
        self.pipeline.add_detector('Statistical', detector)
        
        self.assertIn('Statistical', self.pipeline.detectors)
        self.assertEqual(self.pipeline.detectors['Statistical'], detector)
    
    def test_fit_all(self):
        """Test fitting all detectors."""
        self.pipeline.add_detector('Statistical', StatisticalAnomalyDetector())
        
        # Should not raise an exception
        try:
            self.pipeline.fit_all(self.series)
        except Exception as e:
            self.fail(f"fit_all raised an exception: {e}")
    
    def test_detect_all(self):
        """Test detecting anomalies with all detectors."""
        self.pipeline.add_detector('Statistical', StatisticalAnomalyDetector())
        self.pipeline.fit_all(self.series)
        
        results = self.pipeline.detect_all(self.series)
        
        # Check structure
        self.assertIsInstance(results, dict)
        self.assertIn('Statistical', results)
        
        scores, anomalies = results['Statistical']
        self.assertEqual(len(scores), len(self.series))
        self.assertEqual(len(anomalies), len(self.series))
    
    def test_generate_report(self):
        """Test report generation."""
        self.pipeline.add_detector('Statistical', StatisticalAnomalyDetector())
        self.pipeline.fit_all(self.series)
        results = self.pipeline.detect_all(self.series)
        
        report = self.pipeline.generate_report(self.series, results)
        
        # Check structure
        self.assertIsInstance(report, pd.DataFrame)
        self.assertIn('Detector', report.columns)
        self.assertIn('Anomalies_Detected', report.columns)
        self.assertIn('Anomaly_Rate_Percent', report.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TimeSeriesGenerator(random_state=42)
        self.data = self.generator.generate_multivariate_series(n_points=200)
        self.series = self.data.set_index('Date')['Temperature']
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow."""
        # Data analysis
        analyzer = TimeSeriesAnalyzer()
        analysis_results = analyzer.analyze_multivariate_data(self.data)
        
        # Forecasting
        forecasting_pipeline = ForecastingPipeline()
        forecasting_pipeline.add_model('ARIMA', ARIMAForecaster())
        forecasting_pipeline.fit_all(self.series)
        forecasts = forecasting_pipeline.forecast_all(steps=20)
        
        # Anomaly detection
        anomaly_pipeline = AnomalyDetectionPipeline()
        anomaly_pipeline.add_detector('Statistical', StatisticalAnomalyDetector())
        anomaly_pipeline.fit_all(self.series)
        anomaly_results = anomaly_pipeline.detect_all(self.series)
        
        # Check that all components work together
        self.assertIsNotNone(analysis_results)
        self.assertIsNotNone(forecasts)
        self.assertIsNotNone(anomaly_results)
    
    def test_data_consistency(self):
        """Test that data remains consistent across different operations."""
        original_length = len(self.series)
        
        # Test that series length doesn't change unexpectedly
        analyzer = TimeSeriesAnalyzer()
        outliers = analyzer.detect_outliers(self.data)
        
        # Test forecasting doesn't modify original data
        forecaster = ARIMAForecaster()
        forecaster.fit(self.series)
        forecast, _, _ = forecaster.forecast(steps=10)
        
        self.assertEqual(len(self.series), original_length)
        self.assertEqual(len(forecast), 10)


def run_tests():
    """Run all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTimeSeriesGenerator,
        TestTimeSeriesVisualizer,
        TestTimeSeriesAnalyzer,
        TestARIMAForecaster,
        TestProphetForecaster,
        TestStatisticalAnomalyDetector,
        TestIsolationForestDetector,
        TestForecastingPipeline,
        TestAnomalyDetectionPipeline,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

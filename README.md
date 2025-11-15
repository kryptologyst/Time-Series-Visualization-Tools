# Time Series Visualization Tools

A comprehensive, modern time series analysis toolkit featuring advanced forecasting methods, anomaly detection algorithms, and interactive visualization capabilities.

## Features

### Core Functionality
- **Data Generation**: Synthetic time series with customizable trends, seasonality, and noise
- **Visualization**: Interactive plots using Matplotlib, Seaborn, and Plotly
- **Statistical Analysis**: Comprehensive descriptive statistics and correlation analysis

### Forecasting Methods
- **ARIMA**: AutoRegressive Integrated Moving Average with automatic parameter selection
- **Prophet**: Facebook's forecasting tool for time series with seasonality
- **LSTM**: Deep learning approach using Long Short-Term Memory networks
- **Model Comparison**: Side-by-side evaluation of multiple forecasting approaches

### Anomaly Detection
- **Statistical Methods**: Z-score and IQR-based outlier detection
- **Machine Learning**: Isolation Forest for unsupervised anomaly detection
- **Deep Learning**: Autoencoder-based anomaly detection using PyTorch
- **Comparative Analysis**: Multi-method anomaly detection comparison

### User Interface
- **Streamlit Dashboard**: Interactive web interface for exploration and analysis
- **Command Line Interface**: Script-based analysis for automation
- **Jupyter Notebooks**: Interactive analysis notebooks

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Time-Series-Visualization-Tools.git
cd Time-Series-Visualization-Tools
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Streamlit Dashboard
```bash
streamlit run src/streamlit_app.py
```

### Using Python Scripts
```python
from src.core import TimeSeriesAnalyzer
from src.forecasting import ForecastingPipeline
from src.anomaly_detection import AnomalyDetectionPipeline

# Generate and analyze data
analyzer = TimeSeriesAnalyzer()
results = analyzer.analyze_multivariate_data(n_points=1000)

# Forecasting
forecasting_pipeline = ForecastingPipeline()
forecasting_pipeline.add_model('ARIMA', ARIMAForecaster())
forecasting_pipeline.add_model('Prophet', ProphetForecaster())
forecasting_pipeline.fit_all(results['data'].set_index('Date')['Temperature'])
forecasts = forecasting_pipeline.forecast_all(steps=30)

# Anomaly detection
anomaly_pipeline = AnomalyDetectionPipeline()
anomaly_pipeline.add_detector('Statistical', StatisticalAnomalyDetector())
anomaly_pipeline.add_detector('Isolation Forest', IsolationForestDetector())
anomaly_pipeline.fit_all(results['data'].set_index('Date')['Temperature'])
anomaly_results = anomaly_pipeline.detect_all(results['data'].set_index('Date')['Temperature'])
```

### Running Tests
```bash
python -m pytest tests/ -v
```

## Project Structure

```
time-series-analysis-project/
├── src/                          # Source code
│   ├── core.py                   # Core functionality and data generation
│   ├── forecasting.py           # Forecasting methods (ARIMA, Prophet, LSTM)
│   ├── anomaly_detection.py     # Anomaly detection algorithms
│   └── streamlit_app.py         # Streamlit web interface
├── tests/                        # Unit tests
│   └── test_time_series_analysis.py
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data storage (created automatically)
├── models/                       # Model storage (created automatically)
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Configuration

The project uses YAML configuration files for customizable settings. Key configuration options include:

### Data Settings
- Number of data points for synthetic generation
- Noise levels and seasonality parameters
- Date ranges and frequencies

### Model Settings
- ARIMA parameter ranges
- Prophet seasonality options
- LSTM architecture parameters
- Anomaly detection thresholds

### Visualization Settings
- Figure sizes and styles
- Color palettes
- Plot configurations

## API Reference

### TimeSeriesGenerator
Generate synthetic time series data for testing and demonstration.

```python
generator = TimeSeriesGenerator(random_state=42)
data = generator.generate_multivariate_series(n_points=1000)
series = generator.generate_univariate_series(n_points=500)
```

### ForecastingPipeline
Compare multiple forecasting methods.

```python
pipeline = ForecastingPipeline()
pipeline.add_model('ARIMA', ARIMAForecaster())
pipeline.add_model('Prophet', ProphetForecaster())
pipeline.add_model('LSTM', LSTMForecaster())
pipeline.fit_all(series)
forecasts = pipeline.forecast_all(steps=30)
```

### AnomalyDetectionPipeline
Compare multiple anomaly detection methods.

```python
pipeline = AnomalyDetectionPipeline()
pipeline.add_detector('Statistical', StatisticalAnomalyDetector())
pipeline.add_detector('Isolation Forest', IsolationForestDetector())
pipeline.add_detector('Autoencoder', AutoencoderDetector())
pipeline.fit_all(series)
results = pipeline.detect_all(series)
```

## Examples

### Basic Time Series Analysis
```python
from src.core import TimeSeriesAnalyzer

analyzer = TimeSeriesAnalyzer()
results = analyzer.analyze_multivariate_data(n_points=1000)

# Access results
data = results['data']
statistics = results['statistics']
correlations = results['correlations']
```

### Forecasting Comparison
```python
from src.forecasting import ForecastingPipeline, ARIMAForecaster, ProphetForecaster

# Setup pipeline
pipeline = ForecastingPipeline()
pipeline.add_model('ARIMA', ARIMAForecaster())
pipeline.add_model('Prophet', ProphetForecaster())

# Train and forecast
pipeline.fit_all(series)
forecasts = pipeline.forecast_all(steps=30)

# Evaluate performance
metrics = pipeline.evaluate_forecasts(actual_values, forecasts)
```

### Anomaly Detection
```python
from src.anomaly_detection import AnomalyDetectionPipeline, StatisticalAnomalyDetector

# Setup pipeline
pipeline = AnomalyDetectionPipeline()
pipeline.add_detector('Statistical', StatisticalAnomalyDetector(method='zscore'))

# Detect anomalies
pipeline.fit_all(series)
results = pipeline.detect_all(series)

# Generate report
report = pipeline.generate_report(series, results)
```

## Performance Considerations

### Memory Usage
- Large datasets may require chunked processing
- LSTM models can be memory-intensive for long sequences
- Consider reducing batch sizes for limited memory environments

### Training Time
- ARIMA: Fast training, suitable for real-time applications
- Prophet: Moderate training time, good for batch processing
- LSTM: Longer training time, best for offline analysis

### Scalability
- Statistical methods scale linearly with data size
- Machine learning methods may require optimization for very large datasets
- Consider using GPU acceleration for deep learning models

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest tests/ -v`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## Testing

The project includes comprehensive unit tests covering:
- Data generation and validation
- Model fitting and prediction
- Anomaly detection accuracy
- Integration between components
- Error handling and edge cases

Run tests with:
```bash
python -m pytest tests/ -v --cov=src
```

## Dependencies

### Core Libraries
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.15.0

### Time Series Analysis
- statsmodels >= 0.14.0
- pmdarima >= 2.0.0
- prophet >= 1.1.0
- tslearn >= 0.6.0
- darts >= 0.24.0
- sktime >= 0.20.0

### Machine Learning
- scikit-learn >= 1.3.0
- torch >= 2.0.0
- tensorflow >= 2.13.0

### Web Interface
- streamlit >= 1.25.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Prophet team for the excellent forecasting library
- Statsmodels contributors for comprehensive statistical tools
- PyTorch team for deep learning capabilities
- Streamlit team for the intuitive web framework

## Support

For questions, issues, or contributions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed description
3. Contact the maintainers for urgent matters

## Changelog

### Version 1.0.0
- Initial release with core functionality
- ARIMA, Prophet, and LSTM forecasting
- Statistical and ML-based anomaly detection
- Streamlit web interface
- Comprehensive test suite
- Full documentation and examples
# Time-Series-Visualization-Tools

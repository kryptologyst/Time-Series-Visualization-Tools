"""
Streamlit Web Interface for Time Series Analysis

This module provides a web-based interface for interactive time series analysis
using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core import TimeSeriesGenerator, TimeSeriesVisualizer, TimeSeriesAnalyzer
from src.forecasting import ARIMAForecaster, ProphetForecaster, LSTMForecaster, ForecastingPipeline
from src.anomaly_detection import (
    StatisticalAnomalyDetector, 
    IsolationForestDetector, 
    AutoencoderDetector,
    AnomalyDetectionPipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from YAML file."""
    try:
        with open('config/config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("Configuration file not found. Using default settings.")
        return {}

def main():
    """Main Streamlit application."""
    
    # Load configuration
    config = load_config()
    
    # Title and description
    st.title("📈 Time Series Analysis Dashboard")
    st.markdown("""
    This dashboard provides comprehensive time series analysis capabilities including:
    - Data generation and visualization
    - Forecasting with multiple methods (ARIMA, Prophet, LSTM)
    - Anomaly detection using various algorithms
    - Interactive exploration and comparison tools
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Data generation options
    st.sidebar.subheader("Data Settings")
    n_points = st.sidebar.slider("Number of Data Points", 100, 2000, 1000)
    noise_level = st.sidebar.slider("Noise Level", 0.01, 1.0, 0.1)
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Data Visualization", "Forecasting", "Anomaly Detection", "Comprehensive Analysis"]
    )
    
    # Generate data
    @st.cache_data
    def generate_data(n_points: int, noise_level: float, start_date: str):
        """Generate synthetic time series data."""
        generator = TimeSeriesGenerator()
        return generator.generate_multivariate_series(
            n_points=n_points,
            start_date=start_date.strftime("%Y-%m-%d"),
            noise_level=noise_level
        )
    
    data = generate_data(n_points, noise_level, start_date)
    
    # Main content area
    if analysis_type == "Data Visualization":
        data_visualization_page(data)
    elif analysis_type == "Forecasting":
        forecasting_page(data)
    elif analysis_type == "Anomaly Detection":
        anomaly_detection_page(data)
    elif analysis_type == "Comprehensive Analysis":
        comprehensive_analysis_page(data)

def data_visualization_page(data: pd.DataFrame):
    """Data visualization page."""
    st.header("📊 Data Visualization")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(data))
    
    with col2:
        st.metric("Variables", len(data.columns) - 1)
    
    with col3:
        st.metric("Date Range", f"{(data['Date'].max() - data['Date'].min()).days} days")
    
    with col4:
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Time series plots
    st.subheader("Time Series Plots")
    
    # Multivariate line plot
    fig = px.line(data, x='Date', y=data.columns[1:], 
                  title="Multivariate Time Series")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Correlation Analysis")
    corr_matrix = data.drop('Date', axis=1).corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)
    
    # Individual variable analysis
    st.subheader("Individual Variable Analysis")
    selected_variable = st.selectbox("Select Variable", data.columns[1:])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution plot
        fig = px.histogram(data, x=selected_variable, 
                          title=f"Distribution of {selected_variable}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(data, y=selected_variable, 
                    title=f"Box Plot of {selected_variable}")
        st.plotly_chart(fig, use_container_width=True)

def forecasting_page(data: pd.DataFrame):
    """Forecasting page."""
    st.header("🔮 Forecasting Analysis")
    
    # Select variable for forecasting
    selected_variable = st.selectbox("Select Variable for Forecasting", data.columns[1:])
    series = data.set_index('Date')[selected_variable]
    
    # Forecasting parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_steps = st.slider("Forecast Steps", 10, 100, 30)
        train_size = st.slider("Training Set Size (%)", 70, 90, 80)
    
    with col2:
        selected_models = st.multiselect(
            "Select Forecasting Models",
            ["ARIMA", "Prophet", "LSTM"],
            default=["ARIMA", "Prophet"]
        )
    
    if st.button("Run Forecasting Analysis"):
        with st.spinner("Running forecasting analysis..."):
            # Split data
            split_idx = int(len(series) * train_size / 100)
            train_series = series[:split_idx]
            test_series = series[split_idx:]
            
            # Initialize pipeline
            pipeline = ForecastingPipeline()
            
            # Add selected models
            if "ARIMA" in selected_models:
                pipeline.add_model('ARIMA', ARIMAForecaster())
            
            if "Prophet" in selected_models:
                pipeline.add_model('Prophet', ProphetForecaster())
            
            if "LSTM" in selected_models:
                pipeline.add_model('LSTM', LSTMForecaster(epochs=50))
            
            # Fit models
            pipeline.fit_all(train_series)
            
            # Generate forecasts
            forecasts = pipeline.forecast_all(steps=forecast_steps)
            
            # Display results
            st.subheader("Forecast Results")
            
            # Create forecast plot
            fig = go.Figure()
            
            # Plot training data
            fig.add_trace(go.Scatter(
                x=train_series.index,
                y=train_series.values,
                mode='lines',
                name='Training Data',
                line=dict(color='blue')
            ))
            
            # Plot test data
            fig.add_trace(go.Scatter(
                x=test_series.index,
                y=test_series.values,
                mode='lines',
                name='Actual Test Data',
                line=dict(color='green')
            ))
            
            # Plot forecasts
            colors = ['red', 'orange', 'purple']
            for i, (model_name, forecast) in enumerate(forecasts.items()):
                future_dates = pd.date_range(
                    start=train_series.index[-1],
                    periods=len(forecast) + 1,
                    freq=train_series.index.freq
                )[1:]
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast,
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ))
            
            fig.update_layout(
                title=f"Forecasting Results for {selected_variable}",
                xaxis_title="Date",
                yaxis_title="Value",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Evaluation metrics
            if len(test_series) >= forecast_steps:
                st.subheader("Model Performance")
                
                # Calculate metrics
                test_subset = test_series[:forecast_steps]
                metrics = pipeline.evaluate_forecasts(test_subset.values, forecasts)
                
                # Display metrics table
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df, use_container_width=True)
                
                # Best model
                best_model = metrics_df['RMSE'].idxmin()
                st.success(f"Best performing model: **{best_model}** (Lowest RMSE)")

def anomaly_detection_page(data: pd.DataFrame):
    """Anomaly detection page."""
    st.header("🚨 Anomaly Detection Analysis")
    
    # Select variable for anomaly detection
    selected_variable = st.selectbox("Select Variable for Anomaly Detection", data.columns[1:])
    series = data.set_index('Date')[selected_variable]
    
    # Detection parameters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_detectors = st.multiselect(
            "Select Anomaly Detection Methods",
            ["Statistical (Z-Score)", "Statistical (IQR)", "Isolation Forest", "Autoencoder"],
            default=["Statistical (Z-Score)", "Isolation Forest"]
        )
    
    with col2:
        contamination = st.slider("Expected Anomaly Rate (%)", 1, 20, 10) / 100
        threshold = st.slider("Statistical Threshold", 1.0, 5.0, 3.0)
    
    if st.button("Run Anomaly Detection"):
        with st.spinner("Running anomaly detection analysis..."):
            # Initialize pipeline
            pipeline = AnomalyDetectionPipeline()
            
            # Add selected detectors
            if "Statistical (Z-Score)" in selected_detectors:
                pipeline.add_detector('Statistical (Z-Score)', 
                                    StatisticalAnomalyDetector('zscore', threshold))
            
            if "Statistical (IQR)" in selected_detectors:
                pipeline.add_detector('Statistical (IQR)', 
                                    StatisticalAnomalyDetector('iqr', threshold))
            
            if "Isolation Forest" in selected_detectors:
                pipeline.add_detector('Isolation Forest', 
                                    IsolationForestDetector(contamination=contamination))
            
            if "Autoencoder" in selected_detectors:
                pipeline.add_detector('Autoencoder', 
                                    AutoencoderDetector(epochs=50))
            
            # Fit detectors
            pipeline.fit_all(series)
            
            # Detect anomalies
            results = pipeline.detect_all(series)
            
            # Display results
            st.subheader("Anomaly Detection Results")
            
            # Create comparison plot
            n_detectors = len(results)
            fig = make_subplots(
                rows=n_detectors + 1, cols=1,
                subplot_titles=['Original Time Series'] + list(results.keys()),
                vertical_spacing=0.05
            )
            
            # Plot original series
            fig.add_trace(
                go.Scatter(x=series.index, y=series.values, 
                          mode='lines', name='Original', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Plot detection results
            colors = ['red', 'orange', 'purple', 'green']
            for i, (name, (scores, anomalies)) in enumerate(results.items()):
                row = i + 2
                
                # Normal points
                normal_mask = ~anomalies
                fig.add_trace(
                    go.Scatter(x=series.index[normal_mask], y=series.values[normal_mask],
                              mode='lines', name=f'{name} Normal', 
                              line=dict(color='blue'), showlegend=False),
                    row=row, col=1
                )
                
                # Anomalies
                if np.any(anomalies):
                    fig.add_trace(
                        go.Scatter(x=series.index[anomalies], y=series.values[anomalies],
                                  mode='markers', name=f'{name} Anomalies',
                                  marker=dict(color=colors[i % len(colors)], size=8),
                                  showlegend=False),
                        row=row, col=1
                    )
            
            fig.update_layout(height=300 * (n_detectors + 1), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary report
            st.subheader("Detection Summary")
            report = pipeline.generate_report(series, results)
            st.dataframe(report, use_container_width=True)

def comprehensive_analysis_page(data: pd.DataFrame):
    """Comprehensive analysis page."""
    st.header("🔍 Comprehensive Analysis")
    
    st.markdown("""
    This page provides a complete analysis combining visualization, forecasting, and anomaly detection.
    """)
    
    # Select variable for comprehensive analysis
    selected_variable = st.selectbox("Select Variable for Comprehensive Analysis", data.columns[1:])
    series = data.set_index('Date')[selected_variable]
    
    if st.button("Run Comprehensive Analysis"):
        with st.spinner("Running comprehensive analysis..."):
            
            # Data visualization
            st.subheader("📊 Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic statistics
                stats = series.describe()
                st.dataframe(stats, use_container_width=True)
            
            with col2:
                # Time series plot
                fig = px.line(x=series.index, y=series.values, 
                              title=f"Time Series: {selected_variable}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Forecasting analysis
            st.subheader("🔮 Forecasting Analysis")
            
            # Split data
            train_size = int(len(series) * 0.8)
            train_series = series[:train_size]
            test_series = series[train_size:]
            
            # Quick forecasting
            pipeline = ForecastingPipeline()
            pipeline.add_model('ARIMA', ARIMAForecaster())
            pipeline.add_model('Prophet', ProphetForecaster())
            
            pipeline.fit_all(train_series)
            forecasts = pipeline.forecast_all(steps=len(test_series))
            
            # Forecast plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_series.index, y=train_series.values,
                                   mode='lines', name='Training Data'))
            fig.add_trace(go.Scatter(x=test_series.index, y=test_series.values,
                                   mode='lines', name='Actual'))
            
            for model_name, forecast in forecasts.items():
                future_dates = pd.date_range(start=train_series.index[-1],
                                            periods=len(forecast) + 1,
                                            freq=train_series.index.freq)[1:]
                fig.add_trace(go.Scatter(x=future_dates, y=forecast,
                                       mode='lines', name=f'{model_name} Forecast',
                                       line=dict(dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly detection
            st.subheader("🚨 Anomaly Detection")
            
            detector = IsolationForestDetector()
            detector.fit(series)
            scores, anomalies = detector.detect(series)
            
            # Anomaly plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                   mode='lines', name='Time Series'))
            
            if np.any(anomalies):
                fig.add_trace(go.Scatter(x=series.index[anomalies], y=series.values[anomalies],
                                       mode='markers', name='Anomalies',
                                       marker=dict(color='red', size=8)))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.subheader("📋 Analysis Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Points", len(series))
                st.metric("Training Points", len(train_series))
                st.metric("Test Points", len(test_series))
            
            with col2:
                st.metric("Mean Value", f"{series.mean():.2f}")
                st.metric("Std Deviation", f"{series.std():.2f}")
                st.metric("Min Value", f"{series.min():.2f}")
            
            with col3:
                st.metric("Anomalies Detected", np.sum(anomalies))
                st.metric("Anomaly Rate", f"{np.sum(anomalies)/len(series)*100:.1f}%")
                
                if forecasts:
                    best_model = min(forecasts.keys(), 
                                   key=lambda x: np.sqrt(np.mean((test_series.values - forecasts[x])**2)))
                    st.metric("Best Forecast Model", best_model)

if __name__ == "__main__":
    main()

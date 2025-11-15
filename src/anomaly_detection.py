"""
Anomaly Detection Module

This module provides various anomaly detection methods for time series data
including statistical methods, machine learning approaches, and deep learning.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')

# Anomaly detection libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("Warning: PyTorch not available. Install with: pip install torch")

logger = logging.getLogger(__name__)


class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection."""
    
    def __init__(self, method: str = "zscore", threshold: float = 3.0):
        """
        Initialize statistical anomaly detector.
        
        Args:
            method: Detection method ('zscore', 'iqr', 'modified_zscore')
            threshold: Threshold for anomaly detection
        """
        self.method = method
        self.threshold = threshold
        self.fitted_params = {}
    
    def fit(self, series: pd.Series) -> 'StatisticalAnomalyDetector':
        """
        Fit the statistical model to the time series.
        
        Args:
            series: Time series data
            
        Returns:
            Self for method chaining
        """
        if self.method == "zscore":
            self.fitted_params = {
                'mean': series.mean(),
                'std': series.std()
            }
        elif self.method == "modified_zscore":
            median = series.median()
            mad = np.median(np.abs(series - median))
            self.fitted_params = {
                'median': median,
                'mad': mad
            }
        elif self.method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            self.fitted_params = {
                'q1': q1,
                'q3': q3,
                'iqr': iqr
            }
        
        return self
    
    def detect(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in the time series.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (anomaly_scores, anomaly_labels)
        """
        if not self.fitted_params:
            raise ValueError("Model must be fitted before detection")
        
        if self.method == "zscore":
            scores = np.abs((series - self.fitted_params['mean']) / self.fitted_params['std'])
            anomalies = scores > self.threshold
        
        elif self.method == "modified_zscore":
            scores = 0.6745 * (series - self.fitted_params['median']) / self.fitted_params['mad']
            scores = np.abs(scores)
            anomalies = scores > self.threshold
        
        elif self.method == "iqr":
            lower_bound = self.fitted_params['q1'] - self.threshold * self.fitted_params['iqr']
            upper_bound = self.fitted_params['q3'] + self.threshold * self.fitted_params['iqr']
            scores = np.maximum(
                series - upper_bound,
                lower_bound - series
            )
            scores = np.maximum(scores, 0)  # Only positive scores
            anomalies = (series < lower_bound) | (series > upper_bound)
        
        return scores, anomalies
    
    def plot_anomalies(
        self,
        series: pd.Series,
        anomalies: np.ndarray,
        title: str = "Statistical Anomaly Detection"
    ) -> None:
        """
        Plot the time series with detected anomalies highlighted.
        
        Args:
            series: Time series data
            anomalies: Boolean array indicating anomalies
            title: Plot title
        """
        plt.figure(figsize=(15, 6))
        
        # Plot normal points
        normal_mask = ~anomalies
        plt.plot(series.index[normal_mask], series.values[normal_mask], 
                'b-', alpha=0.7, label='Normal')
        
        # Plot anomalies
        if np.any(anomalies):
            plt.scatter(series.index[anomalies], series.values[anomalies], 
                       color='red', s=50, label='Anomalies', zorder=5)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class IsolationForestDetector:
    """Isolation Forest-based anomaly detection."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.scaler = StandardScaler()
    
    def fit(self, series: pd.Series) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model.
        
        Args:
            series: Time series data
            
        Returns:
            Self for method chaining
        """
        # Create features from time series
        features = self._create_features(series)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit the model
        self.model.fit(features_scaled)
        
        return self
    
    def _create_features(self, series: pd.Series) -> np.ndarray:
        """
        Create features from time series for anomaly detection.
        
        Args:
            series: Time series data
            
        Returns:
            Feature matrix
        """
        features = []
        
        for i in range(len(series)):
            feature_row = []
            
            # Current value
            feature_row.append(series.iloc[i])
            
            # Rolling statistics
            window_sizes = [5, 10, 20]
            for window in window_sizes:
                if i >= window - 1:
                    window_data = series.iloc[i-window+1:i+1]
                    feature_row.extend([
                        window_data.mean(),
                        window_data.std(),
                        window_data.min(),
                        window_data.max()
                    ])
                else:
                    feature_row.extend([0, 0, 0, 0])
            
            # Lag features
            for lag in [1, 2, 3]:
                if i >= lag:
                    feature_row.append(series.iloc[i-lag])
                else:
                    feature_row.append(0)
            
            # Difference features
            for diff in [1, 2]:
                if i >= diff:
                    feature_row.append(series.iloc[i] - series.iloc[i-diff])
                else:
                    feature_row.append(0)
            
            features.append(feature_row)
        
        return np.array(features)
    
    def detect(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (anomaly_scores, anomaly_labels)
        """
        # Create features
        features = self._create_features(series)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict anomalies
        predictions = self.model.predict(features_scaled)
        scores = self.model.score_samples(features_scaled)
        
        # Convert to boolean (1 = normal, -1 = anomaly)
        anomalies = predictions == -1
        
        return scores, anomalies


class AutoencoderDetector:
    """Autoencoder-based anomaly detection using PyTorch."""
    
    def __init__(
        self,
        sequence_length: int = 20,
        encoding_dim: int = 32,
        hidden_dims: List[int] = [64, 32],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Initialize Autoencoder detector.
        
        Args:
            sequence_length: Length of input sequences
            encoding_dim: Dimension of encoded representation
            hidden_dims: Hidden layer dimensions
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = None
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences for autoencoder training.
        
        Args:
            data: Time series data
            
        Returns:
            Array of sequences
        """
        sequences = []
        
        for i in range(self.sequence_length, len(data)):
            sequences.append(data[i-self.sequence_length:i])
        
        return np.array(sequences)
    
    def build_autoencoder(self, input_dim: int) -> nn.Module:
        """
        Build autoencoder model.
        
        Args:
            input_dim: Input dimension (sequence_length)
            
        Returns:
            PyTorch autoencoder model
        """
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim, hidden_dims):
                super(Autoencoder, self).__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Decoder
                decoder_layers = []
                prev_dim = encoding_dim
                
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return Autoencoder(input_dim, self.encoding_dim, self.hidden_dims)
    
    def fit(self, series: pd.Series) -> 'AutoencoderDetector':
        """
        Fit the autoencoder model.
        
        Args:
            series: Time series data
            
        Returns:
            Self for method chaining
        """
        # Prepare data
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data).flatten()
        
        # Create sequences
        sequences = self.create_sequences(scaled_data)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(sequences)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Build model
        self.model = self.build_autoencoder(self.sequence_length).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f'Epoch [{epoch}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        # Calculate threshold based on reconstruction error
        self.model.eval()
        with torch.no_grad():
            reconstruction_errors = []
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                reconstructed = self.model(batch_X)
                error = torch.mean((batch_X - reconstructed) ** 2, dim=1)
                reconstruction_errors.extend(error.cpu().numpy())
            
            # Set threshold as 95th percentile of reconstruction errors
            self.threshold = np.percentile(reconstruction_errors, 95)
        
        return self
    
    def detect(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using autoencoder reconstruction error.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (anomaly_scores, anomaly_labels)
        """
        if self.model is None or self.threshold is None:
            raise ValueError("Model must be fitted before detection")
        
        # Prepare data
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data).flatten()
        
        # Create sequences
        sequences = self.create_sequences(scaled_data)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(sequences)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Calculate reconstruction errors
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                reconstructed = self.model(batch_X)
                error = torch.mean((batch_X - reconstructed) ** 2, dim=1)
                reconstruction_errors.extend(error.cpu().numpy())
        
        # Pad with zeros for the first sequence_length points
        scores = np.zeros(len(series))
        scores[self.sequence_length:] = reconstruction_errors
        
        # Detect anomalies
        anomalies = scores > self.threshold
        
        return scores, anomalies


class AnomalyDetectionPipeline:
    """Pipeline for comparing multiple anomaly detection methods."""
    
    def __init__(self):
        """Initialize the anomaly detection pipeline."""
        self.detectors = {}
        self.results = {}
    
    def add_detector(self, name: str, detector) -> None:
        """
        Add an anomaly detector to the pipeline.
        
        Args:
            name: Name of the detector
            detector: Anomaly detector instance
        """
        self.detectors[name] = detector
    
    def fit_all(self, series: pd.Series) -> None:
        """
        Fit all detectors in the pipeline.
        
        Args:
            series: Time series data
        """
        for name, detector in self.detectors.items():
            logger.info(f"Fitting {name} detector...")
            try:
                detector.fit(series)
                logger.info(f"{name} detector fitted successfully")
            except Exception as e:
                logger.error(f"Error fitting {name} detector: {e}")
    
    def detect_all(self, series: pd.Series) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect anomalies using all detectors.
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with detection results from each detector
        """
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                scores, anomalies = detector.detect(series)
                results[name] = (scores, anomalies)
                logger.info(f"{name} detection completed")
            except Exception as e:
                logger.error(f"Error detecting anomalies with {name}: {e}")
        
        return results
    
    def plot_comparison(
        self,
        series: pd.Series,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "Anomaly Detection Comparison"
    ) -> None:
        """
        Plot comparison of different anomaly detection methods.
        
        Args:
            series: Time series data
            results: Dictionary with detection results
            title: Plot title
        """
        n_detectors = len(results)
        fig, axes = plt.subplots(n_detectors + 1, 1, figsize=(15, 4 * (n_detectors + 1)))
        
        # Plot original series
        axes[0].plot(series.index, series.values, 'b-', linewidth=1)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Plot detection results
        for i, (name, (scores, anomalies)) in enumerate(results.items()):
            ax = axes[i + 1]
            
            # Plot normal points
            normal_mask = ~anomalies
            ax.plot(series.index[normal_mask], series.values[normal_mask], 
                   'b-', alpha=0.7, linewidth=1)
            
            # Plot anomalies
            if np.any(anomalies):
                ax.scatter(series.index[anomalies], series.values[anomalies], 
                         color='red', s=30, label='Anomalies', zorder=5)
            
            ax.set_title(f'{name} Detection (Anomalies: {np.sum(anomalies)})')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            if i == len(results) - 1:
                ax.set_xlabel('Date')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def generate_report(
        self,
        series: pd.Series,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Generate a summary report of anomaly detection results.
        
        Args:
            series: Time series data
            results: Dictionary with detection results
            
        Returns:
            DataFrame with summary statistics
        """
        report_data = []
        
        for name, (scores, anomalies) in results.items():
            n_anomalies = np.sum(anomalies)
            anomaly_rate = n_anomalies / len(series) * 100
            
            report_data.append({
                'Detector': name,
                'Total_Points': len(series),
                'Anomalies_Detected': n_anomalies,
                'Anomaly_Rate_Percent': round(anomaly_rate, 2),
                'Mean_Score': round(np.mean(scores), 4),
                'Max_Score': round(np.max(scores), 4)
            })
        
        return pd.DataFrame(report_data)


def main():
    """Main function to demonstrate anomaly detection capabilities."""
    logger.info("Starting Anomaly Detection Demo")
    
    # Generate sample data with anomalies
    from core import TimeSeriesGenerator
    generator = TimeSeriesGenerator()
    series = generator.generate_univariate_series(n_points=500)
    
    # Add some artificial anomalies
    np.random.seed(42)
    anomaly_indices = np.random.choice(len(series), size=20, replace=False)
    series.iloc[anomaly_indices] += np.random.normal(0, 3, 20)
    
    # Initialize anomaly detection pipeline
    pipeline = AnomalyDetectionPipeline()
    
    # Add detectors
    pipeline.add_detector('Statistical (Z-Score)', StatisticalAnomalyDetector('zscore'))
    pipeline.add_detector('Statistical (IQR)', StatisticalAnomalyDetector('iqr'))
    pipeline.add_detector('Isolation Forest', IsolationForestDetector())
    pipeline.add_detector('Autoencoder', AutoencoderDetector(epochs=50))  # Reduced epochs for demo
    
    # Fit all detectors
    pipeline.fit_all(series)
    
    # Detect anomalies
    results = pipeline.detect_all(series)
    
    # Generate report
    report = pipeline.generate_report(series, results)
    print("\nAnomaly Detection Report:")
    print(report.to_string(index=False))
    
    # Plot comparison
    pipeline.plot_comparison(series, results)
    
    logger.info("Anomaly Detection Demo completed")


if __name__ == "__main__":
    main()

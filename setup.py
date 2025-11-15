#!/usr/bin/env python3
"""
Setup script for Time Series Analysis Project

This script helps set up the project environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True


def create_directories():
    """Create necessary directories."""
    directories = ['data', 'models', 'logs', 'notebooks']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory '{directory}' created/verified")
    
    return True


def install_dependencies():
    """Install project dependencies."""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    )


def run_tests():
    """Run unit tests to verify installation."""
    if not os.path.exists('tests/test_time_series_analysis.py'):
        print("⚠️  Test file not found, skipping tests")
        return True
    
    return run_command(
        f"{sys.executable} -m pytest tests/ -v",
        "Running unit tests"
    )


def create_sample_data():
    """Create sample data for demonstration."""
    try:
        from src.core import TimeSeriesGenerator
        
        generator = TimeSeriesGenerator(random_state=42)
        
        # Generate sample multivariate data
        multivariate_data = generator.generate_multivariate_series(n_points=1000)
        multivariate_data.to_csv('data/sample_multivariate.csv', index=False)
        
        # Generate sample univariate data
        univariate_series = generator.generate_univariate_series(n_points=1000)
        univariate_series.to_csv('data/sample_univariate.csv')
        
        print("✅ Sample data created in data/ directory")
        return True
    
    except ImportError as e:
        print(f"⚠️  Could not create sample data: {e}")
        return True  # Not critical for setup


def main():
    """Main setup function."""
    print("🚀 Setting up Time Series Analysis Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        print("You may need to install them manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but setup can continue")
    
    # Create sample data
    create_sample_data()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the Streamlit dashboard:")
    print("   streamlit run src/streamlit_app.py")
    print("\n2. Try the command line interface:")
    print("   python cli.py comprehensive --points 1000")
    print("\n3. Explore the Jupyter notebook:")
    print("   jupyter notebook notebooks/time_series_analysis_demo.ipynb")
    print("\n4. Run tests:")
    print("   python -m pytest tests/ -v")


if __name__ == "__main__":
    main()

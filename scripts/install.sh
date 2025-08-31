#!/bin/bash

# Universal Consciousness Interface - Installation Script

# Exit on any error
set -e

# Check if we're on a supported system
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "This script is intended for Unix-like systems. Please use pip install -r requirements.txt on Windows."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Installing Universal Consciousness Interface dependencies..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies if requested
if [[ "$1" == "--dev" ]]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

echo "Installation completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
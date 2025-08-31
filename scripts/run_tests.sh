#!/bin/bash

# Universal Consciousness Interface - Test Runner Script

# Exit on any error
set -e

# Check if we're in the right directory
if [[ ! -f "setup.py" ]]; then
    echo "Please run this script from the project root directory."
    exit 1
fi

# Activate virtual environment if it exists
if [[ -d "venv" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

echo "Tests completed successfully!"
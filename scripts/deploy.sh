#!/bin/bash

# Universal Consciousness Interface - Deployment Script

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

# Build the package
echo "Building package..."
python setup.py sdist bdist_wheel

# Upload to PyPI (requires twine and credentials)
echo "Uploading to PyPI..."
twine upload dist/*

echo "Deployment completed successfully!"
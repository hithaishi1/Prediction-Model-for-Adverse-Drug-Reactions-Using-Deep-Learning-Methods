#!/bin/bash

# Setup script for ADR Prediction project
# This script creates the necessary directory structure

echo "Setting up ADR Prediction project structure..."

# Get the base directory (where this script is run from)
BASE_DIR=$(pwd)

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p results
mkdir -p notebooks

# Check if files exist
echo ""
echo "Checking required files..."

if [ -f "src/models.py" ]; then
    echo "✓ src/models.py found"
else
    echo "✗ src/models.py NOT FOUND - Please create this file"
fi

if [ -f "src/train.py" ]; then
    echo "✓ src/train.py found"
else
    echo "✗ src/train.py NOT FOUND - Please create this file"
fi

if [ -f "src/evaluate.py" ]; then
    echo "✓ src/evaluate.py found"
else
    echo "✗ src/evaluate.py NOT FOUND - Please create this file"
fi

if [ -f "src/preprocessing.py" ]; then
    echo "✓ src/preprocessing.py found"
else
    echo "✗ src/preprocessing.py NOT FOUND - Please create this file"
fi

if [ -f "notebooks/03_baseline_models.ipynb" ]; then
    echo "✓ notebooks/03_baseline_models.ipynb found"
else
    echo "✗ notebooks/03_baseline_models.ipynb NOT FOUND - Please create this file"
fi

echo ""
echo "Checking processed data..."
if [ -d "processed_data" ] && [ -f "processed_data/X_train.csv" ]; then
    echo "✓ Processed data found"
else
    echo "✗ Processed data NOT FOUND"
    echo "  Run: python src/preprocessing.py"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. If preprocessing not done: python src/preprocessing.py"
echo "3. Train models: python src/train.py"
echo "4. Evaluate models: python src/evaluate.py"
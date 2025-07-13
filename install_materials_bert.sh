#!/bin/bash

echo "Installing MaterialsBERT Service Dependencies..."
echo

# Ensure we're in the correct directory
cd "$(dirname "$0")/backend"

# Upgrade pip and setuptools first
echo "Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

# Install core dependencies first
echo "Installing core dependencies..."
python3 -m pip install flask==3.0.0 requests==2.31.0 numpy==1.24.3 scikit-learn==1.3.0

# Install PyTorch with compatible version
echo "Installing PyTorch..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install transformers and sentence-transformers
echo "Installing NLP libraries..."
python3 -m pip install transformers==4.35.0 sentence-transformers==2.2.2

# Install additional utilities
echo "Installing additional utilities..."
python3 -m pip install pandas==2.0.3 scipy==1.11.1 python-dotenv==1.0.0

echo
echo "Installation complete!"
echo
echo "You can now start the MaterialsBERT service with:"
echo "  python3 start_materials_bert_simple.py"
echo 
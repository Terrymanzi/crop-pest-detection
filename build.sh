#!/usr/bin/env bash
# Render build script

echo "========================================="
echo "Starting Render Build"
echo "========================================="

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install gdown for model downloads
echo "Installing gdown for model downloads..."
pip install gdown

# Download model files
echo "Downloading model files..."
python download_models.py

echo "========================================="
echo "Build Complete!"
echo "========================================="

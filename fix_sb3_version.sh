#!/bin/bash

# Script to fix stable_baselines3 version and install comet_ml inside Docker container
# Run this inside the Docker container after it's started
#
# Current situation:
# - Dockerfile line 49 installs: stable_baselines3==2.0.1
# - Dockerfile line 79 adds to .bashrc: stable_baselines3==1.6.0
# This script will override both and install a working version

echo "Fixing stable_baselines3 version and installing comet_ml..."

# Remove the .bashrc line that reinstalls 1.6.0
sed -i '/pip install stable-baselines3==1.6.0/d' ~/.bashrc 2>/dev/null || true

# Uninstall current version (could be 2.0.1 from build or 1.6.0 from .bashrc)
pip uninstall -y stable_baselines3 2>/dev/null || true

# Try different versions to find one that works
# Version 1.6.0 is mentioned in dockerfile comments as working
echo "Installing stable_baselines3==1.6.0..."
pip install stable_baselines3==1.6.0 --no-deps

# Install comet_ml if not already installed
echo "Installing comet_ml..."
pip install comet_ml

echo "stable_baselines3 version fixed to 1.6.0"
echo "Verifying installations..."
python -c "import stable_baselines3; print(f'stable_baselines3 version: {stable_baselines3.__version__}')" || echo "Warning: Could not verify stable_baselines3 version"
python -c "import comet_ml; print(f'comet_ml installed successfully')" || echo "Warning: Could not verify comet_ml installation"

echo "Done!"

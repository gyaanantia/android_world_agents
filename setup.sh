#!/bin/bash

# Setup script for Android World Agents environment

set -e  # Exit on any error

echo "Setting up Android World Agents environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Mambaforge or Miniconda first."
    echo "Visit: https://github.com/conda-forge/miniforge#mambaforge"
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Function to create __init__.py files
create_init_files() {
    local base_dir="$1"
    echo "Creating necessary __init__.py files..."
    
    # List of directories that need __init__.py files
    local dirs=(
        "android_world"
        "android_world/agents"
        "android_world/utils"
        "android_world/env"
        "android_world/env/setup_device"
        "android_world/task_evals"
        "android_world/task_evals/utils"
        "android_world/task_evals/single"
        "android_world/task_evals/composite"
        "android_world/task_evals/information_retrieval"
        "android_world/task_evals/common_validators"
        "android_world/task_evals/miniwob"
        "android_world/task_evals/robustness_study"
    )
    
    for dir in "${dirs[@]}"; do
        local init_file="$base_dir/$dir/__init__.py"
        if [ ! -f "$init_file" ]; then
            echo "  Creating $init_file"
            touch "$init_file"
        fi
    done
}

# Create conda environment
echo "Creating conda environment..."
conda create -n android_world python=3.11.8 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate android_world

# Clone and install android_world from GitHub
echo "Cloning Android World from GitHub..."
if [ -d "android_world" ]; then
    echo "  android_world directory already exists, pulling latest changes..."
    cd android_world
    git pull
    cd ..
else
    git clone https://github.com/google-research/android_world.git
fi

# Create necessary __init__.py files before installation
create_init_files "android_world"

# Install android_world in editable mode
echo "Installing Android World in editable mode..."
pip install -e android_world/

# Install this project and its dependencies
echo "Installing Android World Agents package..."
pip install -e .

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "The environment includes:"
echo "  - Android World (editable install from GitHub)"
echo "  - Android World Agents (enhanced T3A with prompting capabilities)"
echo "  - All necessary __init__.py files"
echo "  - All required dependencies"
echo ""
echo "To activate the environment, run:"
echo "  conda activate android_world"
echo ""
echo "To test the installation, run:"
echo "  python -c \"import android_world; print('Android World installed successfully')\""
echo "  python -c \"import android_world.agents; print('Android World agents module available')\""
echo "  python -c \"from src.agent import EnhancedT3A; print('Enhanced T3A agent available')\""

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

# Clone Text2Grad from GitHub
echo "Cloning Text2Grad from GitHub..."
if [ -d "Text2Grad" ]; then
    echo "  Text2Grad directory already exists, pulling latest changes..."
    cd Text2Grad
    git pull
    cd ..
else
    git clone https://github.com/EdWangLoDaSc/Text2Grad-Reinforcement-Learning-from-Natural-Language-Feedback.git Text2Grad
fi

# Install this project and its dependencies
echo "Installing Android World Agents package..."
pip install -e .

# Install Text2Grad dependencies (macOS compatible)
echo "Installing Text2Grad dependencies..."

# Install PyTorch with CPU/MPS support for macOS (no CUDA)
echo "  Installing PyTorch with macOS support (CPU/MPS)..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# Fix NumPy compatibility with PyTorch 2.2.2
echo "  Ensuring NumPy compatibility..."
pip install "numpy<2"

# Install Text2Grad-specific packages
echo "  Installing Text2Grad ML packages..."
pip install trl==0.10.1
pip install scikit-learn pandas
pip install peft --no-dependencies
pip install transformers>=4.35.0
pip install accelerate
pip install bitsandbytes
pip install rouge rouge_score
pip install bert_score

# Install evaluation packages
echo "  Installing evaluation packages..."
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"

# Install additional utility packages
echo "  Installing utility packages..."
pip install wandb
pip install huggingface_hub

# Configure git credentials (optional)
echo "Configuring git credentials..."
git config --global credential.helper store

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "The environment includes:"
echo "  - Android World (editable install from GitHub)"
echo "  - Text2Grad (cloned from GitHub)"
echo "  - Android World Agents (enhanced T3A with prompting capabilities)"
echo "  - Text2Grad dependencies (macOS compatible)"
echo "  - PyTorch with CPU/MPS support"
echo "  - All necessary __init__.py files"
echo "  - All required dependencies"
echo ""
echo "To activate the environment, run:"
echo "  conda activate android_world"
echo ""
echo "Optional setup commands:"
echo "  # Set up Weights & Biases (interactive):"
echo "  wandb login"
echo ""
echo "  # Set up Hugging Face access:"
echo "  python -c \"from huggingface_hub import login; login('YOUR_HF_TOKEN')\""
echo ""
echo "To test the installation, run:"
echo "  python -c \"import android_world; print('Android World installed successfully')\""
echo "  python -c \"import android_world.agents; print('Android World agents module available')\""
echo "  python -c \"from src.agent import EnhancedT3A; print('Enhanced T3A agent available')\""
echo "  python -c \"import torch; print(f'PyTorch {torch.__version__} installed with device: {torch.device(\\\"mps\\\" if torch.backends.mps.is_available() else \\\"cpu\\\")}')\""
echo "  python verify_text2grad.py  # Test Text2Grad integration"

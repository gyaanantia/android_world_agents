#!/bin/bash

# Post-install script to fix missing __init__.py files in android_world

set -e

echo "Fixing missing __init__.py files in android_world..."

# Find the android_world installation directory
ANDROID_WORLD_DIR=$(python -c "import android_world; import os; print(os.path.dirname(android_world.__file__))" 2>/dev/null || echo "")

if [ -z "$ANDROID_WORLD_DIR" ]; then
    echo "Error: android_world not found. Please install it first."
    exit 1
fi

echo "Found android_world at: $ANDROID_WORLD_DIR"

# List of directories that need __init__.py files (relative to android_world root)
dirs=(
    "agents"
    "utils" 
    "env"
    "env/setup_device"
    "task_evals"
    "task_evals/utils"
    "task_evals/single"
    "task_evals/composite"
    "task_evals/information_retrieval"
    "task_evals/common_validators"
    "task_evals/miniwob"
    "task_evals/robustness_study"
)

created_count=0

for dir in "${dirs[@]}"; do
    full_dir="$ANDROID_WORLD_DIR/$dir"
    init_file="$full_dir/__init__.py"
    
    if [ -d "$full_dir" ] && [ ! -f "$init_file" ]; then
        echo "  Creating $init_file"
        touch "$init_file"
        ((created_count++))
    fi
done

echo ""
if [ $created_count -eq 0 ]; then
    echo "✅ All __init__.py files already exist."
else
    echo "✅ Created $created_count missing __init__.py files."
fi

echo ""
echo "Testing imports..."
python -c "import android_world; print('✅ android_world imports successfully')"
python -c "import android_world.agents; print('✅ android_world.agents imports successfully')" || echo "❌ android_world.agents import failed"

echo ""
echo "Done!"

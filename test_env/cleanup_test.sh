#!/bin/bash

# Cleanup script for the test environment

echo "Cleaning up android_world_test environment..."

# Deactivate if currently active
if [[ "$CONDA_DEFAULT_ENV" == "android_world_test" ]]; then
    echo "Deactivating android_world_test environment..."
    conda deactivate
fi

# Remove conda environment
echo "Removing conda environment..."
conda env remove -n android_world_test -y

# Remove cloned repository
if [ -d "android_world_test_repo" ]; then
    echo "Removing cloned repository..."
    rm -rf android_world_test_repo
fi

echo "✅ Cleanup complete!"
echo ""
echo "The following files were removed:"
echo "  - Conda environment: android_world_test"
echo "  - Directory: android_world_test_repo"
echo ""
echo "Your original android_world environment was not affected."

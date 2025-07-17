# Test Scripts

This directory contains scripts to safely test the environment setup without affecting your existing `android_world` environment.

## Scripts

- **`setup_test.sh`** - Creates a test environment (`android_world_test`) with the full setup
- **`test_installation.sh`** - Runs comprehensive tests to verify the installation works
- **`cleanup_test.sh`** - Removes the test environment and cloned repository

## Usage

```bash
# Run the test setup
./setup_test.sh

# Activate test environment and run tests
conda activate android_world_test
./test_installation.sh

# Clean up when done
./cleanup_test.sh
```

The test environment uses:
- Environment name: `android_world_test` 
- Repository clone: `android_world_test_repo`

Your original `android_world` environment remains untouched.

# Android LLM Agent

## Environment Setup

This project requires Android World installed as an editable package from GitHub, with additional `__init__.py` files to make the agents module importable.

### Option 1: Automated Setup (Recommended)

Run the setup script that handles everything automatically:

```bash
./setup.sh
```

This will:
- Create a new conda environment with Python 3.11.8
- Clone Android World from GitHub
- Install it in editable mode
- Create all necessary `__init__.py` files
- Install all required dependencies

### Option 2: Manual Setup

1. Create conda environment:
   ```bash
   conda create -n android_world python=3.11.8
   conda activate android_world
   ```

2. Clone and install Android World:
   ```bash
   git clone https://github.com/google-research/android_world.git
   ./fix_init_files.sh  # Creates missing __init__.py files
   pip install -e android_world/
   ```

3. Install additional requirements:
   ```bash
   pip install -r requirements-github.txt
   ```

### Troubleshooting

If you get import errors for `android_world.agents`, run:
```bash
./fix_init_files.sh
```

### Verification

Test that everything works:
```bash
conda activate android_world
python -c "import android_world; print('Android World installed successfully')"
python -c "import android_world.agents; print('Agents module available')"
```

### Testing

To test the setup process safely without affecting an existing environment:
```bash
cd test/
./setup_test.sh      # Creates android_world_test environment
./test_installation.sh   # Tests the installation
./cleanup_test.sh    # Cleans up test environment
```

## Project Structure

- `src/` - Custom agent implementations
- `prompts/` - Prompt templates and examples
- `tests/` - Test files
- `test/` - Setup testing scripts
- `log_episode.py` - Script to run and log episodes

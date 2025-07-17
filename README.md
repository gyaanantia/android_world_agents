# AndroidWorld Enhanced T3A Agent Evaluation Framework

A comprehensive evaluation framework for AndroidWorld that extends the Text-to-Action (T3A) agent with enhanced prompting capabilities including few-shot learning and self-reflection.

## Features

- **Enhanced T3A Agent**: Extends AndroidWorld's base T3A agent with improved prompting strategies
- **Multiple Prompting Variants**: 
  - Base: Original T3A prompting
  - Few-shot: Learning from examples
  - Reflective: Self-reflection on failures
- **Comprehensive Evaluation**: Detailed episode recording and analysis
- **Modular Design**: Easy to extend with new prompting strategies
- **Results Tracking**: Automatic saving of evaluation results and screenshots

## Installation

### Prerequisites

1. **Set up the Android Emulator**
   - Download Android Studio from [here](https://developer.android.com/studio)
   - Create an Android Virtual Device (AVD) with **Pixel 6** hardware and **Tiramisu, API Level 33** system image
   - Name your AVD as **AndroidWorldAvd** (or remember your chosen name)

2. **Launch the Android Emulator**
   ```bash
   # Launch from command line with required flags
   EMULATOR_NAME=AndroidWorldAvd
   ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
   ```

3. **Install ffmpeg** (if not already installed):
   ```bash
   # For macOS
   brew install ffmpeg

   # For Ubuntu
   sudo apt update && sudo apt install ffmpeg
   ```

### Option 1: Automated Setup (Recommended)

Run the setup script that handles everything automatically:

```bash
./setup.sh
```

This will:
- Create a new conda environment with Python 3.11.8
- Clone AndroidWorld from GitHub
- Install it in editable mode
- Create all necessary `__init__.py` files for proper module imports
- Install all required dependencies

### Option 2: Manual Setup

1. **Create conda environment**:
   ```bash
   conda create -n android_world python=3.11.8
   conda activate android_world
   ```

2. **Clone and install AndroidWorld**:
   ```bash
   git clone https://github.com/google-research/android_world.git
   ./fix_init_files.sh  # Creates missing __init__.py files
   pip install -e android_world/
   ```

3. **Install additional requirements**:
   ```bash
   pip install -r requirements.txt
   ```

### LLM API Setup

For LLM evaluation, set up your API key:

```bash
# For OpenAI models (GPT-3.5, GPT-4)
export OPENAI_API_KEY="your-api-key"

# For Anthropic models (Claude)
export ANTHROPIC_API_KEY="your-api-key"
```

### Verify Installation

After setup, verify everything works:

```bash
conda activate android_world
python verify_framework.py
```

## Quick Start

### Basic Usage

```bash
# Activate environment
conda activate android_world

# Start Android emulator (in separate terminal)
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554

# Run evaluation
python run_evaluation.py --task "single_task_name" --agent_type "base"
```

### Advanced Usage

```bash
python run_evaluation.py \
  --task "single_task_name" \
  --agent_type "few_shot" \
  --model_name "gpt-4" \
  --max_steps 50 \
  --num_episodes 3 \
  --results_dir "my_results" \
  --save_screenshots \
  --log_level "DEBUG"
  --results_dir "my_results" \
  --save_screenshots \
  --log_level "DEBUG"
```

### Available Options

- `--task`: Task name to evaluate (required)
- `--agent_type`: Prompting variant (`base`, `few_shot`, `reflective`)
- `--model_name`: OpenAI model to use (default: `gpt-4`)
- `--max_steps`: Maximum steps per episode (default: 30)
- `--num_episodes`: Number of episodes to run (default: 1)
- `--results_dir`: Directory to save results (default: `results`)
- `--device_id`: Android device ID (auto-detected if not provided)
- `--adb_path`: Path to ADB directory (auto-detected if not provided)
- `--timeout`: Timeout per episode in seconds (default: 300)
- `--save_screenshots`: Save screenshots during evaluation
- `--log_level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)


## Project Structure


```
android_world_agents/
├── src/
│   ├── agent.py           # Enhanced T3A agent with prompting variants
│   ├── evaluator.py       # Episode evaluation and result recording
│   ├── main.py           # Main entry point
│   ├── prompts.py        # Prompt management utilities
│   ├── run_episode.py    # Episode execution logic
│   └── utils.py          # AndroidWorld integration utilities
├── prompts/
│   ├── base_prompt.txt   # Base prompting template
│   ├── few_shot_v1.md    # Few-shot prompting examples
│   └── reflective_v1.md  # Self-reflection prompting template
├── results/              # Evaluation results (auto-created)
├── requirements.txt      # Python dependencies
├── run_evaluation.py     # Main launcher script
└── README.md            # This file
```

## Agent Types

### Base Agent (`base`)
- Uses AndroidWorld's original T3A prompting
- Good baseline for comparison
- Minimal prompt engineering

### Few-Shot Agent (`few_shot`)
- Includes example task completions in prompts
- Learns from successful interaction patterns
- Better performance on similar tasks

### Reflective Agent (`reflective`)
- Self-reflects on failures and adjusts strategy
- Maintains context of previous attempts
- Improved error recovery

## Prompting System

The framework uses a modular prompting system:

1. **Base Prompts** (`prompts/base_prompt.txt`): Core instruction templates
2. **Few-Shot Examples** (`prompts/few_shot_v1.txt`): Example interactions
3. **Reflection Templates** (`prompts/reflective_v1.txt`): Self-reflection patterns

### Custom Prompts

You can customize prompts by editing files in the `prompts/` directory or creating new ones:

```python
# Example: Loading and using prompts
from src.prompts import load_prompt, get_prompt_template, format_prompt

# Load a specific prompt file
custom_prompt = load_prompt("my_custom_prompt.txt")

# Get a prompt template for an agent type
base_prompt = get_prompt_template("base")

# Format a prompt with variables
formatted = format_prompt(base_prompt, goal="Open Settings", ui_elements="Button: Settings")
```

All prompt templates support variable substitution using `{variable_name}` syntax:
- `{goal}`: The current task goal
- `{ui_elements}`: Description of current UI elements
- `{previous_actions}`: Previous actions taken (reflective agent)
- `{reflection}`: Reflection context (reflective agent)

## Results and Analysis

### Result Files

Each evaluation generates:
- `episode_results.json`: Detailed step-by-step results
- `screenshots/`: Screenshots at each step (if enabled)
- `logs/`: Detailed execution logs

### Result Structure

```json
{
  "task_name": "example_task",
  "agent_type": "few_shot",
  "success": true,
  "steps_taken": 12,
  "total_time": 45.2,
  "steps": [
    {
      "step_number": 1,
      "action": "CLICK(button)",
      "success": true,
      "timestamp": "2024-01-15T10:30:00Z",
      "state": {...}
    }
  ]
}
```

## Extending the Framework

### Adding New Agent Types

1. **Create prompting template**:
   ```bash
   touch prompts/my_agent_v1.md
   ```

2. **Extend EnhancedT3A class**:
   ```python
   def _enhance_with_my_agent(self, base_prompt: str, state: dict) -> str:
       # Your custom prompting logic
       return enhanced_prompt
   ```

3. **Register agent type**:
   ```python
   # In src/agent.py
   elif self.agent_type == "my_agent":
       return self._enhance_with_my_agent(base_prompt, state)
   ```

### Adding New Evaluation Metrics

Extend the `EpisodeEvaluator` class in `src/evaluator.py`:

```python
def add_custom_metric(self, metric_name: str, value: float):
    """Add custom evaluation metric."""
    self.custom_metrics[metric_name] = value
```

## Troubleshooting

### Common Issues

1. **AndroidWorld not found**:
   ```bash
   # Run the setup script
   ./setup.sh
   conda activate android_world
   ```

2. **Missing `__init__.py` files**:
   ```bash
   # Run the fix script
   ./fix_init_files.sh
   ```

3. **ADB not found**:
   - Install Android SDK via Android Studio
   - Add ADB to PATH or use `--adb_path` flag

4. **Device not detected**:
   ```bash
   # Check connected devices
   adb devices
   
   # Start emulator with required flags
   ~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
   ```

5. **OpenAI API errors**:
   - Set `OPENAI_API_KEY` environment variable
   - Check API quota and rate limits

6. **Conda environment issues**:
   ```bash
   # Recreate environment
   conda remove -n android_world --all
   ./setup.sh
   ```

### Debug Mode

Enable detailed logging:
```bash
python run_evaluation.py --task "my_task" --log_level "DEBUG"
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines. Use:
```bash
black src/
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project extends AndroidWorld and follows the same licensing terms.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{android_world_enhanced_t3a,
  title={AndroidWorld Enhanced T3A Agent Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/android_world_agents}
}
```

## Support

For issues and questions:
- Run `python verify_framework.py` to check your setup
- Check the troubleshooting section above  
- Review AndroidWorld documentation at [google-research/android_world](https://github.com/google-research/android_world)
- Open an issue on this repository's GitHub page

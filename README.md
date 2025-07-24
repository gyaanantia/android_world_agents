# AndroidWorld Enhanced T3A Agent Evaluation Framework

A comprehensive evaluation framework for AndroidWorld that extends the Text-to-Action (T3A) agent with enhanced prompting capabilities including few-shot learning, self-reflection, OpenAI function calling, and **Gemini 2.5 visual prompt generation with TextGrad optimization**.

## Features

- **Enhanced T3A Agent**: Extends AndroidWorld's base T3A agent with improved prompting strategies
- **Multiple Prompting Variants**: 
  - Base: Original T3A prompting
  - Few-shot: Learning from examples
  - Reflective: Self-reflection on failures
- **🆕 Gemini 2.5 Visual Prompting**: AI-generated contextual prompts based on real-time UI analysis
- **🆕 TextGrad Optimization**: Gradient-based optimization to improve Gemini's visual analysis quality for better agent understanding
- **🆕 Emulator Snapshot System**: Save and restore Android emulator states for debugging and rollback functionality
- **Function Calling Support**: Optional OpenAI function calling for structured output
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
- Install the enhanced agents package

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

3. **Install the enhanced agents package**:
   ```bash
   pip install -e .
   ```

   This installs the package in editable mode with all dependencies including:
   - OpenAI API client for LLM integration
   - Image processing libraries (PIL, OpenCV)
   - Data analysis tools (NumPy, Pandas)
   - Testing framework (pytest)

### LLM API Setup

For LLM evaluation, set up your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

### Gemini 2.5 Visual Prompting Setup (Optional)

For enhanced visual prompting with Gemini 2.5 Flash:

1. **Get Google API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key

2. **Set environment variable**:
   ```bash
   export GOOGLE_API_KEY="your-google-api-key"
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

# Run evaluation with basic agent
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant "base"

# Run evaluation with Gemini 2.5 visual prompting (enhances base prompting)
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant "base" --gemini

# Run evaluation with Gemini and TextGrad optimization (best performance)
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant "base" --gemini --textgrad

# Run evaluation with function calling
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant "base" --function-calling

# Combine all features: few-shot prompting + Gemini + TextGrad + function calling
python run_evaluation.py --task SystemBrightnessMax --prompt-variant few-shot --gemini --textgrad --function-calling
```

### Gemini 2.5 Visual Prompting with TextGrad Optimization

The `--gemini` flag enables visual UI analysis using Google's Gemini 2.5 Flash model. The `--textgrad` flag further optimizes Gemini's analysis using gradient-based optimization for improved agent understanding. Both flags work with any `--prompt-variant` to provide enhanced versions of your chosen prompting strategy.

```bash
# Standard evaluation with different prompt variants
python run_evaluation.py --task SystemBrightnessMax --prompt-variant base --num-episodes 5
python run_evaluation.py --task SystemBrightnessMax --prompt-variant few-shot --num-episodes 5
python run_evaluation.py --task SystemBrightnessMax --prompt-variant reflective --num-episodes 5

# Enhanced with Gemini visual analysis (works with any prompt variant)
python run_evaluation.py --task SystemBrightnessMax --prompt-variant base --gemini --num-episodes 5
python run_evaluation.py --task SystemBrightnessMax --prompt-variant few-shot --gemini --num-episodes 5

# Enhanced with Gemini and TextGrad optimization for best performance
python run_evaluation.py --task SystemBrightnessMax --prompt-variant base --gemini --textgrad --num-episodes 5
python run_evaluation.py --task SystemBrightnessMax --prompt-variant reflective --gemini --textgrad --num-episodes 5
```

**How it works:**
1. **Screenshot Capture**: Android emulator screenshot captured before each step
2. **Visual Analysis**: Gemini analyzes the UI and task goal  
3. **TextGrad Optimization**: If enabled, optimizes Gemini's analysis for better agent understanding
4. **Enhanced Prompting**: Combines Gemini's visual insights with your chosen prompt variant (base/few-shot/reflective)
5. **Action Execution**: Agent returns action based on optimized Gemini-enhanced prompt
6. **Results Tracking**: Gemini and TextGrad usage tracked in evaluation results

**Key Benefits:**
- **`--gemini`** adds visual UI understanding to any prompt variant
- **`--textgrad`** automatically enables `--gemini` and optimizes the visual analysis
- Each prompt variant has a corresponding Gemini-enhanced version that preserves the variant's characteristics
- You can combine any prompt variant with Gemini and TextGrad for maximum effectiveness
```

### Function Calling Demo

To see the difference between function calling and regular text parsing:

```bash
python demo_function_calling.py
```

This script demonstrates:
- How function calling structures LLM output
- Comparison with traditional text parsing
- Action compatibility between both modes

### Advanced Usage

```bash
# Combine all features: few-shot prompting + Gemini visual analysis + TextGrad optimization + function calling
python run_evaluation.py \
  --task "single_task_name" \
  --prompt-variant "few-shot" \
  --gemini \
  --textgrad \
  --model-name "gpt-4o-mini" \
  --max-steps 50 \
  --num-episodes 3 \
  --results-dir "my_results" \
  --log-level "DEBUG" \
  --function-calling \
  --disable-memory
```
NOTE: for `o-series` models, you MUST use the `--function-calling` flag to enable structured output.

### Available Options

- `--task`: Task name to evaluate (random if not specified)
- `--prompt-variant`: Prompting variant (`base`, `few-shot`, `reflective`) - **Works with or without `--gemini`**
- `--gemini`: Enable Gemini 2.5 visual prompting (enhances any prompt variant) - **Can be combined with any `--prompt-variant`**
- `--textgrad`: Enable TextGrad optimization for improved Gemini analysis (automatically enables `--gemini`)
- `--model-name`: OpenAI model to use (default: `gpt-4o-mini`)
- `--max-steps`: Maximum steps per episode (default: 30)
- `--num-episodes`: Number of episodes to run (default: 1)
- `--results-dir`: Directory to save results (default: `results`)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, default: `INFO`)
- `--function-calling`: Enable OpenAI function calling for structured output
- `--disable-memory`: Disable memory (step history) in agent prompts


## Project Structure


```
android_world_agents/
├── src/
│   ├── agent.py              # Enhanced T3A agent with prompting variants
│   ├── evaluator.py          # Episode evaluation and result recording
│   ├── function_calling_llm.py # OpenAI function calling LLM wrapper
│   ├── gemini_prompting.py   # 🆕 Gemini 2.5 visual prompt generation
│   ├── gemini_enhanced_agent.py # 🆕 Gemini-enhanced T3A agent with seamless integration
│   ├── textgrad_opt.py       # 🆕 TextGrad optimization for improving Gemini analysis
│   ├── main.py               # Main entry point
│   ├── prompts.py            # Prompt management utilities
│   ├── run_episode.py        # Episode execution logic
│   ├── test_utils.py         # Testing utilities (moved to tests/)
│   └── utils.py              # AndroidWorld integration utilities
├── prompts/
│   ├── base_prompt.txt       # Base prompting template
│   ├── few_shot_v1.txt       # Few-shot prompting examples
│   ├── reflective_v1.txt     # Self-reflection prompting template
│   └── gemini_base_prompt.txt # 🆕 Gemini system prompt for UI analysis
├── tests/                    # Test suite
│   ├── test_agent.py         # Agent functionality tests
│   ├── test_androidworld_compatibility.py # AndroidWorld integration tests
│   ├── test_evaluator.py     # Evaluator tests
│   ├── test_function_calling.py # Function calling tests
│   ├── test_function_calling_mock.py # Function calling mock tests
│   ├── test_full_serialization.py # Full serialization tests
│   ├── test_imports.py       # Import verification tests
│   ├── test_json_serialization.py # JSON serialization tests
│   ├── test_prompt_actions.py # Prompt action tests
│   └── test_prompts.py       # Prompt system tests
├── android_world/            # AndroidWorld clone (added by setup script)
├── results/                  # Evaluation results 
├── test_env/                 # Test environment setup
├── advanced_replay.py        # Advanced replay features
├── batch_analysis.py         # Batch episode analysis
├── demo_function_calling.py  # Function calling demonstration
├── demo_gemini_prompting.py  # 🆕 Gemini prompting demonstration
├── demo_gemini_agent.py      # 🆕 Gemini-enhanced agent demonstration
├── evaluate_with_gemini.py   # 🆕 AndroidWorld evaluation with Gemini
├── evaluate_with_gemini_integration.py # 🆕 Full Gemini integration evaluation
├── test_gemini_integration.py # 🆕 Integration testing for Gemini functionality
├── replay_episode.py         # Episode replay system
├── fix_init_files.sh         # Script to create missing __init__.py files
├── pyproject.toml            # Package configuration
├── run_tests.py              # Test runner script
├── run_evaluation.py         # Main launcher script
├── setup.sh                  # Automated setup script
├── verify_framework.py       # Framework verification script
├── GEMINI_README.md          # 🆕 Complete Gemini documentation
└── README.md                 # This file
```

## Agent Types

### Base Agent (`base`)
- Uses AndroidWorld's original T3A prompting
- Good baseline for comparison
- Minimal prompt engineering

### Few-Shot Agent (`few-shot`)
- Includes example task completions in prompts
- Learns from successful interaction patterns
- Better performance on similar tasks

### Reflective Agent (`reflective`)
- Self-reflects on failures and adjusts strategy
- Maintains context of previous attempts
- Improved error recovery

### Function Calling Mode (must use for `o-series` models)
- Available for all agent types with `--function-calling` flag
- Uses OpenAI function calling for structured JSON output
- Improved action parsing and validation
- Compatible with all existing agent variants

## 🆕 Gemini 2.5 Visual Prompt Generation with TextGrad Optimization

This framework includes cutting-edge visual prompt generation using Google's Gemini 2.5 model, enhanced with TextGrad optimization for improved agent understanding. The system can enhance any existing prompt variant (base, few-shot, reflective) with visual UI analysis and gradient-based optimization.

**Key Architectural Design:**
- **Prompt Variants**: Each variant (base/few-shot/reflective) has both standard and Gemini-enhanced versions
- **Gemini Enhancement**: Adds visual UI analysis to any prompt variant without replacing the variant's characteristics  
- **TextGrad Optimization**: Optimizes Gemini's visual analysis for better agent understanding
- **Full Compatibility**: All features work together seamlessly

### Key Features

- **🔍 Visual UI Analysis**: Gemini 2.5 analyzes Android screenshots to understand current interface
- **🎯 Context-Aware Prompting**: Generates enhanced versions of any prompt variant using visual UI understanding
- **🚀 TextGrad Optimization**: Uses gradient-based optimization to improve Gemini's visual analysis for better agent understanding
- **🤖 Prompt Variant Compatibility**: Works with all existing prompt variants (base, few-shot, reflective) to create enhanced versions
- **🔄 Seamless Integration**: Drop-in enhancement for any agent configuration with graceful fallback
- **📊 Performance Comparison**: Compare standard vs Gemini-enhanced vs TextGrad-optimized versions of any prompt variant
- **💪 Full Compatibility**: Works with all existing features including function calling and memory

### Quick Start with Gemini and TextGrad

#### 1. Setup API Keys
```bash
export GOOGLE_API_KEY="your-google-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # For TextGrad optimization
```

#### 2. Install Dependencies
```bash
pip install google-generativeai pillow textgrad
```

#### 3. Run with Gemini and TextGrad Optimization
```bash
# Basic prompting with Gemini enhancement
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant base --gemini

# Few-shot prompting with Gemini visual analysis
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant few-shot --gemini

# Enhanced with TextGrad optimization for improved agent understanding
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant base --gemini --textgrad

# Multiple episodes with reflective prompting, Gemini, and TextGrad
python run_evaluation.py --task "SystemBrightnessMax" --prompt-variant reflective --gemini --textgrad --num-episodes 3
```

#### 3. Run Gemini Prompting Demo
```bash
# Interactive demo - enter goals and see generated prompts
python demo_gemini_prompting.py --mode interactive

# Test with sample data
python demo_gemini_prompting.py --mode sample

# Analyze a real screenshot
python demo_gemini_prompting.py --mode screenshot --screenshot path/to/android_screenshot.png --goal "Turn on Wi-Fi"
```

#### 4. Run Gemini-Enhanced Agent Demo
```bash
# Run agent demo with Gemini visual analysis
python demo_gemini_agent.py --task "Navigate to settings and adjust brightness"

# Interactive demo to customize settings
python demo_gemini_agent.py --interactive

# Compare with standard agent
python demo_gemini_agent.py --no-gemini --task "Navigate to settings"
```

#### 5. Evaluate with Gemini Integration
```bash
# Single task with Gemini-enhanced agent
python evaluate_with_gemini_integration.py --tasks "SystemBrightnessMax" --agent-model "gpt-4o-mini"

# Multiple tasks with performance comparison
python evaluate_with_gemini_integration.py --tasks "SystemBrightnessMax,FilesDeleteFile,TasksHighPriorityTasks" --max-steps 30

# Standard evaluation for comparison
python evaluate_with_gemini_integration.py --tasks "SystemBrightnessMax" --no-gemini
```

#### 6. Test Integration
```bash
# Verify everything works correctly
python test_gemini_integration.py
```

### How It Works

1. **Screenshot Analysis**: Gemini 2.5 receives the current Android UI screenshot
2. **Context Understanding**: The model analyzes visible elements, current app, and interface state  
3. **Task Integration**: Combines UI analysis with the specific task goal
4. **TextGrad Optimization**: If enabled, optimizes the visual analysis using gradient-based techniques for better agent understanding
5. **Prompt Generation**: Creates a tailored prompt that guides the agent to complete the task

### Gemini Integration Architecture

The Gemini integration is designed for seamless compatibility with existing AndroidWorld agents:

#### Core Components

1. **`GeminiPromptGenerator`** (`src/gemini_prompting.py`)
   - Handles communication with Gemini 2.5 Flash model
   - Processes screenshots and generates contextual prompts
   - Includes safety checks and error handling

2. **`GeminiEnhancedT3A`** (`src/gemini_enhanced_agent.py`)
   - Extends existing Enhanced T3A agent
   - Optionally uses Gemini for visual analysis
   - Integrates TextGrad optimization when enabled
   - Gracefully falls back to standard prompting if Gemini fails
   - Maintains full compatibility with all existing features

3. **`TextGradOptimizer`** (`src/textgrad_opt.py`)
   - Uses TextGrad to optimize Gemini's visual analysis
   - Preserves Gemini's structured bullet-point format
   - Improves clarity for better agent understanding
   - Keeps all UI elements while highlighting relevant ones

3. **Integration Functions**
   - `create_gemini_enhanced_agent()`: Factory for Gemini-enabled agents
   - `create_standard_agent_with_gemini_fallback()`: Automatic fallback logic
   - TextGrad optimization integration when enabled
   - Status tracking and usage statistics

#### Smart Fallback System

The integration includes multiple layers of fallback to ensure reliability:

```python
# Priority order for prompt generation:
1. Gemini 2.5 visual analysis with TextGrad optimization (if both enabled and working)
2. Gemini 2.5 visual analysis (if enabled and working)
3. Standard enhanced prompting (few-shot, reflective, etc.)
4. Base AndroidWorld prompting (ultimate fallback)
```
2. Standard enhanced prompting (few-shot, reflective, etc.)
3. Base AndroidWorld prompting (ultimate fallback)
```

#### Usage in Your Code

```python
from src.gemini_enhanced_agent import create_gemini_enhanced_agent

# Create agent with Gemini integration and TextGrad optimization
agent = create_gemini_enhanced_agent(
    env=env,
    model_name="gpt-4o-mini",
    prompt_variant="base",           # Works with any variant
    use_function_calling=True,       # Compatible with function calling
    use_gemini=True,                 # Enable Gemini visual analysis
    use_textgrad=True,               # Enable TextGrad optimization
    gemini_model="gemini-2.5-flash"
)

# Check if Gemini and TextGrad are working
status = agent.get_gemini_status()
print(f"Gemini enabled: {status['gemini_enabled']}")
print(f"TextGrad enabled: {status.get('textgrad_enabled', False)}")

# Use exactly like any other agent
result = agent.step("Turn on Wi-Fi")
```

#### Performance Monitoring

The enhanced agents track Gemini and TextGrad usage automatically:

```python
# After evaluation, check usage statistics
step_data = agent.history[-1]  # Last step
gemini_used = step_data.get('used_gemini', False)
textgrad_used = step_data.get('used_textgrad', False)
print(f"Gemini was used: {gemini_used}")
print(f"TextGrad optimization was used: {textgrad_used}")
```
1. **Task Integration**: Combines UI analysis with the specific task goal
2. **TextGrad Optimization**: If enabled, optimizes the visual analysis using gradient-based techniques for better agent understanding  
3. **Prompt Generation**: Creates a tailored prompt that guides the agent to complete the task

### Example Output

**Input**: Screenshot of Android Settings + Goal: "Turn on Wi-Fi" + Prompt Variant: "base"

**Standard Base Prompting**:
```
Your task is to turn on Wi-Fi. Look at the screen and determine the next action to take.
Available actions: [list of UI elements]
```

**Gemini-Enhanced Base Prompting**:
```
● Goal: Turn on Wi-Fi
● Current screen: Android Settings main screen showing network and connectivity options
● Options: ["Wi-Fi" (currently OFF), "Bluetooth", "Display", "Sound", "Network & internet"]
● Action: TAP on the "Wi-Fi" option to access Wi-Fi settings

Your task is to turn on Wi-Fi. Use the visual analysis above to determine the next action.
```

**TextGrad Optimized Gemini-Enhanced Base Prompting**:
```
● Goal: Turn on Wi-Fi
● Current screen: Android Settings main screen - you need to access Wi-Fi controls
● Options: ["Wi-Fi" (currently OFF - this is your target), "Bluetooth", "Display", "Sound", "Network & internet"]  
● Action: TAP on the "Wi-Fi" option to access Wi-Fi settings where you can toggle it ON

Your task is to turn on Wi-Fi. The visual analysis above shows the Wi-Fi option is currently off and provides clear next steps.
```

### Troubleshooting

**Common Issues:**

1. **API Keys Not Found**: Set both `GOOGLE_API_KEY` and `OPENAI_API_KEY` environment variables
2. **Connection Failed**: Check internet connection and verify both API keys are valid
3. **TextGrad Import Error**: Run `pip install textgrad` to install TextGrad optimization
4. **Empty Responses**: Verify image contains visible content, try different temperature
5. **Import Errors**: Run `pip install google-generativeai pillow textgrad`

**Debug Mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging for troubleshooting
generator = create_gemini_generator()
```

## Prompting System

The framework uses a modular prompting system:

1. **Base Prompts** (`prompts/base_prompt.txt`): Core instruction templates
2. **Few-Shot Examples** (`prompts/few_shot_v1.txt`): Example interactions
3. **Reflection Templates** (`prompts/reflective_v1.txt`): Self-reflection patterns

### Custom Prompts

You can customize prompts by editing files in the `prompts/` directory or creating new ones:

```python
# Example: Loading and using prompts
from prompts import load_prompt, get_prompt_template, format_prompt

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
- `{memory}`: Step history (when memory is enabled)
- `{reflection_context}`: Reflection context (reflective agent)

## Episode Replay System

The framework allows you to replay any evaluated episode step-by-step in the Android emulator, making it easy to analyze agent behavior and debug issues.

### Quick Start

```bash
# Activate environment and ensure emulator is running
conda activate android_world
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554

# Replay an episode in interactive mode (step-by-step with user input)
python replay_episode.py results/my_episode.json

# Replay non-interactively with 1 second delay between steps
python replay_episode.py results/my_episode.json --non-interactive --delay 1.0

# Replay with custom delay
python replay_episode.py results/my_episode.json --non-interactive --delay 0.5
```

### Features

- **Visual Playback**: Watch the exact actions the LLM took in the Android emulator
- **Step-by-Step Analysis**: See LLM reasoning and action parsing for each step
- **Interactive Mode**: Pause between steps to examine the UI state
- **Task Initialization**: Automatically extracts task name from JSON and sets up the correct environment
- **Action Validation**: Shows successful/failed action executions
- **Comprehensive Logging**: Detailed output showing action coordinates, UI interactions, and emulator state

### Command Line Options

```bash
python replay_episode.py <json_file> [options]

Positional Arguments:
  json_file             Path to the JSON result file to replay

Optional Arguments:
  --non-interactive     Run without waiting for user input between steps
  --delay SECONDS       Delay in seconds between steps (non-interactive only, default: 2.0)
  --log-level LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR, default: INFO)
```

### Interactive Mode Controls

When running in interactive mode, you can:
- **Press Enter**: Continue to next step
- **Type 'q'**: Quit replay immediately
- **Type 's'**: Switch to non-interactive mode for the rest of the replay

### What You'll See

The replay system shows:

1. **Episode Information**:
   ```
   📁 Loaded episode data from: results/my_episode.json
      Task: SystemCopyToClipboard
      Model: gpt-4o-mini
      Prompt Variant: few-shot
      Success: False
      Steps Taken: 30
   ```

2. **Task Setup**:
   ```
   🎯 Setting up task: SystemCopyToClipboard
      Goal: Copy the following text to the clipboard: Sara's Bakery
   
   📱 Environment set up successfully
      ADB Path: /Users/user/Library/Android/sdk/platform-tools/adb
   ```

3. **Step-by-Step Execution**:
   ```
   ═══ Step 5 ═══
   📝 LLM Response:
      💭 Reason: Long-press on the note text to enter selection mode and bring up copy options.
      🎬 Action: {"action_type": "long_press", "index": 0}
   🔧 Parsed Action: JSONAction(action_type='long_press', index=0)
   
   ⚡ Executing action: long_press
   ✅ Action executed successfully
   ```

4. **Final Results**:
   ```
   🎉 Replay completed!
      Original Success: False
      Original Steps: 30
      Current Task State: ❌ Not Successful
   ```

### Use Cases

- **Debugging Agent Behavior**: See exactly where and why the agent failed
- **UI Interaction Analysis**: Understand how the agent interprets and interacts with Android UI
- **Prompt Engineering**: Analyze LLM reasoning to improve prompts
- **Task Validation**: Verify that tasks are set up correctly
- **Training Data Generation**: Create visual examples of successful/failed interactions
- **Research Analysis**: Study agent behavior patterns across different tasks and models

### Finding Episodes to Replay

Episodes are saved in your results directory (specified by `--results-dir`) with descriptive filenames:

```bash
# List available episodes
ls -la results*/

# Example files:
# SystemCopyToClipboard_few-shot_gpt-4o-mini_trial1.json
# MarkorCreateNoteAndSms_reflective_gpt-4-turbo_trial2.json
# FilesDeleteFile_base_gpt-4o-mini_trial1.json

# Find episodes with more steps for interesting replays
find results*/ -name "*.json" -exec jq '.steps_taken' {} \; -print | grep -B1 -E "[1-9][0-9]+"
```

### Replay Logs

Replay sessions generate detailed logs in `<results_dir>/replay_logs/` showing:
- AndroidWorld environment setup
- ADB commands executed
- UI element interactions
- Action execution success/failure
- Task state changes

### Requirements

- Android emulator must be running on port 5554 with gRPC on 8554
- Same conda environment as evaluation (`android_world`)
- JSON episode files from previous evaluations

### Troubleshooting Replay

1. **Emulator not detected**:
   ```bash
   # Ensure emulator is running with correct flags
   ~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
   ```

2. **Task not found**:
   - The replay system automatically detects task names from JSON files
   - Ensure the JSON file contains a valid `task_name` field

3. **Action execution failures**:
   - Some actions may fail due to UI state differences
   - This is normal and the replay continues with the next action

4. **Environment issues**:
   ```bash
   # Verify AndroidWorld is properly installed
   python verify_framework.py
   ```

## Results and Analysis

### Result Files

Each evaluation generates:
- `episode_results.json`: Detailed step-by-step results
- `android_world_agents.log`: System logs and framework activity
- Episode results are saved in the specified `--results-dir` directory (default: `results/`)

### Result Structure

```json
{
  "task_name": "ExpenseAddMultiple",
  "model_name": "o3-mini",
  "prompt_variant": "base",
  "success": false,
  "steps_taken": 1,
  "evaluation_time": 12.63,
  "goal": "Add the following expenses...",
  "task_class": "ExpenseAddMultiple",
  "task_complexity": 6,
  "agent_claimed_done": true,
  "episode_terminated_early": false,
  "task_actually_successful": false,
  "max_steps": 30,
  "max_steps_allowed": 30,
  "total_ui_elements": 1234,
  "evaluation_timestamp": "2025-01-15T10:30:00Z",
  "actions": [
    "Reason: The home screen does not show...\nAction: {\"action_type\": \"status\", \"goal_status\": \"infeasible\"}"
  ],
  "responses": [
    "Reason: The home screen does not show...\nAction: {\"action_type\": \"status\", \"goal_status\": \"infeasible\"}"
  ],
  "prompts": ["System prompt and user input for each step"],
  "step_records": [
    {
      "step_num": 1,
      "state": {
        "pixels": "<screenshot_excluded>",
        "forest": "<protobuf: AndroidAccessibilityForest>",
        "ui_elements": [
          {
            "text": "Phone",
            "content_description": "Phone",
            "class_name": "android.widget.TextView",
            "bbox_pixels": {"x_min": 76, "x_max": 249, "y_min": 1873, "y_max": 2068},
            "is_clickable": true,
            "is_editable": false,
            "package_name": "com.google.android.apps.nexuslauncher"
          }
        ]
      },
      "action": "Reason: ...\nAction: {...}",
      "agent_data": {
        "before_screenshot": "<screenshot_excluded>",
        "after_screenshot": "<large_array>",
        "before_element_list": [...],
        "after_element_list": [...],
        "action_prompt": "Full prompt sent to LLM",
        "action_output": "LLM's formatted response",
        "action_raw_response": "Raw OpenAI API response object",
        "summary_prompt": "Prompt for step summary",
        "summary": "Brief summary of what happened in this step",
        "summary_raw_response": "Raw summary response"
      },
      "timestamp": 1752980063.680765
    }
  ],
  "step_timings": [1.23, 2.45, ...],
  "initial_state": {...},
  "final_state": {...}
}
```

## Extending the Framework

### Adding New Agent Types

1. **Create prompting template**:
   ```bash
   touch prompts/my_agent_v1.txt
   ```

2. **Extend EnhancedT3A class**:
   ```python
   def _enhance_with_my_agent(self, goal: str, ui_elements: str, memory: str) -> str:
       # Your custom prompting logic
       return format_prompt(
           self.system_prompt,
           goal=goal,
           ui_elements=ui_elements,
           memory=memory,
           custom_context="your custom context"
       )
   ```

3. **Register agent type**:
   ```python
   # In src/agent.py, in _get_action_prompt method
   elif self.prompt_variant == "my_agent":
       return self._enhance_with_my_agent(goal, ui_elements_description, formatted_memory)
   ```

### Adding Function Calling Support

Function calling is automatically available for all agent types. To create custom function schemas:

```python
# In src/function_calling_llm.py
def create_custom_schema():
    return {
        "name": "custom_action",
        "description": "Custom action description",
        "parameters": {
            "type": "object",
            "properties": {
                "custom_field": {
                    "type": "string",
                    "description": "Custom field description"
                }
            }
        }
    }
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
   - The framework auto-detects ADB location

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

7. **Function calling errors**:
   - Ensure you're using a compatible OpenAI model (gpt-4o-mini, gpt-4o, gpt-4-turbo)
   - Function calling requires OpenAI API key to be set
   - Check that `src/function_calling_llm.py` exists in your installation

### Debug Mode

Enable detailed logging:
```bash
python run_evaluation.py --task "my_task" --log-level "DEBUG"
```

## Development

### Running Tests

```bash
# Run all tests
python run_tests.py

# Or run individual tests
python tests/test_imports.py
python tests/test_prompts.py
python tests/test_agent.py
python tests/test_evaluator.py
python tests/test_function_calling.py
python tests/test_androidworld_compatibility.py
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
  author={Gyaan Antia}, 
  year={2025},
  url={https://github.com/gyaanantia/android_world_agents},
  note={Enhanced framework with function calling and advanced prompting strategies}
}
```

## Support

For issues and questions:
- Run `python verify_framework.py` to check your setup
- Check the troubleshooting section above  
- Review AndroidWorld documentation at [google-research/android_world](https://github.com/google-research/android_world)
- Review function calling implementation in `src/function_calling_llm.py`
- Open an issue on this repository's GitHub page

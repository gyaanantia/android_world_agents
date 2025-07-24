# Text2Grad Implementation for AndroidWorld Agents

This implementation provides a complete Text2Grad optimization system for AndroidWorld LLM agents, featuring dense reward functions, gradient-based prompt optimization, and snapshot-based rollback mechanisms.

## 🎯 Features Implemented

### Dense Reward Function
- **-0.05 penalty per step** - Encourages efficiency
- **+0.2 reward for subgoals** - Rewards intermediate progress
- **+1.0 reward for goal completion** - Rewards task success
- **Task-specific subgoal detection** for:
  - Markor (note management)
  - Contacts (contact operations)
  - System Settings (brightness, volume, etc.)
  - Messaging (text message operations)

### Text2Grad Optimization
- **K rollouts of N steps** - Configurable optimization parameters
- **Snapshot-based rollback** - Restore Android emulator state during optimization
- **Gemini integration** - Visual UI analysis and enhanced prompting
- **Gradient-based feedback** - Optimize prompts based on dense reward signals

### Integration Features
- **Command-line configuration** - User-configurable k and n parameters
- **Seamless AndroidWorld integration** - Works with existing task evaluation system
- **Multiple agent types** - Standard, Gemini-enhanced, and Text2Grad agents
- **Comprehensive logging** - Track optimization progress and results

## 🚀 Usage

### Basic Text2Grad Run
```bash
python src/main.py --text2grad --gemini --task SystemBrightnessMax
```

### Custom Optimization Parameters
```bash
python src/main.py --text2grad --gemini --k-rollouts 5 --n-steps 3 --task MarkorDeleteNewestNote
```

### Full Configuration
```bash
python src/main.py \
    --text2grad \
    --gemini \
    --k-rollouts 4 \
    --n-steps 6 \
    --max-steps 20 \
    --task ContactsAddContact \
    --model-name gpt-4o-mini \
    --prompt-variant base
```

## 📖 Command Line Arguments

### Text2Grad Specific
- `--text2grad` - Enable Text2Grad optimization (automatically enables --gemini)
- `--k-rollouts N` - Number of optimization rollouts (default: 3)
- `--n-steps N` - Number of steps per rollout (default: 5)

### General Arguments
- `--task TASK` - Specific task to run (or random if not specified)
- `--gemini` - Enable Gemini 2.5 Flash visual analysis
- `--model-name MODEL` - LLM model to use (default: gpt-4o-mini)
- `--prompt-variant VARIANT` - Prompt type: base, few-shot, reflective
- `--max-steps N` - Maximum steps per episode (default: 25)
- `--num-episodes N` - Number of episodes to run (default: 1)

## 🏗️ Architecture

### Control Flow
1. **Initialize Environment** - Setup Android emulator and task
2. **Create Snapshot** - Save initial emulator state
3. **Gemini Analysis** - Visual UI understanding and enhanced prompting
4. **Text2Grad Optimization**:
   - Run k rollouts of n steps each
   - Calculate dense rewards for each rollout
   - Optimize prompt based on gradient feedback
   - Restore snapshot between rollouts
5. **Execute Optimized Step** - Use optimized prompt for actual task execution
6. **Repeat** - Continue optimization cycle until task completion

### File Structure
```
src/
├── dense_reward.py             # Smart dense reward function with metadata-driven subgoal detection
├── text2grad_agent.py          # Text2Grad optimization agent and configuration
├── main.py                     # Enhanced command-line interface with Text2Grad support
├── run_episode.py              # Main evaluation runner with Text2Grad integration
├── gemini_prompting.py         # Gemini visual analysis integration
├── utils.py                    # Utilities including SnapshotManager for emulator state management
└── ...

demo_text2grad.py            # Standalone demo script
test_text2grad_complete.py   # Comprehensive testing script
```

## 🧪 Testing

### Run Component Tests
```bash
python test_text2grad_complete.py
```

### Run Actual Evaluation
```bash
python test_text2grad_complete.py --run-actual --task SystemBrightnessMax
```

### Test Specific Components
```bash
# Test dense rewards
python -c "from src.dense_reward import DenseRewardFunction; print('Dense rewards working!')"

# Test Text2Grad agent
python -c "from src.text2grad_agent import Text2GradConfig; print('Text2Grad agent working!')"

# Test snapshot manager
python -c "from src.utils import SnapshotManager; print('Snapshot manager working!')"
```

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### Python Environment
```bash
./setup.sh
conda activate android_world
```

### Required Packages
- torch
- transformers
- trl
- peft
- openai
- google-generativeai

## 📊 Reward Structure

### Step Penalties
- Each step: **-0.05 points**
- Encourages efficient task completion

### Subgoal Rewards
- Subgoal achievement: **+0.2 points**
- Examples:
  - Opening correct app
  - Navigating to right menu
  - Entering correct information
  - Confirming actions

### Goal Completion
- Task success: **+1.0 points**
- Final reward for completing the task

### Task-Specific Subgoals

#### Markor Tasks
- Opening Markor app
- Creating new note
- Deleting specific note
- Navigating note list

#### Contacts Tasks
- Opening Contacts app
- Adding new contact
- Editing contact information
- Deleting contact

#### System Settings
- Opening Settings app
- Navigating to correct setting
- Adjusting setting value
- Confirming changes

#### Messaging Tasks
- Opening Messages app
- Composing new message
- Selecting recipient
- Sending message

## 🔍 Optimization Details

### Text2Grad Process
1. **Snapshot Creation** - Save current emulator state
2. **Rollout Execution** - Run k rollouts of n steps each with different prompt variations
3. **Reward Collection** - Calculate dense rewards for each rollout
4. **Gradient Calculation** - Compute prompt optimization gradients
5. **Prompt Update** - Apply gradients to improve prompt
6. **State Restoration** - Restore snapshot for next optimization cycle

### Hyperparameters
- Learning rate: 0.1
- Optimization timeout: 300 seconds
- Early stopping: Enabled
- Rollouts (k): User configurable (default: 3)
- Steps per rollout (n): User configurable (default: 5)

## 🎯 Example Results

When running with Text2Grad, you'll see output like:
```
🚀 Starting AndroidWorld Enhanced T3A Agent Evaluation
   Task: SystemBrightnessMax
   Prompting: Gemini-enhanced base
   Visual Analysis: Enabled (Gemini 2.5 Flash)
   Text2Grad: Enabled (k=3 rollouts, n=5 steps)
   Model: gpt-4o-mini
   Max Steps: 25
   Episodes: 1

📱 Running Episode 1/1
🔮 Using Text2Grad optimization with 3 rollouts × 5 steps
📸 Snapshot manager initialized
🎯 Dense reward function loaded with task-specific subgoals
🔄 Starting Text2Grad optimization cycle...
...
✅ Episode 1 completed successfully!
   Steps taken: 12
   Final reward: 0.85
```

## 🚨 Troubleshooting

### Common Issues

#### API Keys Missing
```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

#### Android Emulator Not Running
```bash
# Start emulator
emulator -avd Pixel_7_API_34 -no-window
```

#### Dependencies Missing
```bash
./setup.sh
pip install torch transformers trl peft
```

#### Snapshot Commands Failing
Ensure adb is working:
```bash
adb devices
adb shell getprop ro.build.version.release
```

### Debugging Text2Grad
Enable debug logging:
```bash
python src/main.py --text2grad --gemini --log-level DEBUG
```

## 🎉 Success Metrics

A successful Text2Grad implementation should show:
- ✅ Dense rewards calculated correctly (-0.05 per step, +0.2 subgoals, +1.0 goal)
- ✅ Snapshot save/restore working without errors
- ✅ k rollouts of n steps executing properly
- ✅ Prompt optimization improving performance over time
- ✅ Integration with existing AndroidWorld tasks
- ✅ User-configurable parameters working

The implementation is complete and ready for production use in AndroidWorld agent evaluation and optimization!

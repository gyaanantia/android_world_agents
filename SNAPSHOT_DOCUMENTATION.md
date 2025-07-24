# AndroidWorld Emulator Snapshot Functionality

This document explains the new snapshot functionality added to the AndroidWorld agents framework. This feature enables saving and restoring Android emulator states, allowing agents to return to previous steps during episode execution.

## Overview

The snapshot functionality leverages the Android emulator's built-in snapshot capabilities to create checkpoints during episode execution. This enables:

1. **Debugging**: Return to specific steps to analyze agent behavior
2. **Rollback**: Recover from failed actions by restoring previous states
3. **Branch exploration**: Try different action sequences from the same starting point
4. **Analysis**: Compare agent performance across different decision points

## Features

### Core Functions

All snapshot functions are available in `src/utils.py`:

#### Basic Snapshot Operations

```python
from src.utils import save_emulator_snapshot, load_emulator_snapshot

# Save current emulator state
success = save_emulator_snapshot("my_checkpoint")

# Restore to saved state
success = load_emulator_snapshot("my_checkpoint")
```

#### Snapshot Management

```python
from src.utils import list_emulator_snapshots, delete_emulator_snapshot

# List all available snapshots
snapshots = list_emulator_snapshots()

# Delete a specific snapshot
success = delete_emulator_snapshot("old_checkpoint")
```

#### Step-Specific Operations

```python
from src.utils import create_step_snapshot, restore_to_step

# Save snapshot for episode step
snapshot_name = create_step_snapshot(step_num=5, episode_id="eval_001")

# Restore to specific step
success = restore_to_step(step_num=5, episode_id="eval_001")
```

### SnapshotManager Class

For automatic snapshot management during episodes:

```python
from src.utils import SnapshotManager

with SnapshotManager("episode_001") as sm:
    # Execute episode steps
    sm.save_step(1)
    # ... run step 1 ...
    
    sm.save_step(2)
    # ... run step 2 ...
    
    # Rollback to step 1 if needed
    sm.restore_step(1)
    
    # Snapshots automatically cleaned up on exit
```

## Technical Details

### How It Works

The snapshot functionality uses the Android emulator's console interface:

1. **Save**: Connects to emulator console (port 5554) and executes `avd snapshot save <name>`
2. **Load**: Executes `avd snapshot load <name>` to restore state
3. **List**: Executes `avd snapshot list` to enumerate available snapshots
4. **Delete**: Executes `avd snapshot delete <name>` to remove snapshots

### Requirements

- Android emulator running with console access (default port 5554)
- `telnet` available in system PATH
- Emulator launched with gRPC support (port 8554)

### Limitations

- **Performance**: Snapshot operations can take 5-30 seconds depending on emulator state
- **Storage**: Each snapshot consumes disk space (typically 1-4 GB)
- **State overwrite**: Loading a snapshot completely overwrites current state
- **Persistence**: Snapshots are stored with the AVD and persist between emulator sessions

## Usage Patterns

### 1. Episode Debugging

Save snapshots at each step to enable detailed analysis:

```python
from src.utils import SnapshotManager

def debug_episode(task_name):
    with SnapshotManager(f"debug_{task_name}") as sm:
        for step in range(1, max_steps + 1):
            # Save state before action
            sm.save_step(step)
            
            # Execute agent action
            result = agent.step(goal)
            
            # Check for failure
            if action_failed:
                # Go back and try different approach
                sm.restore_step(step - 1)
```

### 2. Comparative Analysis

Compare different strategies from the same starting point:

```python
def compare_strategies(task_name, strategies):
    # Save initial state
    save_emulator_snapshot("baseline")
    
    results = {}
    for strategy in strategies:
        # Restore to baseline
        load_emulator_snapshot("baseline")
        
        # Run episode with this strategy
        result = run_episode_with_strategy(strategy)
        results[strategy] = result
    
    return results
```

### 3. Interactive Analysis

Enable manual exploration of episode execution:

```python
def interactive_analysis(episode_id):
    with SnapshotManager(episode_id, auto_cleanup=False) as sm:
        # Load available snapshots for this episode
        steps = sm.list_episode_snapshots()
        
        while True:
            step = input("Go to step (or 'quit'): ")
            if step == 'quit':
                break
            
            if sm.restore_step(int(step)):
                print(f"Restored to step {step}")
                # User can now examine state manually
```

## Integration Examples

### Adding Snapshots to Existing Evaluation

Modify `run_episode.py` to include snapshot support:

```python
def run_episode_with_snapshots(task_name, **kwargs):
    episode_id = f"{task_name}_{timestamp}"
    
    with SnapshotManager(episode_id) as sm:
        evaluator = EpisodeEvaluator(...)
        
        for step in range(max_steps):
            # Save snapshot before action
            sm.save_step(step)
            
            # Execute step
            result = agent.step(goal)
            
            # Record step
            evaluator.record_step(step, state, action, result)
            
            # Check for rollback conditions
            if should_rollback(result):
                sm.restore_step(step - 1)
```

### Enhanced Replay System

Extend `replay_episode.py` with snapshot restoration:

```python
class SnapshotEnabledReplayer(EpisodeReplayer):
    def replay_with_snapshots(self):
        episode_id = self.get_episode_id()
        
        with SnapshotManager(episode_id, auto_cleanup=False) as sm:
            for step_data in self.episode_steps:
                step_num = step_data['step_num']
                
                # Create snapshot before executing step
                sm.save_step(step_num)
                
                # Execute step
                self.execute_step(step_data)
                
                # Allow user to rollback if desired
                if self.interactive and self.should_offer_rollback():
                    previous_step = input("Rollback to step (or Enter to continue): ")
                    if previous_step:
                        sm.restore_step(int(previous_step))
```

## Performance Considerations

### Snapshot Creation Time

- **Fast**: Empty/simple apps (5-10 seconds)
- **Medium**: Apps with data (10-20 seconds)  
- **Slow**: Complex states/multiple apps (20-30 seconds)

### Storage Usage

- Each snapshot: 1-4 GB depending on emulator state
- Recommend cleaning up old snapshots regularly
- Monitor disk space when using many snapshots

### Optimization Tips

1. **Selective snapshots**: Only save at critical decision points
2. **Cleanup strategy**: Remove old snapshots when no longer needed
3. **Parallel execution**: Consider snapshot overhead in timing measurements
4. **Batch operations**: Group multiple actions between snapshots when possible

## Error Handling

### Common Issues

1. **Emulator not responding**: Increase timeout values
2. **Console connection failed**: Check emulator is running on correct port
3. **Snapshot save failed**: Ensure sufficient disk space
4. **Telnet not available**: Install telnet or modify connection method

### Robust Error Handling

```python
from src.utils import save_emulator_snapshot

def robust_snapshot_save(name, retries=3):
    for attempt in range(retries):
        if save_emulator_snapshot(name, timeout=45):
            return True
        
        if attempt < retries - 1:
            print(f"Snapshot attempt {attempt + 1} failed, retrying...")
            time.sleep(5)
    
    print(f"Failed to save snapshot {name} after {retries} attempts")
    return False
```

## Testing

Run the test suite to verify functionality:

```bash
# Basic functionality test
python test_snapshots.py

# Interactive demonstration
python demo_snapshots.py --mode interactive

# Full integration example
python example_snapshot_integration.py
```

## Future Enhancements

Potential improvements for future versions:

1. **Faster snapshots**: Explore emulator APIs for quicker state saving
2. **Incremental snapshots**: Save only changes between steps
3. **Cloud snapshots**: Store snapshots remotely for team debugging
4. **Snapshot metadata**: Include UI state descriptions with snapshots
5. **Automated rollback**: AI-driven rollback decisions based on failure patterns

## Troubleshooting

### Emulator Console Connection

If snapshot operations fail:

```bash
# Test emulator console manually
telnet localhost 5554
> help
> avd snapshot list
> quit
```

### Disk Space Issues

Monitor and clean up snapshots:

```python
# Check available snapshots
snapshots = list_emulator_snapshots()
print(f"Found {len(snapshots)} snapshots")

# Clean up old snapshots
for snapshot in old_snapshots:
    delete_emulator_snapshot(snapshot)
```

### Performance Issues

If snapshots are too slow:

1. Reduce emulator complexity (close unnecessary apps)
2. Use faster storage (SSD recommended)
3. Increase emulator RAM allocation
4. Consider selective snapshotting strategy

## Conclusion

The snapshot functionality provides powerful debugging and analysis capabilities for AndroidWorld agents. By enabling rollback to previous states, researchers can:

- Debug failed episodes more effectively
- Compare different agent strategies
- Implement more sophisticated exploration algorithms
- Provide better interactive analysis tools

The system is designed to be backward-compatible with existing code while adding these new capabilities for enhanced agent development and analysis.

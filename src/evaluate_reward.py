#!/usr/bin/env python3
"""
Reward evaluation script for Android World agent episodes.

This script calculates rewards for episode trajectories based on:
- -0.05 per step
- +0.2 per subgoal achieved  
- +1.0 for task complete

Usage:
    python src/evaluate_reward.py <episode_file.json>
    python src/evaluate_reward.py --help
"""

import argparse
import json
import os
import sys
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import task metadata for subgoal extraction
def load_task_metadata() -> Dict[str, Dict]:
    """Load Android World task metadata for subgoal extraction."""
    try:
        metadata_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "android_world", "android_world", "task_metadata.json"
        )
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        # Convert list to dict keyed by task_name
        metadata_dict = {task['task_name']: task for task in metadata_list}
        return metadata_dict
    except Exception as e:
        print(f"Warning: Could not load task metadata: {e}")
        return {}


def extract_subgoals_from_task(task_name: str, goal: str, metadata: Dict[str, Dict]) -> List[str]:
    """
    Extract potential subgoals from task name and goal description.
    
    Args:
        task_name: Name of the Android World task
        goal: Natural language goal description
        metadata: Task metadata from Android World
        
    Returns:
        List of potential subgoals that could be achieved during the task
    """
    subgoals = []
    
    # Get task metadata if available
    task_info = metadata.get(task_name, {})
    optimal_steps = int(task_info.get('optimal_steps', 0))
    
    # Common subgoal patterns based on task analysis
    
    # App opening subgoals
    if 'settings' in goal.lower() or 'SystemBrightness' in task_name or 'SystemWifi' in task_name:
        subgoals.append("Open Settings app")
        
    if 'contacts' in goal.lower() or 'ContactsAdd' in task_name:
        subgoals.append("Open Contacts app")
        
    if 'camera' in goal.lower() or 'CameraTake' in task_name:
        subgoals.append("Open Camera app")
        
    if 'audio' in goal.lower() or 'AudioRecorder' in task_name:
        subgoals.append("Open Audio Recorder app")
        
    if 'calendar' in goal.lower() or 'Calendar' in task_name:
        subgoals.append("Open Calendar app")
        
    if 'browser' in goal.lower() or 'Browser' in task_name or 'Chrome' in task_name:
        subgoals.append("Open Browser app")
        
    if 'files' in goal.lower() or 'Files' in task_name:
        subgoals.append("Open Files app")
        
    if 'markor' in goal.lower() or 'Markor' in task_name:
        subgoals.append("Open Markor app")
        
    # Navigation subgoals
    if 'settings' in goal.lower():
        if 'brightness' in goal.lower():
            subgoals.extend([
                "Navigate to Display settings",
                "Access Brightness controls"
            ])
        elif 'wifi' in goal.lower():
            subgoals.extend([
                "Navigate to Network settings",
                "Access WiFi controls"
            ])
    
    # Action-specific subgoals
    if 'create' in goal.lower() or 'add' in goal.lower():
        if 'contact' in goal.lower():
            subgoals.extend([
                "Access new contact form",
                "Enter contact information"
            ])
        elif 'note' in goal.lower():
            subgoals.extend([
                "Create new note",
                "Enter note content"
            ])
    
    if 'record' in goal.lower():
        subgoals.extend([
            "Start recording",
            "Stop recording"
        ])
        
    if 'take' in goal.lower() and 'photo' in goal.lower():
        subgoals.append("Capture photo")
        
    if 'brightness' in goal.lower():
        if 'min' in goal.lower():
            subgoals.append("Set brightness to minimum")
        elif 'max' in goal.lower():
            subgoals.append("Set brightness to maximum")
            
    if 'wifi' in goal.lower():
        if 'on' in goal.lower() or 'enable' in goal.lower():
            subgoals.append("Enable WiFi")
        elif 'off' in goal.lower() or 'disable' in goal.lower():
            subgoals.append("Disable WiFi")
    
    # File operations
    if 'delete' in goal.lower():
        subgoals.extend([
            "Locate target file",
            "Delete file"
        ])
        
    if 'save' in goal.lower():
        subgoals.append("Save file")
    
    # Browser-specific tasks
    if 'Browser' in task_name:
        if 'Draw' in task_name:
            subgoals.extend([
                "Open HTML file",
                "Access drawing interface",
                "Create drawing",
                "Submit drawing"
            ])
        elif 'Maze' in task_name:
            subgoals.extend([
                "Open HTML file", 
                "Navigate maze",
                "Reach target position"
            ])
        elif 'Multiply' in task_name:
            subgoals.extend([
                "Open HTML file",
                "Click button multiple times",
                "Calculate product",
                "Enter result"
            ])
    
    # Adjust subgoal count based on optimal steps
    if optimal_steps > 0:
        # Aim for roughly 1 subgoal per 2-3 optimal steps
        target_subgoals = max(1, optimal_steps // 3)
        if len(subgoals) > target_subgoals * 2:
            # Keep the most important subgoals
            subgoals = subgoals[:target_subgoals * 2]
    
    return subgoals


def detect_subgoal_achievement(
    step_data: Dict, 
    action: Dict, 
    subgoals: List[str], 
    achieved_subgoals: set
) -> Tuple[List[str], set]:
    """
    Detect if any subgoals were achieved in this step.
    
    Args:
        step_data: Step information from episode
        action: Action taken in this step
        subgoals: List of all possible subgoals for this task
        achieved_subgoals: Set of already achieved subgoals
        
    Returns:
        Tuple of (newly achieved subgoals, updated achieved subgoals set)
    """
    newly_achieved = []
    
    # Get action type and details
    action_type = action.get('action_type', '')
    
    # Analysis of UI elements or screen state if available
    ui_elements = step_data.get('ui_elements', [])
    ui_text = " ".join([elem.get('text', '') for elem in ui_elements if elem.get('text')])
    
    # Check for app opening
    if action_type == 'open_app':
        app_name = action.get('app_name', '').lower()
        for subgoal in subgoals:
            if subgoal not in achieved_subgoals:
                if (f"open {app_name}" in subgoal.lower() or 
                    app_name in subgoal.lower()):
                    newly_achieved.append(subgoal)
                    achieved_subgoals.add(subgoal)
    
    # Check for clicking on app icons
    if action_type == 'click':
        for subgoal in subgoals:
            if subgoal not in achieved_subgoals and "open" in subgoal.lower():
                # Check if UI contains app names mentioned in subgoal
                subgoal_words = subgoal.lower().split()
                for word in subgoal_words:
                    if len(word) > 3 and word in ui_text.lower():
                        newly_achieved.append(subgoal)
                        achieved_subgoals.add(subgoal)
                        break
    
    # Check for navigation-based subgoals
    if "settings" in ui_text.lower():
        nav_subgoals = [sg for sg in subgoals if "settings" in sg.lower() and sg not in achieved_subgoals]
        for subgoal in nav_subgoals:
            if "display" in subgoal.lower() and "display" in ui_text.lower():
                newly_achieved.append(subgoal)
                achieved_subgoals.add(subgoal)
            elif "network" in subgoal.lower() and ("network" in ui_text.lower() or "wifi" in ui_text.lower()):
                newly_achieved.append(subgoal)
                achieved_subgoals.add(subgoal)
    
    # Check for brightness/wifi controls
    if "brightness" in ui_text.lower():
        brightness_subgoals = [sg for sg in subgoals if "brightness" in sg.lower() and sg not in achieved_subgoals]
        for subgoal in brightness_subgoals:
            if "access" in subgoal.lower() or "controls" in subgoal.lower():
                newly_achieved.append(subgoal)
                achieved_subgoals.add(subgoal)
    
    # Check for form entry
    if action_type == 'input_text':
        form_subgoals = [sg for sg in subgoals if ("enter" in sg.lower() or "information" in sg.lower()) and sg not in achieved_subgoals]
        if form_subgoals:
            newly_achieved.append(form_subgoals[0])
            achieved_subgoals.add(form_subgoals[0])
    
    # Check for recording actions
    if action_type in ['click', 'long_press']:
        recording_subgoals = [sg for sg in subgoals if ("start recording" in sg.lower() or "stop recording" in sg.lower()) and sg not in achieved_subgoals]
        for subgoal in recording_subgoals:
            if "record" in ui_text.lower():
                newly_achieved.append(subgoal)
                achieved_subgoals.add(subgoal)
                break
    
    # Check for successful actions (swipe for brightness, etc.)
    if action_type == 'swipe':
        swipe_subgoals = [sg for sg in subgoals if ("set brightness" in sg.lower() or "brightness" in sg.lower()) and sg not in achieved_subgoals]
        if swipe_subgoals and "brightness" in ui_text.lower():
            newly_achieved.append(swipe_subgoals[0])
            achieved_subgoals.add(swipe_subgoals[0])
    
    return newly_achieved, achieved_subgoals


def calculate_episode_reward(episode_data: Dict) -> Dict:
    """
    Calculate the total reward for an episode based on the reward function.
    
    Args:
        episode_data: Complete episode data from JSON file
        
    Returns:
        Dictionary with reward breakdown and step-by-step rewards
    """
    task_name = episode_data.get('task_name', '')
    goal = episode_data.get('goal', '')
    success = episode_data.get('success', False)
    steps_taken = episode_data.get('steps_taken', 0)
    
    # Load task metadata for subgoal extraction
    metadata = load_task_metadata()
    
    # Extract potential subgoals for this task
    subgoals = extract_subgoals_from_task(task_name, goal, metadata)
    
    # Initialize reward tracking
    total_reward = 0.0
    step_rewards = []
    achieved_subgoals = set()
    
    # Get episode steps from step_records (preferred) or fallback to other formats
    episode_steps = []
    if 'step_records' in episode_data:
        episode_steps = episode_data['step_records']
    elif 'actions' in episode_data:
        episode_steps = episode_data['actions']
    elif 'step_data' in episode_data:
        episode_steps = episode_data['step_data']
    else:
        # Try to reconstruct from responses
        responses = episode_data.get('responses', [])
        actions = episode_data.get('actions', [])
        if responses and actions:
            episode_steps = [{'action': action, 'response': response} 
                           for action, response in zip(actions, responses)]
    
    # Process each step
    for i, step in enumerate(episode_steps):
        step_reward = -0.05  # Base step penalty
        
        # Extract action and UI state from step data
        action = {}
        ui_elements = []
        
        if isinstance(step, dict):
            # Handle step_records format
            if 'action' in step:
                if isinstance(step['action'], str):
                    # Parse JSON action if it's a string
                    import re
                    json_match = re.search(r'\{.*\}', step['action'])
                    if json_match:
                        try:
                            action = json.loads(json_match.group())
                        except:
                            pass
                else:
                    action = step.get('action', {})
            
            # Get UI elements from state if available
            if 'state' in step and isinstance(step['state'], dict):
                ui_elements = step['state'].get('ui_elements', [])
            elif 'ui_elements' in step:
                ui_elements = step['ui_elements']
                
        elif isinstance(step, str):
            # Handle raw action string
            import re
            json_match = re.search(r'\{.*\}', step)
            if json_match:
                try:
                    action = json.loads(json_match.group())
                except:
                    pass
        
        # Create step data for subgoal detection
        step_data = {
            'ui_elements': ui_elements,
            'action': action,
            'step_num': i + 1
        }
        
        # Check for subgoal achievements
        newly_achieved, achieved_subgoals = detect_subgoal_achievement(
            step_data, action, subgoals, achieved_subgoals
        )
        
        # Add subgoal rewards
        subgoal_reward = len(newly_achieved) * 0.2
        step_reward += subgoal_reward
        
        total_reward += step_reward
        step_rewards.append({
            'step': i + 1,
            'base_reward': -0.05,
            'subgoal_reward': subgoal_reward,
            'newly_achieved_subgoals': newly_achieved,
            'total_step_reward': step_reward,
            'cumulative_reward': total_reward
        })
    
    # Add task completion reward
    completion_reward = 1.0 if success else 0.0
    total_reward += completion_reward
    
    # Calculate final metrics
    subgoals_achieved_count = len(achieved_subgoals)
    subgoals_possible_count = len(subgoals)
    subgoal_completion_rate = subgoals_achieved_count / max(1, subgoals_possible_count)
    
    return {
        'task_name': task_name,
        'goal': goal,
        'success': success,
        'steps_taken': steps_taken,
        'total_reward': round(total_reward, 3),
        'completion_reward': completion_reward,
        'step_penalty': steps_taken * -0.05,
        'subgoal_reward': subgoals_achieved_count * 0.2,
        'subgoals_achieved': list(achieved_subgoals),
        'subgoals_possible': subgoals,
        'subgoal_completion_rate': round(subgoal_completion_rate, 3),
        'step_rewards': step_rewards,
        'reward_breakdown': {
            'step_penalties': steps_taken * -0.05,
            'subgoal_rewards': subgoals_achieved_count * 0.2,
            'completion_reward': completion_reward,
            'total': round(total_reward, 3)
        }
    }


def update_episode_with_rewards(episode_file: str, reward_data: Dict) -> None:
    """
    Update the episode JSON file with reward information.
    
    Args:
        episode_file: Path to the episode JSON file
        reward_data: Reward calculation results
    """
    try:
        # Load original episode data
        with open(episode_file, 'r') as f:
            episode_data = json.load(f)
        
        # Add reward information
        episode_data['reward_evaluation'] = reward_data
        
        # Save back to file
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
            
        print(f"✅ Updated {episode_file} with reward information")
        
    except Exception as e:
        print(f"❌ Failed to update {episode_file}: {e}")


def print_reward_summary(reward_data: Dict) -> None:
    """Print a formatted summary of the reward calculation."""
    
    print("\n" + "="*60)
    print("🏆 EPISODE REWARD SUMMARY")
    print("="*60)
    
    print(f"Task: {reward_data['task_name']}")
    print(f"Goal: {reward_data['goal']}")
    print(f"Success: {'✅ Yes' if reward_data['success'] else '❌ No'}")
    print(f"Steps Taken: {reward_data['steps_taken']}")
    
    print(f"\n📊 REWARD BREAKDOWN:")
    breakdown = reward_data['reward_breakdown']
    print(f"  Step Penalties:    {breakdown['step_penalties']:+.3f} ({reward_data['steps_taken']} × -0.05)")
    print(f"  Subgoal Rewards:   {breakdown['subgoal_rewards']:+.3f} ({len(reward_data['subgoals_achieved'])} × +0.2)")
    print(f"  Completion Reward: {breakdown['completion_reward']:+.3f} ({'1.0' if reward_data['success'] else '0.0'})")
    print(f"  {'─'*30}")
    print(f"  TOTAL REWARD:      {breakdown['total']:+.3f}")
    
    print(f"\n🎯 SUBGOAL ANALYSIS:")
    achieved = len(reward_data['subgoals_achieved'])
    possible = len(reward_data['subgoals_possible'])
    rate = reward_data['subgoal_completion_rate']
    print(f"  Subgoals Achieved: {achieved}/{possible} ({rate:.1%})")
    
    if reward_data['subgoals_achieved']:
        print(f"  ✅ Achieved:")
        for subgoal in reward_data['subgoals_achieved']:
            print(f"     • {subgoal}")
    
    remaining = [sg for sg in reward_data['subgoals_possible'] if sg not in reward_data['subgoals_achieved']]
    if remaining:
        print(f"  ❌ Not Achieved:")
        for subgoal in remaining:
            print(f"     • {subgoal}")
    
    print(f"\n📈 STEP-BY-STEP REWARDS:")
    for step_reward in reward_data['step_rewards'][-5:]:  # Show last 5 steps
        step_num = step_reward['step']
        total = step_reward['total_step_reward']
        cumulative = step_reward['cumulative_reward']
        newly_achieved = step_reward['newly_achieved_subgoals']
        
        status = ""
        if newly_achieved:
            status = f" (🎯 {len(newly_achieved)} subgoal{'s' if len(newly_achieved) > 1 else ''})"
        
        print(f"  Step {step_num:2d}: {total:+.3f} → {cumulative:+.3f}{status}")
    
    if len(reward_data['step_rewards']) > 5:
        print(f"  ... (showing last 5 of {len(reward_data['step_rewards'])} steps)")


def main():
    """Main function for the reward evaluation script."""
    parser = argparse.ArgumentParser(
        description="Calculate rewards for Android World agent episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reward Function:
  -0.05 per step (efficiency penalty)
  +0.2 per subgoal achieved (progress reward)  
  +1.0 for task complete (success reward)

Examples:
  python src/evaluate_reward.py results/episode.json
  python src/evaluate_reward.py results/*.json --update
        """
    )
    
    parser.add_argument(
        'episode_files',
        nargs='+',
        help='Path(s) to episode JSON file(s)'
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update the JSON file(s) with reward information'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed step-by-step analysis'
    )
    
    args = parser.parse_args()
    
    # Process each episode file
    for episode_file in args.episode_files:
        if not os.path.exists(episode_file):
            print(f"❌ File not found: {episode_file}")
            continue
            
        try:
            print(f"\n🔍 Analyzing: {episode_file}")
            
            # Load episode data
            with open(episode_file, 'r') as f:
                episode_data = json.load(f)
            
            # Check if rewards already calculated
            if 'reward_evaluation' in episode_data and not args.update:
                print(f"✅ Rewards already calculated (use --update to recalculate)")
                reward_data = episode_data['reward_evaluation']
            else:
                # Calculate rewards
                print("📊 Calculating rewards...")
                reward_data = calculate_episode_reward(episode_data)
                
                # Update file if requested
                if args.update:
                    update_episode_with_rewards(episode_file, reward_data)
            
            # Print summary
            print_reward_summary(reward_data)
            
        except Exception as e:
            print(f"❌ Error processing {episode_file}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()

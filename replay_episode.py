#!/usr/bin/env python3
"""
Episode Replay System for AndroidWorld Enhanced T3A Agent.

This script allows you to replay episodes from JSON result files, stepping through
each action that was taken during the original evaluation.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root and src to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from android_world import registry
from android_world.env import env_launcher, json_action
from android_world.task_evals import task_eval

from utils import (
    suppress_grpc_logging,
    setup_logging, 
    validate_android_world_env,
    find_adb_directory
)

# Suppress gRPC verbose logging before any gRPC communication
suppress_grpc_logging()


class EpisodeReplayer:
    """Replays an episode from a JSON result file."""
    
    def __init__(self, json_file_path: str, interactive: bool = True, delay: float = 1.0):
        """Initialize the replayer.
        
        Args:
            json_file_path: Path to the JSON result file
            interactive: Whether to wait for user input between steps
            delay: Delay in seconds between steps (if not interactive)
        """
        self.json_file_path = json_file_path
        self.interactive = interactive
        self.delay = delay
        self.episode_data = None
        self.env = None
        self.task = None
        
    def load_episode_data(self) -> Dict[str, Any]:
        """Load episode data from JSON file.
        
        Returns:
            Dictionary containing episode data
        """
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
            
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.episode_data = json.load(f)
            
        print(f"📁 Loaded episode data from: {self.json_file_path}")
        print(f"   Task: {self.episode_data.get('task_name', 'Unknown')}")
        print(f"   Model: {self.episode_data.get('model_name', 'Unknown')}")
        print(f"   Prompt Variant: {self.episode_data.get('prompt_variant', 'Unknown')}")
        print(f"   Success: {self.episode_data.get('success', False)}")
        print(f"   Steps Taken: {self.episode_data.get('steps_taken', 0)}")
        print()
        
        return self.episode_data
        
    def setup_environment(self):
        """Set up the AndroidWorld environment and task."""
        task_name = self.episode_data.get('task_name')
        if not task_name:
            raise ValueError("No task_name found in episode data")
            
        # Get task registry and task class
        task_registry = registry.TaskRegistry().get_registry(
            registry.TaskRegistry.ANDROID_WORLD_FAMILY
        )
        
        if task_name not in task_registry:
            available_tasks = list(task_registry.keys())
            print(f"❌ Task '{task_name}' not found. Available tasks:")
            for task in available_tasks[:10]:  # Show first 10
                print(f"   - {task}")
            if len(available_tasks) > 10:
                print(f"   ... and {len(available_tasks) - 10} more")
            raise ValueError(f"Task '{task_name}' not found in registry")
            
        task_cls = task_registry[task_name]
        
        # Initialize task with random parameters
        self.task = task_cls(task_cls.generate_random_params())
            
        print(f"🎯 Setting up task: {task_name}")
        print(f"   Goal: {self.task.goal}")
        print()
        
        # Set up environment
        adb_path = find_adb_directory()
        self.env = env_launcher.load_and_setup_env(
            console_port=5554,
            adb_path=adb_path
        )
        self.env.reset(go_home=True)
        
        # Initialize task in environment
        self.task.initialize_task(self.env)
        
        print(f"📱 Environment set up successfully")
        print(f"   ADB Path: {adb_path}")
        print()
        
    def parse_action_json(self, action_str: str) -> Optional[json_action.JSONAction]:
        """Parse action string into JSONAction object.
        
        Args:
            action_str: JSON string representation of the action
            
        Returns:
            JSONAction object or None if parsing fails
        """
        try:
            # Clean up the action string - sometimes it includes extra text
            action_str = action_str.strip()
            if action_str.startswith('Reason:'):
                # Extract just the JSON part from "Reason: ... Action: {...}"
                lines = action_str.split('\n')
                for line in lines:
                    if line.strip().startswith('Action:'):
                        action_str = line.strip()[7:].strip()  # Remove "Action:" prefix
                        break
                        
            # Parse JSON
            action_dict = json.loads(action_str)
            return json_action.JSONAction(**action_dict)
            
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"❌ Failed to parse action: {action_str}")
            print(f"   Error: {e}")
            return None
            
    def display_step_info(self, step_num: int, action_data: Dict[str, Any]):
        """Display information about the current step.
        
        Args:
            step_num: Current step number
            action_data: Action data from episode
        """
        print(f"═══ Step {step_num} ═══")
        
        # Show the full response if available
        if 'responses' in self.episode_data and step_num <= len(self.episode_data['responses']):
            response = self.episode_data['responses'][step_num - 1]
            print(f"📝 LLM Response:")
            
            # Split response into reason and action
            lines = response.split('\n')
            for line in lines:
                if line.strip().startswith('Reason:'):
                    print(f"   💭 {line.strip()}")
                elif line.strip().startswith('Action:'):
                    print(f"   🎬 {line.strip()}")
                    
        # Show parsed action
        if 'actions' in self.episode_data and step_num <= len(self.episode_data['actions']):
            action_str = self.episode_data['actions'][step_num - 1]
            parsed_action = self.parse_action_json(action_str)
            if parsed_action:
                print(f"🔧 Parsed Action: {parsed_action}")
            else:
                print(f"⚠️  Raw Action String: {action_str}")
                
        print()
        
    def execute_action(self, action: json_action.JSONAction) -> bool:
        """Execute an action in the environment.
        
        Args:
            action: JSONAction to execute
            
        Returns:
            True if action was executed successfully, False otherwise
        """
        try:
            # Special handling for status actions
            if action.action_type == 'status':
                print(f"📊 Status Action: {action.goal_status}")
                return True
                
            # Special handling for answer actions
            if action.action_type == 'answer':
                print(f"💬 Answer Action: {action.text}")
                return True
                
            # Execute the action in the environment
            print(f"⚡ Executing action: {action.action_type}")
            self.env.execute_action(action)
            print(f"✅ Action executed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to execute action: {e}")
            return False
            
    def wait_for_user_input(self, step_num: int, total_steps: int):
        """Wait for user input to continue (if in interactive mode).
        
        Args:
            step_num: Current step number
            total_steps: Total number of steps
        """
        if self.interactive:
            prompt = f"Press Enter to continue to step {step_num + 1}/{total_steps}, 'q' to quit, 's' to skip interaction: "
            user_input = input(prompt).strip().lower()
            
            if user_input == 'q':
                print("🛑 Replay stopped by user")
                sys.exit(0)
            elif user_input == 's':
                print("🏃 Switching to non-interactive mode")
                self.interactive = False
        else:
            if self.delay > 0:
                print(f"⏳ Waiting {self.delay} seconds...")
                time.sleep(self.delay)
                
    def replay_episode(self):
        """Replay the entire episode step by step."""
        if not self.episode_data:
            raise ValueError("No episode data loaded. Call load_episode_data() first.")
            
        actions = self.episode_data.get('actions', [])
        if not actions:
            print("⚠️  No actions found in episode data")
            return
            
        total_steps = len(actions)
        print(f"🎬 Starting replay of {total_steps} steps")
        print(f"   Interactive Mode: {'ON' if self.interactive else 'OFF'}")
        if not self.interactive:
            print(f"   Delay: {self.delay}s between steps")
        print()
        
        for step_num in range(1, total_steps + 1):
            self.display_step_info(step_num, self.episode_data)
            
            # Parse and execute action
            action_str = actions[step_num - 1]
            parsed_action = self.parse_action_json(action_str)
            
            if parsed_action:
                success = self.execute_action(parsed_action)
                if not success:
                    print(f"⚠️  Action execution failed, but continuing...")
            else:
                print(f"⚠️  Could not parse action, skipping...")
                
            # Check if this is a terminal action
            if parsed_action and parsed_action.action_type == 'status':
                print(f"🏁 Episode ended with status: {parsed_action.goal_status}")
                break
                
            # Wait for user input or delay
            if step_num < total_steps:
                self.wait_for_user_input(step_num, total_steps)
                
        print()
        print(f"🎉 Replay completed!")
        print(f"   Original Success: {self.episode_data.get('success', False)}")
        print(f"   Original Steps: {self.episode_data.get('steps_taken', 0)}")
        
        # Optionally check current task state
        if self.task and self.env:
            try:
                current_success = self.task.is_successful(self.env) == 1
                print(f"   Current Task State: {'✅ Successful' if current_success else '❌ Not Successful'}")
            except Exception as e:
                print(f"   Current Task State: ⚠️  Could not evaluate ({e})")
                
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            try:
                self.env.close()
            except Exception as e:
                print(f"⚠️  Error closing environment: {e}")


def main():
    """Main entry point for the replay script."""
    parser = argparse.ArgumentParser(
        description="Replay AndroidWorld Enhanced T3A Agent Episodes"
    )
    
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON result file to replay"
    )
    
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (don't wait for user input)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between steps (non-interactive mode only)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if not validate_android_world_env():
        print("❌ AndroidWorld environment validation failed!")
        print("Please ensure you've run the setup script:")
        print("  ./setup.sh")
        print("  conda activate android_world")
        sys.exit(1)
        
    # Set up logging
    log_dir = os.path.join(os.path.dirname(args.json_file), 'replay_logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(args.log_level, log_dir)
    
    print(f"🚀 AndroidWorld Episode Replayer")
    print(f"   JSON File: {args.json_file}")
    print(f"   Interactive: {not args.non_interactive}")
    if args.non_interactive:
        print(f"   Delay: {args.delay}s")
    print()
    
    # Create and run replayer
    replayer = EpisodeReplayer(
        json_file_path=args.json_file,
        interactive=not args.non_interactive,
        delay=args.delay
    )
    
    try:
        # Load episode data
        replayer.load_episode_data()
        
        # Set up environment
        replayer.setup_environment()
        
        # Replay episode
        replayer.replay_episode()
        
    except KeyboardInterrupt:
        print("\n🛑 Replay interrupted by user")
    except Exception as e:
        print(f"❌ Replay failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        replayer.cleanup()


if __name__ == "__main__":
    main()

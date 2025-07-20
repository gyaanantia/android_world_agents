#!/usr/bin/env python3
"""
Advanced Episode Replay System with debugging and analysis features.

This script provides advanced replay capabilities including:
- Step-by-step debugging
- Screenshot comparison
- Action validation
- Performance analysis
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import tempfile

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


class AdvancedEpisodeReplayer:
    """Advanced episode replayer with debugging and analysis features."""
    
    def __init__(
        self, 
        json_file_path: str, 
        mode: str = "replay",
        save_screenshots: bool = False,
        validate_actions: bool = True,
        compare_states: bool = False
    ):
        """Initialize the advanced replayer.
        
        Args:
            json_file_path: Path to the JSON result file
            mode: Replay mode ('replay', 'debug', 'analyze')
            save_screenshots: Whether to save screenshots at each step
            validate_actions: Whether to validate actions before executing
            compare_states: Whether to compare states with original episode
        """
        self.json_file_path = json_file_path
        self.mode = mode
        self.save_screenshots = save_screenshots
        self.validate_actions = validate_actions
        self.compare_states = compare_states
        self.episode_data = None
        self.env = None
        self.task = None
        self.screenshots_dir = None
        
        if self.save_screenshots:
            self.screenshots_dir = os.path.join(
                os.path.dirname(json_file_path), 
                'replay_screenshots', 
                os.path.splitext(os.path.basename(json_file_path))[0]
            )
            os.makedirs(self.screenshots_dir, exist_ok=True)
            
    def load_episode_data(self) -> Dict[str, Any]:
        """Load and validate episode data from JSON file."""
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
            
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.episode_data = json.load(f)
            
        # Validate required fields
        required_fields = ['task_name', 'actions']
        missing_fields = [field for field in required_fields if field not in self.episode_data]
        if missing_fields:
            raise ValueError(f"Missing required fields in episode data: {missing_fields}")
            
        self._print_episode_summary()
        return self.episode_data
        
    def _print_episode_summary(self):
        """Print a summary of the episode data."""
        print(f"📁 Episode Summary")
        print(f"   File: {self.json_file_path}")
        print(f"   Task: {self.episode_data.get('task_name', 'Unknown')}")
        print(f"   Model: {self.episode_data.get('model_name', 'Unknown')}")
        print(f"   Prompt: {self.episode_data.get('prompt_variant', 'Unknown')}")
        print(f"   Success: {self.episode_data.get('success', False)}")
        print(f"   Steps: {self.episode_data.get('steps_taken', 0)}")
        print(f"   Time: {self.episode_data.get('evaluation_time', 0):.2f}s")
        
        # Analyze action types
        actions = self.episode_data.get('actions', [])
        action_types = {}
        for action_str in actions:
            try:
                action_dict = json.loads(action_str.split('Action: ')[-1] if 'Action: ' in action_str else action_str)
                action_type = action_dict.get('action_type', 'unknown')
                action_types[action_type] = action_types.get(action_type, 0) + 1
            except:
                action_types['parse_error'] = action_types.get('parse_error', 0) + 1
                
        print(f"   Action Types: {dict(action_types)}")
        print()
        
    def setup_environment(self):
        """Set up the AndroidWorld environment and task."""
        task_name = self.episode_data.get('task_name')
        
        # Get task
        self.task = registry.get_random_task() if task_name == 'random' else registry.get_task(task_name)
        if not self.task:
            raise ValueError(f"Task '{task_name}' not found in registry")
            
        print(f"🎯 Task Setup")
        print(f"   Name: {task_name}")
        print(f"   Goal: {self.task.goal}")
        print(f"   Complexity: {getattr(self.task, 'complexity', 'Unknown')}")
        print()
        
        # Set up environment
        adb_path = find_adb_directory()
        self.env = env_launcher.run_task(self.task, adb_path=adb_path)
        
        print(f"📱 Environment Ready")
        print(f"   ADB: {adb_path}")
        print()
        
    def parse_action_with_context(self, action_str: str, step_num: int) -> Tuple[Optional[json_action.JSONAction], str, str]:
        """Parse action string and extract reason and action separately.
        
        Args:
            action_str: Action string from episode data
            step_num: Current step number
            
        Returns:
            Tuple of (parsed_action, reason, action_json)
        """
        reason = ""
        action_json = ""
        
        try:
            # Handle different action string formats
            if 'Reason:' in action_str and 'Action:' in action_str:
                # Standard format: "Reason: ... Action: {...}"
                parts = action_str.split('\n')
                for part in parts:
                    if part.strip().startswith('Reason:'):
                        reason = part.strip()[7:].strip()
                    elif part.strip().startswith('Action:'):
                        action_json = part.strip()[7:].strip()
            else:
                # Assume it's just JSON
                action_json = action_str.strip()
                reason = f"Step {step_num} action"
                
            # Parse the action JSON
            if action_json:
                action_dict = json.loads(action_json)
                parsed_action = json_action.JSONAction(**action_dict)
                return parsed_action, reason, action_json
            else:
                return None, reason, action_json
                
        except Exception as e:
            print(f"❌ Parse error for step {step_num}: {e}")
            return None, reason, action_str
            
    def validate_action(self, action: json_action.JSONAction, step_num: int) -> Tuple[bool, List[str]]:
        """Validate an action before execution.
        
        Args:
            action: Action to validate
            step_num: Current step number
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check action type
        valid_types = [
            'click', 'double_tap', 'long_press', 'scroll', 'swipe',
            'input_text', 'keyboard_enter', 'navigate_home', 'navigate_back',
            'open_app', 'status', 'answer', 'wait'
        ]
        
        if action.action_type not in valid_types:
            issues.append(f"Invalid action type: {action.action_type}")
            
        # Validate parameters based on action type
        if action.action_type in ['click', 'double_tap', 'long_press', 'input_text']:
            if action.index is None:
                issues.append(f"Missing required 'index' parameter for {action.action_type}")
            elif not isinstance(action.index, int) or action.index < 0:
                issues.append(f"Invalid index value: {action.index}")
                
        if action.action_type == 'input_text':
            if not action.text:
                issues.append("Missing required 'text' parameter for input_text")
                
        if action.action_type in ['scroll', 'swipe']:
            if action.direction not in ['up', 'down', 'left', 'right']:
                issues.append(f"Invalid direction: {action.direction}")
                
        if action.action_type == 'open_app':
            if not action.app_name:
                issues.append("Missing required 'app_name' parameter for open_app")
                
        if action.action_type == 'status':
            if action.goal_status not in ['complete', 'infeasible']:
                issues.append(f"Invalid goal_status: {action.goal_status}")
                
        if action.action_type == 'answer':
            if not action.text:
                issues.append("Missing required 'text' parameter for answer")
                
        return len(issues) == 0, issues
        
    def capture_screenshot(self, step_num: int, prefix: str = "") -> Optional[str]:
        """Capture and save a screenshot.
        
        Args:
            step_num: Current step number
            prefix: Optional prefix for filename
            
        Returns:
            Path to saved screenshot or None if failed
        """
        if not self.screenshots_dir:
            return None
            
        try:
            state = self.env.get_state(include_ui_tree=False)
            if hasattr(state, 'pixels') and state.pixels:
                filename = f"{prefix}step_{step_num:03d}.png" if prefix else f"step_{step_num:03d}.png"
                filepath = os.path.join(self.screenshots_dir, filename)
                
                # Save screenshot (this depends on the state.pixels format)
                # You might need to adjust this based on how AndroidWorld stores pixels
                with open(filepath, 'wb') as f:
                    f.write(state.pixels)
                    
                return filepath
        except Exception as e:
            print(f"⚠️  Failed to capture screenshot: {e}")
            
        return None
        
    def execute_step(self, step_num: int, action: json_action.JSONAction, reason: str) -> Dict[str, Any]:
        """Execute a single step and collect results.
        
        Args:
            step_num: Current step number
            action: Action to execute
            reason: Reason for the action
            
        Returns:
            Dictionary with step execution results
        """
        step_result = {
            'step_num': step_num,
            'action': action,
            'reason': reason,
            'executed': False,
            'success': False,
            'error': None,
            'pre_screenshot': None,
            'post_screenshot': None,
            'execution_time': 0
        }
        
        # Capture pre-action screenshot
        if self.save_screenshots:
            step_result['pre_screenshot'] = self.capture_screenshot(step_num, "pre_")
            
        # Validate action if enabled
        if self.validate_actions:
            is_valid, issues = self.validate_action(action, step_num)
            if not is_valid:
                step_result['error'] = f"Validation failed: {', '.join(issues)}"
                return step_result
                
        # Execute action
        start_time = time.time()
        try:
            if action.action_type == 'status':
                print(f"📊 Status: {action.goal_status}")
                step_result['executed'] = True
                step_result['success'] = True
            elif action.action_type == 'answer':
                print(f"💬 Answer: {action.text}")
                step_result['executed'] = True
                step_result['success'] = True
            else:
                self.env.execute_action(action)
                step_result['executed'] = True
                step_result['success'] = True
                print(f"✅ Executed: {action.action_type}")
                
        except Exception as e:
            step_result['error'] = str(e)
            print(f"❌ Execution failed: {e}")
            
        step_result['execution_time'] = time.time() - start_time
        
        # Capture post-action screenshot
        if self.save_screenshots:
            step_result['post_screenshot'] = self.capture_screenshot(step_num, "post_")
            
        return step_result
        
    def debug_mode_step(self, step_num: int, action: json_action.JSONAction, reason: str):
        """Interactive debugging for a single step.
        
        Args:
            step_num: Current step number
            action: Action to execute
            reason: Reason for the action
        """
        print(f"🐛 DEBUG MODE - Step {step_num}")
        print(f"   Reason: {reason}")
        print(f"   Action: {action}")
        
        if self.validate_actions:
            is_valid, issues = self.validate_action(action, step_num)
            if issues:
                print(f"   ⚠️  Validation Issues: {', '.join(issues)}")
                
        while True:
            choice = input("\n[e]xecute, [s]kip, [q]uit, [i]nspect environment: ").strip().lower()
            
            if choice == 'e':
                result = self.execute_step(step_num, action, reason)
                break
            elif choice == 's':
                print("⏭️  Skipped step")
                break
            elif choice == 'q':
                print("🛑 Debug session ended")
                sys.exit(0)
            elif choice == 'i':
                self.inspect_environment()
            else:
                print("Invalid choice. Use 'e', 's', 'q', or 'i'.")
                
    def inspect_environment(self):
        """Inspect current environment state."""
        try:
            state = self.env.get_state(include_ui_tree=True)
            print("\n🔍 Environment State:")
            print(f"   UI Elements: {len(state.ui_elements) if state.ui_elements else 0}")
            
            if state.ui_elements:
                print("   Top 10 UI Elements:")
                for i, element in enumerate(state.ui_elements[:10]):
                    text = element.get('text', '') or element.get('content_description', '') or element.get('class_name', '')
                    clickable = element.get('is_clickable', False)
                    print(f"     {i}: {text[:50]}... {'[clickable]' if clickable else ''}")
                    
            # Check task state
            if self.task:
                task_successful = self.task.is_successful(self.env) == 1
                print(f"   Task Status: {'✅ Successful' if task_successful else '❌ Not Complete'}")
                
        except Exception as e:
            print(f"   ❌ Failed to inspect environment: {e}")
            
    def analyze_episode(self):
        """Analyze the episode for patterns and insights."""
        print(f"📊 Episode Analysis")
        
        actions = self.episode_data.get('actions', [])
        responses = self.episode_data.get('responses', [])
        
        # Action type distribution
        action_types = {}
        action_timings = []
        
        for i, action_str in enumerate(actions):
            parsed_action, reason, _ = self.parse_action_with_context(action_str, i + 1)
            if parsed_action:
                action_types[parsed_action.action_type] = action_types.get(parsed_action.action_type, 0) + 1
                
        print(f"   Action Distribution: {dict(action_types)}")
        
        # Analyze success patterns
        success = self.episode_data.get('success', False)
        steps_taken = self.episode_data.get('steps_taken', 0)
        max_steps = self.episode_data.get('max_steps', 30)
        
        print(f"   Completion: {steps_taken}/{max_steps} steps")
        print(f"   Success Rate: {'100%' if success else '0%'}")
        
        # Find failure points
        if not success:
            last_action = actions[-1] if actions else None
            if last_action:
                parsed_action, reason, _ = self.parse_action_with_context(last_action, len(actions))
                if parsed_action and parsed_action.action_type == 'status':
                    print(f"   Failure Reason: {parsed_action.goal_status}")
                    
        print()
        
    def replay_episode(self):
        """Main replay logic based on selected mode."""
        if not self.episode_data:
            raise ValueError("No episode data loaded")
            
        actions = self.episode_data.get('actions', [])
        if not actions:
            print("⚠️  No actions found in episode data")
            return
            
        total_steps = len(actions)
        print(f"🎬 Starting {self.mode} mode with {total_steps} steps\n")
        
        if self.mode == "analyze":
            self.analyze_episode()
            return
            
        step_results = []
        
        for step_num in range(1, total_steps + 1):
            action_str = actions[step_num - 1]
            parsed_action, reason, action_json = self.parse_action_with_context(action_str, step_num)
            
            if not parsed_action:
                print(f"⚠️  Step {step_num}: Could not parse action, skipping")
                continue
                
            print(f"═══ Step {step_num}/{total_steps} ═══")
            print(f"💭 Reason: {reason}")
            print(f"🎬 Action: {action_json}")
            
            if self.mode == "debug":
                self.debug_mode_step(step_num, parsed_action, reason)
            else:
                result = self.execute_step(step_num, parsed_action, reason)
                step_results.append(result)
                
            # Check for terminal actions
            if parsed_action.action_type == 'status':
                print(f"🏁 Episode ended at step {step_num}")
                break
                
            print()
            
        # Final summary
        if step_results:
            successful_steps = sum(1 for r in step_results if r['success'])
            print(f"📈 Replay Summary:")
            print(f"   Successful Steps: {successful_steps}/{len(step_results)}")
            print(f"   Total Execution Time: {sum(r['execution_time'] for r in step_results):.2f}s")
            
        if self.screenshots_dir:
            print(f"   Screenshots Saved: {self.screenshots_dir}")
            
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            try:
                self.env.close()
            except Exception as e:
                print(f"⚠️  Error closing environment: {e}")


def main():
    """Main entry point for the advanced replay script."""
    parser = argparse.ArgumentParser(
        description="Advanced AndroidWorld Episode Replayer"
    )
    
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON result file to replay"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["replay", "debug", "analyze"],
        default="replay",
        help="Replay mode: replay (automatic), debug (interactive), analyze (statistics only)"
    )
    
    parser.add_argument(
        "--screenshots",
        action="store_true",
        help="Save screenshots at each step"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip action validation"
    )
    
    parser.add_argument(
        "--compare-states",
        action="store_true",
        help="Compare states with original episode (experimental)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Validate environment for modes that need it
    if args.mode in ["replay", "debug"]:
        if not validate_android_world_env():
            print("❌ AndroidWorld environment validation failed!")
            sys.exit(1)
            
    # Set up logging
    log_dir = os.path.join(os.path.dirname(args.json_file), 'replay_logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(args.log_level, log_dir)
    
    print(f"🚀 Advanced AndroidWorld Episode Replayer")
    print(f"   Mode: {args.mode.upper()}")
    print(f"   File: {args.json_file}")
    if args.screenshots:
        print(f"   Screenshots: Enabled")
    print()
    
    # Create and run replayer
    replayer = AdvancedEpisodeReplayer(
        json_file_path=args.json_file,
        mode=args.mode,
        save_screenshots=args.screenshots,
        validate_actions=not args.no_validation,
        compare_states=args.compare_states
    )
    
    try:
        # Load episode data
        replayer.load_episode_data()
        
        # Set up environment if needed
        if args.mode in ["replay", "debug"]:
            replayer.setup_environment()
            
        # Run replay
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

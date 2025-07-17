#!/usr/bin/env python3
"""
Main entry point for AndroidWorld Enhanced T3A Agent evaluation framework.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    setup_logging, 
    validate_android_world_env, 
    ensure_results_dir,
    find_adb_directory
)
from run_episode import run_episode


def main():
    """Main entry point for the evaluation framework."""
    parser = argparse.ArgumentParser(
        description="AndroidWorld Enhanced T3A Agent Evaluation Framework"
    )
    
    # Task configuration
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        help="Task name to evaluate (e.g., 'single_task_name' or 'all')"
    )
    
    parser.add_argument(
        "--task_config", 
        type=str,
        help="Path to task configuration file"
    )
    
    # Agent configuration
    parser.add_argument(
        "--agent_type", 
        type=str, 
        default="base",
        choices=["base", "few_shot", "reflective"],
        help="Type of agent prompting to use"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4",
        help="OpenAI model name to use"
    )
    
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=30,
        help="Maximum number of steps per episode"
    )
    
    # Environment configuration
    parser.add_argument(
        "--device_id", 
        type=str,
        help="Android device ID (use 'adb devices' to list)"
    )
    
    parser.add_argument(
        "--adb_path", 
        type=str,
        help="Path to ADB executable directory"
    )
    
    # Output configuration
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Evaluation options
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=1,
        help="Number of episodes to run per task"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=300,
        help="Timeout in seconds per episode"
    )
    
    parser.add_argument(
        "--save_screenshots", 
        action="store_true",
        help="Save screenshots during evaluation"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Validate environment
    if not validate_android_world_env():
        print("❌ AndroidWorld environment validation failed!")
        print("Please ensure you've run the setup script:")
        print("  ./setup.sh")
        print("  conda activate android_world")
        sys.exit(1)
    
    # Find ADB if not provided
    if not args.adb_path:
        adb_path = find_adb_directory()
        if not adb_path:
            print("❌ ADB not found! Please install Android SDK or specify --adb_path")
            sys.exit(1)
        args.adb_path = adb_path
    
    # Ensure results directory exists
    results_dir = ensure_results_dir(args.results_dir)
    
    print(f"🚀 Starting AndroidWorld Enhanced T3A Agent Evaluation")
    print(f"   Task: {args.task}")
    print(f"   Agent Type: {args.agent_type}")
    print(f"   Model: {args.model_name}")
    print(f"   Max Steps: {args.max_steps}")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   Results Dir: {results_dir}")
    print(f"   ADB Path: {args.adb_path}")
    print()
    
    # Run evaluation
    try:
        for episode in range(args.num_episodes):
            print(f"📱 Running Episode {episode + 1}/{args.num_episodes}")
            
            result = run_episode(
                task_name=args.task,
                agent_type=args.agent_type,
                model_name=args.model_name,
                max_steps=args.max_steps,
                results_dir=results_dir,
                device_id=args.device_id,
                adb_path=args.adb_path,
                timeout=args.timeout,
                save_screenshots=args.save_screenshots,
                task_config=args.task_config
            )
            
            if result.get("success"):
                print(f"✅ Episode {episode + 1} completed successfully!")
            else:
                print(f"❌ Episode {episode + 1} failed: {result.get('error', 'Unknown error')}")
            
            print(f"   Steps taken: {result.get('steps_taken', 0)}")
            print(f"   Result file: {result.get('result_file', 'Not saved')}")
            print()
    
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)
    
    print("🎉 Evaluation complete!")


if __name__ == "__main__":
    main()

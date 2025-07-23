#!/usr/bin/env python3
"""
Main entry point for AndroidWorld Enhanced T3A Agent evaluation framework.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import List, Optional

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    suppress_grpc_logging,
    setup_logging, 
    validate_android_world_env, 
    ensure_results_dir
)
from run_episode import run_episode

# Suppress gRPC verbose logging before any imports that use gRPC
# Comment this out to enable gRPC logging for debugging
suppress_grpc_logging()


def main():
    """Main entry point for the evaluation framework."""
    parser = argparse.ArgumentParser(
        description="AndroidWorld Enhanced T3A Agent Evaluation Framework"
    )
    
    # Task configuration
    parser.add_argument(
        "--task", 
        type=str, 
        default=None,
        help="Task name to evaluate (random if not specified)"
    )
    
    # Agent configuration
    parser.add_argument(
        "--prompt-variant", 
        type=str, 
        default="base",
        choices=["base", "few-shot", "reflective"],
        help="Type of agent prompting to use (works with or without --gemini)"
    )
    
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="gpt-4o-mini",
        help="OpenAI model name to use"
    )
    
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=30,
        help="Maximum number of steps per episode"
    )
    
    # Output configuration
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Agent configuration
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable memory (step history) in agent prompts"
    )
    
    parser.add_argument(
        "--function-calling",
        action="store_true",
        help="Use OpenAI function calling for structured output"
    )
    
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini 2.5 Flash for visual UI analysis and enhanced prompting (works with any --prompt-variant)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--num-episodes", 
        type=int, 
        default=1,
        help="Number of episodes to run per task"
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations - no mutual exclusivity needed anymore
    # Both --prompt-variant and --gemini can be used together
    
    # Check for required API keys
    if args.gemini and not os.getenv("GOOGLE_API_KEY"):
        print("❌ Gemini requires GOOGLE_API_KEY environment variable!")
        print("Please set your Google API key:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("Get an API key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    # Validate environment
    if not validate_android_world_env():
        print("❌ AndroidWorld environment validation failed!")
        print("Please ensure you've run the setup script:")
        print("  ./setup.sh")
        print("  conda activate android_world")
        sys.exit(1)
    
    # Ensure results directory exists
    results_dir = ensure_results_dir(args.results_dir)
    
    # Set up logging (after results directory is created)
    setup_logging(args.log_level, results_dir)
    
    print(f"🚀 Starting AndroidWorld Enhanced T3A Agent Evaluation")
    print(f"   Task: {args.task if args.task else 'Random'}")
    if args.gemini:
        print(f"   Prompting: Gemini-enhanced {args.prompt_variant}")
        print(f"   Visual Analysis: Enabled (Gemini 2.5 Flash)")
    else:
        print(f"   Prompting: Standard {args.prompt_variant}")
        print(f"   Visual Analysis: Disabled")
    print(f"   Model: {args.model_name}")
    print(f"   Max Steps: {args.max_steps}")
    print(f"   Episodes: {args.num_episodes}")
    print(f"   Results Dir: {results_dir}")
    print()
    
    # Run evaluation
    try:
        for episode in range(args.num_episodes):
            print(f"📱 Running Episode {episode + 1}/{args.num_episodes}")
            
            result = run_episode(
                task_name=args.task,
                prompt_variant=args.prompt_variant,
                model_name=args.model_name,
                max_steps=args.max_steps,
                output_dir=results_dir,
                use_memory=not args.disable_memory,
                use_function_calling=args.function_calling,
                use_gemini=args.gemini
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

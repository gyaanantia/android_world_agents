"""Main evaluation runner for AndroidWorld LLM agents."""

import argparse
import json
import os
import random
import time
import uuid
from typing import Dict, Any, Optional

from android_world import registry
from android_world.env import env_launcher
from android_world.task_evals import task_eval

from agent import create_agent
from evaluator import EpisodeEvaluator
from utils import find_adb_directory, ensure_results_dir


def run_episode(
    task_name: Optional[str] = None,
    model_name: str = "gpt-4",
    prompt_variant: str = "base",
    max_steps: int = 25,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """Run a single episode evaluation.
    
    Args:
        task_name: Specific task name, or None for random.
        model_name: LLM model name.
        prompt_variant: Prompting variant ("base", "few-shot", "reflective").
        max_steps: Maximum number of steps.
        output_dir: Output directory for results.
        
    Returns:
        Episode results dictionary.
    """
    print(f"🚀 Starting evaluation:")
    print(f"   Model: {model_name}")
    print(f"   Prompt variant: {prompt_variant}")
    print(f"   Max steps: {max_steps}")
    
    # Ensure output directory exists
    ensure_results_dir(output_dir)
    
    # Setup environment
    env = env_launcher.load_and_setup_env(
        console_port=5554, 
        adb_path=find_adb_directory()
    )
    env.reset(go_home=True)
    
    # Select task
    task_registry = registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    
    if task_name:
        if task_name not in task_registry:
            available_tasks = list(task_registry.keys())
            print(f"❌ Task '{task_name}' not found. Available tasks:")
            for task in available_tasks[:10]:  # Show first 10
                print(f"   - {task}")
            if len(available_tasks) > 10:
                print(f"   ... and {len(available_tasks) - 10} more")
            raise ValueError(f"Task '{task_name}' not found")
        task_cls = task_registry[task_name]
    else:
        task_cls = random.choice(list(task_registry.values()))
        task_name = task_cls.__name__
    
    # Initialize task
    task = task_cls(task_cls.generate_random_params())
    task.initialize_task(env)
    
    # Create agent
    agent = create_agent(env, model_name, prompt_variant)
    
    print(f"📱 Task: {task_name}")
    print(f"🎯 Goal: {task.goal}")
    
    # Create evaluator
    evaluator = EpisodeEvaluator(
        task_name=task_name,
        goal=task.goal,
        model_name=model_name,
        prompt_variant=prompt_variant,
        max_steps=max_steps
    )
    
    # Run episode
    start_time = time.time()
    
    try:
        result = agent.step(task.goal)
        step_count = 0
        
        while not result.done and step_count < max_steps:
            # Record step
            evaluator.record_step(
                step_num=step_count,
                state=env.get_state(True),
                action=result.action if hasattr(result, 'action') else None,
                agent_data=result.data
            )
            
            # Take next step
            result = agent.step(task.goal)
            step_count += 1
            
            # For reflective agent, add reflection after failed actions
            if prompt_variant == "reflective" and hasattr(result, 'action') and not result.done:
                if step_count > 1:  # Add reflection after first step
                    reflection = f"Step {step_count}: Attempted action, continuing towards goal"
                    agent.add_reflection(reflection)
        
        # Record final step
        evaluator.record_step(
            step_num=step_count,
            state=env.get_state(True),
            action=result.action if hasattr(result, 'action') else None,
            agent_data=result.data
        )
        
        # Evaluate success
        success = result.done
        evaluation_time = time.time() - start_time
        
        # Generate results
        results = evaluator.generate_results(
            success=success,
            steps_taken=step_count,
            evaluation_time=evaluation_time,
            final_state=env.get_state(True)
        )
        
        # Save results
        output_file = os.path.join(
            output_dir, 
            f"{task_name}_{prompt_variant}_{model_name.replace('/', '_')}_{uuid.uuid4().hex[:8]}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n📊 Episode completed!")
        print(f"✅ Success: {success}")
        print(f"📝 Steps taken: {step_count}")
        print(f"⏱️ Time: {evaluation_time:.2f}s")
        print(f"💾 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise
    finally:
        env.close()


def main():
    """Main entry point for episode evaluation."""
    parser = argparse.ArgumentParser(
        description="Run AndroidWorld LLM agent evaluation"
    )
    
    parser.add_argument(
        "--task", 
        type=str, 
        help="Specific task name (default: random)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model name (default: gpt-4)"
    )
    
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=["base", "few-shot", "reflective"],
        default="base",
        help="Prompting variant (default: base)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Maximum number of steps (default: 25)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_episode(
            task_name=args.task,
            model_name=args.model,
            prompt_variant=args.prompt_variant,
            max_steps=args.max_steps,
            output_dir=args.output_dir
        )
        
        exit(0 if results["success"] else 1)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_json", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--prompt_variant", choices=["base", "few-shot", "reflective"], default="base"
    )
    args = parser.parse_args()
    path = run_episode(args.episode_json, args.model, args.prompt_variant)
    print(f"Log written to {path}")


if __name__ == "__main__":
    main()

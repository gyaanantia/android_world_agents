#!/usr/bin/env python3
"""
Demonstration of Text2Grad integration with AndroidWorld agents.

This script shows how to use the Text2Grad optimization system to improve
Gemini prompts through dense reward feedback in AndroidWorld tasks.

Usage:
    python demo_text2grad.py --task "SystemBrightnessMax" --k-rollouts 3 --n-steps 5

The system will:
1. Take a snapshot of the current environment
2. Run Text2Grad optimization with k rollouts of n steps each
3. Use the optimized Gemini prompt for the actual agent step
4. Continue until task completion
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from android_world import registry
from android_world.env import env_launcher
from src.text2grad_agent import Text2GradAgent, Text2GradConfig, create_text2grad_agent
from src.gemini_prompting import create_gemini_generator
from src.dense_reward import create_dense_reward_function
from src.utils import find_adb_directory, suppress_grpc_logging

# Suppress gRPC verbose logging
suppress_grpc_logging()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('text2grad_demo.log')
    ]
)
logger = logging.getLogger(__name__)


def run_text2grad_demo(task_name: str, 
                      k_rollouts: int = 3, 
                      n_steps: int = 5,
                      max_episode_steps: int = 30,
                      model_name: str = "gpt-4o-mini"):
    """
    Run a demonstration of Text2Grad optimization.
    
    Args:
        task_name: AndroidWorld task to run
        k_rollouts: Number of optimization rollouts per step
        n_steps: Number of steps per rollout
        max_episode_steps: Maximum steps for the main episode
        model_name: LLM model to use for the agent
    """
    logger.info("🚀 Starting Text2Grad Demo")
    logger.info(f"Task: {task_name}")
    logger.info(f"Optimization: {k_rollouts} rollouts × {n_steps} steps")
    logger.info(f"Model: {model_name}")
    
    try:
        # 1. Setup environment
        logger.info("📱 Setting up AndroidWorld environment...")
        adb_path = find_adb_directory()
        env = env_launcher.load_and_setup_env(
            console_port=5554,
            adb_path=adb_path
        )
        env.reset(go_home=True)
        logger.info("✅ Environment ready")
        
        # 2. Setup task
        logger.info(f"🎯 Setting up task: {task_name}")
        task_registry = registry.TaskRegistry()
        aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
        
        if task_name not in aw_registry:
            available_tasks = list(aw_registry.keys())
            logger.error(f"❌ Task '{task_name}' not found")
            logger.info("Available tasks:")
            for task in available_tasks[:10]:
                logger.info(f"   - {task}")
            return False
        
        task_cls = aw_registry[task_name]
        params = task_cls.generate_random_params()
        task = task_cls(params)
        task.initialize_task(env)
        
        logger.info(f"Goal: {task.goal}")
        logger.info(f"Complexity: {task.complexity}")
        
        # 3. Setup Gemini generator
        logger.info("🔮 Initializing Gemini generator...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("❌ GOOGLE_API_KEY not set. Please set it to use Gemini integration.")
            return False
        
        gemini_generator = create_gemini_generator(
            api_key=api_key,
            model_name="gemini-2.5-flash"
        )
        
        if not gemini_generator:
            logger.error("❌ Failed to create Gemini generator")
            return False
        
        logger.info("✅ Gemini generator ready")
        
        # 4. Setup Text2Grad configuration
        text2grad_config = Text2GradConfig(
            k_rollouts=k_rollouts,
            n_steps=n_steps,
            learning_rate=0.1,
            optimization_timeout=300.0,
            enable_early_stopping=True
        )
        
        # 5. Create Text2Grad agent
        logger.info("🧠 Creating Text2Grad agent...")
        agent = create_text2grad_agent(
            env=env,
            model_name=model_name,
            prompt_variant="base",
            use_memory=True,
            use_function_calling=False,
            text2grad_config=text2grad_config,
            gemini_generator=gemini_generator
        )
        logger.info("✅ Text2Grad agent ready")
        
        # 6. Setup dense reward function for monitoring
        reward_function = create_dense_reward_function()
        reward_function.reset()
        
        # 7. Run episode with Text2Grad optimization
        logger.info("🎬 Starting episode with Text2Grad optimization...")
        logger.info("=" * 60)
        
        episode_start_time = time.time()
        step_count = 0
        total_episode_reward = 0.0
        
        while step_count < max_episode_steps:
            step_count += 1
            logger.info(f"\n📍 Episode Step {step_count}/{max_episode_steps}")
            
            # Take step with Text2Grad optimization
            step_start_time = time.time()
            result = agent.step(task.goal)
            step_time = time.time() - step_start_time
            
            # Calculate episode reward for monitoring
            action = result.data.get('action_output') if result.data else {}
            step_reward, reward_info = reward_function.calculate_step_reward(
                env=env,
                task=task,
                action=action,
                action_history=[],  # Simplified for demo
                is_terminal=result.done
            )
            total_episode_reward += step_reward
            
            logger.info(f"⏱️  Step completed in {step_time:.2f}s")
            logger.info(f"💰 Step reward: {step_reward:.3f} (total: {total_episode_reward:.3f})")
            
            # Log optimization results if available
            if result.data and 'text2grad_optimization' in result.data:
                opt_results = result.data['text2grad_optimization']
                logger.info(f"🔬 Optimization: {opt_results.get('rollouts_completed', 0)} rollouts, "
                          f"best reward: {opt_results.get('best_reward', 0.0):.3f}")
            
            # Check if done
            if result.done:
                logger.info("🏁 Agent declared task complete")
                break
        
        # 8. Evaluate final success
        episode_time = time.time() - episode_start_time
        final_success = task.is_successful(env) > 0.5
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 EPISODE RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Success: {final_success}")
        logger.info(f"📝 Steps taken: {step_count}")
        logger.info(f"⏱️  Total time: {episode_time:.2f}s")
        logger.info(f"💰 Total reward: {total_episode_reward:.3f}")
        
        # Get optimization summary
        opt_summary = agent.get_optimization_summary()
        if opt_summary.get('status') != 'no_optimizations':
            logger.info(f"🔬 Optimizations: {opt_summary['total_optimizations']}")
            logger.info(f"📈 Success rate: {opt_summary['success_rate']:.1%}")
            logger.info(f"📊 Avg improvement: {opt_summary['average_improvement']:.3f}")
        
        # Get reward breakdown
        reward_summary = reward_function.get_episode_summary()
        logger.info(f"🎯 Subgoals achieved: {reward_summary['subgoals_achieved']}")
        logger.info(f"💸 Step penalties: {reward_summary['total_step_penalty']:.3f}")
        logger.info(f"🎁 Subgoal rewards: {reward_summary['total_subgoal_reward']:.3f}")
        
        logger.info("\n🎉 Text2Grad demo completed!")
        return final_success
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            env.close()
            logger.info("🔌 Environment closed")
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Text2Grad AndroidWorld Demo")
    parser.add_argument(
        "--task", 
        default="SystemBrightnessMax",
        help="AndroidWorld task to run (default: SystemBrightnessMax)"
    )
    parser.add_argument(
        "--k-rollouts", 
        type=int, 
        default=3,
        help="Number of optimization rollouts per step (default: 3)"
    )
    parser.add_argument(
        "--n-steps", 
        type=int, 
        default=5,
        help="Number of steps per rollout (default: 5)"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=30,
        help="Maximum episode steps (default: 30)"
    )
    parser.add_argument(
        "--model", 
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run demo
    success = run_text2grad_demo(
        task_name=args.task,
        k_rollouts=args.k_rollouts,
        n_steps=args.n_steps,
        max_episode_steps=args.max_steps,
        model_name=args.model
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import time
    main()

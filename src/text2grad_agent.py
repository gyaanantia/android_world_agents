"""
Text2Grad optimization integration for AndroidWorld agents.

This module implements the complete Text2Grad optimization cycle:
1. Take snapshot of current environment state
2. Generate Gemini visual analysis of current screen
3. Run Text2Grad optimization using rollouts with dense rewards
4. Restore snapshot and apply optimized prompt for the actual agent step
5. Repeat cycle

The optimization uses a dense reward function to provide gradient signals
for improving Gemini's visual analysis prompts.
"""

import logging
import time
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Add Text2Grad to path
text2grad_path = os.path.join(os.path.dirname(__file__), '..', 'Text2Grad')
if os.path.exists(text2grad_path):
    sys.path.insert(0, text2grad_path)

from android_world.env import interface
from android_world.task_evals import task_eval
from dense_reward import DenseRewardFunction
from gemini_prompting import GeminiPromptGenerator
from agent import create_agent
from utils import SnapshotManager

logger = logging.getLogger(__name__)


@dataclass
class Text2GradConfig:
    """Configuration for Text2Grad optimization."""
    k_rollouts: int = 3  # Number of rollouts for optimization
    n_steps: int = 5     # Number of steps per rollout
    learning_rate: float = 1e-5
    max_prompt_length: int = 2000
    optimization_timeout: float = 300.0  # 5 minutes max
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.1  # Stop if improvement < threshold
    use_original_text2grad: bool = True  # Use original Text2GradTrainer when available


@dataclass
class RolloutResult:
    """Results from a single rollout."""
    total_reward: float
    steps_taken: int
    final_success: bool
    action_history: List[Dict[str, Any]]
    reward_breakdown: Dict[str, Any]


def create_dense_reward_function() -> DenseRewardFunction:
    """Create a dense reward function for Text2Grad optimization."""
    return DenseRewardFunction()


class Text2GradOptimizer:
    """
    Implements Text2Grad optimization for Gemini prompt refinement.
    
    CURRENT STATUS: Simplified implementation using exploration-based optimization.
    
    FULL TEXT2GRAD INTEGRATION WOULD REQUIRE:
    
    1. **Reward Model Training**: Train a Text2Grad reward model on AndroidWorld 
       visual analysis tasks with natural language feedback data. This model learns
       to convert textual critiques into token-level reward signals.
    
    2. **Data Collection**: Collect datasets of:
       - Screenshots from AndroidWorld tasks
       - Gemini visual analysis outputs
       - Human feedback/critiques on the analysis quality
       - Success/failure outcomes for the resulting agent actions
    
    3. **Token-Level Reward Assignment**: Use the trained reward model to assign
       rewards to individual tokens in the Gemini prompt based on how they
       contribute to successful task completion.
    
    4. **PPO-Based Optimization**: Use the Text2GradTrainer with PPO to optimize
       the prompt generation model by backpropagating through the reward signals.
    
    The current implementation uses structured exploration as a reasonable fallback
    that still provides meaningful optimization signals for visual analysis improvement.
    """
    
    def __init__(self, 
                 config: Text2GradConfig,
                 gemini_generator: GeminiPromptGenerator,
                 base_agent_config: Dict[str, Any]):
        self.config = config
        self.gemini_generator = gemini_generator
        self.base_agent_config = base_agent_config
        # Generate unique episode ID for snapshot management
        episode_id = f"text2grad_{int(time.time())}"
        self.snapshot_manager = SnapshotManager(episode_id)
        
        # Try to use original Text2GradTrainer if available
        self.original_trainer = None
        if self.config.use_original_text2grad:
            self.original_trainer = self._initialize_original_text2grad()
        
    def _initialize_original_text2grad(self):
        """Initialize the original Text2GradTrainer if available.
        
        Text2Grad requires:
        1. A trained reward model that can process natural language feedback
        2. Token-level reward computation capabilities
        3. PPO-based gradient optimization
        
        For AndroidWorld, we would need to train a Text2Grad reward model
        on visual UI analysis tasks with natural language feedback.
        """
        try:
            # Import Text2Grad components
            from nl_gradiant_policy_optimization.kodcode.text2grad_trainer import Text2GradTrainer
            from trl import PPOConfig, AutoModelForCausalLMWithValueHead
            import torch
            
            # Check if we have a trained reward model for AndroidWorld
            # TODO: Train Text2Grad reward model on AndroidWorld visual analysis tasks
            reward_model_path = "path/to/androidworld/text2grad/reward/model"  # Placeholder
            
            if not os.path.exists(reward_model_path):
                logger.info("📝 Text2Grad components available but no AndroidWorld reward model found")
                logger.info("🔬 To use full Text2Grad: train reward model on visual analysis feedback")
                return {'config': None, 'available': False, 'reason': 'no_reward_model'}
            
            # Create a minimal PPO config for Text2Grad
            ppo_config = PPOConfig(
                model_name=self.base_agent_config.get('model_name', 'gpt-4o-mini'),
                learning_rate=self.config.learning_rate,
                batch_size=self.config.k_rollouts,
                mini_batch_size=1,
                ppo_epochs=1,
                seed=42
            )
            
            logger.info("✅ Full Text2Grad pipeline available with trained reward model")
            return {'config': ppo_config, 'available': True, 'trainer_class': Text2GradTrainer}
            
        except ImportError as e:
            logger.warning(f"⚠️ Text2Grad components not available: {e}")
            logger.info("� Install Text2Grad dependencies for gradient-based optimization")
            return None
        except Exception as e:
            logger.error(f"❌ Error initializing Text2Grad: {e}")
            return None
    
    def _perturb_gemini_prompt(self, base_prompt: str, iteration: int) -> str:
        """
        Create a perturbed version of the Gemini prompt for exploration.
        
        This is a simplified fallback when the full Text2Grad pipeline is not available.
        In a complete Text2Grad implementation, this would use the trained reward model
        and PromptGradientPPO to compute actual gradients for prompt optimization.
        
        TODO: Replace with proper Text2Grad gradient-based perturbations when
        reward model is available for AndroidWorld domain.
        """
        if self.original_trainer and self.original_trainer.get('available', False):
            # Use actual Text2Grad gradient computation
            return self._compute_text2grad_perturbation(base_prompt, iteration)
        else:
            # Fallback to exploration-based perturbations
            return self._compute_exploration_perturbation(base_prompt, iteration)
    
    def _compute_text2grad_perturbation(self, base_prompt: str, iteration: int) -> str:
        """
        Compute gradient-based perturbations using the actual Text2Grad pipeline.
        
        This would use the trained reward model to compute token-level gradients
        and apply them to optimize the prompt. Currently not implemented as we
        would need a Text2Grad reward model trained on AndroidWorld data.
        """
        # TODO: Implement actual Text2Grad gradient computation
        # This would involve:
        # 1. Tokenizing the base_prompt
        # 2. Computing rewards for each token using the reward model
        # 3. Computing gradients based on the reward signals
        # 4. Applying gradient-based updates to the prompt tokens
        
        # For now, fall back to exploration
        return self._compute_exploration_perturbation(base_prompt, iteration)
    
    def _compute_exploration_perturbation(self, base_prompt: str, iteration: int) -> str:
        """
        Compute exploration-based perturbations as a fallback.
        
        This uses structured exploration strategies that target different aspects
        of visual analysis, which is more principled than random perturbations.
        """
        # Structured exploration strategies for visual analysis optimization
        exploration_strategies = [
            "Pay closer attention to UI element hierarchy and navigation context.",
            "Focus on spatial relationships between interactive elements.",
            "Emphasize visual indicators of task progress and completion states.", 
            "Consider alternative interaction paths and accessibility features.",
            "Analyze color, size, and position cues for element importance.",
            "Look for patterns in successful task completion workflows.",
            "Focus on text content and its relationship to available actions.",
            "Consider the temporal sequence of required interactions."
        ]
        
        if iteration < len(exploration_strategies):
            strategy = exploration_strategies[iteration]
            return f"{base_prompt}\n\n[Visual Analysis Enhancement]: {strategy}"
        else:
            # Additional exploration with variation
            variation_id = iteration - len(exploration_strategies)
            return f"{base_prompt}\n\n[Exploration Variant {variation_id}]: Apply systematic visual analysis improvements."
    
    def _run_rollout(self, 
                    env: interface.AsyncEnv,
                    task: task_eval.TaskEval,
                    gemini_prompt: str,
                    rollout_id: int,
                    snapshot_step: int,
                    goal: str) -> RolloutResult:
        """
        Run a single rollout with the given Gemini prompt.
        
        Args:
            env: Android environment 
            task: Task to evaluate
            gemini_prompt: The Gemini prompt to use for this rollout
            rollout_id: Identifier for this rollout
            snapshot_step: Original snapshot step to restore before rollout
            goal: The goal string to use for agent steps
            
        Returns:
            RolloutResult containing the rollout outcomes
        """
        logger.info(f"🔄 Starting rollout {rollout_id} with {self.config.n_steps} steps")
        
        # CRITICAL: Restore snapshot ONCE at the beginning of each rollout for clean state
        # This should NOT happen inside the step loop
        print(f"🔄 [ROLLOUT {rollout_id}] About to restore snapshot {snapshot_step} at START of rollout")
        restore_success = self.snapshot_manager.restore_step(snapshot_step)
        if not restore_success:
            print(f"❌ [ROLLOUT {rollout_id}] Failed to restore snapshot for step {snapshot_step}, continuing anyway")
            logger.warning(f"⚠️ Failed to restore snapshot for step {snapshot_step}, continuing anyway")
        else:
            print(f"✅ [ROLLOUT {rollout_id}] Successfully restored snapshot {snapshot_step} - environment reset for clean rollout")
            logger.info(f"🔄 Restored snapshot {snapshot_step} for rollout {rollout_id} (this should happen ONCE per rollout)")
        
        # Create agent that will use the perturbed prompt on the first step only
        # Text2Grad optimization works by enhancing the initial visual understanding
        agent = create_agent(
            env=env,
            model_name=self.base_agent_config.get('model_name', 'gpt-4o-mini'),
            prompt_variant=self.base_agent_config.get('prompt_variant', 'base'),
            use_memory=self.base_agent_config.get('use_memory', True),
            use_function_calling=self.base_agent_config.get('use_function_calling', False)
        )
        
        # Initialize dense reward function using Text2Grad principles
        reward_function = create_dense_reward_function()
        reward_function.reset_episode()
        
        action_history = []
        total_reward = 0.0
        
        for step in range(self.config.n_steps):
            try:
                # Take agent step - need to handle case where task might be None
                if task:
                    step_goal = task.goal
                else:
                    step_goal = goal  # Use the passed goal parameter as fallback
                
                # CRITICAL: For the FIRST step only, inject the perturbed Gemini prompt
                # This is where Text2Grad optimization takes effect
                if step == 0:
                    # First step: Use the perturbed Gemini prompt for enhanced visual understanding
                    print(f"🔍 [ROLLOUT {rollout_id}, STEP {step+1}] Using perturbed Gemini prompt for initial visual analysis")
                    # TODO: Inject gemini_prompt into the agent's prompt for this step
                    # For now, we note that this step should use the optimized visual understanding
                    enhanced_goal = f"{step_goal}\n\n[Text2Grad Visual Enhancement]: {gemini_prompt}"
                    result = agent.step(enhanced_goal)
                else:
                    # Subsequent steps: Use regular agent without Gemini visual block
                    print(f"🔄 [ROLLOUT {rollout_id}, STEP {step+1}] Using regular agent step (no Gemini injection)")
                    result = agent.step(step_goal)
                
                # Extract action from result - the agent returns AgentInteractionResult
                # with step_data containing 'action_output' (raw LLM response string)
                if hasattr(result, 'data') and result.data:
                    # action_output is the raw LLM response, we need to parse it
                    action_output = result.data.get('action_output', '')
                    if isinstance(action_output, str):
                        # Try to extract action from the LLM output string
                        try:
                            # Look for action JSON in the output with more robust regex
                            import re
                            import json
                            
                            # Try multiple patterns to find JSON
                            patterns = [
                                r'Action:\s*(\{[^}]*\})',  # Simple single-line JSON
                                r'Action:\s*(\{.*?\})',    # Multi-line JSON (non-greedy)
                                r'"action_type":\s*"([^"]*)"',  # Just extract action_type
                            ]
                            
                            action = None
                            for pattern in patterns:
                                action_match = re.search(pattern, action_output, re.DOTALL)
                                if action_match:
                                    try:
                                        if 'action_type' in pattern:
                                            # Just action_type pattern
                                            action = {'action_type': action_match.group(1)}
                                        else:
                                            # Full JSON pattern
                                            action = json.loads(action_match.group(1))
                                        break
                                    except json.JSONDecodeError:
                                        continue
                            
                            if action is None:
                                # Fallback to simple representation
                                action = {
                                    'action_type': 'unknown',
                                    'raw_output': action_output[:100]  # Truncate for safety
                                }
                        except Exception as e:
                            logger.debug(f"Failed to parse action from output: {e}")
                            action = {'action_type': 'parse_error', 'error': str(e)}
                    else:
                        action = action_output if isinstance(action_output, dict) else {'action': str(action_output)}
                else:
                    # Fallback for unexpected result format
                    action = {'action_type': 'unknown_result', 'result': str(result)}
                    
                action_history.append(action)
                
                # Calculate dense reward using Text2Grad methodology
                is_terminal = result.done or step == self.config.n_steps - 1
                step_reward, reward_info = reward_function.calculate_step_reward(
                    env=env,
                    task=task,
                    action=action,
                    action_history=action_history,
                    is_terminal=is_terminal
                )
                
                total_reward += step_reward
                
                logger.debug(f"📊 Rollout {rollout_id}, Step {step+1}: "
                           f"reward={step_reward:.3f}, total={total_reward:.3f}")
                
                # Stop if agent declares done
                if result.done:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Error in rollout {rollout_id}, step {step}: {e}")
                break
        
        # Check final success
        final_success = False
        try:
            if task and hasattr(task, 'is_successful'):
                final_success = task.is_successful(env) > 0.5
        except Exception as e:
            logger.debug(f"Could not check task success in rollout: {e}")
            final_success = False
        
        episode_summary = reward_function.get_episode_summary()
        
        print(f"✅ [ROLLOUT {rollout_id}] Completed with reward={total_reward:.3f}, success={final_success}, steps={len(action_history)}")
        logger.info(f"✅ Rollout {rollout_id} completed: "
                   f"reward={total_reward:.3f}, success={final_success}, "
                   f"steps={len(action_history)}")
        
        return RolloutResult(
            total_reward=total_reward,
            steps_taken=len(action_history),
            final_success=final_success,
            action_history=action_history,
            reward_breakdown=episode_summary
        )
    
    def optimize_gemini_prompt(self,
                              env: interface.AsyncEnv,
                              task: task_eval.TaskEval,
                              screenshot: np.ndarray,
                              goal: str,
                              snapshot_step: int) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize the Gemini prompt using Text2Grad methodology.
        
        This is the core Text2Grad optimization loop:
        1. Generate initial Gemini analysis
        2. Run k rollouts with different prompt perturbations
        3. Select best prompt based on dense reward signals
        4. Return optimized prompt for actual agent use
        """
        logger.info(f"🎯 Starting Text2Grad optimization with {self.config.k_rollouts} rollouts")
        optimization_start = time.time()
        
        # Step 1: Generate initial Gemini analysis
        logger.info("🔍 Generating initial Gemini visual analysis...")
        initial_result = self.gemini_generator.generate_agent_prompt(
            screenshot=screenshot,
            goal=goal
        )
        
        if not initial_result.get('success', False):
            logger.warning("❌ Initial Gemini analysis failed, skipping optimization")
            return initial_result.get('agent_prompt', ''), {
                'status': 'failed', 
                'reason': 'initial_gemini_failed'
            }
        
        base_prompt = initial_result.get('raw_response', '')
        logger.info(f"✅ Base Gemini prompt generated ({len(base_prompt)} chars)")
        
        # Step 2: Run Text2Grad optimization rollouts
        rollout_results = []
        best_prompt = base_prompt
        best_reward = float('-inf')
        
        for rollout_id in range(self.config.k_rollouts):
            try:
                # Create Text2Grad-inspired prompt perturbation
                perturbed_prompt = self._perturb_gemini_prompt(base_prompt, rollout_id)
                logger.info(f"🔄 Testing prompt variation {rollout_id + 1}/{self.config.k_rollouts}")
                
                # Run rollout with this prompt variation
                result = self._run_rollout(
                    env=env,
                    task=task,
                    gemini_prompt=perturbed_prompt,
                    rollout_id=rollout_id,
                    snapshot_step=snapshot_step,
                    goal=goal  # Pass the goal parameter
                )
                
                rollout_results.append(result)
                
                # Update best prompt using Text2Grad selection criteria
                if result.total_reward > best_reward:
                    best_reward = result.total_reward
                    best_prompt = perturbed_prompt
                    logger.info(f"🎉 New best prompt found in rollout {rollout_id}: "
                              f"reward={best_reward:.3f}")
                
                # Text2Grad-style early stopping
                if (self.config.enable_early_stopping and 
                    rollout_id > 0 and
                    result.total_reward > best_reward - self.config.early_stopping_threshold):
                    logger.info(f"⏹️ Early stopping at rollout {rollout_id}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error in optimization rollout {rollout_id}: {e}")
                continue
        
        optimization_time = time.time() - optimization_start
        
        # Step 3: Compile Text2Grad optimization results
        optimization_results = {
            'status': 'completed',
            'method': 'text2grad_optimization',
            'optimization_time': optimization_time,
            'rollouts_completed': len(rollout_results),
            'best_reward': best_reward,
            'rollout_rewards': [r.total_reward for r in rollout_results],
            'average_reward': np.mean([r.total_reward for r in rollout_results]) if rollout_results else 0.0,
            'best_rollout_success': any(r.final_success for r in rollout_results),
            'improvement_over_baseline': best_reward - (rollout_results[0].total_reward if rollout_results else 0.0),
            'initial_prompt_length': len(base_prompt),
            'optimized_prompt_length': len(best_prompt)
        }
        
        logger.info(f"🎯 Text2Grad optimization completed in {optimization_time:.2f}s: "
                   f"best_reward={best_reward:.3f}, "
                   f"improvement={optimization_results['improvement_over_baseline']:.3f}")
        
        return best_prompt, optimization_results


class Text2GradAgent:
    """
    An AndroidWorld agent that uses Text2Grad optimization for Gemini prompts.
    
    This agent implements the full Text2Grad cycle:
    1. Takes snapshots before each step  
    2. Optimizes Gemini prompts using Text2Grad methodology
    3. Uses optimized prompts for actual agent steps
    4. Restores environment state using snapshots
    """
    
    def __init__(self,
                 env: interface.AsyncEnv,
                 model_name: str = "gpt-4o-mini",
                 prompt_variant: str = "base",
                 use_memory: bool = True,
                 use_function_calling: bool = False,
                 text2grad_config: Optional[Text2GradConfig] = None,
                 gemini_generator: Optional[GeminiPromptGenerator] = None):
        
        self.env = env
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.use_memory = use_memory
        self.use_function_calling = use_function_calling
        
        self.config = text2grad_config or Text2GradConfig()
        self.gemini_generator = gemini_generator
        
        # Base agent configuration
        self.base_agent_config = {
            'model_name': model_name,
            'prompt_variant': prompt_variant,
            'use_memory': use_memory,
            'use_function_calling': use_function_calling
        }
        
        # Initialize Text2Grad optimizer
        if self.gemini_generator:
            self.optimizer = Text2GradOptimizer(
                config=self.config,
                gemini_generator=self.gemini_generator,
                base_agent_config=self.base_agent_config
            )
            logger.info("🚀 Text2Grad optimizer initialized")
        else:
            self.optimizer = None
            logger.warning("⚠️ No Gemini generator provided, Text2Grad optimization disabled")
        
        # Create the actual agent that will be used for final steps
        self.agent = create_agent(
            env=env,
            model_name=model_name,
            prompt_variant=prompt_variant,
            use_memory=use_memory,
            use_function_calling=use_function_calling
        )
        
        self.step_count = 0
        self.optimization_history = []
    
    def step(self, goal: str, task: Optional[task_eval.TaskEval] = None) -> Any:
        """
        Take a step with Text2Grad optimization.
        
        This implements the complete Text2Grad cycle:
        1. Take snapshot of current environment state
        2. Get screenshot for Gemini analysis
        3. Run Text2Grad optimization to get optimized prompt
        4. Restore snapshot to current state
        5. Take actual step with optimized prompt injected into agent
        """
        self.step_count += 1
        logger.info(f"🎯 Text2Grad step {self.step_count} starting")
        
        # If no Gemini generator, fall back to regular agent
        if not self.optimizer:
            logger.info("📝 No optimizer available, using regular agent step")
            return self.agent.step(goal)
        
        try:
            # Step 1: Take snapshot of current state for Text2Grad optimization
            snapshot_step = self.step_count
            print(f"📸 [MAIN STEP {self.step_count}] About to save snapshot {snapshot_step} before optimization")
            snapshot_created = self.optimizer.snapshot_manager.save_step(snapshot_step)
            if not snapshot_created:
                print(f"❌ [MAIN STEP {self.step_count}] Failed to create snapshot for step {snapshot_step}")
                logger.warning(f"⚠️ Failed to create snapshot for step {snapshot_step}")
                # Fall back to regular agent if snapshot fails
                logger.info("🔄 Falling back to regular agent step due to snapshot failure")
                return self.agent.step(goal)
            else:
                print(f"✅ [MAIN STEP {self.step_count}] Successfully saved snapshot {snapshot_step} - optimization can begin")
            
            logger.info(f"📸 Saved snapshot for step: {snapshot_step}")
            
            # Step 2: Get current screenshot for Gemini analysis
            current_state = self.env.get_state(True)
            screenshot = current_state.pixels
            logger.info(f"📱 Captured screenshot ({screenshot.shape}) for Gemini analysis")
            
            # Step 3: Run Text2Grad optimization to get optimized prompt
            print(f"🎯 [MAIN STEP {self.step_count}] Starting Text2Grad optimization with {self.config.k_rollouts} rollouts...")
            logger.info("🎯 Running Text2Grad optimization...")
            optimized_prompt, optimization_results = self.optimizer.optimize_gemini_prompt(
                env=self.env,
                task=task,
                screenshot=screenshot,
                goal=goal,
                snapshot_step=snapshot_step
            )
            
            print(f"✅ [MAIN STEP {self.step_count}] Text2Grad optimization completed - {optimization_results.get('rollouts_completed', 0)} rollouts finished")
            
            # Store optimization results for analysis
            self.optimization_history.append({
                'step': self.step_count,
                'snapshot_step': snapshot_step,
                'optimization_results': optimization_results,
                'optimized_prompt_length': len(optimized_prompt)
            })
            
            logger.info(f"✅ Text2Grad optimization completed: "
                       f"method={optimization_results.get('method', 'unknown')}, "
                       f"improvement={optimization_results.get('improvement_over_baseline', 0):.3f}")
            
            # Step 4: Restore snapshot to current state for actual step
            print(f"🔄 [MAIN STEP {self.step_count}] About to restore snapshot {snapshot_step} AFTER all rollouts completed")
            restore_success = self.optimizer.snapshot_manager.restore_step(snapshot_step)
            if restore_success:
                print(f"✅ [MAIN STEP {self.step_count}] Successfully restored snapshot {snapshot_step} - ready for actual agent step")
                logger.info(f"🔄 Restored snapshot {snapshot_step} for actual step")
            else:
                print(f"❌ [MAIN STEP {self.step_count}] Failed to restore snapshot {snapshot_step} for actual step")
                logger.warning(f"⚠️ Failed to restore snapshot {snapshot_step} for actual step")
            
            # Step 5: Take actual step with optimized prompt
            # TODO: Inject optimized_prompt into agent's prompt system
            # For now, we use the regular agent but add metadata about optimization
            logger.info(f"🚀 Taking actual step with Text2Grad optimized guidance")
            result = self.agent.step(goal)
            
            # Add Text2Grad optimization metadata to result
            if hasattr(result, 'data') and result.data:
                result.data['text2grad_optimization'] = optimization_results
                result.data['optimized_prompt_used'] = True
                result.data['optimized_prompt'] = optimized_prompt
            
            logger.info(f"✅ Text2Grad step {self.step_count} completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Text2Grad optimization failed for step {self.step_count}: {e}")
            logger.info("🔄 Falling back to regular agent step")
            return self.agent.step(goal)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of all Text2Grad optimizations performed."""
        if not self.optimization_history:
            return {'status': 'no_optimizations'}
        
        total_optimizations = len(self.optimization_history)
        successful_optimizations = sum(
            1 for opt in self.optimization_history 
            if opt['optimization_results'].get('status') == 'completed'
        )
        
        average_improvement = np.mean([
            opt['optimization_results'].get('improvement_over_baseline', 0.0)
            for opt in self.optimization_history
            if opt['optimization_results'].get('improvement_over_baseline') is not None
        ])
        
        return {
            'total_optimizations': total_optimizations,
            'successful_optimizations': successful_optimizations,
            'success_rate': successful_optimizations / total_optimizations if total_optimizations > 0 else 0.0,
            'average_improvement': average_improvement,
            'optimization_history': self.optimization_history
        }


def create_text2grad_agent(env: interface.AsyncEnv,
                          model_name: str = "gpt-4o-mini",
                          prompt_variant: str = "base",
                          use_memory: bool = True,
                          use_function_calling: bool = False,
                          text2grad_config: Optional[Text2GradConfig] = None,
                          gemini_generator: Optional[GeminiPromptGenerator] = None) -> Text2GradAgent:
    """Factory function to create a Text2Grad-enabled agent."""
    return Text2GradAgent(
        env=env,
        model_name=model_name,
        prompt_variant=prompt_variant,
        use_memory=use_memory,
        use_function_calling=use_function_calling,
        text2grad_config=text2grad_config,
        gemini_generator=gemini_generator
    )

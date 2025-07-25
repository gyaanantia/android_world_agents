"""
Gemini-enhanced agent that can optionally use Gemini 2.5 Flash for UI analysis and prompt generation.

This module provides an enhanced T3A agent that can leverage Google's Gemini 2.5 Flash model
to analyze Android UI screenshots and generate contextual prompts. The agent maintains full
backward compatibility with the existing agent system.

Key Features:
- Optional Gemini integration with graceful fallback
- Visual UI analysis using Gemini 2.5 Flash
- Enhanced contextual prompts based on visual understanding
- Full compatibility with existing AndroidWorld framework
- Maintains all existing prompt variants (base, few-shot, reflective)
"""

import os
import sys
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Tuple
from PIL import Image

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from src import prompts
from src.prompts import get_gemini_enhanced_prompt, format_prompt
from src.function_calling_llm import create_llm
from textgrad_opt import create_textgrad_optimizer
from android_world.agents.m3a import _summarize_prompt
from src.agent import (
    EnhancedT3A, 
    create_agent,
    _generate_seeact_ui_elements_description
)
from android_world.env import (
    interface,
    json_action
)
from android_world.agents import (
    m3a_utils,
    base_agent,
    agent_utils,
    infer
)
 
# Try to import Gemini functionality with graceful fallback
try:
    from src.gemini_prompting import create_gemini_generator, GeminiPromptGenerator
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiPromptGenerator = None


class GeminiEnhancedT3A(EnhancedT3A):
    """T3A agent with optional Gemini 2.5 Flash integration for visual UI analysis.
    
    This agent extends the EnhancedT3A agent to optionally use Google's Gemini 2.5 Flash
    model for analyzing Android UI screenshots and generating enhanced contextual prompts.
    When Gemini is unavailable or disabled, it gracefully falls back to standard prompting.
    """
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        llm,
        prompt_variant: str = "base",
        use_memory: bool = True,
        use_function_calling: bool = False,
        use_gemini: bool = True,
        use_textgrad: bool = False,
        gemini_model: str = "gemini-2.5-flash",
        name: str = "GeminiEnhancedT3A",
    ):
        """Initialize Gemini-enhanced T3A agent.
        
        Args:
            env: The environment.
            llm: The text-only LLM for agent execution.
            prompt_variant: Type of prompting ("base", "few-shot", "reflective").
            use_memory: Whether to use memory (step history) in prompts.
            use_function_calling: Whether to use OpenAI function calling for structured output.
            use_gemini: Whether to use Gemini for visual UI analysis (if available).
            use_textgrad: Whether to use TextGrad optimization on Gemini output.
            gemini_model: The Gemini model to use for visual analysis.
            name: The agent name.
        """
        super().__init__(env, llm, prompt_variant, use_memory, use_function_calling, name)
        
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.use_textgrad = use_textgrad and use_gemini  # TextGrad requires Gemini
        self.gemini_generator = None
        self.textgrad_optimizer = None
        
        # Initialize Gemini generator if requested and available
        if self.use_gemini:
            try:
                # Always pass the API key explicitly to avoid configuration issues
                api_key = os.getenv("GOOGLE_API_KEY")
                self.gemini_generator = create_gemini_generator(
                    api_key=api_key,
                    model_name=gemini_model
                )
                if self.gemini_generator is None:
                    print("⚠️ Gemini dependencies not available, falling back to standard prompting")
                    self.use_gemini = False
                    self.use_textgrad = False
                else:
                    print(f"✅ Gemini {gemini_model} initialized for visual UI analysis")
                    
                    # Initialize TextGrad optimizer if requested
                    if self.use_textgrad:
                        try:
                            self.textgrad_optimizer = create_textgrad_optimizer(enabled=True)
                            if self.textgrad_optimizer.is_available():
                                print("✅ TextGrad optimizer initialized for gradient-based optimization")
                            else:
                                print("⚠️ TextGrad initialization failed, using standard Gemini analysis")
                                self.use_textgrad = False
                        except ImportError:
                            print("⚠️ TextGrad module not available")
                            self.use_textgrad = False
                        except Exception as e:
                            print(f"⚠️ Failed to initialize TextGrad: {e}")
                            self.use_textgrad = False
                            
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
                print("⚠️ Falling back to standard prompting")
                self.use_gemini = False
                self.use_textgrad = False
                self.gemini_generator = None
        else:
            if not GEMINI_AVAILABLE:
                print("⚠️ Gemini dependencies not installed, using standard prompting")
            else:
                print("ℹ️ Gemini disabled by user preference, using standard prompting")
    
    def _screenshot_to_pil(self, screenshot: np.ndarray) -> Image.Image:
        """Convert numpy screenshot to PIL Image.
        
        Args:
            screenshot: Screenshot as numpy array.
            
        Returns:
            PIL Image object.
        """
        # Ensure the screenshot is in the right format
        if screenshot.dtype != np.uint8:
            screenshot = (screenshot * 255).astype(np.uint8)
        
        # Handle different array shapes
        if len(screenshot.shape) == 3:
            if screenshot.shape[2] == 3:  # RGB
                return Image.fromarray(screenshot, 'RGB')
            elif screenshot.shape[2] == 4:  # RGBA
                return Image.fromarray(screenshot, 'RGBA')
        
        # Fallback: assume RGB
        return Image.fromarray(screenshot, 'RGB')
    
    def _get_gemini_enhanced_prompt(
        self, 
        goal: str, 
        screenshot: np.ndarray, 
        ui_elements_description: str,
        memory: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """Generate an enhanced prompt using Gemini visual analysis.
        
        Args:
            goal: The task goal.
            screenshot: The current UI screenshot.
            ui_elements_description: Description of UI elements.
            memory: Previous step memory.
            
        Returns:
            Tuple of (enhanced prompt string, whether TextGrad was used).
            Returns (standard prompt, False) if Gemini fails.
        """
        if not self.use_gemini or not self.gemini_generator:
            # Fallback to standard prompting
            return self._get_action_prompt(goal, ui_elements_description, memory), False
        
        try:
            # Generate Gemini visual analysis
            gemini_result = self.gemini_generator.generate_agent_prompt(
                screenshot=screenshot,
                goal=goal
            )
            
            # Extract the Gemini analysis from the result
            if gemini_result and gemini_result.get('success', False):
                # Get the raw Gemini response which should be in the ● format
                raw_response = gemini_result.get('raw_response', '')
                
                if raw_response and raw_response.strip():
                    # Apply TextGrad optimization if enabled
                    processed_analysis = raw_response.strip()
                    textgrad_used = False
                    if self.use_textgrad and self.textgrad_optimizer:
                        try:
                            processed_analysis = self.textgrad_optimizer.optimize_visual_analysis(
                                gemini_analysis=raw_response.strip(),
                                task_goal=goal,
                                ui_elements=ui_elements_description,
                                feedback_context={
                                    'goal': goal,
                                    'ui_elements': ui_elements_description,
                                    'memory': memory
                                }
                            )
                            print("✅ Applied TextGrad optimization to Gemini analysis")
                            textgrad_used = True
                        except Exception as e:
                            print(f"⚠️ TextGrad optimization failed: {e}, using original analysis")
                            processed_analysis = raw_response.strip()
                            textgrad_used = False
                    
                    # Use the Gemini-enhanced prompt template
                    gemini_prompt_template = prompts.get_gemini_enhanced_prompt(self.prompt_variant)
                    
                    # Format memory properly
                    if self.use_memory and memory:
                        formatted_memory = '\n'.join(memory)
                    elif self.use_memory:
                        formatted_memory = "You just started, no action has been performed yet."
                    else:
                        formatted_memory = "Memory is disabled for this session."
                    
                    # Format the enhanced prompt with processed Gemini analysis
                    enhanced_prompt = prompts.format_prompt(
                        gemini_prompt_template,
                        goal=goal,
                        ui_elements=ui_elements_description,
                        memory=formatted_memory,
                        gemini_analysis=processed_analysis
                    )
                    
                    if textgrad_used:
                        print("✅ Using Gemini-enhanced prompt with TextGrad optimization")
                    else:
                        print("✅ Using Gemini-enhanced prompt")
                    return enhanced_prompt, textgrad_used
                else:
                    print("⚠️ Gemini returned empty analysis, using standard prompting")
                    return self._get_action_prompt(goal, ui_elements_description, memory), False
            else:
                error_msg = gemini_result.get('error', 'Unknown error') if gemini_result else 'No result returned'
                print(f"⚠️ Gemini visual analysis failed: {error_msg}")
                print("⚠️ Falling back to standard prompting")
                return self._get_action_prompt(goal, ui_elements_description, memory), False
                
        except Exception as e:
            print(f"⚠️ Gemini enhanced prompting failed: {e}")
            print("⚠️ Falling back to standard prompting")
            return self._get_action_prompt(goal, ui_elements_description, memory), False
    
    def step(self, goal: str):
        """Override step method to optionally use Gemini-enhanced prompting.
        
        This method extends the parent step method to optionally use Gemini 2.5 Flash
        for visual UI analysis and prompt generation. If Gemini is unavailable or fails,
        it gracefully falls back to the standard prompting approach.
        
        Args:
            goal: The task goal string.
            
        Returns:
            AgentInteractionResult with step outcome.
        """
        # Use parent class step method but with our enhanced prompt generation
        step_data = {
            'before_screenshot': None,
            'after_screenshot': None,
            'before_element_list': None,
            'after_element_list': None,
            'action_prompt': None,
            'action_output': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
            'used_gemini': False,  # Track whether Gemini was used
            'used_textgrad': False,  # Track whether TextGrad optimization was used
        }
        
        # Get current state
        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        ui_elements = state.ui_elements
        screenshot = state.pixels
        
        # Generate UI element descriptions using SeeAct approach
        
        
        before_element_list = _generate_seeact_ui_elements_description(
            ui_elements,
            logical_screen_size,
        )
        
        # Generate memory (step history)
        memory = [
            'Step ' + str(i + 1) + ': ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ]
        
        # Use Gemini-enhanced prompt generation if available
        if self.use_gemini and self.gemini_generator:
            action_prompt, textgrad_used = self._get_gemini_enhanced_prompt(
                goal,
                screenshot,
                before_element_list,
                memory
            )
            step_data['used_gemini'] = True
            step_data['used_textgrad'] = textgrad_used
        else:
            # Use standard prompt generation
            action_prompt = self._get_action_prompt(
                goal,
                before_element_list,
                memory
            )
            step_data['used_gemini'] = False
            step_data['used_textgrad'] = False
        
        # Save state data
        step_data['before_screenshot'] = screenshot.copy()
        step_data['before_element_list'] = ui_elements
        step_data['action_prompt'] = action_prompt
        
        # Continue with the rest of the step method from parent class
        # Get LLM response
        action_output, is_safe, raw_response = self.llm.predict(action_prompt)
        
        # Handle safety check
        if is_safe == False:
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER} -- {action_output}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""
        
        if not raw_response:
            raise RuntimeError('Error calling LLM in action selection phase.')
        
        step_data['action_output'] = action_output
        step_data['action_raw_response'] = raw_response
        
        # Parse the response
        if self.use_function_calling:
            reason, action = self._parse_function_calling_output(action_output)
        else:
            reason, action = m3a_utils.parse_reason_action_output(action_output)
        
        # If the output is not in the right format, add it to step summary
        if (not reason) or (not action):
            print('❌ Action prompt output is not in the correct format.')
            print(f'❌ Raw output was: {action_output[:500]}...')
            print(f'❌ Parsed reason: {reason}')
            print(f'❌ Parsed action: {action}')
            step_data['summary'] = (
                'Output for action selection is not in the correct format, so no'
                ' action is performed.'
            )
            self.history.append(step_data)
            
            return base_agent.AgentInteractionResult(False, step_data)
        
        print('Action: ' + action)
        print('Reason: ' + reason)
        
        # Convert action to JSON
        try:
            converted_action = json_action.JSONAction(
                **agent_utils.extract_json(action),
            )
        except Exception as e:
            print('Failed to convert the output to a valid action.')
            print(str(e))
            step_data['summary'] = (
                'Can not parse the output to a valid action. Please make sure to pick'
                ' the action from the list with the correct json format!'
            )
            self.history.append(step_data)
            
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Validate index for certain actions
        if converted_action.action_type in ['click', 'long_press', 'input_text']:
            if converted_action.index is not None and converted_action.index >= len(ui_elements):
                print('Index out of range.')
                step_data['summary'] = (
                    'The parameter index is out of range. Remember the index must be in'
                    ' the UI element list!'
                )
                self.history.append(step_data)   
                return base_agent.AgentInteractionResult(False, step_data)
        
        # Handle status actions
        if converted_action.action_type == 'status':
            if converted_action.goal_status == 'infeasible':
                print('🛑 Agent stopped since it thinks mission impossible.')
                step_data['summary'] = 'Agent thinks the task is infeasible.'
            elif converted_action.goal_status == 'complete':
                print('✅ Agent stopped since it thinks the task is complete.')
                step_data['summary'] = 'Agent thinks the request has been completed.'
            else:
                print(f'⚠️ Agent stopped with status: {converted_action.goal_status}')
                step_data['summary'] = f'Agent stopped with status: {converted_action.goal_status}'
            
            self.history.append(step_data)
            
            return base_agent.AgentInteractionResult(True, step_data)
        
        # Handle answer actions
        if converted_action.action_type == 'answer':
            print('Agent answered with: ' + converted_action.text)
        
        # Execute the action
        try:
            self.env.execute_action(converted_action)
        except Exception as e:
            print('Some error happened executing the action ', converted_action.action_type)
            print(str(e))
            step_data['summary'] = (
                'Some error happened executing the action ' + converted_action.action_type
            )
            self.history.append(step_data)
            
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Get post-action state
        state = self.get_post_transition_state()
        ui_elements = state.ui_elements
        
        after_element_list = _generate_seeact_ui_elements_description(
            ui_elements,
            self.env.logical_screen_size,
        )
        
        # Save post-action state
        step_data['after_screenshot'] = state.pixels.copy()
        step_data['after_element_list'] = ui_elements
        
        # Generate summary using regular text LLM (not function calling)
        
        summary_prompt = _summarize_prompt(
            goal,
            action,
            reason,
            before_element_list,
            after_element_list,
        )
        
        summary, is_safe, raw_response = self.summary_llm.predict(summary_prompt)
        
        if is_safe == False:
            summary = """Summary triggered LLM safety classifier."""
        
        step_data['summary_prompt'] = summary_prompt
        step_data['summary'] = (
            f'Action selected: {action}. {summary}'
            if raw_response
            else 'Error calling LLM in summarization phase.'
        )
        print('Summary: ' + summary)
        step_data['summary_raw_response'] = raw_response
        
        # Add reflection for reflective agent
        if self.prompt_variant == "reflective":
            self.add_reflection(f"Goal: {goal}, Action: {action}, Result: {summary}")
        
        self.history.append(step_data)
        
        return base_agent.AgentInteractionResult(False, step_data)
    
    def get_gemini_status(self) -> Dict[str, Any]:
        """Get current Gemini integration status.
        
        Returns:
            Dictionary with Gemini status information.
        """
        return {
            'gemini_available': GEMINI_AVAILABLE,
            'gemini_enabled': self.use_gemini,
            'gemini_generator_initialized': self.gemini_generator is not None,
            'model': self.gemini_generator.model_name if self.gemini_generator else None,
        }


def create_gemini_enhanced_agent(
    env: interface.AsyncEnv,
    model_name: str = "gpt-4o-mini",
    prompt_variant: str = "base",
    use_memory: bool = True,
    use_function_calling: bool = False,
    use_gemini: bool = True,
    use_textgrad: bool = False,
    gemini_model: str = "gemini-2.5-flash"
) -> GeminiEnhancedT3A:
    """Factory function to create a Gemini-enhanced T3A agent.
    
    Args:
        env: The environment.
        model_name: The LLM model name for agent execution.
        prompt_variant: The prompting variant ("base", "few-shot", "reflective").
        use_memory: Whether to use memory (step history) in prompts.
        use_function_calling: Whether to use OpenAI function calling for structured output.
        use_gemini: Whether to use Gemini for visual UI analysis (if available).
        use_textgrad: Whether to use TextGrad optimization on Gemini output.
        gemini_model: The Gemini model to use for visual analysis.
        
    Returns:
        A GeminiEnhancedT3A agent instance.
    """
    # Create the LLM wrapper
    if use_function_calling:
        llm = create_llm(model_name, use_function_calling=True)
    else:
        llm = infer.Gpt4Wrapper(model_name)
    
    return GeminiEnhancedT3A(
        env=env,
        llm=llm,
        prompt_variant=prompt_variant,
        use_memory=use_memory,
        use_function_calling=use_function_calling,
        use_gemini=use_gemini,
        use_textgrad=use_textgrad,
        gemini_model=gemini_model
    )


def create_standard_agent_with_gemini_fallback(
    env: interface.AsyncEnv,
    model_name: str = "gpt-4o-mini",
    prompt_variant: str = "base",
    use_memory: bool = True,
    use_function_calling: bool = False,
    try_gemini: bool = True,
    gemini_model: str = "gemini-1.5-flash"
):
    """Create an agent that tries to use Gemini but gracefully falls back to standard agent.
    
    This function attempts to create a Gemini-enhanced agent, but if Gemini is unavailable
    or fails to initialize, it creates a standard EnhancedT3A agent instead.
    
    Args:
        env: The environment.
        model_name: The LLM model name for agent execution.
        prompt_variant: The prompting variant ("base", "few-shot", "reflective").
        use_memory: Whether to use memory (step history) in prompts.
        use_function_calling: Whether to use OpenAI function calling for structured output.
        try_gemini: Whether to attempt Gemini integration.
        gemini_model: The Gemini model to use for visual analysis.
        
    Returns:
        Either a GeminiEnhancedT3A or EnhancedT3A agent instance.
    """
    if try_gemini and GEMINI_AVAILABLE:
        try:
            return create_gemini_enhanced_agent(
                env=env,
                model_name=model_name,
                prompt_variant=prompt_variant,
                use_memory=use_memory,
                use_function_calling=use_function_calling,
                use_gemini=True,
                gemini_model=gemini_model
            )
        except Exception as e:
            print(f"⚠️ Failed to create Gemini-enhanced agent: {e}")
            print("⚠️ Falling back to standard agent")
    
    # Fallback to standard agent
    return create_agent(
        env=env,
        model_name=model_name,
        prompt_variant=prompt_variant,
        use_memory=use_memory,
        use_function_calling=use_function_calling
    )

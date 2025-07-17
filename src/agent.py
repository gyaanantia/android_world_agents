"""Enhanced T3A agent with few-shot and self-reflection capabilities."""

import os
import sys
from pathlib import Path

# Add android_world to Python path
android_world_path = Path(__file__).parent.parent / "android_world"
if android_world_path.exists() and str(android_world_path) not in sys.path:
    sys.path.insert(0, str(android_world_path))

import openai
from typing import Dict, List, Optional

from android_world.agents import t3a
from android_world.agents import infer
from android_world.agents import base_agent
from android_world.env import interface

# Import prompts module
try:
    from .prompts import get_prompt_template, format_prompt
except ImportError:
    # Fallback for direct execution
    import prompts
    get_prompt_template = prompts.get_prompt_template
    format_prompt = prompts.format_prompt

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EnhancedT3A(t3a.T3A):
    """T3A agent with enhanced prompting capabilities."""
    
    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.LlmWrapper,
        prompt_variant: str = "base",
        name: str = "EnhancedT3A",
    ):
        """Initialize enhanced T3A agent.
        
        Args:
            env: The environment.
            llm: The text-only LLM.
            prompt_variant: Type of prompting ("base", "few-shot", "reflective").
            name: The agent name.
        """
        super().__init__(env, llm, name)
        self.prompt_variant = prompt_variant
        self.reflection_history = []
        
        # Load appropriate prompt
        try:
            self.system_prompt = get_prompt_template(self.prompt_variant)
        except ValueError:
            # Fallback to base prompt if agent type is not recognized
            self.system_prompt = get_prompt_template("base")
    
    def _get_action_prompt(self, goal: str, ui_elements_description: str, history: Optional[List[str]] = None) -> str:
        """Generate action prompt based on variant."""
        # Format history properly
        if history:
            formatted_history = '\n'.join([f'Step {i+1}: {step}' for i, step in enumerate(history)])
        else:
            formatted_history = "You just started, no action has been performed yet."
        
        if self.prompt_variant == "base":
            return format_prompt(
                self.system_prompt, 
                goal=goal, 
                ui_elements=ui_elements_description,
                history=formatted_history
            )
        elif self.prompt_variant == "few_shot":
            return self._enhance_with_few_shot(goal, ui_elements_description, formatted_history)
        elif self.prompt_variant == "reflective":
            return self._enhance_with_reflection(goal, ui_elements_description, formatted_history)
        else:
            return format_prompt(
                self.system_prompt, 
                goal=goal, 
                ui_elements=ui_elements_description,
                history=formatted_history
            )
    
    def _enhance_with_few_shot(self, goal: str, ui_elements: str, history: str) -> str:
        """Add few-shot examples to the prompt."""
        return format_prompt(
            self.system_prompt, 
            goal=goal, 
            ui_elements=ui_elements,
            history=history
        )
    
    def _enhance_with_reflection(self, goal: str, ui_elements: str, history: str) -> str:
        """Add reflection to the prompt."""
        reflection_context = ""
        if self.reflection_history:
            reflection_context = "\n## Previous Reflections:\n" + "\n".join(self.reflection_history[-3:])
        
        return format_prompt(
            self.system_prompt, 
            goal=goal, 
            ui_elements=ui_elements,
            history=history,
            reflection_context=reflection_context
        )
    
    def add_reflection(self, reflection: str):
        """Add a reflection to the agent's history."""
        self.reflection_history.append(reflection)
    
    def reset(self, go_home_on_reset: bool = False):
        """Reset the agent state."""
        super().reset(go_home_on_reset)
        if self.prompt_variant == "reflective":
            # Keep some reflection history across episodes for learning
            self.reflection_history = self.reflection_history[-5:]  # Keep last 5
    
    def step(self, goal: str):
        """Override step method to use enhanced prompting."""
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
        }
        
        # Get current state
        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        ui_elements = state.ui_elements
        
        # Generate UI element descriptions
        from android_world.agents.t3a import _generate_ui_elements_description_list_full
        before_element_list = _generate_ui_elements_description_list_full(
            ui_elements,
            logical_screen_size,
        )
        
        # Generate history
        history = [
            'Step ' + str(i + 1) + ': ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ]
        
        # Use our enhanced prompt generation
        action_prompt = self._get_action_prompt(
            goal,
            before_element_list,
            history
        )
        
        # Save state data
        step_data['before_screenshot'] = state.pixels.copy()
        step_data['before_element_list'] = ui_elements
        step_data['action_prompt'] = action_prompt
        
        # Get LLM response
        action_output, is_safe, raw_response = self.llm.predict(action_prompt)
        
        # Handle safety check
        if is_safe == False:
            from android_world.agents import m3a_utils
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""
        
        if not raw_response:
            raise RuntimeError('Error calling LLM in action selection phase.')
        
        step_data['action_output'] = action_output
        step_data['action_raw_response'] = raw_response
        
        # Parse the response
        from android_world.agents import m3a_utils
        reason, action = m3a_utils.parse_reason_action_output(action_output)
        
        # If the output is not in the right format, add it to step summary
        if (not reason) or (not action):
            print('Action prompt output is not in the correct format.')
            step_data['summary'] = (
                'Output for action selection is not in the correct format, so no'
                ' action is performed.'
            )
            self.history.append(step_data)
            
            from android_world.agents import base_agent
            return base_agent.AgentInteractionResult(False, step_data)
        
        print('Action: ' + action)
        print('Reason: ' + reason)
        
        # Convert action to JSON
        try:
            from android_world.agents import agent_utils
            from android_world.env import json_action
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
            
            from android_world.agents import base_agent
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Validate index for certain actions
        if converted_action.action_type in ['click', 'long-press', 'input-text']:
            if converted_action.index is not None and converted_action.index >= len(ui_elements):
                print('Index out of range.')
                step_data['summary'] = (
                    'The parameter index is out of range. Remember the index must be in'
                    ' the UI element list!'
                )
                self.history.append(step_data)
                
                from android_world.agents import base_agent
                return base_agent.AgentInteractionResult(False, step_data)
        
        # Handle status actions
        if converted_action.action_type == 'status':
            if converted_action.goal_status == 'infeasible':
                print('Agent stopped since it thinks mission impossible.')
            step_data['summary'] = 'Agent thinks the request has been completed.'
            self.history.append(step_data)
            
            from android_world.agents import base_agent
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
            
            from android_world.agents import base_agent
            return base_agent.AgentInteractionResult(False, step_data)
        
        # Get post-action state
        state = self.get_post_transition_state()
        ui_elements = state.ui_elements
        
        after_element_list = _generate_ui_elements_description_list_full(
            ui_elements,
            self.env.logical_screen_size,
        )
        
        # Save post-action state
        step_data['after_screenshot'] = state.pixels.copy()
        step_data['after_element_list'] = ui_elements
        
        # Generate summary
        from android_world.agents.t3a import _summarize_prompt
        summary_prompt = _summarize_prompt(
            goal,
            action,
            reason,
            before_element_list,
            after_element_list,
        )
        
        summary, is_safe, raw_response = self.llm.predict(summary_prompt)
        if is_safe == False:
            summary = """Summary triggered LLM safety classifier."""
        
        step_data['summary_prompt'] = summary_prompt
        step_data['summary'] = (
            f'Action selected: {action}. {summary}'
            if raw_response
            else 'Error calling LLM in summerization phase.'
        )
        print('Summary: ' + summary)
        step_data['summary_raw_response'] = raw_response
        
        # Add reflection for reflective agent
        if self.prompt_variant == "reflective":
            self.add_reflection(f"Goal: {goal}, Action: {action}, Result: {summary}")
        
        self.history.append(step_data)
        
        from android_world.agents import base_agent
        return base_agent.AgentInteractionResult(False, step_data)
def create_agent(
    env: interface.AsyncEnv,
    model_name: str = "gpt-4",
    prompt_variant: str = "base"
) -> EnhancedT3A:
    """Factory function to create an enhanced T3A agent.
    
    Args:
        env: The environment.
        model_name: The LLM model name.
        prompt_variant: The prompting variant ("base", "few-shot", "reflective").
        
    Returns:
        An EnhancedT3A agent instance.
    """
    llm = infer.Gpt4Wrapper(model_name)
    return EnhancedT3A(env, llm, prompt_variant)

"""Enhanced LLM wrapper with OpenAI function calling support."""

import json
import openai
from typing import Dict, List, Optional, Tuple, Any
from android_world.agents import infer


class FunctionCallingLLM(infer.LlmWrapper):
    """LLM wrapper that uses OpenAI function calling for structured output."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize function calling LLM.
        
        Args:
            model_name: OpenAI model name that supports function calling.
        """
        super().__init__()
        self.model_name = model_name
        self.client = openai.OpenAI()
        
        # Define the function schema for Android actions
        self.function_schema = {
            "name": "execute_android_action",
            "description": "Execute an action on the Android device",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning for why this action is being taken"
                    },
                    "action": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "enum": [
                                    "status", "answer", "click", "long_press", 
                                    "input_text", "keyboard_enter", "navigate_home", 
                                    "navigate_back", "scroll", "open_app", "wait",
                                    "double_tap", "swipe", "unknown"
                                ],
                                "description": "The type of action to perform"
                            },
                            "goal_status": {
                                "type": "string",
                                "enum": ["complete", "infeasible"],
                                "description": "Goal status (only for status action_type)"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text content (for answer or input_text actions)"
                            },
                            "index": {
                                "type": "integer",
                                "description": "UI element index (for click, long_press, input_text, scroll actions)"
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["up", "down", "left", "right"],
                                "description": "Scroll direction (for scroll action)"
                            },
                            "app_name": {
                                "type": "string",
                                "description": "App name to open (for open_app action)"
                            },
                            "x": {
                                "type": "integer",
                                "description": "X coordinate for click/tap actions (alternative to index)"
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate for click/tap actions (alternative to index)"
                            },
                            "keycode": {
                                "type": "string",
                                "description": "Android keycode for special key actions (e.g., KEYCODE_ENTER)"
                            },
                            "clear_text": {
                                "type": "boolean",
                                "description": "Whether to clear existing text before input (for input_text actions)"
                            }
                        },
                        "required": ["action_type"],
                        "additionalProperties": False
                    }
                },
                "required": ["reasoning", "action"],
                "additionalProperties": False
            }
        }
    
    def predict(self, prompt: str) -> Tuple[str, bool, str]:
        """Make prediction using function calling.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            Tuple of (formatted_output, is_safe, raw_response).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                functions=[self.function_schema],
                function_call={"name": "execute_android_action"},
                temperature=0.0
            )
            
            # Extract function call
            message = response.choices[0].message
            if message.function_call:
                function_args = json.loads(message.function_call.arguments)
                reasoning = function_args.get("reasoning", "")
                action_data = function_args.get("action", {})
                
                # Format output to match expected format
                formatted_output = f"Reason: {reasoning}\nAction: {json.dumps(action_data)}"
                
                return formatted_output, True, json.dumps(response.model_dump())
            else:
                # Fallback if no function call (shouldn't happen with function_call parameter)
                content = message.content or ""
                return content, True, json.dumps(response.model_dump())
                
        except Exception as e:
            error_msg = f"Error in function calling: {str(e)}"
            return error_msg, False, error_msg


class HybridLLM(infer.LlmWrapper):
    """Hybrid LLM that can switch between function calling and regular text generation."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", use_function_calling: bool = True):
        """Initialize hybrid LLM.
        
        Args:
            model_name: OpenAI model name.
            use_function_calling: Whether to use function calling for structured output.
        """
        super().__init__()
        self.use_function_calling = use_function_calling
        
        if use_function_calling:
            self.llm = FunctionCallingLLM(model_name)
        else:
            self.llm = infer.Gpt4Wrapper(model_name)
    
    def predict(self, prompt: str) -> Tuple[str, bool, str]:
        """Make prediction using the configured LLM approach."""
        return self.llm.predict(prompt)


def create_llm(model_name: str = "gpt-4o-mini", use_function_calling: bool = True) -> infer.LlmWrapper:
    """Factory function to create LLM with optional function calling.
    
    Args:
        model_name: OpenAI model name.
        use_function_calling: Whether to enable function calling.
        
    Returns:
        LLM wrapper instance.
    """
    return HybridLLM(model_name, use_function_calling)

"""Episode evaluation and result recording."""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class StepRecord:
    """Record of a single step in an episode."""
    step_num: int
    state: Dict[str, Any]
    action: Optional[Dict[str, Any]]
    agent_data: Dict[str, Any]
    timestamp: float


class EpisodeEvaluator:
    """Evaluates and records episode results."""
    
    def __init__(
        self,
        task_name: str,
        goal: str,
        model_name: str,
        prompt_variant: str,
        max_steps: int
    ):
        self.task_name = task_name
        self.goal = goal
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.max_steps = max_steps
        
        self.steps: List[StepRecord] = []
        self.start_time = time.time()
    
    def record_step(
        self,
        step_num: int,
        state: Any,
        action: Optional[Dict[str, Any]],
        agent_data: Dict[str, Any]
    ):
        """Record a single step."""
        # Convert state to serializable format
        state_dict = self._state_to_dict(state)
        
        # Convert agent_data to serializable format (it may contain numpy arrays)
        agent_data_dict = self._serialize_agent_data(agent_data)
        
        step_record = StepRecord(
            step_num=step_num,
            state=state_dict,
            action=action,
            agent_data=agent_data_dict,
            timestamp=time.time()
        )
        
        self.steps.append(step_record)
    
    def _serialize_agent_data(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize agent data, handling numpy arrays and other complex objects."""
        serialized = {}
        for key, value in agent_data.items():
            if key == 'before_screenshot' and isinstance(value, np.ndarray):
                # Exclude screenshot data like we do with state pixels
                serialized[key] = "<screenshot_excluded>"
            elif isinstance(value, np.ndarray):
                # Handle other numpy arrays
                if value.size > 100:
                    serialized[key] = f"<large_array: {value.shape} {value.dtype}>"
                else:
                    serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                                   np.uint8, np.uint16, np.uint32, np.uint64)):
                serialized[key] = int(value)
            elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                serialized[key] = float(value)
            elif isinstance(value, np.bool_):
                serialized[key] = bool(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_agent_data(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = [self._serialize_agent_data(item) if isinstance(item, dict) else self._serialize_value_simple(item) for item in value]
            else:
                # For other types, try JSON serialization test
                try:
                    json.dumps(value)
                    serialized[key] = value
                except (TypeError, ValueError):
                    serialized[key] = str(value)
        return serialized
    
    def _serialize_value_simple(self, value):
        """Simple serialization for values that might be complex objects."""
        if isinstance(value, np.ndarray):
            if value.size > 100:
                return f"<large_array: {value.shape} {value.dtype}>"
            return value.tolist()
        elif isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                               np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif hasattr(value, '__dict__'):
            # Handle objects like UIElement by converting to dict representation
            return {k: self._serialize_value_simple(v) for k, v in value.__dict__.items() 
                   if not k.startswith('_')}
        else:
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)
    
    def _state_to_dict(self, state) -> Dict[str, Any]:
        """Convert environment state to JSON-serializable dictionary.
        
        Excludes screenshot pixels to avoid huge JSON files.
        """
        
        def _serialize_value(value, key=None):
            """Recursively serialize values, handling numpy arrays."""
            # Skip screenshot pixels - they're always (2400, 1080, 3) and not needed for analysis
            if key == 'pixels' and isinstance(value, np.ndarray):
                return "<screenshot_excluded>"
            # Skip protobuf forest objects as they're not JSON serializable
            elif key == 'forest' and hasattr(value, 'DESCRIPTOR'):
                return f"<protobuf: {type(value).__name__}>"
            elif isinstance(value, np.ndarray):
                # For other numpy arrays (small ones), convert to list
                if value.size > 100:  # Skip large arrays
                    return f"<large_array: {value.shape} {value.dtype}>"
                return value.tolist()
            elif isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                                   np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, dict):
                return {k: _serialize_value(v, k) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [_serialize_value(item) for item in value]
            elif hasattr(value, '__dict__') and not hasattr(value, 'DESCRIPTOR'):
                # Convert objects with __dict__ to dictionary, but skip protobuf objects
                return {k: _serialize_value(v, k) for k, v in value.__dict__.items()
                       if not k.startswith('_')}
            elif hasattr(value, 'DESCRIPTOR'):
                # Handle protobuf objects
                return f"<protobuf: {type(value).__name__}>"
            else:
                # For primitive types or unknown objects, try to convert to string
                try:
                    # Test if it's JSON serializable
                    json.dumps(value)
                    return value
                except (TypeError, ValueError):
                    return str(value)
        
        if hasattr(state, '__dict__'):
            return _serialize_value(state.__dict__)
        else:
            return _serialize_value(state)
    
    def generate_results(
        self,
        success: bool,
        steps_taken: int,
        evaluation_time: float,
        final_state: Any
    ) -> Dict[str, Any]:
        """Generate comprehensive results dictionary."""
        
        # Extract prompts and responses from agent data
        prompts = []
        responses = []
        actions = []
        
        for step in self.steps:
            if 'action_prompt' in step.agent_data:
                prompts.append(step.agent_data['action_prompt'])
            if 'action_output' in step.agent_data:
                responses.append(step.agent_data['action_output'])
            if step.action:
                actions.append(step.action)
        
        # Calculate step timing
        step_timings = []
        for i, step in enumerate(self.steps):
            if i == 0:
                step_time = step.timestamp - self.start_time
            else:
                step_time = step.timestamp - self.steps[i-1].timestamp
            step_timings.append(step_time)
        
        # Calculate Gemini and TextGrad usage statistics
        gemini_usage_count = 0
        textgrad_usage_count = 0
        for step in self.steps:
            if step.agent_data.get('used_gemini', False):
                gemini_usage_count += 1
            if step.agent_data.get('used_textgrad', False):
                textgrad_usage_count += 1
        
        results = {
            # Basic episode info
            "task_name": self.task_name,
            "goal": self.goal,
            "model_name": self.model_name,
            "prompt_variant": self.prompt_variant,
            "max_steps": self.max_steps,
            
            # Results
            "success": success,
            "steps_taken": steps_taken,
            "evaluation_time": evaluation_time,
            
            # Detailed data
            "prompts": prompts,
            "responses": responses,
            "actions": actions,
            "step_timings": step_timings,
            
            # State information
            "initial_state": self._state_to_dict(self.steps[0].state) if self.steps else None,
            "final_state": self._state_to_dict(final_state),
            
            # Full step records (for detailed analysis)
            "step_records": [
                {
                    "step_num": step.step_num,
                    "state": self._state_to_dict(step.state),
                    "action": step.action,
                    "agent_data": step.agent_data,
                    "timestamp": step.timestamp
                }
                for step in self.steps
            ],
            
            # Metadata
            "evaluation_timestamp": time.time(),
            "total_ui_elements": sum(
                len(step.state.get("ui_elements", []))
                for step in self.steps
            ),
            
            # Usage statistics
            "gemini_usage_count": gemini_usage_count,
            "textgrad_usage_count": textgrad_usage_count,
            "gemini_usage_percentage": round(100 * gemini_usage_count / max(1, steps_taken), 1),
            "textgrad_usage_percentage": round(100 * textgrad_usage_count / max(1, steps_taken), 1)
        }
        
        return results
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the episode."""
        if not self.steps:
            return {}
        
        return {
            "total_steps": len(self.steps),
            "avg_step_time": sum(
                step.timestamp - self.steps[i-1].timestamp
                for i, step in enumerate(self.steps[1:], 1)
            ) / max(1, len(self.steps) - 1),
            "total_prompts": sum(
                1 for step in self.steps 
                if 'action_prompt' in step.agent_data
            ),
            "total_ui_elements": sum(
                len(step.state.get("ui_elements", []))
                for step in self.steps
            )
        }


def step_accuracy(predicted_actions: List[str], gold_actions: List[str]) -> float:
    """Calculate step-level accuracy."""
    if not predicted_actions or not gold_actions:
        return 0.0
    
    correct = 0
    for pred, gold in zip(predicted_actions, gold_actions):
        if pred.strip() == gold.strip():
            correct += 1
    
    return correct / len(predicted_actions)


def episode_success(predicted_actions: List[str], gold_actions: List[str]) -> bool:
    """Determine if episode was successful."""
    if not predicted_actions:
        return False
    
    # Check if the last action indicates completion
    last_action = predicted_actions[-1].strip().lower()
    return any(completion_word in last_action for completion_word in [
        "complete", "done", "finished", "success", "accomplished"
    ])

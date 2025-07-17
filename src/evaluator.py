"""Episode evaluation and result recording."""

import json
import time
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
        
        step_record = StepRecord(
            step_num=step_num,
            state=state_dict,
            action=action,
            agent_data=agent_data,
            timestamp=time.time()
        )
        
        self.steps.append(step_record)
    
    def _state_to_dict(self, state) -> Dict[str, Any]:
        """Convert state to JSON-serializable dictionary."""
        if hasattr(state, 'ui_elements'):
            ui_elements = []
            for elem in state.ui_elements:
                elem_dict = {
                    "text": elem.text,
                    "content_desc": elem.content_description,
                    "class": elem.class_name,
                    "resource_id": getattr(elem, 'resource_id', None),
                    "clickable": getattr(elem, 'clickable', False),
                    "enabled": getattr(elem, 'enabled', True)
                }
                
                if hasattr(elem, 'bbox_pixels'):
                    elem_dict["bbox_pixels"] = [
                        elem.bbox_pixels.x_min,
                        elem.bbox_pixels.y_min,
                        elem.bbox_pixels.x_max,
                        elem.bbox_pixels.y_max
                    ]
                
                ui_elements.append(elem_dict)
            
            return {
                "ui_elements": ui_elements,
                "screen_width": getattr(state, 'screen_width', None),
                "screen_height": getattr(state, 'screen_height', None)
            }
        
        # Fallback for unknown state format
        return {"raw_state": str(state)}
    
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
            "initial_state": self.steps[0].state if self.steps else None,
            "final_state": self._state_to_dict(final_state),
            
            # Full step records (for detailed analysis)
            "step_records": [
                {
                    "step_num": step.step_num,
                    "state": step.state,
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
            )
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

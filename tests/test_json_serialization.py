#!/usr/bin/env python3
"""Test JSON serialization without NumpyEncoder."""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluator import EpisodeEvaluator

def test_json_serialization():
    """Test that results can be serialized to JSON without NumpyEncoder."""
    
    # Create a simple evaluator
    evaluator = EpisodeEvaluator(
        task_name="test_task",
        goal="test goal",
        model_name="test_model",
        prompt_variant="base",
        max_steps=1
    )
    
    # Create mock results
    results = {
        "task_name": "test_task",
        "success": True,
        "steps_taken": 1,
        "evaluation_time": 5.0,
        "prompts": ["test prompt"],
        "responses": ["test response"],
        "actions": ["test action"],
        "final_state": "<screenshot_excluded>",
        "step_records": [
            {
                "step_num": 1,
                "state": "<screenshot_excluded>",
                "action": "test action",
                "agent_data": {},
                "timestamp": 1234567890.0
            }
        ]
    }
    
    try:
        # Test JSON serialization
        json_str = json.dumps(results, indent=2)
        print("✅ JSON serialization successful!")
        print(f"JSON size: {len(json_str)} characters")
        
        # Test deserialization
        parsed = json.loads(json_str)
        print("✅ JSON deserialization successful!")
        print(f"Parsed keys: {list(parsed.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_json_serialization()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Test actual AndroidWorld state serialization."""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from evaluator import EpisodeEvaluator
from android_world.env import env_launcher
from utils import find_adb_directory

def test_full_serialization():
    """Test serialization with actual AndroidWorld state."""
    print("🧪 Testing full AndroidWorld state serialization...")
    
    try:
        # Setup environment
        env = env_launcher.load_and_setup_env(
            console_port=5554, 
            adb_path=find_adb_directory()
        )
        env.reset(go_home=True)
        state = env.get_state(True)
        
        print(f"✅ Got AndroidWorld state: {type(state)}")
        print(f"   State attributes: {list(state.__dict__.keys())}")
        
        # Test state serialization
        evaluator = EpisodeEvaluator(
            task_name="test",
            goal="test goal",
            model_name="test",
            prompt_variant="base",
            max_steps=1
        )
        
        # Test _state_to_dict
        serialized_state = evaluator._state_to_dict(state)
        print(f"✅ _state_to_dict successful: {type(serialized_state)}")
        print(f"   Serialized keys: {list(serialized_state.keys())}")
        
        # Test JSON serialization of state
        json_str = json.dumps(serialized_state)
        print(f"✅ State JSON serialization successful: {len(json_str)} chars")
        
        # Test full results generation
        results = evaluator.generate_results(
            success=True,
            steps_taken=1,
            evaluation_time=1.0,
            final_state=state
        )
        print(f"✅ generate_results successful: {type(results)}")
        
        # Test full results JSON serialization
        json_str = json.dumps(results)
        print(f"✅ Full results JSON serialization successful: {len(json_str)} chars")
        
        env.close()
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_full_serialization()
    sys.exit(0 if success else 1)

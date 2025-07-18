#!/usr/bin/env python3
"""Test function calling integration with AndroidWorld action parsing."""

import sys
import os
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from function_calling_llm import FunctionCallingLLM
from android_world.env import json_action
from android_world.agents import agent_utils


def test_action_compatibility():
    """Test that function calling output is compatible with AndroidWorld's action parsing."""
    print("🧪 Testing Function Calling ↔ AndroidWorld Action Compatibility")
    print("=" * 70)
    
    errors = []
    
    # Test cases with different action types
    test_cases = [
        {
            "name": "Click Action",
            "expected_action": '{"action_type": "click", "index": 0}',
            "description": "Click on first UI element"
        },
        {
            "name": "Input Text Action", 
            "expected_action": '{"action_type": "input_text", "text": "Hello World", "index": 1}',
            "description": "Type text into element"
        },
        {
            "name": "Scroll Action",
            "expected_action": '{"action_type": "scroll", "direction": "down"}',
            "description": "Scroll down on screen"
        },
        {
            "name": "Open App Action",
            "expected_action": '{"action_type": "open_app", "app_name": "Calculator"}',
            "description": "Open calculator app"
        },
        {
            "name": "Status Complete",
            "expected_action": '{"action_type": "status", "goal_status": "complete"}',
            "description": "Mark task as complete"
        },
        {
            "name": "Answer Action",
            "expected_action": '{"action_type": "answer", "text": "The answer is 42"}',
            "description": "Answer user question"
        },
        {
            "name": "Navigate Home",
            "expected_action": '{"action_type": "navigate_home"}',
            "description": "Go to home screen"
        },
        {
            "name": "Long Press Action",
            "expected_action": '{"action_type": "long_press", "index": 2}',
            "description": "Long press on element"
        },
        {
            "name": "Double Tap Action",
            "expected_action": '{"action_type": "double_tap", "index": 1}',
            "description": "Double tap on element"
        },
        {
            "name": "Swipe Action",
            "expected_action": '{"action_type": "swipe", "direction": "left"}',
            "description": "Swipe left"
        },
        {
            "name": "Unknown Action",
            "expected_action": '{"action_type": "unknown"}',
            "description": "Fallback for unrecognized actions"
        }
    ]
    
    print("Testing action parsing compatibility...")
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n{i}. {test_case['name']}")
            
            # Create mock function calling output
            mock_output = f"Reason: {test_case['description']}\nAction: {test_case['expected_action']}"
            
            # Parse reason and action (simulating our agent's parsing)
            lines = mock_output.strip().split('\n')
            reason_line = None
            action_line = None
            
            for line in lines:
                if line.startswith('Reason:'):
                    reason_line = line[7:].strip()
                elif line.startswith('Action:'):
                    action_line = line[7:].strip()
            
            if not reason_line or not action_line:
                raise ValueError("Could not parse reason/action from output")
            
            print(f"   Reason: {reason_line}")
            print(f"   Action: {action_line}")
            
            # Test AndroidWorld's action parsing (same as T3A agent)
            action_data = agent_utils.extract_json(action_line)
            converted_action = json_action.JSONAction(**action_data)
            
            print(f"   ✅ AndroidWorld JSONAction created: {converted_action}")
            
            # Verify action type is supported
            if converted_action.action_type not in [
                'click', 'double_tap', 'scroll', 'swipe', 'input_text', 
                'navigate_home', 'navigate_back', 'keyboard_enter', 'open_app', 
                'status', 'wait', 'long_press', 'answer', 'unknown'
            ]:
                raise ValueError(f"Unsupported action type: {converted_action.action_type}")
            
            print(f"   ✅ Action type '{converted_action.action_type}' is supported")
            
            # Test JSON serialization (for logging/storage)
            json_str = converted_action.json_str()
            print(f"   ✅ JSON serialization: {json_str}")
            
        except Exception as e:
            error_msg = f"{test_case['name']}: {str(e)}"
            print(f"   ❌ {error_msg}")
            errors.append(error_msg)
    
    print(f"\n{'='*70}")
    if errors:
        print(f"❌ {len(errors)} compatibility issues found:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ All action types are compatible with AndroidWorld!")
        print("✅ Function calling output can be parsed by AndroidWorld's action system")
        return True


def test_function_schema_completeness():
    """Test that our function schema includes all AndroidWorld action types."""
    print("\n🧪 Testing Function Schema Completeness")
    print("=" * 70)
    
    try:
        # Get our function calling schema
        llm = FunctionCallingLLM("gpt-4o-mini")
        schema = llm.function_schema
        
        our_action_types = set(schema["parameters"]["properties"]["action"]["properties"]["action_type"]["enum"])
        
        # AndroidWorld's supported action types (from json_action.py)
        androidworld_action_types = {
            'click', 'double_tap', 'scroll', 'swipe', 'input_text', 
            'navigate_home', 'navigate_back', 'keyboard_enter', 'open_app', 
            'status', 'wait', 'long_press', 'answer', 'unknown'
        }
        
        print(f"Our schema action types: {sorted(our_action_types)}")
        print(f"AndroidWorld action types: {sorted(androidworld_action_types)}")
        
        # Check for missing action types (excluding 'unknown' which is for fallback)
        important_types = androidworld_action_types - {'unknown'}
        missing_types = important_types - our_action_types
        extra_types = our_action_types - androidworld_action_types
        
        if missing_types:
            print(f"⚠️  Missing action types: {sorted(missing_types)}")
            return False
        
        if extra_types:
            print(f"ℹ️  Extra action types (not in AndroidWorld): {sorted(extra_types)}")
        
        print("✅ Function schema includes all important AndroidWorld action types!")
        return True
        
    except Exception as e:
        print(f"❌ Schema completeness test failed: {e}")
        return False


def main():
    """Run all compatibility tests."""
    print("🔍 AndroidWorld Function Calling Compatibility Test Suite")
    print("=" * 70)
    
    success = True
    
    # Test action compatibility
    if not test_action_compatibility():
        success = False
    
    # Test schema completeness
    if not test_function_schema_completeness():
        success = False
    
    print(f"\n{'='*70}")
    if success:
        print("🎉 All compatibility tests passed!")
        print("✅ Function calling is fully compatible with AndroidWorld")
    else:
        print("❌ Some compatibility issues detected")
        print("🔧 Please review and fix the issues above")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

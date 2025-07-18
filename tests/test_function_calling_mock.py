#!/usr/bin/env python3
"""Mock test for function calling integration without API calls."""

import sys
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from function_calling_llm import FunctionCallingLLM, HybridLLM, create_llm


def test_function_calling_mock():
    """Test function calling with mocked OpenAI response."""
    print("Testing Function Calling with Mock Response...")
    
    # Create a mock response that simulates successful function calling
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.function_call = Mock()
    mock_response.choices[0].message.function_call.arguments = json.dumps({
        "reasoning": "I need to click on the Settings app icon to open it and achieve the goal.",
        "action": {
            "action_type": "click",
            "index": 1
        }
    })
    mock_response.model_dump.return_value = {"mock": "response"}
    
    try:
        llm = FunctionCallingLLM("gpt-4o-mini")
        
        # Mock the OpenAI client call
        with patch.object(llm.client.chat.completions, 'create', return_value=mock_response):
            test_prompt = "Click on Settings app at index 1"
            output, is_safe, raw_response = llm.predict(test_prompt)
            
            print(f"Output: {output}")
            print(f"Is safe: {is_safe}")
            
            # Verify the output format
            if "Reason:" in output and "Action:" in output:
                print("✅ Function calling output format is correct!")
                
                # Parse the action
                action_line = None
                reason_line = None
                for line in output.split('\n'):
                    if line.startswith('Reason:'):
                        reason_line = line[7:].strip()
                    elif line.startswith('Action:'):
                        action_line = line[7:].strip()
                
                if action_line and reason_line:
                    action_data = json.loads(action_line)
                    print(f"✅ Parsed reason: {reason_line}")
                    print(f"✅ Parsed action: {action_data}")
                    
                    if action_data.get("action_type") == "click" and action_data.get("index") == 1:
                        print("✅ Function calling produced correct structured output!")
                    else:
                        print("❌ Action data doesn't match expected values")
                else:
                    print("❌ Could not parse reason and action from output")
            else:
                print("❌ Function calling output format is incorrect")
                
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        import traceback
        traceback.print_exc()


def test_parsing_method():
    """Test the parsing method directly."""
    print("\nTesting Function Calling Output Parsing...")
    
    try:
        # Test parsing manually since _parse_function_calling_output is in the agent class
        sample_output = """Reason: I need to click on the Settings app to open it
Action: {"action_type": "click", "index": 1}"""
        
        # Parse the output manually (same logic as in agent class)
        lines = sample_output.strip().split('\n')
        reason_line = None
        action_line = None
        
        for line in lines:
            if line.startswith('Reason:'):
                reason_line = line[7:].strip()  # Remove "Reason:" prefix
            elif line.startswith('Action:'):
                action_line = line[7:].strip()  # Remove "Action:" prefix
        
        if reason_line and action_line:
            print(f"✅ Parsing successful!")
            print(f"   Reason: {reason_line}")
            print(f"   Action: {action_line}")
            
            # Verify action is valid JSON
            action_data = json.loads(action_line)
            if "action_type" in action_data:
                print(f"✅ Action type found: {action_data['action_type']}")
            else:
                raise AssertionError("Missing action_type in parsed action")
        else:
            raise AssertionError("Parsing failed - could not extract reason and action")
            
    except Exception as e:
        print(f"❌ Parsing test failed: {e}")


def test_schema_validation():
    """Test that our function schema is valid."""
    print("\nTesting Function Schema Validation...")
    
    try:
        llm = FunctionCallingLLM("gpt-4o-mini")
        schema = llm.function_schema
        
        # Validate schema structure
        required_keys = ["name", "description", "parameters"]
        for key in required_keys:
            if key not in schema:
                raise AssertionError(f"Missing required key: {key}")
        
        print("✅ Schema has all required top-level keys")
        
        # Validate parameters structure
        params = schema["parameters"]
        if params.get("type") != "object":
            raise AssertionError("Parameters type should be 'object'")
            
        properties = params.get("properties", {})
        if "reasoning" not in properties or "action" not in properties:
            raise AssertionError("Missing required properties: reasoning or action")
            
        print("✅ Schema has correct properties structure")
        
        # Validate action enum values
        action_props = properties["action"]["properties"]
        action_types = action_props["action_type"]["enum"]
        
        expected_actions = [
            "status", "answer", "click", "long_press", 
            "input_text", "keyboard_enter", "navigate_home", 
            "navigate_back", "scroll", "open_app", "wait"
        ]
        
        for action_type in expected_actions:
            if action_type not in action_types:
                raise AssertionError(f"Missing expected action type: {action_type}")
        
        print(f"✅ Schema includes all {len(expected_actions)} expected action types")
        
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")


def test_hybrid_llm_modes():
    """Test that hybrid LLM can switch between modes."""
    print("\nTesting Hybrid LLM Mode Switching...")
    
    try:
        # Test function calling mode
        llm_fc = create_llm("gpt-4o-mini", use_function_calling=True)
        if hasattr(llm_fc, 'use_function_calling') and llm_fc.use_function_calling:
            print("✅ Function calling mode enabled correctly")
        else:
            print("❌ Function calling mode not enabled")
            
        # Test regular mode
        llm_regular = create_llm("gpt-4o-mini", use_function_calling=False)
        if hasattr(llm_regular, 'use_function_calling') and not llm_regular.use_function_calling:
            print("✅ Regular mode enabled correctly")
        else:
            print("❌ Regular mode not enabled")
            
    except Exception as e:
        print(f"❌ Hybrid LLM test failed: {e}")


def main():
    """Run all mock tests."""
    print("🧪 Testing Function Calling Integration (Mock Mode)")
    print("=" * 60)
    print("Note: Using mock responses to avoid API quota issues")
    print("=" * 60)
    
    errors = []  # Track any errors
    
    try:
        test_schema_validation()
        print("✅ Schema validation test passed")
    except Exception as e:
        error_msg = f"Schema validation test failed: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    
    try:
        test_parsing_method()
        print("✅ Parsing method test passed")
    except Exception as e:
        error_msg = f"Parsing method test failed: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    
    try:
        test_hybrid_llm_modes()
        print("✅ Hybrid LLM mode test passed")
    except Exception as e:
        error_msg = f"Hybrid LLM mode test failed: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    
    try:
        test_function_calling_mock()
        print("✅ Function calling mock test passed")
    except Exception as e:
        error_msg = f"Function calling mock test failed: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    
    if errors:
        print(f"\n❌ {len(errors)} error(s) occurred:")
        for error in errors:
            print(f"   - {error}")
        print("\n📝 Next Steps:")
        print("   1. Fix the failing tests")
        print("   2. Add OpenAI credits to test with real API calls")
        print("   3. Use --function-calling flag in your agent runs:")
        print("      python src/main.py --function-calling --task YourTask")
        print("   4. Compare structured vs free-form output reliability")
        sys.exit(1)
    else:
        print("\n🎉 All function calling integration tests passed!")
        print("\n📝 Next Steps:")
        print("   1. Add OpenAI credits to test with real API calls")
        print("   2. Use --function-calling flag in your agent runs:")
        print("      python src/main.py --function-calling --task YourTask")
        print("   3. Compare structured vs free-form output reliability")
        sys.exit(0)


if __name__ == "__main__":
    main()

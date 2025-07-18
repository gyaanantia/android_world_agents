#!/usr/bin/env python3
"""Test function calling integration."""

import sys
import os
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from function_calling_llm import FunctionCallingLLM, HybridLLM, create_llm
from test_utils import confirm_api_usage, check_api_key, show_cost_summary


def test_function_calling_llm():
    """Test the function calling LLM wrapper."""
    print("Testing Function Calling LLM...")
    
    # Ask for user confirmation before making API calls
    if not confirm_api_usage("Function Calling LLM Test", "gpt-4o-mini", "~$0.001-0.003"):
        return
    
    try:
        # Create function calling LLM
        llm = FunctionCallingLLM("gpt-4o-mini")
        
        # Test with a simple prompt
        test_prompt = """
        You are an Android automation agent. The current goal is to open the Calculator app.
        
        Current UI elements:
        0: Home screen launcher icon for "Calculator"
        1: Search button
        2: Settings icon
        
        Based on the goal and UI elements, what action should you take?
        """
        
        print("Sending test prompt to function calling LLM...")
        output, is_safe, raw_response = llm.predict(test_prompt)
        
        print(f"Output: {output}")
        print(f"Is safe: {is_safe}")
        print(f"Raw response length: {len(raw_response)} characters")
        
        # Verify output format
        if "Reason:" in output and "Action:" in output:
            print("✅ Function calling output format is correct!")
            
            # Try to parse the action JSON
            action_line = None
            for line in output.split('\n'):
                if line.startswith('Action:'):
                    action_line = line[7:].strip()
                    break
            
            if action_line:
                try:
                    action_data = json.loads(action_line)
                    print(f"✅ Action JSON is valid: {action_data}")
                    
                    if "action_type" in action_data:
                        print(f"✅ Action type found: {action_data['action_type']}")
                    else:
                        print("❌ Missing action_type in action data")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Action JSON is invalid: {e}")
            else:
                print("❌ Could not find Action line in output")
        else:
            print("❌ Function calling output format is incorrect")
            
    except Exception as e:
        print(f"❌ Function calling test failed: {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_llm():
    """Test the hybrid LLM that can switch between modes."""
    print("\nTesting Hybrid LLM...")
    
    try:
        # Test function calling mode
        print("Testing function calling mode...")
        llm_fc = create_llm(use_function_calling=True)
        
        # Test regular mode
        print("Testing regular mode...")
        llm_regular = create_llm(use_function_calling=False)
        
        print("✅ Hybrid LLM creation successful!")
        
    except Exception as e:
        print(f"❌ Hybrid LLM test failed: {e}")


def test_function_schema():
    """Test the function schema definition."""
    print("\nTesting Function Schema...")
    
    try:
        llm = FunctionCallingLLM("gpt-4o-mini")
        schema = llm.function_schema
        
        # Verify required fields
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in schema:
                raise AssertionError(f"Missing required field: {field}")
        
        print("✅ Function schema has all required fields")
        
        # Verify action types
        action_types = schema["parameters"]["properties"]["action"]["properties"]["action_type"]["enum"]
        expected_types = ["status", "answer", "click", "long_press", "input_text", "keyboard_enter", 
                         "navigate_home", "navigate_back", "scroll", "open_app", "wait"]
        
        for action_type in expected_types:
            if action_type not in action_types:
                raise AssertionError(f"Missing action type: {action_type}")
        
        print(f"✅ Function schema has all {len(expected_types)} expected action types")
        
    except Exception as e:
        print(f"❌ Function schema test failed: {e}")


def main():
    """Run all function calling tests."""
    print("🧪 Testing Function Calling Integration")
    print("=" * 50)
    
    errors = []  # Track any errors
    
    # Check if OpenAI API key is set
    if not check_api_key():
        errors.append("OpenAI API key not available")
        print(f"\n❌ {len(errors)} error(s) occurred!")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    
    # Show cost summary
    free_tests = ["Function Schema Validation", "Hybrid LLM Creation"]
    paid_tests = ["Function Calling LLM (Real API calls)"]
    show_cost_summary(free_tests, paid_tests)
    
    # Run tests and track errors
    try:
        test_function_schema()
        print("✅ Function schema test passed")
    except Exception as e:
        error_msg = f"Function schema test failed: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    
    try:
        test_hybrid_llm()
        print("✅ Hybrid LLM test passed")
    except Exception as e:
        error_msg = f"Hybrid LLM test failed: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    
    try:
        test_function_calling_llm()
        print("✅ Function calling LLM test completed")
    except Exception as e:
        error_msg = f"Function calling LLM test failed: {e}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
    
    print("\n🎉 Function calling tests completed!")
    
    if errors:
        print(f"\n❌ {len(errors)} error(s) occurred:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

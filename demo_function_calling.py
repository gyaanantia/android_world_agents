#!/usr/bin/env python3
"""Demo script showing function calling vs regular text parsing."""

import sys
import os
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import suppress_grpc_logging

# Suppress gRPC verbose logging before any imports that use gRPC
suppress_grpc_logging()

from function_calling_llm import create_llm
from android_world.agents import infer
from test_utils import confirm_api_usage, check_api_key


def demo_comparison():
    """Demonstrate the difference between function calling and regular text parsing."""
    print("🔍 Function Calling vs Regular Text Parsing Demo")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not check_api_key():
        return
    
    # Ask for user confirmation before making API calls
    if not confirm_api_usage("Function Calling Demo", "gpt-4o-mini", "~$0.002-0.006 total (2 API calls)"):
        return
    
    # Test prompt
    test_prompt = """
    You are an Android automation agent. Your goal is to open the Settings app.
    
    Current UI elements on the screen:
    0: Calculator app icon
    1: Settings app icon  
    2: Camera app icon
    3: Phone app icon
    4: Messages app icon
    
    You need to click on the Settings app to achieve the goal.
    
    Provide your reasoning and the action to take.
    """
    
    print("Test prompt:")
    print("-" * 30)
    print(test_prompt)
    print("-" * 30)
    
    try:
        # Test with function calling
        print("\n🔧 Using Function Calling:")
        print("-" * 30)
        
        llm_fc = create_llm("gpt-4o-mini", use_function_calling=True)
        output_fc, is_safe_fc, raw_fc = llm_fc.predict(test_prompt)
        
        print("Output:")
        print(output_fc)
        print(f"\nIs safe: {is_safe_fc}")
        print(f"Response type: Structured function call")
        
        # Test with regular text parsing
        print("\n📝 Using Regular Text Parsing:")
        print("-" * 30)
        
        llm_regular = create_llm("gpt-4o-mini", use_function_calling=False)
        output_regular, is_safe_regular, raw_regular = llm_regular.predict(test_prompt)
        
        print("Output:")
        print(output_regular)
        print(f"\nIs safe: {is_safe_regular}")
        print(f"Response type: Free-form text")
        
        # Compare parsing reliability
        print("\n📊 Parsing Comparison:")
        print("-" * 30)
        
        # Try to parse function calling output
        try:
            fc_reason = None
            fc_action = None
            for line in output_fc.split('\n'):
                if line.startswith('Reason:'):
                    fc_reason = line[7:].strip()
                elif line.startswith('Action:'):
                    fc_action_str = line[7:].strip()
                    fc_action = json.loads(fc_action_str)
            
            if fc_reason and fc_action:
                print("✅ Function calling: Parsing successful")
                print(f"   Reason: {fc_reason}")
                print(f"   Action: {fc_action}")
                print(f"   Action type: {fc_action.get('action_type', 'Unknown')}")
            else:
                print("❌ Function calling: Parsing failed")
                
        except Exception as e:
            print(f"❌ Function calling: Parsing error - {e}")
        
        # Try to parse regular output (would use m3a_utils in real code)
        print("\n   Regular text parsing would require:")
        print("   - Complex regex patterns")
        print("   - Error handling for malformed JSON")
        print("   - Manual validation of action types")
        print("   - Fallback mechanisms")
        
        print("\n🎯 Benefits of Function Calling:")
        print("   ✅ Guaranteed JSON structure")
        print("   ✅ Type validation at API level")
        print("   ✅ No regex parsing needed")
        print("   ✅ Consistent output format")
        print("   ✅ Built-in error handling")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the demo."""
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("To use function calling in your agent:")
    print("  python src/main.py --function-calling --task YourTask")
    print("\nTo compare modes:")
    print("  python src/main.py --task YourTask  # Regular mode")
    print("  python src/main.py --function-calling --task YourTask  # Function calling mode")


if __name__ == "__main__":
    main()

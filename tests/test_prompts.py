#!/usr/bin/env python3
"""
Test script for the Enhanced T3A prompting system.
Verifies all prompt variants work correctly.
"""
import sys
import os
from pathlib import Path

# Add the src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from prompts import get_prompt_template, format_prompt, list_available_prompts

def test_prompt_system():
    """Test the enhanced prompting system."""
    
    print("=== Enhanced T3A Prompting System Test ===\n")
    
    errors = []  # Track any errors
    
    # Test 1: Check available prompts
    print("1. Available prompt templates:")
    try:
        available_prompts = list_available_prompts()
        for prompt in available_prompts:
            print(f"   - {prompt}")
        print()
    except Exception as e:
        error_msg = f"Failed to list available prompts: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)
    
    # Test 2: Load and verify each prompt
    print("2. Testing prompt variants:")
    
    test_vars = {
        "goal": "Turn on Wi-Fi",
        "ui_elements": "0: Switch 'Wi-Fi' <clickable, checked: false>\n1: TextView 'Wi-Fi' <not clickable>",
        "memory": "You just started, no action has been performed yet.",
        "reflection_context": "No previous reflection available for this task."
    }
    
    for variant in ['base', 'few-shot', 'reflective']:
        try:
            # Load template
            template = get_prompt_template(variant)
            
            # Format with test variables
            formatted = format_prompt(template, **test_vars)
            
            # Verify content
            required_elements = [
                "You are an agent who can operate an Android phone",
                "action_type",
                "Turn on Wi-Fi",
                "Switch 'Wi-Fi'"
            ]
            
            missing = [elem for elem in required_elements if elem not in formatted]
            
            if missing:
                error_msg = f"{variant.upper()}: Missing elements: {missing}"
                print(f"   ✗ {error_msg}")
                errors.append(error_msg)
            else:
                print(f"   ✓ {variant.upper()}: {len(formatted)} characters, all elements present")
                
        except Exception as e:
            error_msg = f"{variant.upper()}: ERROR - {e}"
            print(f"   ✗ {error_msg}")
            errors.append(error_msg)
    
    print()
    
    # Test 3: Show prompt differences
    print("3. Prompt characteristics:")
    
    base_template = get_prompt_template("base")
    few_shot_template = get_prompt_template("few-shot")
    reflective_template = get_prompt_template("reflective")
    
    print(f"   Base prompt: {len(base_template)} characters")
    print(f"   Few-shot prompt: {len(few_shot_template)} characters (+{len(few_shot_template) - len(base_template)} vs base)")
    print(f"   Reflective prompt: {len(reflective_template)} characters (+{len(reflective_template) - len(base_template)} vs base)")
    
    print()
    
    # Test 4: Verify specific features
    print("4. Feature verification:")
    
    # Check base features
    base_features = ["JSON format", "action_type", "click", "input_text", "scroll"]
    for feature in base_features:
        if feature in base_template:
            print(f"   ✓ Base contains: {feature}")
        else:
            error_msg = f"Base missing: {feature}"
            print(f"   ✗ {error_msg}")
            errors.append(error_msg)
    
    # Check few-shot features
    few_shot_features = ["Example", "learn from these examples", "similar reasoning"]
    for feature in few_shot_features:
        if feature.lower() in few_shot_template.lower():
            print(f"   ✓ Few-shot contains: {feature}")
        else:
            error_msg = f"Few-shot missing: {feature}"
            print(f"   ✗ {error_msg}")
            errors.append(error_msg)
    
    # Check reflective features
    reflective_features = ["self-reflection", "learn from mistakes", "reflect on"]
    for feature in reflective_features:
        if feature.lower() in reflective_template.lower():
            print(f"   ✓ Reflective contains: {feature}")
        else:
            error_msg = f"Reflective missing: {feature}"
            print(f"   ✗ {error_msg}")
            errors.append(error_msg)
    
    print("\n=== Test Complete ===")
    
    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        return True


if __name__ == "__main__":
    try:
        success = test_prompt_system()
        if success:
            print("✅ All prompt tests passed!")
            sys.exit(0)
        else:
            print("❌ Some prompt tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Prompt tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

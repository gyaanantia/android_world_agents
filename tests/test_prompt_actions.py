#!/usr/bin/env python3
"""Verify all prompts have complete and consistent action sets."""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompts import get_prompt_template


def check_action_completeness():
    """Check that all prompts include all required actions."""
    print("🔍 Checking Action Completeness Across All Prompts")
    print("=" * 60)
    
    # Complete list of actions that should be in all prompts
    required_actions = [
        "status",
        "answer", 
        "click",
        "double_tap",
        "long_press",
        "swipe",
        "input_text",
        "keyboard_enter",
        "navigate_home",
        "navigate_back", 
        "scroll",
        "open_app",
        "wait"
    ]
    
    variants = ["base", "few-shot", "reflective"]
    all_good = True
    
    for variant in variants:
        print(f"\n📝 Checking {variant.upper()} prompt:")
        
        try:
            template = get_prompt_template(variant)
            
            missing_actions = []
            for action in required_actions:
                # Check both the action_type format and the action name
                if f'"action_type": "{action}"' not in template and action not in template:
                    missing_actions.append(action)
            
            if missing_actions:
                print(f"   ❌ Missing actions: {missing_actions}")
                all_good = False
            else:
                print(f"   ✅ All {len(required_actions)} actions present")
                
            # Check for specific format requirements
            format_checks = [
                ("JSON format", '"action_type"'),
                ("Goal status complete", '"goal_status": "complete"'),
                ("Goal status infeasible", '"goal_status": "infeasible"'),
                ("Direction parameter", '"direction"'),
                ("Index parameter", '"index"'),
                ("Text parameter", '"text"'),
                ("App name parameter", '"app_name"')
            ]
            
            for check_name, pattern in format_checks:
                if pattern not in template:
                    print(f"   ⚠️  Missing {check_name}: {pattern}")
                    
        except Exception as e:
            print(f"   ❌ Error loading {variant}: {e}")
            all_good = False
    
    print(f"\n{'='*60}")
    if all_good:
        print("✅ All prompts have complete action sets!")
        return True
    else:
        print("❌ Some prompts are missing actions or formats")
        return False


def check_action_consistency():
    """Check that action descriptions are consistent across prompts."""
    print("\n🔍 Checking Action Consistency Across Prompts")
    print("=" * 60)
    
    variants = ["base", "few-shot", "reflective"]
    actions_by_variant = {}
    
    # Extract action patterns from each variant
    for variant in variants:
        try:
            template = get_prompt_template(variant)
            actions_by_variant[variant] = template
            print(f"✅ Loaded {variant} prompt ({len(template)} chars)")
        except Exception as e:
            print(f"❌ Failed to load {variant}: {e}")
            return False
    
    # Check for specific action formats
    action_patterns = [
        ('"action_type": "status"', "Status action"),
        ('"action_type": "click"', "Click action"),
        ('"action_type": "double_tap"', "Double tap action"),
        ('"action_type": "swipe"', "Swipe action"),
        ('"action_type": "input_text"', "Input text action"),
        ('"action_type": "scroll"', "Scroll action")
    ]
    
    print(f"\nChecking key action patterns:")
    all_consistent = True
    
    for pattern, description in action_patterns:
        print(f"  {description}:")
        for variant in variants:
            if pattern in actions_by_variant[variant]:
                print(f"    ✅ {variant}")
            else:
                print(f"    ❌ {variant} - MISSING")
                all_consistent = False
    
    if all_consistent:
        print(f"\n✅ All action patterns are consistent across prompts!")
        return True
    else:
        print(f"\n❌ Some action patterns are inconsistent")
        return False


def main():
    """Run all action verification checks."""
    print("🧪 Prompt Action Verification Suite")
    print("=" * 60)
    
    success = True
    
    if not check_action_completeness():
        success = False
        
    if not check_action_consistency():
        success = False
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All prompts have complete and consistent action sets!")
        print("✅ Ready for AndroidWorld integration")
    else:
        print("❌ Action verification failed")
        print("🔧 Please review and fix the issues above")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

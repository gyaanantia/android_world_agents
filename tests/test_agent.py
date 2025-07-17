#!/usr/bin/env python3
"""
Quick test to verify that the enhanced agent works properly.
"""

import sys
import os
from pathlib import Path

# Add the src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_agent_import():
    """Test that the enhanced agent can be imported and basic functionality works."""
    try:
        from agent import EnhancedT3A, create_agent
        print("✅ Agent classes imported successfully")
        
        from prompts import get_prompt_template, format_prompt
        print("✅ Prompt functions imported successfully")
        
        # Test prompt loading
        base_prompt = get_prompt_template("base")
        print(f"✅ Base prompt loaded ({len(base_prompt)} characters)")
        
        # Test prompt formatting
        formatted = format_prompt(
            base_prompt, 
            goal="Test", 
            ui_elements="Test UI",
            history="You just started, no action has been performed yet."
        )
        print("✅ Prompt formatting works")
        
        print("\n🎉 All agent functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_import()
    sys.exit(0 if success else 1)

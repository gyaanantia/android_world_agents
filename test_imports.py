#!/usr/bin/env python3
"""Test script to verify imports work correctly."""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all imports work correctly."""
    
    print("Testing imports...")
    
    try:
        # Test android_world imports
        import agent
        print("✓ Agent module imported successfully")
        
        # Test EnhancedT3A class
        from agent import EnhancedT3A
        print("✓ EnhancedT3A class imported successfully")
        
        # Test that android_world modules are accessible
        from android_world.agents import t3a
        print("✓ android_world.agents.t3a imported successfully")
        
        from android_world.agents import infer
        print("✓ android_world.agents.infer imported successfully")
        
        from android_world.env import interface
        print("✓ android_world.env.interface imported successfully")
        
        # Test prompts module
        import prompts
        print("✓ prompts module imported successfully")
        
        # Test specific functions
        from prompts import get_prompt_template, format_prompt
        print("✓ prompt functions imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

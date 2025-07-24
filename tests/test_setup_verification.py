"""
Test environment setup verification.
Run this after setup to ensure all dependencies are properly installed.
"""

import importlib
import pytest


def _test_import(module_name):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def test_core_dependencies():
    """Test that all core dependencies can be imported."""
    core_modules = [
        "textgrad",
        "google.generativeai", 
        "openai",
        "numpy",
        "PIL",
        "pandas",
        "matplotlib",
        "json",
        "pathlib",
    ]
    
    for module in core_modules:
        assert _test_import(module), f"Failed to import {module}"


def test_project_modules():
    """Test that all project modules can be imported."""
    import sys
    sys.path.append('src')
    
    project_modules = [
        "agent",
        "gemini_enhanced_agent", 
        "evaluator",
        "function_calling_llm",
        "textgrad_opt",
    ]
    
    for module in project_modules:
        assert _test_import(module), f"Failed to import project module {module}"


def test_textgrad_available():
    """Test that TextGrad is properly installed and functional."""
    try:
        import textgrad
        # Try to create a basic TextGrad object to ensure it's working
        assert hasattr(textgrad, 'get_engine'), "TextGrad missing get_engine function"
    except ImportError:
        pytest.fail("TextGrad not installed")


def test_google_api_available():
    """Test that Google Generative AI is available."""
    try:
        import google.generativeai as genai
        assert hasattr(genai, 'configure'), "Google Generative AI missing configure function"
    except ImportError:
        pytest.fail("Google Generative AI not installed")


if __name__ == "__main__":
    # Allow running as standalone script for backwards compatibility
    import sys
    
    print("Testing Android World Agents environment setup...")
    print("=" * 50)
    
    try:
        test_core_dependencies()
        print("✅ Core dependencies test passed")
    except AssertionError as e:
        print(f"❌ Core dependencies test failed: {e}")
        sys.exit(1)
    
    try:
        test_project_modules()
        print("✅ Project modules test passed")
    except AssertionError as e:
        print(f"❌ Project modules test failed: {e}")
        sys.exit(1)
    
    try:
        test_textgrad_available()
        print("✅ TextGrad availability test passed")
    except Exception as e:
        print(f"❌ TextGrad availability test failed: {e}")
        sys.exit(1)
    
    try:
        test_google_api_available()
        print("✅ Google API availability test passed")
    except Exception as e:
        print(f"❌ Google API availability test failed: {e}")
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 All environment setup tests passed!")

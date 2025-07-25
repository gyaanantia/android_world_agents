"""
Test TextGrad optimizer functionality.
"""

import os
import sys
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_textgrad_optimizer_creation():
    """Test basic TextGrad optimizer creation."""
    from textgrad_opt import create_textgrad_optimizer
    
    # Test optimizer creation
    optimizer = create_textgrad_optimizer(enabled=True)
    assert optimizer is not None, "Failed to create TextGrad optimizer"


def test_textgrad_availability():
    """Test that TextGrad optimizer is available."""
    from textgrad_opt import create_textgrad_optimizer
    
    optimizer = create_textgrad_optimizer(enabled=True)
    # Note: This may return False if TextGrad dependencies aren't available
    # which is okay for testing - we just want to ensure the code doesn't crash
    availability = optimizer.is_available()
    assert isinstance(availability, bool), "is_available() should return a boolean"


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), 
    reason="OpenAI API key not available for TextGrad optimization"
)
def test_textgrad_optimization():
    """Test TextGrad optimization with real API calls (requires API key)."""
    from textgrad_opt import create_textgrad_optimizer
    
    optimizer = create_textgrad_optimizer(enabled=True)
    
    if not optimizer.is_available():
        pytest.skip("TextGrad optimizer not available (missing dependencies or API key)")
    
    # Test optimization with sample data
    sample_analysis = """
    The screen shows a messaging app. There are several conversation threads visible.
    There's a compose button somewhere on the screen. The interface uses Material Design.
    Some conversations are from John, Mom, and Work Team. The layout is typical of messaging apps.
    """
    
    goal = "Send a message to John"
    ui_elements = "0: Button 'Compose' (clickable)\n1: List 'Conversations' (scrollable)\n2: Item 'John Smith' (clickable)"
    
    optimized = optimizer.optimize_visual_analysis(
        gemini_analysis=sample_analysis,
        task_goal=goal,
        ui_elements=ui_elements
    )
    
    assert isinstance(optimized, str), "Optimization should return a string"
    assert len(optimized) > 0, "Optimized analysis should not be empty"
    
    # Check if optimization focuses on relevant elements
    assert "John" in optimized or "john" in optimized.lower(), "Optimization should mention John"


def test_textgrad_disabled():
    """Test TextGrad optimizer when disabled."""
    from textgrad_opt import create_textgrad_optimizer
    
    optimizer = create_textgrad_optimizer(enabled=False)
    assert optimizer is not None, "Should create optimizer even when disabled"
    
    # When disabled, optimization should return original text
    original = "Test analysis"
    result = optimizer.optimize_visual_analysis(original, "test goal", "test ui")
    assert result == original, "Disabled optimizer should return original text"


if __name__ == "__main__":
    # Allow running as standalone script
    print("🧪 Testing TextGrad Optimizer")
    print("=" * 40)
    
    try:
        test_textgrad_optimizer_creation()
        print("✅ TextGrad optimizer creation test passed")
    except Exception as e:
        print(f"❌ TextGrad optimizer creation test failed: {e}")
        exit(1)
    
    try:
        test_textgrad_availability()
        print("✅ TextGrad availability test passed")
    except Exception as e:
        print(f"❌ TextGrad availability test failed: {e}")
        exit(1)
    
    try:
        test_textgrad_disabled()
        print("✅ TextGrad disabled test passed")
    except Exception as e:
        print(f"❌ TextGrad disabled test failed: {e}")
        exit(1)
    
    print("=" * 40)
    print("🎉 Basic TextGrad tests passed!")
    print("Note: Run with pytest to include API-dependent tests")

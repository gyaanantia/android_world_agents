#!/usr/bin/env python3
"""
Test for TextGrad optimization functionality.
"""

import os
import sys
import pytest
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from textgrad_opt import create_textgrad_optimizer, test_textgrad_optimization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_textgrad_optimizer_initialization():
    """Test that TextGrad optimizer initializes correctly."""
    optimizer = create_textgrad_optimizer(enabled=False)
    assert optimizer is not None
    assert not optimizer.is_available()  # Should be disabled
    
    # Test with enabled (may fail if no API key)
    try:
        optimizer_enabled = create_textgrad_optimizer(enabled=True)
        if os.getenv("OPENAI_API_KEY"):
            assert optimizer_enabled.is_available()
        else:
            assert not optimizer_enabled.is_available()
    except Exception as e:
        # Expected if no API key
        logger.info(f"TextGrad initialization failed (expected if no API key): {e}")


def test_textgrad_optimization_with_mock_data():
    """Test TextGrad optimization with mock data."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available for testing")
    
    optimizer = create_textgrad_optimizer(enabled=True)
    if not optimizer.is_available():
        pytest.skip("TextGrad optimizer not available")
    
    # Test optimization
    original_analysis = """
The screen shows a messaging app. There's a compose button in the bottom right.
Click it to send a message.
"""
    
    goal = "Send a message to John saying 'Hello'"
    ui_elements = "0: Button 'Compose' (clickable)\n1: List 'Messages' (scrollable)"
    
    optimized = optimizer.optimize_visual_analysis(
        original_analysis, goal, ui_elements
    )
    
    assert optimized is not None
    assert len(optimized) > 0
    # The optimization may or may not change the text significantly


def test_textgrad_cli_test():
    """Test the built-in test function."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available for testing")
    
    # This should run without throwing exceptions
    result = test_textgrad_optimization()
    assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__])

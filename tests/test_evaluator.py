#!/usr/bin/env python3
"""Test evaluator functionality."""

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluator import step_accuracy, episode_success


def test_step_accuracy():
    """Test step-level accuracy calculation."""
    pred = ["CLICK('A')", "CLICK('B')", "DONE"]
    gold = ["CLICK('A')", "CLICK('C')", "DONE"]
    assert step_accuracy(pred, gold) == 2 / 3


def test_episode_success_with_completion():
    """Test episode success when last action indicates completion."""
    pred = ["CLICK('A')", "CLICK('B')", "DONE"]
    assert episode_success(pred, ["A", "B", "C"])  # gold doesn't matter for current implementation
    
    pred = ["CLICK('A')", "CLICK('B')", "COMPLETE"]
    assert episode_success(pred, ["A", "B", "C"])
    
    pred = ["CLICK('A')", "CLICK('B')", "FINISHED"]
    assert episode_success(pred, ["A", "B", "C"])


def test_episode_success_without_completion():
    """Test episode success when last action doesn't indicate completion."""
    pred = ["CLICK('A')", "CLICK('B')"]
    assert not episode_success(pred, ["A", "B"])
    
    pred = ["CLICK('A')", "CLICK('B')", "CLICK('C')"]
    assert not episode_success(pred, ["A", "B", "C"])


def test_episode_success_empty():
    """Test episode success with empty predictions."""
    pred = []
    assert not episode_success(pred, ["A", "B"])


if __name__ == "__main__":
    try:
        test_step_accuracy()
        test_episode_success_with_completion()
        test_episode_success_without_completion()
        test_episode_success_empty()
        print("✅ All evaluator tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Evaluator tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

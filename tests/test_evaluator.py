import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluator import step_accuracy, episode_success


def test_step_accuracy():
    pred = ["CLICK('A')", "CLICK('B')", "DONE"]
    gold = ["CLICK('A')", "CLICK('C')", "DONE"]
    assert step_accuracy(pred, gold) == 2 / 3


def test_episode_success_true():
    pred = ["A", "B"]
    gold = ["A", "B"]
    assert episode_success(pred, gold)


def test_episode_success_false():
    pred = ["A", "B"]
    gold = ["A", "C"]
    assert not episode_success(pred, gold)

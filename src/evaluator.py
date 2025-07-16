import json
from pathlib import Path
from typing import List, Dict


def step_accuracy(pred: List[str], gold: List[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(p == g for p, g in zip(pred, gold))
    return correct / len(gold)


def episode_success(pred: List[str], gold: List[str]) -> bool:
    return pred[: len(gold)] == gold


def evaluate_dir(pred_dir: str, gold_dir: str) -> Dict[str, float]:
    pred_path = Path(pred_dir)
    gold_path = Path(gold_dir)
    step_accs = []
    successes = []
    for pred_file in pred_path.glob("*.jsonl"):
        gold_file = gold_path / pred_file.name
        if not gold_file.exists():
            continue
        pred_actions = []
        gold_actions = []
        with open(pred_file, "r", encoding="utf-8") as pf:
            for line in pf:
                obj = json.loads(line)
                if "summary" in obj:
                    continue
                pred_actions.append(obj["agent_action"])
                gold_actions.append(obj["gold_action"])
        acc = step_accuracy(pred_actions, gold_actions)
        success = episode_success(pred_actions, gold_actions)
        step_accs.append(acc)
        successes.append(success)
    if not step_accs:
        return {"step_acc": 0.0, "episode_success": 0.0}
    return {
        "step_acc": sum(step_accs) / len(step_accs),
        "episode_success": sum(successes) / len(successes),
    }

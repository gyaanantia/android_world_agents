import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from src.utils import load_episode
from src.agent import next_action
from src.evaluator import step_accuracy, episode_success


def run_episode(ep_path: str, model: str, prompt_variant: str) -> Path:
    episode = load_episode(ep_path)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prompt_variant}.jsonl"
    history: List[str] = []
    pred_actions: List[str] = []
    with open(log_path, "w", encoding="utf-8") as lf:
        for step, obs in enumerate(episode["observations"]):
            action = next_action(
                episode["goal"], obs, history, model=model, prompt_variant=prompt_variant
            )
            pred_actions.append(action)
            history.append(action)
            gold_action = episode["actions"][step] if step < len(episode["actions"]) else ""
            lf.write(
                json.dumps(
                    {
                        "step": step,
                        "observation": obs,
                        "agent_action": action,
                        "gold_action": gold_action,
                    }
                )
                + "\n"
            )
            if action.startswith("DONE"):
                break
        summary = {
            "step_acc": step_accuracy(pred_actions, episode["actions"][: len(pred_actions)]),
            "episode_success": episode_success(pred_actions, episode["actions"]),
        }
        lf.write(json.dumps({"summary": summary}) + "\n")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_json", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--prompt_variant", choices=["base", "few-shot", "reflective"], default="base"
    )
    args = parser.parse_args()
    path = run_episode(args.episode_json, args.model, args.prompt_variant)
    print(f"Log written to {path}")


if __name__ == "__main__":
    main()

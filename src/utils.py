import json
import os
from pathlib import Path
from typing import Dict


def load_episode(path: str) -> Dict:
    """Load an episode JSON file and validate required keys."""
    expanded = os.path.expanduser(path)
    with open(expanded, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ["goal", "observations", "actions"]:
        if key not in data:
            raise ValueError(f"Missing key: {key}")
    return data


def normalize_action(action: str) -> str:
    """Return canonical form of an action string."""
    if not action:
        return action
    action = action.strip()
    if "(" in action and action.endswith(")"):
        verb, rest = action.split("(", 1)
        verb = verb.strip().upper()
        rest = rest.strip()
        return f"{verb}({rest}"
    return action.upper()

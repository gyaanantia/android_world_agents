import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging


def find_adb_directory() -> Optional[str]:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found in the common Android SDK paths. Please install Android'
        " SDK and ensure adb is in one of the expected directories. If it's"
        ' already installed, point to the installed location.'
    )


def ensure_results_dir(results_dir: str) -> str:
    """Ensure results directory exists and return absolute path."""
    abs_path = os.path.abspath(os.path.expanduser(results_dir))
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def format_ui_elements_for_prompt(ui_elements: List[Dict[str, Any]]) -> str:
    """Format UI elements for inclusion in prompts."""
    if not ui_elements:
        return "No UI elements detected."
    
    formatted = []
    for i, element in enumerate(ui_elements):
        text = element.get('text', '')
        content_desc = element.get('content_description', '')
        resource_id = element.get('resource_id', '')
        
        # Build element description
        desc_parts = []
        if text:
            desc_parts.append(f"text='{text}'")
        if content_desc:
            desc_parts.append(f"content_desc='{content_desc}'")
        if resource_id:
            desc_parts.append(f"id='{resource_id}'")
        
        element_desc = f"[{i}] {' '.join(desc_parts)}"
        formatted.append(element_desc)
    
    return "\n".join(formatted)


def setup_logging(log_level: str = "INFO", results_dir: str = ".") -> None:
    """Set up logging configuration."""
    log_file_path = os.path.join(results_dir, 'android_world_agents.log')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path)
        ]
    )


def validate_android_world_env() -> bool:
    """Validate AndroidWorld environment is properly set up."""
    try:
        import android_world
        import android_world.agents
        import android_world.task_evals
        import android_world.env
        import android_world.utils
        return True
    except ImportError:
        logging.error("AndroidWorld not installed. Please install from GitHub.")
        return False


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


def load_episode(path: str) -> Dict:
    """Load an episode JSON file and validate required keys."""
    expanded = os.path.expanduser(path)
    with open(expanded, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ["goal", "observations", "actions"]:
        if key not in data:
            raise ValueError(f"Missing key: {key}")
    return data

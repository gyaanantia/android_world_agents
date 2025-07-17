import os
from pathlib import Path
from typing import Dict, Optional


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _read(name: str) -> str:
    """Read a prompt file from the prompts directory."""
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (e.g., "base_prompt.txt", "few_shot_v1.md")
        
    Returns:
        The prompt content as a string, or empty string if not found.
    """
    return _read(prompt_name)


def list_available_prompts() -> list[str]:
    """List all available prompt files in the prompts directory."""
    if not PROMPTS_DIR.exists():
        return []
    
    return [f.name for f in PROMPTS_DIR.iterdir() if f.is_file()]


def get_prompt_template(agent_type: str) -> str:
    """
    Get the appropriate prompt template for a given agent type.
    
    Args:
        agent_type: Type of agent ("base", "few_shot", "reflective")
        
    Returns:
        The prompt template content.
    """
    prompt_mapping = {
        "base": "base_prompt.txt",
        "few_shot": "few_shot_v1.txt", 
        "reflective": "reflective_v1.txt"
    }
    
    filename = prompt_mapping.get(agent_type)
    if filename:
        return load_prompt(filename)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with given variables.
    
    Args:
        template: The prompt template string
        **kwargs: Variables to substitute in the template
        
    Returns:
        The formatted prompt string.
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required variable in prompt template: {e}")


# Pre-load commonly used prompts
BASE_PROMPT = _read("base_prompt.txt")
FEW_SHOT_EXAMPLES = _read("few_shot_v1.txt")
REFLECTIVE_PROMPT = _read("reflective_v1.txt")

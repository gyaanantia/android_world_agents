import os
from pathlib import Path


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


def get_prompt_template(prompt_variant: str) -> str:
    """
    Get the prompt template for a specific variant.
    
    Args:
        prompt_variant: Type of prompting variant ("base", "few-shot", "reflective", 
                       "gemini-base", "gemini-few-shot", "gemini-reflective")
        
    Returns:
        The prompt template content.
    """
    prompt_mapping = {
        "base": "base_prompt.txt",
        "few-shot": "few_shot_v1.txt", 
        "reflective": "reflective_v1.txt",
        "gemini-base": "gemini_enhanced_base_prompt.txt",
        "gemini-few-shot": "gemini_enhanced_few_shot_v1.txt",
        "gemini-reflective": "gemini_enhanced_reflective_v1.txt"
    }
    
    filename = prompt_mapping.get(prompt_variant)
    if filename:
        return load_prompt(filename)
    else:
        available_variants = list(prompt_mapping.keys())
        raise ValueError(f"Unknown prompt variant: {prompt_variant}. Available variants: {available_variants}")


def get_gemini_enhanced_prompt(prompt_variant: str) -> str:
    """
    Get the Gemini-enhanced version of a prompt variant.
    
    Args:
        prompt_variant: Base prompt variant ("base", "few-shot", "reflective")
        
    Returns:
        The Gemini-enhanced prompt template content.
    """
    gemini_variant = f"gemini-{prompt_variant}"
    return get_prompt_template(gemini_variant)


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

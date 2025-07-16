from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _read(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


BASE_PROMPT = _read("base_prompt.txt")
FEW_SHOT_EXAMPLES = _read("few_shot_v1.md")
REFLECTIVE_PROMPT = _read("reflective_v1.md")

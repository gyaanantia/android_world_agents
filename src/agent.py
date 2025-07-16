import os
import openai
from typing import Dict, List
from .prompts import BASE_PROMPT, FEW_SHOT_EXAMPLES, REFLECTIVE_PROMPT

openai.api_key = os.getenv("OPENAI_API_KEY")


VARIANT_PROMPTS = {
    "base": BASE_PROMPT,
    "few-shot": FEW_SHOT_EXAMPLES,
    "reflective": REFLECTIVE_PROMPT,
}


def next_action(
    goal: str,
    observation: Dict,
    history: List[str],
    model: str = "gpt-4o-mini",
    prompt_variant: str = "base",
) -> str:
    """Call OpenAI to get the next action. Returns a string."""
    prompt = VARIANT_PROMPTS.get(prompt_variant, BASE_PROMPT)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Goal: {goal}"},
        {"role": "user", "content": f"Observation: {observation}"},
    ]
    for h in history:
        messages.append({"role": "assistant", "content": h})
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "action",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["CLICK", "SCROLL", "BACK", "HOME", "DONE"],
                                },
                                "target": {"type": "string"},
                            },
                            "required": ["type"],
                        },
                    },
                }
            ],
        )
        tool_calls = resp.choices[0].message.tool_calls
        if tool_calls:
            args = tool_calls[0].function.arguments
            return (
                f'{args.get("type")}("{args.get("target", "")}")'
                if args.get("type") != "DONE"
                else "DONE"
            )
        return "INVALID"
    except Exception:
        return "INVALID"

# AGENTS.md

> **Project** `android_llm_agent` \
> **Purpose** Evaluate how well a Large‑Language Model (LLM) can autonomously choose *state–action* pairs in the **android\_world** benchmark. The emulator is already running; this repo supplies everything else – agents, prompts, evaluators, and CLIs.

---

## 1. Directory Layout

```
android_llm_agent/
├── prompts/               # external‑text prompt files
│   ├── base_prompt.txt
│   ├── few_shot_v1.md
│   └── reflective_v1.md
├── results/               # JSONL logs written by each run
├── src/
│   ├── __init__.py
│   ├── utils.py           # file I/O & string helpers
│   ├── prompts.py         # prompt strings & few‑shot tuples
│   ├── agent.py           # core next_action() loop
│   ├── evaluator.py       # metrics
│   └── run_episode.py     # CLI driver
└── tests/
    └── test_evaluator.py
```

*Codex must generate every file above unless it already exists.*

---

## 2. Episode Schema

Each episode JSON exchanged with `run_episode.py` **must** match:

```jsonc
{
  "goal": "Uninstall the Slack app",
  "observations": [
    /* ordered list of screen‑level UI dumps;
       each item may come from `env.get_state().ui_elements` */
  ],
  "actions": [
    "CLICK(\"Settings\")",
    "SCROLL(\"Apps\")",
    "CLICK(\"Slack\")",
    "CLICK(\"Uninstall\")",
    "DONE"
  ]
}
```

---

## 3. Agent Variants

| ID             | Prompt Strategy                                                                 | Extra Cost | Recommended File           |
| -------------- | ------------------------------------------------------------------------------- | ---------- | -------------------------- |
| **base**       | Last observation only                                                           | ‑          | `prompts/base_prompt.txt`  |
| **few‑shot**   | Adds 1‑2 exemplars for each task family                                         | +20‑30 %   | `prompts/few_shot_v1.md`   |
| **reflective** | Model *explains* choice before returning JSON (`chain‑of‑thought` but stripped) | +35‑50 %   | `prompts/reflective_v1.md` |

All three share the **function‑calling** schema below.

---

### 3.1 Function‑Calling Specification

```jsonc
{
  "type": "function",
  "function": {
    "name": "action",
    "parameters": {
      "type": "object",
      "properties": {
        "type":  { "type": "string",
                   "enum": ["CLICK","SCROLL","BACK","HOME","DONE"] },
        "target":{ "type": "string" }
      },
      "required": ["type"]
    }
  }
}
```

*Rules*

1. `target` **must** exactly match a string in the current observation’s `"ui_elements"`.
2. `DONE` may omit `target`.
3. Any mismatch → return the literal string `"INVALID"`; `run_episode.py` will reprompt the model with a short error message and the same observation.

---

## 4. Module Responsibilities

| File                    | Key Symbols                                                    | Notes                                                                                                     |
| ----------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **src/utils.py**        | `load_episode(path)`                                           | Loads JSON, expands `~`, validates keys.                                                                  |
|                         | `normalize_action(str)`                                        | Canonicalises verbs, strips whitespace, leaves quotes.                                                    |
| **src/prompts.py**      | `BASE_PROMPT`, `FEW_SHOT_EXAMPLES`                             | Importable constants; also reads external text files under `prompts/`.                                    |
| **src/agent.py**        | `next_action(goal, observation, history, model="gpt-4o-mini")` | *Pure* function: returns one action string, no side effects.                                              |
| **src/evaluator.py**    | `step_accuracy`, `episode_success`, `evaluate_dir`             | Computes per‑step and per‑episode metrics; returns a dict like `{"step_acc":0.82,"episode_success":0.5}`. |
| **src/run\_episode.py** | `main()` (CLI)                                                 | Drives env loop, logs to `results/*.jsonl`, supports `--model` and `--prompt_variant`.                    |

---

## 5. Logging & Results

* Each run writes one **JSON Lines** file: `results/<timestamp>_<variant>.jsonl`.
* Every line = `{"step": 3, "observation": {...}, "agent_action": "...", "gold_action": "..."}`
* At end, a summary dict is appended: `{"summary": {"step_acc": 0.87, "episode_success": true}}`.

---

## 6. Evaluation Protocol

```python
from src.evaluator import evaluate_dir
metrics = evaluate_dir(pred_dir="results/", gold_dir="gold/")
# returns: {'step_acc': 0.83, 'episode_success': 0.4}
```

*Use ≥ 10 episodes.*
Report both metrics in the write‑up.

---

## 7. Extending or Swapping Models

1. Wrap the required provider (OpenAI, Anthropic, Mistral, etc.) behind a `ModelBackend` interface inside `src/agent.py`.
2. Add `--backend` flag to CLI.
3. Ensure function‑calling schema is converted to the provider’s equivalent (e.g., Anthropic `tool_use`).

---

## 8. Testing Checklist (Codex must satisfy)

* `pytest` passes (unit test + any new ones).
* `black .` (line‑length 100) shows no diffs.
* `ruff .` shows no errors.
* `python src/run_episode.py --episode_json sample.json --model gpt-4o-mini` finishes ≤ 15 steps.

---

## 9. Milestones for Codex

1. **Scaffold** full tree with empty or stub files.
2. Implement `utils.py`, `prompts.py` constants, and parse external prompt files.
3. Implement `agent.next_action()` with OpenAI function‑calling.
4. Write evaluator + unit test.
5. Flesh out CLI runner with logging.
6. Provide example prompts in `prompts/`.

Once these steps pass locally, you have a minimal but complete framework to iterate on prompt engineering and LLM choice.

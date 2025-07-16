# 📑 CONTEXT
You are an elite software engineer tasked with scaffolding a research prototype called
“android_llm_agent”. The goal is to evaluate how well a large‑language‑model (LLM)
can autonomously choose state–action pairs inside the **android_world** benchmark
(episodes come as JSON with goal, observations, ground‑truth action trace).

# 🗂️ OUTPUT FORMAT
Respond with **ONLY** a sequence of file blocks.
Each block starts with:
=== <relative/path/filename> ===
and ends with:
=== end ===
Put the file’s full contents in between.
No explanations or extra chat.

Example:
=== README.md ===
# Title
bla bla
=== end ===

# 📁 REQUIRED TREE & CONTENT
(1) README.md  
 • One‑sentence project purpose.  
 • Quickstart (venv, `pip install -r requirements.txt`, run a sample episode).  
 • How to add prompt variants.  
(2) requirements.txt  
 openai>=1.14.0  
 tqdm  
 python‑Levenshtein  
 (keep versions loose, pin only if necessary).  
(3) src/__init__.py (empty).  
(4) src/utils.py  
 • `load_episode(json_path) -> dict`  
 • `normalize_action(str) -> str` (strip spaces, capitalize “CLICK”, “SCROLL”, etc.).  
(5) src/prompts.py  
 • `BASE_PROMPT` string with `{goal}`, `{observation}`, `{history}` placeholders.  
 • `FEW_SHOT_EXAMPLES` list of `(goal, steps)` tuples.  
(6) src/agent.py  
 • `next_action()` function exactly matching this signature:  
  `def next_action(goal: str, observation: dict, history: list[str], model=None) -> str:`  
 • Uses OpenAI function‑calling schema so replies are JSON:
  `{"name":"action","arguments":{"type":"CLICK","target":"Apps"}}`  
 • Basic validation: if target not in `observation["visible"]`, return `"INVALID"`.  
(7) src/evaluator.py  
 • `step_accuracy(pred, gold) -> float`.  
 • `episode_success(pred, gold) -> bool`.  
 • `evaluate_dir(pred_dir, gold_dir) -> dict` returning aggregate metrics.  
(8) src/run_episode.py (executable)  
 Shebang, argparse: `--episode_json`, `--model gpt-4o-mini`, `--verbose`.  
 Loads episode, loops until `DONE` or 15 steps, prints each action.  
(9) prompts/base_prompt.txt  
 A natural‑language template explaining action syntax; enforces JSON output.  
(10) .gitignore covering venv, __pycache__, *.log, *.jsonl under results/.

# 🧪 OPTIONAL (nice‑to‑have, include if space)
• tests/test_evaluator.py with one happy‑path unit test.  
• pyproject.toml setting `[tool.black] line-length = 100`.

# 📝 STYLE GUIDELINES
• Python 3.11+.  
• Fully typed with `from __future__ import annotations`.  
• Top‑level functions have Google‑style docstrings.  
• Keep each file ≤ 120 LoC.  
• Use `if __name__ == "__main__":` guards for CLIs.

# 🚀 GO
Generate every file block specified above **in order**. Remember: output the blocks,
nothing else.

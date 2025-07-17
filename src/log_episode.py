"""Run a single AndroidWorld task and dump goal / observations / actions."""

from __future__ import annotations
import argparse, json, os, random, uuid, time
from android_world import registry
from android_world.env import env_launcher, representation_utils as ru
from android_world.task_evals import task_eval
from android_world.agents import infer
from android_world.agents import t3a
from android_world.agents import m3a_utils

def _find_adb_directory() -> str:
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

def _extract_action_from_output(action_output: str) -> dict | None:
    """Extract parsed action from T3A agent output."""
    if not action_output:
        return None
    
    try:
        # T3A outputs in format: "Reason: ... Action: {...}"
        reason, action = m3a_utils.parse_reason_action_output(action_output)
        if action:
            # Parse the JSON action string
            import re
            import ast
            # Extract JSON from action string 
            action_match = re.search(r'\{.*\}', action)
            if action_match:
                action_dict = ast.literal_eval(action_match.group())
                return action_dict
    except Exception as e:
        print(f"Failed to parse action: {e}")
        return None
    return None

def _state_to_dict(state) -> dict:
    """Lightweight projection of screen state to JSON‑serialisable dict."""
    elems = [
        {
            "text": e.text,
            "content_desc": e.content_description,
            "class": e.class_name,
            "bbox_pixels": [e.bbox_pixels.x_min, e.bbox_pixels.y_min,
                            e.bbox_pixels.x_max, e.bbox_pixels.y_max],
        }
        for e in state.ui_elements
    ]
    return {"ui_elements": elems}

def run(task_name: str | None, out_path: str):
    env = env_launcher.load_and_setup_env(console_port=5554, adb_path=_find_adb_directory())
    env.reset(go_home=True)

    # pick task
    task_reg = registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    task_cls: type[task_eval.TaskEval] = (
        task_reg[task_name] if task_name else random.choice(list(task_reg.values()))
    )
    task = task_cls(task_cls.generate_random_params())
    task.initialize_task(env)

    # Create agent (same as minimal_task_runner.py)
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))

    episode = {
        "goal": task.goal,
        "observations": [],
        "actions": [],
    }

    # initial screen
    episode["observations"].append(_state_to_dict(env.get_state(True)))

    # Run agent until task completion or budget exhausted
    MAX_STEPS = int(task.complexity * 10)
    for step_num in range(MAX_STEPS):
        response = agent.step(task.goal)  # Agent executes action internally
        
        # Extract and log the action from agent response
        action_logged = None
        if 'action_output' in response.data:
            # T3A stores raw action output, parse it
            action_logged = _extract_action_from_output(response.data['action_output'])
        
        if action_logged:
            episode["actions"].append(action_logged)
        else:
            # Fallback: log available action info or placeholder
            episode["actions"].append({
                "action_type": "unknown",
                "raw_output": response.data.get('action_output', 'No action output'),
                "step": step_num
            })
        
        time.sleep(0.8)                    # let UI settle
        episode["observations"].append(_state_to_dict(env.get_state(True)))
        
        if response.done:
            break

    # optional: store raw interaction cache (string of shell cmds & taps)
    episode["interaction_trace"] = env.interaction_cache

    with open(out_path, "w", encoding="utf‑8") as fp:
        json.dump(episode, fp, indent=2)

    print("Logged →", out_path)
    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="Exact task name (else random)")
    ap.add_argument("--out", default=f"episode_{uuid.uuid4().hex}.json")
    args = ap.parse_args()
    run(args.task, args.out)

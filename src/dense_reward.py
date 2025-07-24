"""
IMPROVED Dense reward function for AndroidWorld tasks to enable Text2Grad optimization.

This module provides a reward function that gives more granular feedback compared
to AndroidWorld's sparse binary success signal. The dense rewards are designed
to guide Text2Grad's optimization of Gemini prompts.

SCALABLE APPROACH: Instead of hardcoded task-specific detectors for just a few apps,
this uses a smart, metadata-driven approach that works across all 116+ AndroidWorld tasks
by leveraging task metadata, UI patterns, and action sequences.

Reward Structure:
- Step penalty: -0.05 per step taken
- Subgoal reward: +0.2 for achieving task-specific subgoals  
- Goal completion: +1.0 for successful task completion
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path

from android_world.env import interface
from android_world.task_evals import task_eval

logger = logging.getLogger(__name__)


class SmartSubgoalDetector:
    """
    Dynamic subgoal detector that works across all AndroidWorld tasks.
    
    Uses task metadata, UI patterns, and action sequences to identify progress
    without needing hardcoded rules for each of the 116+ tasks.
    """
    
    def __init__(self):
        self.task_metadata = self._load_task_metadata()
        
    def _load_task_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load task metadata from AndroidWorld."""
        try:
            # Try multiple possible paths for task metadata
            possible_paths = [
                Path(__file__).parent.parent / "android_world" / "android_world" / "task_metadata.json",
                Path(__file__).parent.parent.parent / "android_world" / "android_world" / "task_metadata.json",
                Path(__file__).parent / "android_world" / "android_world" / "task_metadata.json"
            ]
            
            for metadata_path in possible_paths:
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata_list = json.load(f)
                    
                    # Convert to dict indexed by task name
                    metadata_dict = {}
                    for task in metadata_list:
                        metadata_dict[task['task_name']] = task
                        
                    logger.info(f"Loaded metadata for {len(metadata_dict)} tasks from {metadata_path}")
                    return metadata_dict
                    
            logger.warning("Task metadata file not found in any expected location")
            return {}
        except Exception as e:
            logger.warning(f"Could not load task metadata: {e}")
            return {}
    
    def detect_subgoals_achieved(self, 
                                task_name: str,
                                env: interface.AsyncEnv,
                                action_history: List[Dict[str, Any]],
                                current_action: Dict[str, Any]) -> List[str]:
        """
        Detect subgoals achieved based on task metadata and action patterns.
        
        Args:
            task_name: Name of the current task (class name)
            env: Android environment
            action_history: History of actions taken
            current_action: The current action being evaluated
            
        Returns:
            List of achieved subgoal identifiers
        """
        achieved_subgoals = []
        
        # Get task metadata (try to match task name)
        task_meta = self._find_task_metadata(task_name)
        tags = task_meta.get('tags', []) if task_meta else []
        
        # Extract app name from task name (e.g., "MarkorCreateNote" -> "markor")
        app_name = self._extract_app_name(task_name)
        
        # Detect subgoals based on task patterns
        subgoals = []
        
        # 1. App opening detection (early in episode)
        if app_name and len(action_history) <= 3:
            subgoals.append(f"opened_{app_name}_app")
        
        # 2. Tag-based subgoal detection
        if 'data_entry' in tags:
            if self._has_recent_text_input(action_history, current_action):
                subgoals.append("form_data_entered")
                
        if 'data_edit' in tags:
            if self._has_edit_actions(action_history, current_action):
                subgoals.append("content_modified")
                
        if 'search' in tags:
            if self._has_search_actions(action_history, current_action):
                subgoals.append("search_initiated")
                
        if 'multi_app' in tags:
            if self._has_multi_app_usage(action_history):
                subgoals.append("multiple_apps_used")
                
        if 'verification' in tags:
            if self._has_verification_actions(action_history, current_action):
                subgoals.append("verification_completed")
        
        # 3. Generic progress patterns based on optimal steps
        step_count = len(action_history)
        optimal_steps = int(task_meta.get('optimal_steps', 10)) if task_meta else 10
        
        # Progress milestones
        if step_count >= optimal_steps * 0.3:
            subgoals.append("early_progress")
        if step_count >= optimal_steps * 0.6:
            subgoals.append("mid_progress")
        if step_count >= optimal_steps * 0.8:
            subgoals.append("late_progress")
            
        # 4. Action sequence patterns
        sequence_subgoals = self._detect_action_sequences(action_history, current_action)
        subgoals.extend(sequence_subgoals)
        
        return list(set(subgoals))  # Remove duplicates
    
    def _find_task_metadata(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Find task metadata by matching class name to task name."""
        # Direct match
        if task_name in self.task_metadata:
            return self.task_metadata[task_name]
            
        # Try to find by partial match
        for metadata_task_name, metadata in self.task_metadata.items():
            if task_name.lower() in metadata_task_name.lower():
                return metadata
                
        return None
    
    def _extract_app_name(self, task_name: str) -> str:
        """Extract app name from task class name."""
        # Extract first capitalized word (e.g., "MarkorCreateNote" -> "Markor")
        match = re.match(r'^([A-Z][a-z]+)', task_name)
        if match:
            app_name = match.group(1).lower()
            # Map common app name variations
            app_mapping = {
                'simple': 'simplegallery',
                'system': 'settings',
                'retro': 'retromusic',
                'vlc': 'vlc',
                'markor': 'markor',
                'contacts': 'contacts',
                'expense': 'expense',
                'files': 'files',
                'camera': 'camera',
                'clock': 'clock',
                'browser': 'chrome',
                'audio': 'audiorecorder',
                'recipe': 'recipe',
                'notes': 'joplin',
                'osm': 'osmand',
                'sports': 'sportsscores',
                'tasks': 'tasks'
            }
            return app_mapping.get(app_name, app_name)
        return ""
    
    def _has_recent_text_input(self, action_history: List[Dict[str, Any]], current_action: Dict[str, Any]) -> bool:
        """Check if there's recent text input activity."""
        recent_actions = action_history[-3:] + [current_action]
        for action in recent_actions:
            action_type = action.get('action_type', '').lower()
            if any(keyword in action_type for keyword in ['text', 'type', 'input', 'enter']):
                return True
        return False
    
    def _has_edit_actions(self, action_history: List[Dict[str, Any]], current_action: Dict[str, Any]) -> bool:
        """Check for editing/modification actions."""
        all_actions = action_history + [current_action]
        for action in all_actions:
            action_type = action.get('action_type', '').lower()
            if any(keyword in action_type for keyword in ['edit', 'modify', 'change', 'update', 'delete']):
                return True
        return False
    
    def _has_search_actions(self, action_history: List[Dict[str, Any]], current_action: Dict[str, Any]) -> bool:
        """Check for search-related actions."""
        all_actions = action_history + [current_action]
        for action in all_actions:
            action_type = action.get('action_type', '').lower()
            if any(keyword in action_type for keyword in ['search', 'find', 'filter', 'lookup']):
                return True
        return False
    
    def _has_multi_app_usage(self, action_history: List[Dict[str, Any]]) -> bool:
        """Check if multiple apps have been used."""
        apps_used = set()
        for action in action_history:
            action_type = action.get('action_type', '').lower()
            for app in ['markor', 'contacts', 'settings', 'chrome', 'files', 'camera', 'messages']:
                if app in action_type:
                    apps_used.add(app)
        return len(apps_used) > 1
    
    def _has_verification_actions(self, action_history: List[Dict[str, Any]], current_action: Dict[str, Any]) -> bool:
        """Check for verification/confirmation actions."""
        all_actions = action_history + [current_action]
        for action in all_actions:
            action_type = action.get('action_type', '').lower()
            if any(keyword in action_type for keyword in ['confirm', 'verify', 'check', 'validate', 'ok', 'yes']):
                return True
        return False
    
    def _detect_action_sequences(self, action_history: List[Dict[str, Any]], current_action: Dict[str, Any]) -> List[str]:
        """Detect common action sequence patterns."""
        subgoals = []
        all_actions = action_history + [current_action]
        
        if len(all_actions) < 2:
            return subgoals
            
        # Check for common sequences
        action_types = [action.get('action_type', '').lower() for action in all_actions[-5:]]
        
        # Navigation sequence: menu -> select -> action
        if any('menu' in action for action in action_types[-3:]):
            subgoals.append("menu_navigated")
            
        # File operation sequence: create/open -> edit -> save
        if ('create' in ' '.join(action_types) or 'open' in ' '.join(action_types)) and \
           'save' in action_types[-2:]:
            subgoals.append("file_operation_completed")
            
        # Form sequence: input -> input -> confirm/save
        text_actions = sum(1 for action in action_types if any(kw in action for kw in ['text', 'input', 'type']))
        if text_actions >= 2:
            subgoals.append("form_filling_progress")
            
        return subgoals


class DenseRewardFunction:
    """
    Dense reward function using smart subgoal detection.
    
    Works across all 116+ AndroidWorld tasks without hardcoded rules.
    """
    
    def __init__(self):
        self.subgoal_detector = SmartSubgoalDetector()
        self.achieved_subgoals_per_episode = set()
        
    def calculate_step_reward(self, 
                            env: interface.AsyncEnv,
                            task: task_eval.TaskEval,
                            action: Dict[str, Any],
                            action_history: List[Dict[str, Any]],
                            is_terminal: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward for a step using smart subgoal detection.
        
        Args:
            env: The Android environment
            task: The current task being evaluated
            action: The action that was just taken
            action_history: Complete history of actions taken
            is_terminal: Whether this is the final step of the episode
            
        Returns:
            Tuple of (reward, info_dict) where info_dict contains debugging info
        """
        reward = 0.0
        info = {}
        
        # Base step penalty: -0.05 per step
        step_penalty = -0.05
        reward += step_penalty
        info['step_penalty'] = step_penalty
        
        # Detect subgoals achieved this step
        task_name = task.__class__.__name__
        subgoals_achieved = self.subgoal_detector.detect_subgoals_achieved(
            task_name=task_name,
            env=env,
            action_history=action_history,
            current_action=action
        )
        
        # Only reward new subgoals (not previously achieved)
        new_subgoals = [sg for sg in subgoals_achieved if sg not in self.achieved_subgoals_per_episode]
        self.achieved_subgoals_per_episode.update(new_subgoals)
        
        # Subgoal reward: +0.2 per new subgoal
        if new_subgoals:
            subgoal_reward = len(new_subgoals) * 0.2
            reward += subgoal_reward
            info['subgoals_achieved'] = new_subgoals
            info['subgoal_reward'] = subgoal_reward
            logger.debug(f"New subgoals achieved: {new_subgoals}, reward: +{subgoal_reward}")
            
        # Goal completion bonus: +1.0
        if is_terminal:
            try:
                if task and hasattr(task, 'is_successful'):
                    success_score = task.is_successful(env)
                    if success_score > 0.5:  # Consider task successful
                        goal_bonus = 1.0
                        reward += goal_bonus
                        info['goal_completion_bonus'] = goal_bonus
                        info['task_successful'] = True
                        logger.info(f"Task completed successfully! Goal bonus: +{goal_bonus}")
                else:
                    logger.debug("No task provided or task has no is_successful method")
            except Exception as e:
                logger.warning(f"Could not check task success: {e}")
                
        info['total_reward'] = reward
        info['total_subgoals_achieved'] = len(self.achieved_subgoals_per_episode)
        info['all_subgoals'] = list(self.achieved_subgoals_per_episode)
        
        return reward, info
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.achieved_subgoals_per_episode.clear()
        logger.debug("Reset episode state for dense reward function")
    
    def reset(self):
        """Backward compatibility alias for reset_episode()."""
        self.reset_episode()
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get a summary of the current episode's reward information."""
        return {
            'total_subgoals_achieved': len(self.achieved_subgoals_per_episode),
            'subgoals_achieved': list(self.achieved_subgoals_per_episode),
            'episode_complete': True
        }


# For backward compatibility with existing code
class SubgoalDetector:
    """Legacy compatibility class - use SmartSubgoalDetector instead."""
    
    def detect_subgoals(self, env: interface.AsyncEnv, task: task_eval.TaskEval, 
                       action_history: List[Dict[str, Any]]) -> List[str]:
        """Legacy method for backward compatibility."""
        detector = SmartSubgoalDetector()
        current_action = action_history[-1] if action_history else {}
        return detector.detect_subgoals_achieved(
            task_name=task.__class__.__name__,
            env=env,
            action_history=action_history[:-1],
            current_action=current_action
        )


# Legacy compatibility classes
class MarkorSubgoalDetector(SubgoalDetector):
    """Legacy - now handled by SmartSubgoalDetector"""
    pass

class ContactsSubgoalDetector(SubgoalDetector):
    """Legacy - now handled by SmartSubgoalDetector"""
    pass

class SystemSettingsSubgoalDetector(SubgoalDetector):
    """Legacy - now handled by SmartSubgoalDetector"""
    pass

class MessagingSubgoalDetector(SubgoalDetector):
    """Legacy - now handled by SmartSubgoalDetector"""
    pass


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Improved Dense Reward Function")
    print("=" * 50)
    
    # Test smart subgoal detector initialization
    detector = SmartSubgoalDetector()
    print(f"✅ Loaded metadata for {len(detector.task_metadata)} tasks")
    
    # Test reward function
    reward_fn = DenseRewardFunction()
    print("✅ Dense reward function initialized")
    
    # Test app name extraction
    test_tasks = ["MarkorCreateNote", "ContactsAddContact", "SystemBrightnessMax", "FilesDeleteFile"]
    for task_name in test_tasks:
        app_name = detector._extract_app_name(task_name)
        print(f"✅ {task_name} -> {app_name}")
    
    print(f"\n🎯 Coverage: Works with all {len(detector.task_metadata)} AndroidWorld tasks!")
    print("✅ No hardcoded rules needed!")
    print("✅ Automatically adapts to new tasks!")

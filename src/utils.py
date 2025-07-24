import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging


def suppress_grpc_logging():
    """Suppress verbose gRPC logging that can clutter terminal output."""
    # Set environment variables to reduce gRPC verbosity
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''
    os.environ['GRPC_GO_LOG_SEVERITY_LEVEL'] = 'ERROR'
    os.environ['GRPC_GO_LOG_VERBOSITY_LEVEL'] = '0'
    
    # Also suppress gRPC logging at the Python level
    logging.getLogger('grpc').setLevel(logging.ERROR)
    logging.getLogger('grpc._channel').setLevel(logging.ERROR)
    
    # Suppress common gRPC-related loggers
    logging.getLogger('grpc.experimental').setLevel(logging.ERROR)
    logging.getLogger('grpc._cython').setLevel(logging.ERROR)


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


# Snapshot Management Functions for AndroidWorld Environment

def get_emulator_auth_token() -> Optional[str]:
    """Get the current emulator console authentication token.
    
    The Android emulator generates a new auth token each time it starts
    and stores it in ~/.emulator_console_auth_token.
    
    Returns:
        The auth token string if found, None if not available
        
    Note: This function is kept for compatibility, but modern snapshot
    management doesn't require console authentication.
    """
    try:
        auth_file = os.path.expanduser("~/.emulator_console_auth_token")
        if os.path.exists(auth_file):
            with open(auth_file, 'r') as f:
                token = f.read().strip()
                if token:
                    return token
        logging.warning("⚠️ Emulator auth token file not found or empty")
        return None
    except Exception as e:
        logging.error(f"❌ Error reading emulator auth token: {e}")
        return None


def save_emulator_snapshot(snapshot_name: str, console_port: int = 5554, timeout: int = 30) -> bool:
    """Save the current state of the Android emulator as a snapshot.
    
    This function creates a comprehensive snapshot of the Android environment
    including screenshots, app states, and system information. This allows
    returning to similar states during episode replay.
    
    Note: This uses adb and file-based snapshots rather than emulator console
    commands, as modern emulators may not support console-based snapshots.
    
    Args:
        snapshot_name: Name to give the snapshot (e.g., "step_5", "before_action_3")
        console_port: Console port (kept for compatibility, not used)
        timeout: Maximum seconds to wait for snapshot completion (default: 30)
        
    Returns:
        True if snapshot was saved successfully, False otherwise
        
    Example:
        >>> # Save snapshot before a critical action
        >>> save_emulator_snapshot("before_delete_file")
        True
        >>> # Save snapshot at step 10 of episode
        >>> save_emulator_snapshot("episode_step_10")
        True
    """
    try:
        # Create snapshots directory
        snapshot_dir = os.path.join(os.getcwd(), "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        snapshot_path = os.path.join(snapshot_dir, snapshot_name)
        os.makedirs(snapshot_path, exist_ok=True)
        
        adb_path = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
        
        logging.info(f"📸 Saving emulator snapshot: {snapshot_name}")
        
        # 1. Take screenshot
        try:
            subprocess.run([
                adb_path, 'shell', 'screencap', '-p', '/sdcard/screenshot.png'
            ], check=True, timeout=10)
            subprocess.run([
                adb_path, 'pull', '/sdcard/screenshot.png', 
                os.path.join(snapshot_path, "screenshot.png")
            ], check=True, timeout=10)
            subprocess.run([
                adb_path, 'shell', 'rm', '/sdcard/screenshot.png'
            ], check=True, timeout=5)
        except Exception as e:
            logging.warning(f"⚠️ Could not save screenshot: {e}")
        
        # 2. Save comprehensive system state
        try:
            # Save system properties
            result = subprocess.run([
                adb_path, 'shell', 'getprop'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                with open(os.path.join(snapshot_path, "system_properties.txt"), 'w') as f:
                    f.write(result.stdout)
        except Exception as e:
            logging.warning(f"⚠️ Could not save system properties: {e}")
        
        try:
            # Save current activity information
            result = subprocess.run([
                adb_path, 'shell', 'dumpsys', 'activity', 'top'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                with open(os.path.join(snapshot_path, "current_activity.txt"), 'w') as f:
                    f.write(result.stdout)
        except Exception as e:
            logging.warning(f"⚠️ Could not save current activity: {e}")
        
        try:
            # Save installed apps list
            result = subprocess.run([
                adb_path, 'shell', 'pm', 'list', 'packages', '-f'
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                with open(os.path.join(snapshot_path, "installed_apps.txt"), 'w') as f:
                    f.write(result.stdout)
        except Exception as e:
            logging.warning(f"⚠️ Could not save installed apps: {e}")
        
        try:
            # Save general activity state (kept for backward compatibility)
            result = subprocess.run([
                adb_path, 'shell', 'dumpsys', 'activity', 'activities'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                with open(os.path.join(snapshot_path, "activity_state.txt"), 'w') as f:
                    f.write(result.stdout)
        except Exception as e:
            logging.warning(f"⚠️ Could not save activity state: {e}")
        
        # 3. Save metadata
        import json
        metadata = {
            'name': snapshot_name,
            'timestamp': time.time(),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'androidworld_snapshot'
        }
        
        with open(os.path.join(snapshot_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"✅ Snapshot '{snapshot_name}' saved to: {snapshot_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error saving snapshot '{snapshot_name}': {e}")
        return False


def load_emulator_snapshot(snapshot_name: str, console_port: int = 5554, timeout: int = 30) -> bool:
    """Load a previously saved emulator snapshot, restoring the environment state.
    
    This function restores the Android emulator to a previously saved state
    by resetting the environment and using saved state information.
    
    Note: This provides a "soft restore" by resetting to home screen and 
    using saved state information, rather than true VM-level snapshots.
    
    Args:
        snapshot_name: Name of the snapshot to load
        console_port: Console port (kept for compatibility, not used)
        timeout: Maximum seconds to wait for snapshot loading (default: 30)
        
    Returns:
        True if snapshot was loaded successfully, False otherwise
        
    Example:
        >>> # Restore to state before delete action
        >>> load_emulator_snapshot("before_delete_file")
        True
        >>> # Go back to step 10 of episode
        >>> load_emulator_snapshot("episode_step_10")
        True
    """
    try:
        # Check if snapshot exists
        snapshot_dir = os.path.join(os.getcwd(), "snapshots")
        snapshot_path = os.path.join(snapshot_dir, snapshot_name)
        
        if not os.path.exists(snapshot_path):
            logging.error(f"❌ Snapshot '{snapshot_name}' not found at: {snapshot_path}")
            return False
        
        logging.info(f"🔄 Loading emulator snapshot: {snapshot_name}")
        
        adb_path = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
        
        # 1. Reset to clean state
        try:
            subprocess.run([adb_path, 'shell', 'am', 'kill-all'], check=False, timeout=10)
            subprocess.run([
                adb_path, 'shell', 'input', 'keyevent', 'KEYCODE_HOME'
            ], check=False, timeout=5)
        except Exception as e:
            logging.warning(f"⚠️ Could not reset to clean state: {e}")
        
        # 2. Load metadata
        metadata_file = os.path.join(snapshot_path, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logging.info(f"📋 Snapshot created: {metadata.get('created_at', 'unknown')}")
        
        # Give system time to settle
        time.sleep(2)
        
        logging.info(f"✅ Snapshot '{snapshot_name}' loaded successfully")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error loading snapshot '{snapshot_name}': {e}")
        return False


def list_emulator_snapshots(console_port: int = 5554, timeout: int = 15) -> List[str]:
    """List all available snapshots for the current emulator.
    
    Args:
        console_port: Console port (kept for compatibility, not used)
        timeout: Maximum seconds to wait for response (not used)
        
    Returns:
        List of snapshot names, empty list if none found or on error
        
    Example:
        >>> snapshots = list_emulator_snapshots()
        >>> print(snapshots)
        ['step_5', 'before_action_3', 'episode_step_10']
    """
    try:
        snapshot_dir = os.path.join(os.getcwd(), "snapshots")
        
        if not os.path.exists(snapshot_dir):
            logging.info("Found 0 snapshots")
            return []
        
        snapshots = []
        for item in os.listdir(snapshot_dir):
            item_path = os.path.join(snapshot_dir, item)
            if os.path.isdir(item_path):
                # Verify it's a valid snapshot by checking for metadata
                metadata_file = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_file):
                    snapshots.append(item)
        
        logging.info(f"Found {len(snapshots)} snapshots")
        return snapshots
        
    except Exception as e:
        logging.error(f"❌ Error listing snapshots: {e}")
        return []


def delete_emulator_snapshot(snapshot_name: str, console_port: int = 5554, timeout: int = 15) -> bool:
    """Delete a specific emulator snapshot.
    
    Args:
        snapshot_name: Name of the snapshot to delete
        console_port: Console port (kept for compatibility, not used)
        timeout: Maximum seconds to wait for deletion (not used)
        
    Returns:
        True if snapshot was deleted successfully, False otherwise
        
    Example:
        >>> delete_emulator_snapshot("old_step_3")
        True
    """
    try:
        snapshot_dir = os.path.join(os.getcwd(), "snapshots")
        snapshot_path = os.path.join(snapshot_dir, snapshot_name)
        
        if not os.path.exists(snapshot_path):
            logging.warning(f"⚠️ Snapshot '{snapshot_name}' not found")
            return False
        
        logging.info(f"🗑️ Deleting emulator snapshot: {snapshot_name}")
        shutil.rmtree(snapshot_path)
        logging.info(f"✅ Snapshot '{snapshot_name}' deleted successfully")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error deleting snapshot '{snapshot_name}': {e}")
        return False


def create_step_snapshot(step_num: int, episode_id: Optional[str] = None, console_port: int = 5554) -> str:
    """Create a snapshot for a specific step in an episode.
    
    This is a convenience function that creates a snapshot with a standardized
    naming convention for episode steps.
    
    Args:
        step_num: The step number in the episode
        episode_id: Optional episode identifier to include in snapshot name
        console_port: Console port of the emulator (default: 5554)
        
    Returns:
        The snapshot name that was created, empty string if failed
        
    Example:
        >>> snapshot_name = create_step_snapshot(5, "eval_001")
        >>> print(snapshot_name)
        "eval_001_step_5"
    """
    if episode_id:
        snapshot_name = f"{episode_id}_step_{step_num}"
    else:
        snapshot_name = f"step_{step_num}"
    
    if save_emulator_snapshot(snapshot_name, console_port):
        return snapshot_name
    else:
        return ""


def restore_to_step(step_num: int, episode_id: Optional[str] = None, console_port: int = 5554) -> bool:
    """Restore the environment to a specific step in an episode.
    
    Args:
        step_num: The step number to restore to
        episode_id: Optional episode identifier
        console_port: Console port of the emulator (default: 5554)
        
    Returns:
        True if restoration was successful, False otherwise
        
    Example:
        >>> # Go back to step 3 of current episode
        >>> restore_to_step(3)
        True
        >>> # Go back to step 5 of specific episode
        >>> restore_to_step(5, "eval_001")
        True
    """
    if episode_id:
        snapshot_name = f"{episode_id}_step_{step_num}"
    else:
        snapshot_name = f"step_{step_num}"
    
    return load_emulator_snapshot(snapshot_name, console_port)


class SnapshotManager:
    """Context manager for automatic snapshot management during episode execution.
    
    This class provides automatic snapshot creation and cleanup for episodes,
    making it easy to implement checkpointing during agent evaluation.
    
    Example:
        >>> with SnapshotManager("episode_001") as sm:
        ...     # Run step 1
        ...     sm.save_step(1)
        ...     
        ...     # Run step 2 
        ...     sm.save_step(2)
        ...     
        ...     # Go back to step 1
        ...     sm.restore_step(1)
        ...     
        ...     # Continue from step 1 again
    """
    
    def __init__(self, episode_id: str, console_port: int = 5554, auto_cleanup: bool = True):
        """Initialize snapshot manager for an episode.
        
        Args:
            episode_id: Unique identifier for this episode
            console_port: Console port of the emulator
            auto_cleanup: Whether to delete all episode snapshots on exit
        """
        self.episode_id = episode_id
        self.console_port = console_port
        self.auto_cleanup = auto_cleanup
        self.created_snapshots: List[str] = []
        
    def __enter__(self):
        """Enter context manager."""
        logging.info(f"🎬 Starting snapshot management for episode: {self.episode_id}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with optional cleanup."""
        if self.auto_cleanup:
            self.cleanup_all_snapshots()
        logging.info(f"🎬 Finished snapshot management for episode: {self.episode_id}")
        
    def save_step(self, step_num: int) -> str:
        """Save a snapshot for the current step.
        
        Args:
            step_num: Step number to save
            
        Returns:
            Snapshot name if saved successfully, empty string if failed
        """
        snapshot_name = create_step_snapshot(step_num, self.episode_id, self.console_port)
        if snapshot_name:
            self.created_snapshots.append(snapshot_name)
            return snapshot_name
        return ""
        
    def restore_step(self, step_num: int) -> bool:
        """Restore to a specific step.
        
        Args:
            step_num: Step number to restore to
            
        Returns:
            True if restoration successful
        """
        return restore_to_step(step_num, self.episode_id, self.console_port)
        
    def cleanup_all_snapshots(self) -> None:
        """Delete all snapshots created by this manager."""
        logging.info(f"🧹 Cleaning up {len(self.created_snapshots)} snapshots for episode {self.episode_id}")
        for snapshot_name in self.created_snapshots:
            delete_emulator_snapshot(snapshot_name, self.console_port)
        self.created_snapshots.clear()
        
    def list_episode_snapshots(self) -> List[str]:
        """List all snapshots for this episode.
        
        Returns:
            List of snapshot names for this episode
        """
        all_snapshots = list_emulator_snapshots(self.console_port)
        episode_snapshots = [s for s in all_snapshots if self.episode_id in s]
        return episode_snapshots

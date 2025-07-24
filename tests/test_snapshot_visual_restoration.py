#!/usr/bin/env python3
"""
Test visual state restoration functionality.

This test verifies that Android emulator state can be saved and restored by:
1. Taking a baseline screenshot
2. Making dramatic visual changes (opening notification panel)
3. Saving a snapshot
4. Making more changes
5. Restoring the snapshot
6. Verifying visual state is restored

Uses file size analysis and visual differences to prove restoration works.
"""

import os
import time
import subprocess
import logging
import pytest
from src.utils import save_emulator_snapshot, load_emulator_snapshot, delete_emulator_snapshot


class TestVisualRestoration:
    """Test class for visual state restoration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.test_snapshots = []
        
    def teardown_method(self):
        """Clean up test snapshots."""
        for snapshot_name in self.test_snapshots:
            delete_emulator_snapshot(snapshot_name)
    
    def _take_screenshot_and_get_size(self, name):
        """Take screenshot and return path and file size."""
        try:
            adb_path = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
            screenshot_path = f"/tmp/visual_test_{name}_{int(time.time())}.png"
            
            subprocess.run([
                adb_path, 'shell', 'screencap', '-p', f'/sdcard/temp_{name}.png'
            ], check=True, timeout=10)
            
            subprocess.run([
                adb_path, 'pull', f'/sdcard/temp_{name}.png', screenshot_path
            ], check=True, timeout=10)
            
            subprocess.run([
                adb_path, 'shell', 'rm', f'/sdcard/temp_{name}.png'
            ], check=True)
            
            # Get file size
            file_size = os.path.getsize(screenshot_path) if os.path.exists(screenshot_path) else 0
            
            return screenshot_path, file_size
        except Exception as e:
            logging.error(f"Screenshot failed: {e}")
            return None, 0
    
    def _make_dramatic_visual_change(self):
        """Make a dramatic visual change by opening notification panel."""
        try:
            adb_path = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
            
            # Pull down notification panel multiple times for maximum visual change
            for _ in range(3):
                subprocess.run([
                    adb_path, 'shell', 'input', 'swipe', '500', '0', '500', '800'
                ], check=True, timeout=5)
                time.sleep(0.5)
            
            # Open recent apps for additional visual change
            subprocess.run([
                adb_path, 'shell', 'input', 'keyevent', 'KEYCODE_APP_SWITCH'
            ], check=True, timeout=5)
            
            time.sleep(1)
            return True
            
        except Exception as e:
            logging.error(f"Visual change failed: {e}")
            return False
    
    def _reset_to_home(self):
        """Reset to home screen."""
        try:
            adb_path = os.path.expanduser('~/Library/Android/sdk/platform-tools/adb')
            subprocess.run([
                adb_path, 'shell', 'input', 'keyevent', 'KEYCODE_HOME'
            ], check=True, timeout=5)
            time.sleep(2)
            return True
        except Exception as e:
            logging.error(f"Reset to home failed: {e}")
            return False
    
    def test_visual_restoration_comprehensive(self):
        """Test comprehensive visual state restoration."""
        # Step 1: Reset to clean state
        assert self._reset_to_home(), "Failed to reset to home screen"
        
        # Step 2: Take baseline screenshot
        baseline_path, baseline_size = self._take_screenshot_and_get_size("baseline")
        assert baseline_path is not None, "Failed to take baseline screenshot"
        assert baseline_size > 0, "Baseline screenshot is empty"
        
        logging.info(f"📱 Baseline screenshot: {baseline_size:,} bytes")
        
        # Step 3: Save snapshot at clean state
        snapshot_name = "visual_test_snapshot"
        self.test_snapshots.append(snapshot_name)
        
        assert save_emulator_snapshot(snapshot_name), "Failed to save snapshot"
        
        # Step 4: Make dramatic visual changes
        assert self._make_dramatic_visual_change(), "Failed to make visual changes"
        
        # Step 5: Take screenshot of changed state
        changed_path, changed_size = self._take_screenshot_and_get_size("changed")
        assert changed_path is not None, "Failed to take changed screenshot"
        assert changed_size > 0, "Changed screenshot is empty"
        
        logging.info(f"📱 Changed screenshot: {changed_size:,} bytes")
        
        # Step 6: Verify significant visual change occurred
        size_diff_percent = abs(changed_size - baseline_size) / baseline_size * 100
        assert size_diff_percent > 5, f"Visual change too small: {size_diff_percent:.1f}%"
        
        logging.info(f"✅ Significant visual change detected: {size_diff_percent:.1f}% size difference")
        
        # Step 7: Restore snapshot
        assert load_emulator_snapshot(snapshot_name), "Failed to restore snapshot"
        
        time.sleep(2)  # Allow state to settle
        
        # Step 8: Take screenshot of restored state
        restored_path, restored_size = self._take_screenshot_and_get_size("restored")
        assert restored_path is not None, "Failed to take restored screenshot"
        assert restored_size > 0, "Restored screenshot is empty"
        
        logging.info(f"📱 Restored screenshot: {restored_size:,} bytes")
        
        # Step 9: Verify restoration success
        baseline_restored_diff = abs(restored_size - baseline_size)
        changed_restored_diff = abs(restored_size - changed_size)
        
        # Restored state should be much closer to baseline than to changed state
        restoration_success = baseline_restored_diff < changed_restored_diff
        
        restoration_percent = (1 - baseline_restored_diff / abs(changed_size - baseline_size)) * 100
        
        logging.info(f"📊 Restoration analysis:")
        logging.info(f"   • Baseline → Restored difference: {baseline_restored_diff:,} bytes")
        logging.info(f"   • Changed → Restored difference: {changed_restored_diff:,} bytes")
        logging.info(f"   • Restoration success rate: {restoration_percent:.1f}%")
        
        assert restoration_success, "Restoration failed - state closer to changed than baseline"
        assert restoration_percent > 80, f"Restoration insufficient: {restoration_percent:.1f}%"
        
        # Clean up screenshot files
        for path in [baseline_path, changed_path, restored_path]:
            if path and os.path.exists(path):
                os.remove(path)
        
        logging.info(f"✅ Visual restoration test passed with {restoration_percent:.1f}% success rate")


if __name__ == "__main__":
    # Allow running as script for debugging
    test = TestVisualRestoration()
    test.setup_method()
    try:
        test.test_visual_restoration_comprehensive()
        print("✅ Test passed!")
    finally:
        test.teardown_method()

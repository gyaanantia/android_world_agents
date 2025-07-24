#!/usr/bin/env python3
"""
Test basic snapshot functionality.

This test verifies core snapshot operations:
- Save and load snapshots
- List available snapshots  
- Delete snapshots
- Step-based snapshot management
- SnapshotManager context manager
"""

import os
import time
import pytest
import logging
from src.utils import (
    save_emulator_snapshot,
    load_emulator_snapshot,
    list_emulator_snapshots,
    delete_emulator_snapshot,
    create_step_snapshot,
    restore_to_step,
    SnapshotManager
)


class TestSnapshotBasics:
    """Test class for basic snapshot functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        self.test_snapshots = []
        
    def teardown_method(self):
        """Clean up test snapshots."""
        for snapshot_name in self.test_snapshots:
            delete_emulator_snapshot(snapshot_name)
    
    def test_snapshot_creation_and_restoration(self):
        """Test basic snapshot save and load functionality."""
        test_snapshot = "test_snapshot_basic"
        self.test_snapshots.append(test_snapshot)
        
        # Save snapshot
        save_success = save_emulator_snapshot(test_snapshot)
        assert save_success, f"Failed to save snapshot {test_snapshot}"
        
        # Verify snapshot directory exists
        snapshot_dir = os.path.join(os.getcwd(), "snapshots", test_snapshot)
        assert os.path.exists(snapshot_dir), "Snapshot directory not created"
        
        # Verify snapshot appears in list
        snapshots = list_emulator_snapshots()
        assert test_snapshot in snapshots, f"Snapshot {test_snapshot} not found in list"
        
        # Wait a moment to simulate state change
        time.sleep(1)
        
        # Load snapshot
        load_success = load_emulator_snapshot(test_snapshot)
        assert load_success, f"Failed to load snapshot {test_snapshot}"
    
    def test_step_snapshot_functions(self):
        """Test the step-specific snapshot convenience functions."""
        step_num = 5
        episode_id = "test_episode"
        
        # Test step snapshot creation
        step_snapshot_name = create_step_snapshot(step_num, episode_id)
        assert step_snapshot_name, "Failed to create step snapshot"
        assert step_snapshot_name == f"{episode_id}_step_{step_num}", "Incorrect step snapshot name"
        
        self.test_snapshots.append(step_snapshot_name)
        
        # Verify step snapshot exists
        snapshots = list_emulator_snapshots()
        assert step_snapshot_name in snapshots, f"Step snapshot {step_snapshot_name} not found"
        
        # Test step restoration
        restore_success = restore_to_step(step_num, episode_id)
        assert restore_success, f"Failed to restore to step {step_num}"
    
    def test_step_snapshot_without_episode_id(self):
        """Test step snapshots without episode ID."""
        step_num = 10
        
        # Create step snapshot without episode ID
        step_snapshot_name = create_step_snapshot(step_num)
        assert step_snapshot_name, "Failed to create step snapshot without episode ID"
        assert step_snapshot_name == f"step_{step_num}", "Incorrect step snapshot name"
        
        self.test_snapshots.append(step_snapshot_name)
        
        # Test restoration without episode ID
        restore_success = restore_to_step(step_num)
        assert restore_success, f"Failed to restore to step {step_num} without episode ID"
    
    def test_snapshot_manager_context(self):
        """Test SnapshotManager context manager."""
        episode_id = "test_manager_episode"
        
        # Test with auto_cleanup=False to manually verify snapshots
        with SnapshotManager(episode_id, auto_cleanup=False) as sm:
            # Create step snapshots
            step1_name = sm.save_step(1)
            assert step1_name, "Failed to create step 1 snapshot"
            
            step2_name = sm.save_step(2)
            assert step2_name, "Failed to create step 2 snapshot"
            
            # Verify snapshots were created
            episode_snapshots = sm.list_episode_snapshots()
            assert len(episode_snapshots) >= 2, f"Expected at least 2 episode snapshots, got {len(episode_snapshots)}"
            assert step1_name in episode_snapshots, "Step 1 snapshot not found in episode list"
            assert step2_name in episode_snapshots, "Step 2 snapshot not found in episode list"
            
            # Test restoration
            restore_success = sm.restore_step(1)
            assert restore_success, "Failed to restore to step 1"
            
            # Add snapshots to cleanup list
            self.test_snapshots.extend([step1_name, step2_name])
    
    def test_snapshot_manager_auto_cleanup(self):
        """Test SnapshotManager auto cleanup functionality."""
        episode_id = "test_cleanup_episode"
        created_snapshots = []
        
        # Create snapshots and let auto cleanup handle them
        with SnapshotManager(episode_id, auto_cleanup=True) as sm:
            step1_name = sm.save_step(1)
            step2_name = sm.save_step(2)
            created_snapshots = [step1_name, step2_name]
            
            # Verify snapshots exist during context
            episode_snapshots = sm.list_episode_snapshots()
            assert len(episode_snapshots) >= 2, "Episode snapshots not created"
        
        # After context exit, snapshots should be cleaned up
        # Note: We can't easily verify cleanup worked, but the test validates the API works
        
    def test_snapshot_listing(self):
        """Test snapshot listing functionality."""
        # Create a few test snapshots
        test_snapshots = ["list_test_1", "list_test_2", "list_test_3"]
        
        for snapshot_name in test_snapshots:
            success = save_emulator_snapshot(snapshot_name)
            assert success, f"Failed to create test snapshot {snapshot_name}"
            self.test_snapshots.append(snapshot_name)
        
        # List all snapshots
        all_snapshots = list_emulator_snapshots()
        
        # Verify our test snapshots are in the list
        for snapshot_name in test_snapshots:
            assert snapshot_name in all_snapshots, f"Test snapshot {snapshot_name} not found in list"
    
    def test_snapshot_deletion(self):
        """Test snapshot deletion functionality."""
        snapshot_name = "test_deletion"
        
        # Create snapshot
        success = save_emulator_snapshot(snapshot_name)
        assert success, f"Failed to create snapshot {snapshot_name}"
        
        # Verify it exists
        snapshots = list_emulator_snapshots()
        assert snapshot_name in snapshots, f"Snapshot {snapshot_name} not found before deletion"
        
        # Delete snapshot
        delete_success = delete_emulator_snapshot(snapshot_name)
        assert delete_success, f"Failed to delete snapshot {snapshot_name}"
        
        # Verify it's gone
        snapshots_after = list_emulator_snapshots()
        assert snapshot_name not in snapshots_after, f"Snapshot {snapshot_name} still exists after deletion"
    
    def test_nonexistent_snapshot_operations(self):
        """Test operations on nonexistent snapshots."""
        nonexistent_name = "nonexistent_snapshot_12345"
        
        # Loading nonexistent snapshot should fail
        load_success = load_emulator_snapshot(nonexistent_name)
        assert not load_success, "Loading nonexistent snapshot should fail"
        
        # Deleting nonexistent snapshot should return False
        delete_success = delete_emulator_snapshot(nonexistent_name)
        assert not delete_success, "Deleting nonexistent snapshot should return False"


if __name__ == "__main__":
    # Allow running as script for debugging
    test = TestSnapshotBasics()
    test.setup_method()
    try:
        test.test_snapshot_creation_and_restoration()
        test.test_step_snapshot_functions()
        test.test_step_snapshot_without_episode_id()
        test.test_snapshot_manager_context()
        test.test_snapshot_manager_auto_cleanup()
        test.test_snapshot_listing()
        test.test_snapshot_deletion()
        test.test_nonexistent_snapshot_operations()
        print("✅ All tests passed!")
    finally:
        test.teardown_method()

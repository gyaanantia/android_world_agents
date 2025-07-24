#!/usr/bin/env python3
"""
Test comprehensive snapshot functionality.

This test verifies that Android emulator snapshots capture complete state including:
- Visual state (screenshots)
- System properties (device configuration)
- Installed applications (package manager)
- Current app activity (activity manager)
- App stack information (activity state)
- Metadata (timestamps, snapshot info)
"""

import os
import time
import pytest
import logging
from src.utils import save_emulator_snapshot, load_emulator_snapshot, delete_emulator_snapshot, SnapshotManager


class TestComprehensiveSnapshots:
    """Test class for comprehensive snapshot functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.test_snapshots = []
        
    def teardown_method(self):
        """Clean up test snapshots."""
        for snapshot_name in self.test_snapshots:
            delete_emulator_snapshot(snapshot_name)
    
    def test_comprehensive_snapshot_creation(self):
        """Test that comprehensive snapshots are created with all required files."""
        snapshot_name = "test_comprehensive"
        self.test_snapshots.append(snapshot_name)
        
        # Create comprehensive snapshot
        success = save_emulator_snapshot(snapshot_name)
        assert success, "Failed to create comprehensive snapshot"
        
        # Check snapshot directory exists
        snapshot_dir = os.path.join(os.getcwd(), "snapshots", snapshot_name)
        assert os.path.exists(snapshot_dir), f"Snapshot directory not found: {snapshot_dir}"
        
        # Verify all required files are present
        required_files = [
            "screenshot.png",
            "system_properties.txt", 
            "current_activity.txt",
            "installed_apps.txt",
            "activity_state.txt",
            "metadata.json"
        ]
        
        for required_file in required_files:
            file_path = os.path.join(snapshot_dir, required_file)
            assert os.path.exists(file_path), f"Missing required file: {required_file}"
            
            # Verify file has content
            file_size = os.path.getsize(file_path)
            assert file_size > 0, f"Required file is empty: {required_file}"
        
        logging.info(f"✅ All {len(required_files)} required files present and non-empty")
    
    def test_snapshot_data_comprehensiveness(self):
        """Test that snapshot data is comprehensive and meaningful."""
        snapshot_name = "test_data_comprehensive"
        self.test_snapshots.append(snapshot_name)
        
        # Create snapshot
        success = save_emulator_snapshot(snapshot_name)
        assert success, "Failed to create snapshot"
        
        snapshot_dir = os.path.join(os.getcwd(), "snapshots", snapshot_name)
        
        # Test system properties
        sys_props_file = os.path.join(snapshot_dir, "system_properties.txt")
        with open(sys_props_file, 'r') as f:
            props_content = f.read()
            props_lines = props_content.strip().split('\n')
        
        # Should have hundreds of system properties
        assert len(props_lines) > 400, f"Too few system properties: {len(props_lines)}"
        
        # Should contain key Android properties
        assert "ro.build.version.sdk" in props_content, "Missing SDK version property"
        assert "ro.product.model" in props_content, "Missing device model property"
        
        logging.info(f"✅ System properties: {len(props_lines)} properties captured")
        
        # Test installed apps
        apps_file = os.path.join(snapshot_dir, "installed_apps.txt")
        with open(apps_file, 'r') as f:
            apps_content = f.read()
            package_lines = [line for line in apps_content.split('\n') if 'package:' in line]
        
        # Should have many installed packages
        assert len(package_lines) > 100, f"Too few installed packages: {len(package_lines)}"
        
        # Should contain system packages
        assert any('android' in line for line in package_lines), "Missing Android system packages"
        
        logging.info(f"✅ Installed apps: {len(package_lines)} packages captured")
        
        # Test current activity
        activity_file = os.path.join(snapshot_dir, "current_activity.txt")
        with open(activity_file, 'r') as f:
            activity_content = f.read()
            activity_lines = activity_content.strip().split('\n')
        
        # Should have detailed activity information
        assert len(activity_lines) > 500, f"Too little activity data: {len(activity_lines)}"
        
        # Should contain activity stack information
        assert "ActivityRecord" in activity_content, "Missing activity record information"
        
        logging.info(f"✅ Current activity: {len(activity_lines)} lines captured")
        
        # Test metadata
        metadata_file = os.path.join(snapshot_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            import json
            metadata = json.load(f)
        
        # Verify metadata structure
        required_metadata_keys = ['name', 'timestamp', 'created_at', 'type']
        for key in required_metadata_keys:
            assert key in metadata, f"Missing metadata key: {key}"
        
        assert metadata['type'] == 'androidworld_snapshot', "Incorrect snapshot type"
        assert metadata['name'] == snapshot_name, "Incorrect snapshot name in metadata"
        
        logging.info(f"✅ Metadata complete with all required fields")
    
    def test_snapshot_manager_comprehensive(self):
        """Test SnapshotManager creates comprehensive snapshots."""
        episode_id = "test_comprehensive_episode"
        
        with SnapshotManager(episode_id, auto_cleanup=False) as sm:
            # Create step snapshot
            step1_name = sm.save_step(1)
            assert step1_name, "Failed to create step 1 snapshot"
            assert step1_name == f"{episode_id}_step_1", "Incorrect step snapshot name"
            
            # Verify comprehensive snapshot was created
            snapshot_dir = os.path.join(os.getcwd(), "snapshots", step1_name)
            assert os.path.exists(snapshot_dir), "Step snapshot directory not found"
            
            # Check for comprehensive files
            comprehensive_files = [
                "system_properties.txt",
                "current_activity.txt", 
                "installed_apps.txt",
                "activity_state.txt"
            ]
            
            for comp_file in comprehensive_files:
                file_path = os.path.join(snapshot_dir, comp_file)
                assert os.path.exists(file_path), f"Missing comprehensive file: {comp_file}"
            
            # Clean up manually since auto_cleanup=False
            self.test_snapshots.append(step1_name)
            
        logging.info(f"✅ SnapshotManager creates comprehensive snapshots")
    
    def test_snapshot_restoration_preserves_state(self):
        """Test that snapshot restoration preserves comprehensive state."""
        snapshot_name = "test_restoration_state"
        self.test_snapshots.append(snapshot_name)
        
        # Create initial snapshot
        success = save_emulator_snapshot(snapshot_name)
        assert success, "Failed to create initial snapshot"
        
        # Wait a moment, then restore
        time.sleep(1)
        
        restore_success = load_emulator_snapshot(snapshot_name)
        assert restore_success, "Failed to restore snapshot"
        
        # Give time for restoration to complete
        time.sleep(2)
        
        # Create second snapshot to compare
        comparison_name = "test_restoration_comparison"
        self.test_snapshots.append(comparison_name)
        
        success2 = save_emulator_snapshot(comparison_name)
        assert success2, "Failed to create comparison snapshot"
        
        # Compare file sizes (should be similar after restoration)
        original_dir = os.path.join(os.getcwd(), "snapshots", snapshot_name)
        restored_dir = os.path.join(os.getcwd(), "snapshots", comparison_name)
        
        for filename in ["system_properties.txt", "installed_apps.txt"]:
            original_file = os.path.join(original_dir, filename)
            restored_file = os.path.join(restored_dir, filename)
            
            if os.path.exists(original_file) and os.path.exists(restored_file):
                original_size = os.path.getsize(original_file)
                restored_size = os.path.getsize(restored_file)
                
                # Sizes should be very similar (system state shouldn't change much)
                size_diff_percent = abs(original_size - restored_size) / original_size * 100
                assert size_diff_percent < 5, f"Significant state change in {filename}: {size_diff_percent:.1f}%"
        
        logging.info(f"✅ Snapshot restoration preserves comprehensive state")


if __name__ == "__main__":
    # Allow running as script for debugging
    test = TestComprehensiveSnapshots()
    test.setup_method()
    try:
        test.test_comprehensive_snapshot_creation()
        test.test_snapshot_data_comprehensiveness()
        test.test_snapshot_manager_comprehensive()
        test.test_snapshot_restoration_preserves_state()
        print("✅ All tests passed!")
    finally:
        test.teardown_method()

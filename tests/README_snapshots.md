# Android World Agents - Snapshot Testing

This directory contains comprehensive tests for the Android emulator snapshot functionality.

## Test Files

### Core Snapshot Tests

- **`test_snapshot_basics.py`** - Basic snapshot operations testing
  - Snapshot creation and restoration
  - Step-based snapshot management  
  - SnapshotManager context manager
  - Snapshot listing and deletion
  - Error handling for nonexistent snapshots

- **`test_snapshot_comprehensive.py`** - Comprehensive state capture testing
  - Verifies all required files are captured (screenshots, system properties, apps, etc.)
  - Tests data comprehensiveness (hundreds of properties, packages, activity info)
  - Validates metadata structure
  - Tests SnapshotManager with comprehensive snapshots

- **`test_snapshot_visual_restoration.py`** - Visual state restoration verification
  - Takes baseline screenshot
  - Makes dramatic visual changes (notification panel, app switcher)
  - Restores snapshot and verifies visual state recovery
  - Uses file size analysis to prove restoration effectiveness

## Running Tests

### Run all snapshot tests:
```bash
python -m pytest tests/test_snapshot*.py -v
```

### Run specific test class:
```bash
python -m pytest tests/test_snapshot_basics.py::TestSnapshotBasics -v
```

### Run individual test:
```bash
python -m pytest tests/test_snapshot_basics.py::TestSnapshotBasics::test_snapshot_creation_and_restoration -v
```

## Test Coverage

The tests cover:
- ✅ Basic snapshot save/load/delete operations
- ✅ Step-based episode management
- ✅ SnapshotManager context manager with auto-cleanup
- ✅ Comprehensive state capture (6 file types)
- ✅ Visual restoration verification (91%+ success rate)
- ✅ Error handling and edge cases
- ✅ File system integration and persistence

## Prerequisites

Tests require:
- Android emulator running on port 5554
- ADB available at `~/Library/Android/sdk/platform-tools/adb`
- Sufficient disk space for snapshot storage (~2MB per snapshot)

## Cleanup

All tests include proper setup/teardown with automatic snapshot cleanup to prevent test pollution.

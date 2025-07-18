#!/usr/bin/env python3
"""
Test runner for Android World Agents.
Runs all tests and reports results.
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess
import glob

def discover_tests():
    """Discover all test files in the tests directory."""
    test_files = []
    
    # Get all test_*.py files in tests directory
    tests_dir = Path("tests")
    if tests_dir.exists():
        for test_file in tests_dir.glob("test_*.py"):
            test_files.append(str(test_file))
    
    # Add verify_framework.py if it exists
    if Path("verify_framework.py").exists():
        test_files.append("verify_framework.py")
    
    return sorted(test_files)

def get_test_name(test_file):
    """Extract test name from file path (remove test_ prefix and .py suffix)."""
    filename = Path(test_file).stem
    if filename.startswith("test_"):
        return filename[5:]  # Remove "test_" prefix
    return filename

def find_test_file(test_name):
    """Find test file by name (with or without test_ prefix)."""
    # First try with test_ prefix in tests directory
    test_file = f"tests/test_{test_name}.py"
    if Path(test_file).exists():
        return test_file
    
    # Try without prefix in tests directory
    test_file = f"tests/{test_name}.py"
    if Path(test_file).exists():
        return test_file
    
    # Try in root directory
    test_file = f"{test_name}.py"
    if Path(test_file).exists():
        return test_file
    
    return None

def run_test(test_file):
    """Run a single test file and return success status."""
    try:
        print(f"\n{'='*50}")
        print(f"Running: {test_file}")
        print('='*50)
        
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED")
            return True
        else:
            print(f"❌ {test_file} FAILED")
            return False
            
    except Exception as e:
        print(f"❌ {test_file} ERROR: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Android World Agents tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --list             # List available tests
  python run_tests.py imports prompts    # Run specific tests
  python run_tests.py evaluator agent    # Run evaluator and agent tests
        """
    )
    
    parser.add_argument(
        "tests", 
        nargs="*", 
        help="Specific tests to run (use names without test_ prefix, e.g., 'imports', 'prompts')"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available tests"
    )
    
    return parser.parse_args()

def main():
    """Run all tests."""
    args = parse_args()
    
    # Discover all available tests
    all_tests = discover_tests()
    
    if args.list:
        print("📋 Available tests:")
        for test_file in all_tests:
            test_name = get_test_name(test_file)
            print(f"  - {test_name:<15} ({test_file})")
        return
    
    # Determine which tests to run
    if args.tests:
        # Run specific tests
        test_files = []
        for test_name in args.tests:
            test_file = find_test_file(test_name)
            if test_file:
                test_files.append(test_file)
            else:
                print(f"⚠️  Test '{test_name}' not found")
        
        if not test_files:
            print("❌ No valid tests found to run")
            sys.exit(1)
    else:
        # Run all tests
        test_files = all_tests
    
    if not test_files:
        print("⚠️  No tests found to run")
        sys.exit(1)
    
    print("🚀 Running Android World Agents Test Suite")
    print(f"📊 Running {len(test_files)} test(s)")
    
    # Track results
    passed = 0
    failed = 0
    
    # Run each test
    for test_file in test_files:
        if run_test(test_file):
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 {failed} test(s) failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

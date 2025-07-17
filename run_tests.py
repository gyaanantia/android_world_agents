#!/usr/bin/env python3
"""
Test runner for Android World Agents.
Runs all tests and reports results.
"""

import sys
import os
from pathlib import Path
import subprocess

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

def main():
    """Run all tests."""
    print("🚀 Running Android World Agents Test Suite")
    
    # List of test files to run
    test_files = [
        "tests/test_imports.py",
        "tests/test_prompts.py", 
        "tests/test_agent.py",
        "tests/test_evaluator.py",
        "verify_framework.py"
    ]
    
    # Track results
    passed = 0
    failed = 0
    
    # Run each test
    for test_file in test_files:
        if Path(test_file).exists():
            if run_test(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  {test_file} not found, skipping")
    
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

#!/usr/bin/env python3
"""
Verification script to test the AndroidWorld Enhanced T3A Agent framework.
"""

import sys
import os
import importlib.util

def test_imports():
    """Test if all modules can be imported."""
    print("🔍 Testing module imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    modules_to_test = [
        'src.utils',
        'src.prompts', 
        'src.evaluator',
        'src.main'
    ]
    
    for module_name in modules_to_test:
        try:
            spec = importlib.util.find_spec(module_name)
            assert spec is not None, f"{module_name}: Module not found"
            print(f"✅ {module_name}: Found")
        except Exception as e:
            print(f"❌ {module_name}: Error - {e}")
            assert False, f"{module_name}: Error - {e}"

def test_file_structure():
    """Test if all required files exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'src/agent.py',
        'src/evaluator.py', 
        'src/main.py',
        'src/prompts.py',
        'src/run_episode.py',
        'src/utils.py',
        'prompts/base_prompt.txt',
        'prompts/few_shot_v1.txt',
        'prompts/reflective_v1.txt',
        'pyproject.toml',
        'run_evaluation.py',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"❌ {file_path}: Missing")
        else:
            print(f"✅ {file_path}: Found")
    
    assert len(missing_files) == 0, f"Missing files: {missing_files}"

def test_prompt_functionality():
    """Test if prompt loading and formatting works correctly."""
    print("\n🔤 Testing prompt functionality...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    try:
        from prompts import load_prompt, get_prompt_template, format_prompt, list_available_prompts
        
        # Test listing prompts
        available_prompts = list_available_prompts()
        assert available_prompts, "No prompts found in prompts directory"
        print(f"✅ Found {len(available_prompts)} prompt files")
        
        # Test loading specific prompts
        for prompt_variant in ["base", "few-shot", "reflective"]:
            try:
                prompt = get_prompt_template(prompt_variant)
                assert prompt, f"{prompt_variant} prompt is empty"
                print(f"✅ {prompt_variant} prompt loaded successfully")
            except Exception as e:
                print(f"❌ {prompt_variant} prompt loading failed: {e}")
                assert False, f"{prompt_variant} prompt loading failed: {e}"
        
        # Test prompt formatting
        try:
            base_prompt = get_prompt_template("base")
            formatted = format_prompt(base_prompt, 
                                    goal="Test goal", 
                                    ui_elements="Test UI", 
                                    memory="Test memory")
            assert "{goal}" not in formatted and "{ui_elements}" not in formatted and "{memory}" not in formatted, "Prompt formatting failed - variables not substituted"
            print("✅ Prompt formatting works correctly")
        except Exception as e:
            print(f"❌ Prompt formatting failed: {e}")
            assert False, f"Prompt formatting failed: {e}"
        
    except Exception as e:
        print(f"❌ Prompt functionality test failed: {e}")
        assert False, f"Prompt functionality test failed: {e}"


def test_dependencies():
    """Test if dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    # Check if pyproject.toml exists
    assert os.path.exists('pyproject.toml'), "pyproject.toml not found"
    print("✅ pyproject.toml found")
    
    # Test basic imports (skip AndroidWorld for now)
    basic_deps = ['json', 'os', 'sys', 'datetime', 'pathlib', 'typing', 'logging']
    
    for dep in basic_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing")
            assert False, f"{dep}: Missing"

def test_execution_permissions():
    """Test if scripts have execution permissions."""
    print("\n🔐 Testing execution permissions...")
    
    scripts = ['run_evaluation.py']
    
    for script in scripts:
        assert os.access(script, os.X_OK), f"{script}: Not executable"
        print(f"✅ {script}: Executable")

def main():
    """Run all tests."""
    print("🧪 AndroidWorld Enhanced T3A Agent Framework Verification\n")
    
    # Check if we're in the right conda environment
    try:
        import android_world
        print(f"✅ Running in environment with AndroidWorld {android_world.__version__ if hasattr(android_world, '__version__') else 'installed'}")
    except ImportError:
        print("⚠️  AndroidWorld not found. Please run setup first:")
        print("   ./setup.sh")
        print("   conda activate android_world")
        print("")

    try:
        import android_world.agents
        print("✅ AndroidWorld agents module found. Installed correctly.")
    except ImportError:
        print("❌ AndroidWorld agents module not found. Please check your installation. You may need to run:")
        print("   ./fix_init_files.sh")
        print("")
    
    tests = [
        ("Module Imports", test_imports),
        ("File Structure", test_file_structure), 
        ("Prompt Functionality", test_prompt_functionality),
        ("Dependencies", test_dependencies),
        ("Execution Permissions", test_execution_permissions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Ensure conda environment is activated: conda activate android_world")
        print("2. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("3. Start Android emulator with: ~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554")
        print("4. Run evaluation: python run_evaluation.py --task your_task --prompt-variant base")
        return True
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

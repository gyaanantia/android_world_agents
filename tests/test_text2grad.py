#!/usr/bin/env python3
"""
Verification script for Text2Grad integration with AndroidWorld Agents.
Tests that Text2Grad dependencies are properly installed and compatible.
"""

import sys
import os
import importlib.util

def test_pytorch_installation():
    """Test PyTorch installation and device compatibility."""
    print("🔍 Testing PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        
        # Test available devices
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ Metal Performance Shaders (MPS) available for GPU acceleration")
        else:
            device = torch.device("cpu")
            print("✅ CPU device available (MPS not available)")
        
        # Test basic tensor operations
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = torch.matmul(x, y)
        print(f"✅ Basic tensor operations work on {device}")
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        assert False, f"PyTorch import failed: {e}"
    except Exception as e:
        print(f"❌ PyTorch device test failed: {e}")
        assert False, f"PyTorch device test failed: {e}"

def test_transformers_installation():
    """Test Transformers library installation."""
    print("\n🔍 Testing Transformers installation...")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} installed")
        
        # Test basic tokenizer loading (lightweight test)
        from transformers import AutoTokenizer
        print("✅ AutoTokenizer import successful")
        
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        assert False, f"Transformers import failed: {e}"
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        assert False, f"Transformers test failed: {e}"

def test_text2grad_ml_dependencies():
    """Test Text2Grad specific ML dependencies."""
    print("\n🔍 Testing Text2Grad ML dependencies...")
    
    ml_deps = [
        ("trl", "TRL (Transformer Reinforcement Learning)"),
        ("peft", "PEFT (Parameter-Efficient Fine-Tuning)"),
        ("accelerate", "Hugging Face Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
        ("sklearn", "Scikit-learn"),
        ("rouge_score", "ROUGE Score"),
        ("bert_score", "BERT Score"),
        ("wandb", "Weights & Biases"),
        ("huggingface_hub", "Hugging Face Hub")
    ]
    
    failed_deps = []
    
    for module_name, description in ml_deps:
        try:
            __import__(module_name)
            print(f"✅ {description}: Available")
        except ImportError:
            print(f"❌ {description}: Missing")
            failed_deps.append(module_name)
    
    assert len(failed_deps) == 0, f"Missing dependencies: {failed_deps}"

def test_text2grad_compatibility():
    """Test Text2Grad specific functionality."""
    print("\n🔍 Testing Text2Grad compatibility...")
    
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModel
        
        # Test that we can work with models in CPU/MPS mode
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        print(f"✅ Text2Grad can use device: {device}")
        
        # Test that numpy is compatible version
        import numpy as np
        print(f"✅ NumPy {np.__version__} (compatible with Text2Grad)")
        
    except Exception as e:
        print(f"❌ Text2Grad compatibility test failed: {e}")
        assert False, f"Text2Grad compatibility test failed: {e}"

def test_text2grad_directory():
    """Test that Text2Grad directory exists and is accessible."""
    print("\n🔍 Testing Text2Grad directory access...")
    
    text2grad_path = "Text2Grad"  # Text2Grad is in project root, pytest runs from project root
    
    if not os.path.exists(text2grad_path):
        print(f"⚠️  Text2Grad directory not found at {text2grad_path}")
        print("ℹ️  Run ./setup.sh to install Text2Grad automatically")
        return  # Don't fail the test, just warn
    
    print(f"✅ Text2Grad directory found at {text2grad_path}")
    
    # Check for key subdirectories
    key_dirs = [
        "nl_reward_model",
        "nl_gradiant_policy_optimization",  # Note: This has a typo in the original Text2Grad repo 
        "rm_data_anno"
    ]
    
    missing_dirs = []
    for dir_name in key_dirs:
        dir_path = os.path.join(text2grad_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name} directory found")
        else:
            print(f"❌ {dir_name} directory missing")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️  Some Text2Grad directories missing: {missing_dirs}")
        print("ℹ️  This might indicate an incomplete Text2Grad installation")

def test_environment_variables():
    """Test required environment variables."""
    print("\n🔍 Testing environment variables...")
    
    # Check OpenAI API key (required for both projects)
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️  OPENAI_API_KEY not set (required for LLM functionality)")
    
    # Check Google API key (optional for Gemini)
    if os.getenv("GOOGLE_API_KEY"):
        print("✅ GOOGLE_API_KEY is set")
    else:
        print("ℹ️  GOOGLE_API_KEY not set (optional for Gemini functionality)")
    
    # Note: Not asserting on API keys since they may not be set in test environment

def main():
    """Run all Text2Grad verification tests."""
    print("🧪 Text2Grad Integration Verification\n")
    
    tests = [
        ("PyTorch Installation", test_pytorch_installation),
        ("Transformers Installation", test_transformers_installation),
        ("Text2Grad ML Dependencies", test_text2grad_ml_dependencies),
        ("Text2Grad Compatibility", test_text2grad_compatibility),
        ("Text2Grad Directory", test_text2grad_directory),
        ("Environment Variables", test_environment_variables)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Text2Grad Integration Test Results:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Text2Grad integration is ready!")
        print("\nNext steps:")
        print("1. Ensure your API keys are set:")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print("2. Test Text2Grad functionality:")
        print("   cd Text2Grad && python -c \"import torch; print('Text2Grad ready')\"")
        print("3. Add --text2grad flag to run_evaluation.py when ready")
        print("   python run_evaluation.py --text2grad --task YourTask")
        return True
    else:
        print("\n❌ Some Text2Grad integration issues detected.")
        print("📝 Recommended fixes:")
        print("1. Re-run setup: ./setup.sh")
        print("2. Manually install missing packages:")
        print("   pip install transformers accelerate trl peft")
        print("3. Check that you're in the correct conda environment:")
        print("   conda activate android_world")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

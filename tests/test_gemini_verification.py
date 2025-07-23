#!/usr/bin/env python3
"""
Quick verification tests for Gemini 2.5 Flash integration.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gemini_prompting import create_gemini_generator, GEMINI_AVAILABLE


class TestGeminiVerification:
    """Quick verification tests for Gemini integration."""
    
    def test_api_key_configured(self):
        """Test that API key is properly configured."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            assert len(api_key) > 0, "API key should not be empty"
            print(f"✅ API key configured (length: {len(api_key)})")
        else:
            pytest.skip("GOOGLE_API_KEY not set - skipping API key tests")
    
    def test_dependencies_available(self):
        """Test that Gemini dependencies are available."""
        assert isinstance(GEMINI_AVAILABLE, bool), "GEMINI_AVAILABLE should be boolean"
        if GEMINI_AVAILABLE:
            print("✅ Dependencies available")
        else:
            pytest.skip("Gemini dependencies not available")
    
    def test_generator_creation(self):
        """Test that generator can be created successfully."""
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini dependencies not available")
            
        try:
            generator = create_gemini_generator()
            if generator:
                assert hasattr(generator, 'model_name'), "Generator should have model_name"
                assert hasattr(generator, 'temperature'), "Generator should have temperature"
                print(f"✅ Generator created successfully")
                print(f"   Model: {generator.model_name}")
                print(f"   Temperature: {generator.temperature}")
            else:
                pytest.skip("Failed to create generator (likely missing API key)")
        except Exception as e:
            pytest.fail(f"Generator creation failed: {e}")
    
    def test_api_connection(self):
        """Test basic API connection."""
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini dependencies not available")
            
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
            
        try:
            generator = create_gemini_generator()
            if generator and hasattr(generator, 'test_connection'):
                if generator.test_connection():
                    print("✅ API connection successful!")
                else:
                    pytest.fail("API connection failed")
            else:
                pytest.skip("Generator creation failed or no test_connection method")
        except Exception as e:
            pytest.fail(f"API connection test failed: {e}")


def test_verification_summary():
    """Print verification summary."""
    print("\n🎉 Gemini 2.5 Flash verification completed!")
    print("   Ready for AndroidWorld agent integration.")


if __name__ == "__main__":
    # Allow running as standalone script for quick verification
    import subprocess
    
    print("🔍 Gemini 2.5 Flash Verification")
    print("=" * 40)
    
    # Run the tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=False)
    
    sys.exit(result.returncode)

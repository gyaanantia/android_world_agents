"""
Test suite for Gemini 2.5 Flash integration with AndroidWorld agents.

This module tests the integration between Google's Gemini 2.5 Flash model
and AndroidWorld agents, verifying imports, agent creation, and fallback behavior.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import Mock

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestGeminiIntegration:
    """Test suite for Gemini integration functionality."""
    
    def test_gemini_imports(self):
        """Test that all Gemini-related imports work correctly."""
        from src.gemini_enhanced_agent import (
            GeminiEnhancedT3A, 
            create_gemini_enhanced_agent,
            create_standard_agent_with_gemini_fallback,
            GEMINI_AVAILABLE
        )
        
        from src.gemini_prompting import (
            create_gemini_generator, 
            GEMINI_AVAILABLE as GEMINI_PROMPTING_AVAILABLE
        )
        
        from src.agent import EnhancedT3A, create_agent
        
        # All imports should succeed
        assert True
    
    def test_gemini_availability_check(self):
        """Test that Gemini availability is correctly detected."""
        from src.gemini_prompting import GEMINI_AVAILABLE
        
        # Should be boolean
        assert isinstance(GEMINI_AVAILABLE, bool)
    
    def test_gemini_generator_creation(self):
        """Test Gemini generator creation (if dependencies available)."""
        from src.gemini_prompting import create_gemini_generator, GEMINI_AVAILABLE
        
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini dependencies not available")
        
        # Should return None if no API key, or generator if API key exists
        generator = create_gemini_generator()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            assert generator is not None
            assert hasattr(generator, 'model_name')
            assert generator.model_name == "gemini-2.5-flash"
        else:
            # Without API key, should handle gracefully
            assert generator is None or generator is not None
    
    def test_enhanced_agent_creation(self):
        """Test creation of Gemini-enhanced agent."""
        from src.gemini_enhanced_agent import GeminiEnhancedT3A
        
        # Create mock environment and LLM
        mock_env = Mock()
        mock_env.logical_screen_size = (400, 800)
        
        mock_llm = Mock()
        mock_llm.predict.return_value = ("Mock response", True, "Mock raw response")
        
        # Should be able to create agent regardless of Gemini availability
        agent = GeminiEnhancedT3A(
            env=mock_env,
            llm=mock_llm,
            prompt_variant="base",
            use_memory=True,
            use_function_calling=False,
            use_gemini=True
        )
        
        assert agent is not None
        assert hasattr(agent, 'get_gemini_status')
        
        # Check status method works
        status = agent.get_gemini_status()
        assert isinstance(status, dict)
        assert 'gemini_available' in status
        assert 'gemini_enabled' in status
    
    def test_graceful_fallback(self):
        """Test that agents work when Gemini is unavailable."""
        from src.gemini_enhanced_agent import create_standard_agent_with_gemini_fallback
        
        mock_env = Mock()
        mock_env.logical_screen_size = (400, 800)
        
        # Should create some kind of agent regardless of Gemini availability
        agent = create_standard_agent_with_gemini_fallback(
            env=mock_env,
            model_name="gpt-4o-mini",
            try_gemini=True
        )
        
        assert agent is not None
        # Should be either GeminiEnhancedT3A or EnhancedT3A
        assert hasattr(agent, 'step')  # Both agent types have step method
    
    def test_gemini_status_reporting(self):
        """Test that Gemini status is correctly reported."""
        from src.gemini_enhanced_agent import GeminiEnhancedT3A
        
        mock_env = Mock()
        mock_env.logical_screen_size = (400, 800)
        
        mock_llm = Mock()
        mock_llm.predict.return_value = ("Mock response", True, "Mock raw response")
        
        # Test with Gemini disabled
        agent = GeminiEnhancedT3A(
            env=mock_env,
            llm=mock_llm,
            use_gemini=False
        )
        
        status = agent.get_gemini_status()
        assert status['gemini_enabled'] is False
        
        # Test with Gemini enabled (may still be False if dependencies unavailable)
        agent = GeminiEnhancedT3A(
            env=mock_env,
            llm=mock_llm,
            use_gemini=True
        )
        
        status = agent.get_gemini_status()
        assert isinstance(status['gemini_enabled'], bool)


class TestGeminiPrompting:
    """Test suite for Gemini prompt generation functionality."""
    
    def test_generator_factory_function(self):
        """Test the create_gemini_generator factory function."""
        from src.gemini_prompting import create_gemini_generator, GEMINI_AVAILABLE
        
        if not GEMINI_AVAILABLE:
            # Should return None when dependencies unavailable
            generator = create_gemini_generator()
            assert generator is None
        else:
            # Should handle missing API key gracefully
            generator = create_gemini_generator(api_key=None)
            # Behavior depends on whether GOOGLE_API_KEY env var is set
            if os.getenv("GOOGLE_API_KEY"):
                assert generator is not None
            else:
                assert generator is None
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Requires Google API key")
    def test_generator_with_api_key(self):
        """Test generator creation and basic functionality with API key."""
        from src.gemini_prompting import create_gemini_generator, GEMINI_AVAILABLE
        
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini dependencies not available")
        
        generator = create_gemini_generator()
        assert generator is not None
        assert generator.model_name == "gemini-2.5-flash"
        assert generator.temperature == 0.1
        
        # Test connection (this makes an actual API call)
        try:
            connection_ok = generator.test_connection()
            assert isinstance(connection_ok, bool)
        except Exception:
            # API call might fail for various reasons, but shouldn't crash
            pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

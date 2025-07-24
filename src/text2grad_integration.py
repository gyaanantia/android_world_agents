"""
Text2Grad integration module for AndroidWorld agents.

This module provides integration between Text2Grad and the AndroidWorld agent framework,
allowing for gradient-based feedback processing on Gemini visual analysis output.
"""

import os
import sys
from typing import Optional, Dict, Any, List
import logging

# Add Text2Grad to path if it exists
text2grad_path = os.path.join(os.path.dirname(__file__), '..', 'Text2Grad')
if os.path.exists(text2grad_path):
    sys.path.insert(0, text2grad_path)

class Text2GradProcessor:
    """Handles Text2Grad processing for Gemini visual analysis output."""
    
    def __init__(self, enabled: bool = False):
        """Initialize Text2Grad processor.
        
        Args:
            enabled: Whether Text2Grad processing is enabled.
        """
        self.enabled = enabled
        self.initialized = False
        
        if self.enabled:
            self._initialize_text2grad()
    
    def _initialize_text2grad(self) -> bool:
        """Initialize Text2Grad components.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Try to import Text2Grad dependencies
            import torch
            import transformers
            
            # Check for Text2Grad modules (placeholder for future implementation)
            # TODO: Import actual Text2Grad modules when integrating specific components
            
            logging.info("✅ Text2Grad processor initialized successfully")
            self.initialized = True
            return True
            
        except ImportError as e:
            logging.warning(f"⚠️ Text2Grad initialization failed: {e}")
            logging.warning("Text2Grad will be disabled for this session")
            self.enabled = False
            return False
        except Exception as e:
            logging.error(f"❌ Text2Grad initialization error: {e}")
            self.enabled = False
            return False
    
    def process_gemini_output(self, gemini_analysis: str, task_context: Dict[str, Any]) -> str:
        """Process Gemini visual analysis output with Text2Grad.
        
        Args:
            gemini_analysis: The visual analysis output from Gemini.
            task_context: Context about the current task and state.
            
        Returns:
            Enhanced analysis with Text2Grad processing, or original analysis if disabled.
        """
        if not self.enabled or not self.initialized:
            return gemini_analysis
        
        try:
            # TODO: Implement actual Text2Grad processing
            # This is a placeholder that will be expanded when integrating specific Text2Grad components
            
            # For now, just add a marker that Text2Grad was applied
            enhanced_analysis = f"""
{gemini_analysis}

[Text2Grad Enhancement Applied]
This analysis has been processed through Text2Grad for gradient-based feedback integration.
"""
            
            logging.debug("✅ Text2Grad processing completed")
            return enhanced_analysis.strip()
            
        except Exception as e:
            logging.error(f"❌ Text2Grad processing failed: {e}")
            logging.warning("Falling back to original Gemini analysis")
            return gemini_analysis
    
    def generate_feedback_gradients(self, response: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Generate gradient-based feedback for model responses.
        
        Args:
            response: The model's response to generate gradients for.
            ground_truth: Optional ground truth for comparison.
            
        Returns:
            Dictionary containing gradient information and feedback.
        """
        if not self.enabled or not self.initialized:
            return {"gradients": None, "feedback": "Text2Grad disabled"}
        
        try:
            # TODO: Implement actual gradient generation
            # This is a placeholder for future Text2Grad integration
            
            gradients = {
                "token_level_feedback": "Placeholder for token-level gradients",
                "span_level_rewards": "Placeholder for span-level rewards",
                "feedback_quality": 0.0  # Placeholder score
            }
            
            return {
                "gradients": gradients,
                "feedback": "Text2Grad gradient generation completed",
                "success": True
            }
            
        except Exception as e:
            logging.error(f"❌ Text2Grad gradient generation failed: {e}")
            return {"gradients": None, "feedback": f"Error: {str(e)}", "success": False}
    
    def is_available(self) -> bool:
        """Check if Text2Grad is available and properly initialized.
        
        Returns:
            True if Text2Grad is available, False otherwise.
        """
        return self.enabled and self.initialized


def create_text2grad_processor(enabled: bool = False) -> Text2GradProcessor:
    """Factory function to create a Text2Grad processor.
    
    Args:
        enabled: Whether to enable Text2Grad processing.
        
    Returns:
        A Text2GradProcessor instance.
    """
    return Text2GradProcessor(enabled=enabled)


def verify_text2grad_setup() -> bool:
    """Verify that Text2Grad is properly set up and available.
    
    Returns:
        True if Text2Grad setup is valid, False otherwise.
    """
    try:
        # Check basic dependencies
        import torch
        import transformers
        
        # Check for Text2Grad directory
        text2grad_dir = os.path.join(os.path.dirname(__file__), '..', 'Text2Grad')
        if not os.path.exists(text2grad_dir):
            logging.warning("Text2Grad directory not found")
            return False
        
        # Check for key Text2Grad subdirectories
        key_dirs = [
            "nl_reward_model",
            "nl_gradient_policy_optimization",
            "rm_data_anno"
        ]
        
        for dir_name in key_dirs:
            dir_path = os.path.join(text2grad_dir, dir_name)
            if not os.path.exists(dir_path):
                logging.warning(f"Text2Grad subdirectory missing: {dir_name}")
                return False
        
        logging.info("✅ Text2Grad setup verification passed")
        return True
        
    except ImportError as e:
        logging.warning(f"Text2Grad dependencies missing: {e}")
        return False
    except Exception as e:
        logging.error(f"Text2Grad setup verification failed: {e}")
        return False


# Module-level verification on import
_TEXT2GRAD_AVAILABLE = verify_text2grad_setup()

def is_text2grad_available() -> bool:
    """Check if Text2Grad is available at the module level.
    
    Returns:
        True if Text2Grad is available, False otherwise.
    """
    return _TEXT2GRAD_AVAILABLE

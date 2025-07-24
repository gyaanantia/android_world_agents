#!/usr/bin/env python3
"""
Gemini 2.5 Prompt Generation System for AndroidWorld Agents.

This module uses Google's Gemini 2.5 model to analyze Android UI screenshots
and task goals to generate optimized prompts for AndroidWorld agents.

The system takes a screenshot and task goal as input and produces a contextual
prompt that will help agents successfully complete the task.
"""

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging
import argparse
from PIL import Image

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from prompts import load_prompt

try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError as e:
    genai = None
    Image = None
    GEMINI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Gemini dependencies not available: {e}")
    logger.warning("Install with: pip install google-generativeai pillow")

logger = logging.getLogger(__name__)


class GeminiPromptGenerator:
    """Gemini-powered prompt generator for AndroidWorld agents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        max_image_size: Tuple[int, int] = (1024, 1024),
        temperature: float = 0.1,
    ):
        """Initialize the Gemini prompt generator.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            model_name: Gemini model to use.
            max_image_size: Maximum image dimensions (width, height) for API.
            temperature: Model temperature for generation (lower = more consistent).
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Gemini dependencies not available. "
                "Install with: pip install google-generativeai pillow"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or provide api_key parameter."
            )
        
        logger.debug(f"API key first 10 chars: {self.api_key[:10] if self.api_key else 'None'}")
        logger.debug(f"API key last 10 chars: {self.api_key[-10:] if self.api_key else 'None'}")
        logger.debug(f"API key length: {len(self.api_key) if self.api_key else 0}")
        
        self.model_name = model_name
        self.max_image_size = max_image_size
        self.temperature = temperature
        
        # Configure Gemini with retry logic
        logger.debug(f"Configuring Gemini with API key ending in: ...{self.api_key[-4:] if self.api_key else 'None'}")
        
        # Load the system prompt template first
        self.system_prompt = load_prompt("gemini_base_prompt.txt")
        if not self.system_prompt:
            raise ValueError("Could not load gemini_base_prompt.txt")
        
        # Configure and test with retry
        max_config_attempts = 3
        self.model = None
        
        for attempt in range(max_config_attempts):
            try:
                logger.debug(f"Configuration attempt {attempt + 1}/{max_config_attempts}")
                
                # Configure Gemini
                genai.configure(api_key=self.api_key)
                
                # Initialize the model
                logger.debug(f"Creating GenerativeModel: {self.model_name}")
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        top_p=0.8,
                        top_k=20,
                        max_output_tokens=8192,
                    ),
                )
                
                # Test the configuration
                logger.debug("Testing Gemini configuration...")
                test_response = self.model.generate_content("test")
                logger.debug("Gemini configuration test successful")
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.warning(f"Configuration attempt {attempt + 1} failed: {e}")
                if attempt == max_config_attempts - 1:
                    logger.error(f"All configuration attempts failed. Last error: {e}")
                    raise ValueError(f"Failed to configure Gemini after {max_config_attempts} attempts: {e}")
                else:
                    # Wait before retry
                    time.sleep(0.5)
        
        logger.info(f"Initialized GeminiPromptGenerator with model: {self.model_name}")
    
    def _prepare_image(self, screenshot: np.ndarray) -> Image.Image:
        """Prepare screenshot for Gemini API.
        
        Args:
            screenshot: Screenshot as numpy array (H, W, 3) uint8.
            
        Returns:
            PIL Image ready for API.
        """
        # Convert numpy array to PIL Image
        if screenshot.dtype != np.uint8:
            screenshot = (screenshot * 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
            image = Image.fromarray(screenshot, mode='RGB')
        elif len(screenshot.shape) == 3 and screenshot.shape[2] == 4:
            # Convert RGBA to RGB
            image = Image.fromarray(screenshot, mode='RGBA').convert('RGB')
        else:
            raise ValueError(f"Unexpected screenshot shape: {screenshot.shape}")
        
        # Resize if too large (Gemini has size limits)
        max_dimension = 2048  # Conservative limit for Gemini
        if image.size[0] > max_dimension or image.size[1] > max_dimension:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_dimension / image.size[0], max_dimension / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {screenshot.shape[:2]} to {image.size}")
        
        # Ensure image is in RGB mode and properly formatted
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.debug(f"Prepared image: size={image.size}, mode={image.mode}")
        return image
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API.
        
        Args:
            image: PIL Image to convert.
            
        Returns:
            Base64 encoded image string.
        """
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _format_prompt(self, goal: str) -> str:
        """Format the system prompt with the task goal.
        
        Args:
            goal: The task goal string.
            
        Returns:
            Formatted prompt string.
        """
        return self.system_prompt.format(goal=goal)
    
    def generate_agent_prompt(
        self,
        screenshot: np.ndarray,
        goal: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate an agent prompt using Gemini.
        
        Args:
            screenshot: Current Android UI screenshot as numpy array.
            goal: Task goal description.
            max_retries: Maximum number of API call retries.
            
        Returns:
            Dictionary containing:
            - 'agent_prompt': Generated prompt for the agent
            - 'analysis': Gemini's analysis of the current UI state
            - 'approach': Recommended approach for the task
            - 'success': Whether generation was successful
            - 'raw_response': Raw Gemini response
            - 'error': Error message if generation failed
        """
        try:
            # Prepare inputs
            image = self._prepare_image(screenshot)
            prompt = self._format_prompt(goal)
            
            logger.debug(f"Generating prompt for goal: {goal}")
            logger.debug(f"Image size: {image.size}, mode: {image.mode}")
            logger.debug(f"Prompt length: {len(prompt)}")
            
            # Generate response
            for attempt in range(max_retries):
                try:
                    # Simple approach that works
                    response = self.model.generate_content([prompt, image])
                    if response.text:
                        logger.debug("Successfully generated content")
                        return self._parse_response(response.text, goal)
                    else:
                        logger.warning(f"Empty response on attempt {attempt + 1}")
                        if attempt == max_retries - 1:
                            raise ValueError("Gemini returned empty response")
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    logger.debug(f"Full error details: {type(e).__name__}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
        
        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            return {
                'agent_prompt': self._fallback_prompt(goal),
                'analysis': "Failed to analyze UI",
                'approach': "Using fallback approach",
                'success': False,
                'raw_response': None,
                'error': str(e)
            }
    
    def _parse_response(self, response_text: str, goal: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured components.
        
        Args:
            response_text: Raw response from Gemini.
            goal: Original task goal.
            
        Returns:
            Structured response dictionary.
        """
        try:
            # Extract sections using markers
            analysis = self._extract_section(response_text, "TASK ANALYSIS:", "RECOMMENDED APPROACH:")
            approach = self._extract_section(response_text, "RECOMMENDED APPROACH:", "GENERATED AGENT PROMPT:")
            agent_prompt = self._extract_section(response_text, "GENERATED AGENT PROMPT:", None)
            
            # Clean up extracted content
            analysis = analysis.strip() if analysis else "UI analysis not provided"
            approach = approach.strip() if approach else "Approach not specified"
            agent_prompt = agent_prompt.strip() if agent_prompt else self._fallback_prompt(goal)
            
            return {
                'agent_prompt': agent_prompt,
                'analysis': analysis,
                'approach': approach,
                'success': True,
                'raw_response': response_text,
                'error': None
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse response, using fallback: {e}")
            return {
                'agent_prompt': self._fallback_prompt(goal),
                'analysis': "Failed to parse analysis",
                'approach': "Using fallback approach",
                'success': False,
                'raw_response': response_text,
                'error': f"Parse error: {e}"
            }
    
    def _extract_section(self, text: str, start_marker: str, end_marker: Optional[str]) -> str:
        """Extract a section from the response text.
        
        Args:
            text: Full response text.
            start_marker: Section start marker.
            end_marker: Section end marker (None for end of text).
            
        Returns:
            Extracted section content.
        """
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return ""
        
        start_idx += len(start_marker)
        
        if end_marker:
            end_idx = text.find(end_marker, start_idx)
            if end_idx == -1:
                return text[start_idx:].strip()
            return text[start_idx:end_idx].strip()
        else:
            return text[start_idx:].strip()
    
    def _fallback_prompt(self, goal: str) -> str:
        """Generate a basic fallback prompt when Gemini fails.
        
        Args:
            goal: Task goal.
            
        Returns:
            Simple fallback prompt with proper format requirements.
        """
        return f"""Your task is to complete the following goal: {goal}

Analyze the current screen carefully and take appropriate actions to accomplish this task. 
Look for relevant UI elements like buttons, text fields, and navigation options.
If you can't find what you need immediately, try scrolling or navigating to other screens.
Use the most direct path to complete the task efficiently.

IMPORTANT: You must respond in the exact format required by AndroidWorld:

Reason: [Explain why you're taking this action]
Action: {{"action_type": "...", ...}}

Available actions include:
- click, double_tap, long_press (with index)
- input_text (with text and index)
- swipe, scroll (with direction)
- navigate_home, navigate_back
- open_app (with app_name)
- status (with goal_status: "complete" or "infeasible")
- answer (with text)
- wait

Use the status action with "complete" when the task is finished successfully."""
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Create a simple test image
            test_image = Image.new('RGB', (100, 100), color='white')
            test_prompt = "Describe this image in one word."
            
            response = self.model.generate_content([test_prompt, test_image])
            return bool(response.text)
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def create_gemini_generator(
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    **kwargs
) -> Optional[GeminiPromptGenerator]:
    """Factory function to create a Gemini prompt generator.
    
    Args:
        api_key: Google API key.
        model_name: Gemini model to use.
        **kwargs: Additional arguments for GeminiPromptGenerator.
        
    Returns:
        Configured GeminiPromptGenerator instance, or None if dependencies unavailable.
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini dependencies not available, returning None")
        return None
        
    try:
        return GeminiPromptGenerator(
            api_key=api_key,
            model_name=model_name,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create Gemini generator: {e}")
        return None


def main():
    """CLI interface for testing the Gemini prompt generator."""
    
    
    parser = argparse.ArgumentParser(description="Test Gemini Prompt Generator")
    parser.add_argument("--test-connection", action="store_true", help="Test API connection")
    parser.add_argument("--goal", type=str, help="Test goal for prompt generation")
    parser.add_argument("--screenshot", type=str, help="Path to test screenshot")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        generator = create_gemini_generator()
        
        if args.test_connection:
            print("Testing Gemini API connection...")
            if generator.test_connection():
                print("✅ Connection successful!")
            else:
                print("❌ Connection failed!")
                return
        
        if args.goal and args.screenshot:
            print(f"Generating prompt for goal: {args.goal}")
            
            # Load screenshot
            image = Image.open(args.screenshot)
            screenshot = np.array(image)
            
            # Generate prompt
            result = generator.generate_agent_prompt(screenshot, args.goal)
            
            print("\n" + "="*60)
            print("ANALYSIS:")
            print(result['analysis'])
            print("\n" + "="*60)
            print("APPROACH:")
            print(result['approach'])
            print("\n" + "="*60)
            print("GENERATED AGENT PROMPT:")
            print(result['agent_prompt'])
            print("\n" + "="*60)
            print(f"Success: {result['success']}")
            if result['error']:
                print(f"Error: {result['error']}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

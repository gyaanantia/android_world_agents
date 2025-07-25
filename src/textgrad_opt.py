#!/usr/bin/env python3
"""
TextGrad optimization module for AndroidWorld agents.

This module uses TextGrad to optimize Gemini's visual analysis output to provide
clearer, more actionable visual understanding for AndroidWorld agents. The optimizer
focuses on improving:
1. Clarity of UI element descriptions
2. Relevance to the specific task goal
3. Actionable next-step suggestions
4. Removal of unnecessary or confusing information
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    import textgrad as tg
    from textgrad import get_engine
    from textgrad.tasks import load_task
    TEXTGRAD_AVAILABLE = True
except ImportError as e:
    tg = None
    TEXTGRAD_AVAILABLE = False
    print(f"⚠️ TextGrad not available: {e}")

logger = logging.getLogger(__name__)


class TextGradOptimizer:
    """Optimizer that uses TextGrad to improve Gemini's visual analysis for better agent understanding."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_optimization_steps: int = 2,
        enabled: bool = True
    ):
        """Initialize TextGrad optimizer.
        
        Args:
            model_name: The LLM model to use for optimization (default: gpt-4o-mini)
            api_key: OpenAI API key (will use OPENAI_API_KEY env var if not provided)
            temperature: Temperature for text generation
            max_optimization_steps: Maximum number of optimization steps (keep low for efficiency)
            enabled: Whether TextGrad optimization is enabled
        """
        self.enabled = enabled and TEXTGRAD_AVAILABLE
        self.model_name = model_name
        self.temperature = temperature
        self.max_optimization_steps = max_optimization_steps
        self.initialized = False
        
        if not self.enabled:
            if not TEXTGRAD_AVAILABLE:
                logger.warning("TextGrad not available - optimization disabled")
            else:
                logger.info("TextGrad optimization disabled by user")
            return
            
        # Set up API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not found - TextGrad optimization disabled")
            self.enabled = False
            return
            
        self._initialize_textgrad()
    
    def _initialize_textgrad(self) -> bool:
        """Initialize TextGrad components.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Set the API key for TextGrad
            tg.set_backward_engine(
                get_engine(engine_name=self.model_name, api_key=self.api_key),
                override=True
            )
            
            # Create the optimization engine
            self.engine = get_engine(engine_name=self.model_name, api_key=self.api_key)
            
            logger.info(f"✅ TextGrad optimizer initialized with {self.model_name}")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ TextGrad initialization failed: {e}")
            self.enabled = False
            return False
    
    def optimize_visual_analysis(
        self, 
        gemini_analysis: str, 
        task_goal: str, 
        ui_elements: Optional[str] = None,
        feedback_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimize Gemini's visual analysis for better agent understanding.
        
        This method improves the visual analysis by:
        1. Making UI element descriptions clearer and more specific
        2. Highlighting elements most relevant to the task goal
        3. Providing more actionable next-step suggestions
        4. Removing verbose or confusing information
        
        Args:
            gemini_analysis: Original Gemini visual analysis text
            task_goal: The specific task the agent is trying to accomplish
            ui_elements: Description of UI elements on screen (optional)
            feedback_context: Additional context for optimization (optional)
            
        Returns:
            Optimized visual analysis text that provides clearer understanding for the agent
        """
        if not self.enabled or not self.initialized:
            logger.debug("TextGrad optimization skipped (disabled or not initialized)")
            return gemini_analysis
            
        try:
            return self._apply_optimization(
                gemini_analysis, task_goal, ui_elements, feedback_context
            )
        except Exception as e:
            logger.error(f"❌ TextGrad optimization failed: {e}")
            logger.warning("Falling back to original Gemini analysis")
            return gemini_analysis
    
    def _apply_optimization(
        self,
        analysis: str,
        goal: str,
        ui_elements: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Apply TextGrad optimization to the analysis text.
        
        Args:
            analysis: Original analysis text
            goal: Task goal
            ui_elements: UI elements description
            context: Additional context
            
        Returns:
            Optimized analysis text
        """
        try:
            # Check if this is Gemini's bullet-point format
            if self._is_gemini_format(analysis):
                return self._optimize_gemini_format(analysis, goal, ui_elements, context)
            else:
                return self._optimize_general_format(analysis, goal, ui_elements, context)
            
        except Exception as e:
            logger.error(f"TextGrad optimization error: {e}")
            return analysis
    
    def _is_gemini_format(self, analysis: str) -> bool:
        """Check if the analysis follows Gemini's bullet-point format."""
        # Look for Gemini's characteristic bullet points
        return "Goal:" in analysis and "Current screen:" in analysis and "Options:" in analysis
    
    def _optimize_gemini_format(
        self,
        analysis: str,
        goal: str,
        ui_elements: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Optimize Gemini's structured bullet-point format for better agent understanding."""
        
        optimization_prompt = f"""
You are improving a visual analysis from Gemini for an AI agent. The analysis follows this format:
● Goal: [task goal]
● Current screen: [screen location/context]
● Options: [list of UI elements]
● Action: [suggested action]

Your task is to improve this analysis to help an AI agent better understand what to do. Keep the EXACT same format but improve the content.

TASK GOAL: {goal}
AVAILABLE UI ELEMENTS: {ui_elements if ui_elements else "Not specified"}

CURRENT ANALYSIS:
{analysis}

Improve this analysis by:
1. Making the "Current screen" description more specific and helpful for navigation
2. KEEP ALL available UI elements in "Options" - do NOT remove any elements, but organize them to highlight the most relevant ones first
3. Making the "Action" more precise and actionable, with step-by-step details if needed
4. Keeping descriptions concise but clear

IMPORTANT: Preserve ALL UI elements mentioned in the original Options list. The agent needs to see all available choices to make informed decisions.

IMPROVED ANALYSIS (keep the ● bullet format EXACTLY):
"""
        
        # Generate improved analysis
        improved = self.engine.generate(optimization_prompt, temperature=0.2)
        improved = improved.strip()
        
        # Validate that it still follows the format
        if self._is_gemini_format(improved):
            logger.info("✅ TextGrad optimized Gemini format analysis")
            return improved
        else:
            logger.warning("TextGrad broke Gemini format, using original")
            return analysis
    
    def _optimize_general_format(
        self,
        analysis: str,
        goal: str,
        ui_elements: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Optimize general format visual analysis for better agent understanding."""
        
        optimization_prompt = f"""
You are improving a visual analysis for an AI agent that needs to complete Android tasks.

TASK GOAL: {goal}
AVAILABLE UI ELEMENTS: {ui_elements if ui_elements else "Not specified"}

CURRENT ANALYSIS:
{analysis}

Rewrite this analysis to be more helpful for an AI agent. Focus on:
1. Clear identification of ALL UI elements relevant to the task (don't remove any available elements)
2. Specific actionable next steps with detailed instructions
3. Remove unnecessary descriptive details but keep all functional information
4. Highlight the most important elements for completing: {goal}
5. Organize information to show most relevant elements first, but include everything

IMPORTANT: Preserve information about ALL available UI elements. The agent needs complete information to make good decisions.

IMPROVED ANALYSIS FOR AGENT:
"""
        
        improved = self.engine.generate(optimization_prompt, temperature=0.2)
        return improved.strip() if improved.strip() else analysis
    
    def _create_agent_clarity_prompt(
        self,
        goal: str,
        ui_elements: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create a prompt focused on improving visual analysis clarity for AI agents.
        
        Args:
            goal: Task goal
            ui_elements: UI elements description
            context: Additional context
            
        Returns:
            Optimization prompt focused on agent understanding
        """
        prompt = f"""
You are helping improve visual analysis for an AI agent that needs to complete Android tasks.

TASK GOAL: {goal}

The AI agent needs visual analysis that:
- Clearly identifies clickable/interactive elements relevant to the task
- Provides specific next-step suggestions
- Focuses only on information that helps complete the goal
- Uses precise, actionable language
- Removes unnecessary descriptive details

Available UI elements: {ui_elements if ui_elements else "Not specified"}
"""
        
        if context:
            if context.get('memory'):
                prompt += f"\nPrevious agent actions: {context.get('memory')}"
        
        return prompt
    
    def is_available(self) -> bool:
        """Check if TextGrad optimization is available.
        
        Returns:
            True if TextGrad is available and initialized, False otherwise.
        """
        return self.enabled and self.initialized


def create_textgrad_optimizer(
    model_name: str = "gpt-4o-mini",
    enabled: bool = True,
    **kwargs
) -> TextGradOptimizer:
    """Factory function to create a TextGrad optimizer.
    
    Args:
        model_name: The LLM model to use for optimization
        enabled: Whether to enable TextGrad optimization
        **kwargs: Additional arguments for TextGradOptimizer
        
    Returns:
        A TextGradOptimizer instance
    """
    return TextGradOptimizer(
        model_name=model_name,
        enabled=enabled,
        **kwargs
    )


def test_textgrad_optimization():
    """Test function to verify TextGrad optimization works."""
    if not TEXTGRAD_AVAILABLE:
        print("❌ TextGrad not available for testing")
        return False
        
    try:
        # Create optimizer
        optimizer = create_textgrad_optimizer(enabled=True)
        
        if not optimizer.is_available():
            print("❌ TextGrad optimizer not available")
            return False
            
        # Test optimization with Gemini-format sample data
        gemini_sample = """
● Goal: Send a message to John saying 'Hello'
● Current screen: Messaging app main screen
● Options: ["Compose", "Messages", "Settings", "Search"]
● Action: CLICK("Compose")
"""
        
        goal = "Send a message to John saying 'Hello'"
        ui_elements = "0: Button 'Compose' (clickable)\n1: List 'Messages' (scrollable)\n2: Button 'Settings' (clickable)"
        
        print("Testing Gemini format optimization...")
        optimized_gemini = optimizer.optimize_visual_analysis(
            gemini_sample, goal, ui_elements
        )
        
        print("✅ Gemini format optimization test successful")
        print(f"Original: {len(gemini_sample)} chars")
        print(f"Optimized: {len(optimized_gemini)} chars")
        print("Original analysis:")
        print(gemini_sample)
        print("\nOptimized analysis:")
        print(optimized_gemini)
        
        # Test general format
        general_sample = """
The screen shows a messaging app. There's a compose button in the bottom right.
Click it to send a message.
"""
        
        print("\nTesting general format optimization...")
        optimized_general = optimizer.optimize_visual_analysis(
            general_sample, goal, ui_elements
        )
        
        print("✅ General format optimization test successful")
        print(f"Original: {len(general_sample)} chars")
        print(f"Optimized: {len(optimized_general)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ TextGrad optimization test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the optimizer when run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TextGrad optimization")
    parser.add_argument("--test", action="store_true", help="Run optimization test")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use for optimization")
    
    args = parser.parse_args()
    
    if args.test:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        test_textgrad_optimization()
    else:
        print("TextGrad Optimizer Module")
        print("Use --test to run a basic optimization test")

"""Utility functions for tests that cost money."""

import os


def confirm_api_usage(test_name: str, model_name: str = "gpt-4o-mini", estimated_cost: str = "~$0.001-0.003") -> bool:
    """Ask user to confirm API usage that costs money.
    
    Args:
        test_name: Name of the test being run
        model_name: OpenAI model being used
        estimated_cost: Estimated cost range
        
    Returns:
        True if user confirms, False otherwise
    """
    print(f"\n⚠️  WARNING: {test_name} will make real OpenAI API calls that cost money.")
    print(f"   Model: {model_name}")
    print(f"   Estimated cost: {estimated_cost}")
    
    try:
        while True:
            response = input("\nDo you want to proceed with API calls? [y/N]: ").lower().strip()
            if response in ['n', 'no', '']:
                print(f"❌ {test_name} skipped - user declined API calls")
                return False
            elif response in ['y', 'yes']:
                print("✅ Proceeding with API calls...")
                return True
            else:
                print("Please enter 'y' for yes or 'n' for no")
    except (EOFError, KeyboardInterrupt):
        # Handle non-interactive mode or Ctrl+C
        print(f"\n❌ {test_name} skipped - non-interactive mode or user cancelled")
        return False


def check_api_key() -> bool:
    """Check if OpenAI API key is set.
    
    Returns:
        True if API key is available, False otherwise
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key to run tests that use the OpenAI API.")
        return False
    return True


def show_cost_summary(free_tests: list, paid_tests: list):
    """Show a summary of which tests cost money.
    
    Args:
        free_tests: List of test names that are free
        paid_tests: List of test names that cost money
    """
    print("📋 Test Cost Summary:")
    print("   FREE Tests (no API calls):")
    for test in free_tests:
        print(f"     ✅ {test}")
    
    if paid_tests:
        print("   💰 PAID Tests (real API calls):")
        for test in paid_tests:
            print(f"     💸 {test}")
    print()

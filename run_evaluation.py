#!/usr/bin/env python3
"""
Simple launcher script for AndroidWorld Enhanced T3A Agent evaluation.
"""

import sys
import os

# Add project root and src to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from main import main

if __name__ == "__main__":
    main()

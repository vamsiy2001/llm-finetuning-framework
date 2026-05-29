"""
Shared pytest configuration and fixtures.
"""

import sys
from pathlib import Path

# make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

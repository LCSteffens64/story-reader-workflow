"""
Entry point for running the package as a module.

Usage:
    python -m story_reader -i narration.wav -o output/
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())

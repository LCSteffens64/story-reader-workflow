#!/usr/bin/env python3
"""
Story Reader Pipeline - Backward Compatible Entry Point

This file provides backward compatibility with the original main.py interface.
For new usage, prefer:
    python -m story_reader -i narration.wav -o output/

Or use the package directly in Python:
    from story_reader import StoryReaderPipeline, PipelineConfig
    config = PipelineConfig(input_audio="narration.wav")
    pipeline = StoryReaderPipeline(config)
    pipeline.run()
"""

import sys

# Use the new modular package
from story_reader.cli import main

if __name__ == "__main__":
    sys.exit(main())

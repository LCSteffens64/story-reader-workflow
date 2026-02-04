"""
Story Reader Workflow - Automated video generation from narration audio.

This package provides a pipeline for converting spoken narration into
visual "story reader" videos with AI-generated imagery.
"""

from .config import PipelineConfig
from .pipeline import StoryReaderPipeline
from .core.cache import CacheManager
from .core.job import Job

__version__ = "0.2.0"
__all__ = [
    "PipelineConfig",
    "StoryReaderPipeline", 
    "CacheManager",
    "Job",
]

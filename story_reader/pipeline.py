"""
Main pipeline orchestrator for the Story Reader workflow.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from .config import PipelineConfig
from .core.cache import CacheManager, NullCacheManager
from .core.job import Job
from .steps import (
    TranscriberStep,
    SegmenterStep,
    ImageGeneratorStep,
    VideoComposerStep,
    AudioMixerStep,
)


class StoryReaderPipeline:
    """
    Main pipeline orchestrator that coordinates all processing steps.
    
    The pipeline transforms narration audio into a visual video with:
    1. Speech-to-text transcription (Whisper)
    2. Paragraph segmentation
    3. AI image generation (Stable Diffusion)
    4. Ken Burns video effect
    5. Audio mixing
    
    Example:
        ```python
        config = PipelineConfig(input_audio="story.wav")
        pipeline = StoryReaderPipeline(config)
        result = pipeline.run()
        print(f"Video created: {result}")
        ```
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache manager
        if config.clear_cache:
            cache = CacheManager(config.output_dir)
            cache.clear()
            self.cache = CacheManager(config.output_dir)
        elif config.use_cache:
            self.cache = CacheManager(config.output_dir)
        else:
            self.cache = NullCacheManager()
        
        # Initialize steps
        self._init_steps()
    
    def _init_steps(self) -> None:
        """Initialize all pipeline steps."""
        self.transcriber = TranscriberStep(self.config, self.cache)
        self.segmenter = SegmenterStep(self.config, self.cache)
        self.image_generator = ImageGeneratorStep(self.config, self.cache)
        self.video_composer = VideoComposerStep(self.config, self.cache)
        self.audio_mixer = AudioMixerStep(self.config, self.cache)
    
    def run(self) -> Path:
        """
        Execute the full pipeline.
        
        Returns:
            Path to the final video file
            
        Raises:
            FileNotFoundError: If input audio doesn't exist
            RuntimeError: If any pipeline step fails
        """
        # Validate input
        if not self.config.input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {self.config.input_audio}")
        
        print(f"Input audio: {self.config.input_audio}")
        print(f"Output directory: {self.config.output_dir}")
        print("-" * 50)
        
        # Step 1: Transcribe audio
        segments = self.transcriber.execute(self.config.input_audio)
        
        # Step 2: Segment into paragraphs
        paragraphs = self.segmenter.execute(segments)
        
        # Step 3: Generate images
        images = self.image_generator.execute(paragraphs)
        
        # Step 4: Create Ken Burns video
        visuals_video = self.video_composer.execute((images, paragraphs))
        
        # Step 5: Mux audio (optional)
        if not self.config.skip_audio_mux:
            final_video = self.audio_mixer.execute(visuals_video)
        else:
            final_video = visuals_video
            print("Skipped audio muxing (--no-audio flag)")
        
        print("-" * 50)
        print(f"Pipeline complete! Output: {final_video}")
        
        return final_video
    
    def run_step(self, step_name: str, input_data: Any) -> Any:
        """
        Run a single pipeline step (useful for testing/debugging).
        
        Args:
            step_name: Name of the step to run
            input_data: Input data for the step
            
        Returns:
            Output from the step
        """
        steps = {
            'transcribe': self.transcriber,
            'segment': self.segmenter,
            'generate_images': self.image_generator,
            'compose_video': self.video_composer,
            'mix_audio': self.audio_mixer,
        }
        
        if step_name not in steps:
            raise ValueError(f"Unknown step: {step_name}. Available: {list(steps.keys())}")
        
        return steps[step_name].execute(input_data)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached items."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
    
    @staticmethod
    def clear_model_caches() -> None:
        """Clear all model caches to free memory."""
        TranscriberStep.clear_model_cache()
        ImageGeneratorStep.clear_pipeline_cache()

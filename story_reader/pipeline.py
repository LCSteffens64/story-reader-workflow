"""
Main pipeline orchestrator for the Story Reader workflow.
"""

from pathlib import Path
import shutil
import subprocess
from typing import Optional, Dict, Any

from .config import PipelineConfig
from .core.cache import CacheManager, NullCacheManager
from .core.job import Job
from .steps import (
    TranscriberStep,
    SegmenterStep,
    ImageGeneratorStep,
    HybridImageGeneratorStep,
    LegnextImageGeneratorStep,
    ImageUpscalerStep,
    UpscaleMethod,
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
        
        # Setup output directories
        self._init_output_dirs()
        
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

    def _init_output_dirs(self) -> None:
        """Initialize output directories and clear old assets."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Clear old scenes on each run to avoid stale assets
        if self.config.scenes_dir.exists():
            if not self.config.keep_scenes:
                shutil.rmtree(self.config.scenes_dir)
        if not self.config.scenes_dir.exists():
            self.config.scenes_dir.mkdir(parents=True, exist_ok=True)

        # Preserve images when cache is enabled to allow resuming
        if self.config.images_dir.exists():
            if not self.config.use_cache or self.config.clear_cache:
                shutil.rmtree(self.config.images_dir)
                self.config.images_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.config.images_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_steps(self) -> None:
        """Initialize all pipeline steps."""
        self.transcriber = TranscriberStep(self.config, self.cache)
        self.segmenter = SegmenterStep(self.config, self.cache)

        if self.config.use_legnext and self.config.use_pexels:
            raise ValueError("Image sources are mutually exclusive: use only one of --use-legnext or --use-pexels.")
        
        # Choose image source (Legnext, Pexels, or Stable Diffusion)
        if self.config.use_legnext:
            self.image_generator = LegnextImageGeneratorStep(self.config, self.cache)
        elif self.config.use_pexels:
            self.image_generator = HybridImageGeneratorStep(self.config, self.cache)
        else:
            self.image_generator = ImageGeneratorStep(self.config, self.cache)
        
        # Initialize upscaler if enabled
        if self.config.upscale_images:
            method = UpscaleMethod(self.config.upscale_method)
            self.image_upscaler = ImageUpscalerStep(
                self.config, 
                self.cache,
                scale_factor=self.config.upscale_factor,
                method=method,
                enhance_sharpness=self.config.upscale_sharpness,
                enhance_contrast=self.config.upscale_contrast,
            )
        else:
            self.image_upscaler = None
        
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
        # Prepare input audio (concatenate narration takes if present)
        self._prepare_input_audio()
        self._prepare_music_audio()
        self._validate_music_length()

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
        
        # Step 3b: Upscale images (optional)
        if self.image_upscaler:
            images = self.image_upscaler.execute(images)
        
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

    def _prepare_input_audio(self) -> None:
        """
        If narration takes exist (narration/1.wav, 2.wav, ...),
        concatenate them into a single file for transcription and muxing.
        """
        narration_dir = self.config.narration_dir
        if not narration_dir.exists() or not narration_dir.is_dir():
            return

        takes = sorted(
            narration_dir.glob("*.wav"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
        )
        if not takes:
            return

        concat_sources = takes
        if self.config.normalize_narration:
            normalized_dir = self.config.output_dir / "narration_normalized"
            normalized_dir.mkdir(parents=True, exist_ok=True)
            concat_sources = []
            for take in takes:
                normalized = normalized_dir / take.name
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", str(take),
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-c:a", "pcm_s16le",
                    str(normalized),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"FFmpeg loudnorm stderr: {result.stderr}")
                    raise RuntimeError(f"FFmpeg loudnorm failed for {take.name}: {result.stderr}")
                concat_sources.append(normalized)

        concat_list = self.config.output_dir / "narration_concat.txt"
        with open(concat_list, "w") as f:
            for take in concat_sources:
                f.write(f"file '{take.resolve()}'\n")

        combined = self.config.output_dir / "narration_combined.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(combined)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg concat stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        self.config.input_audio = combined

    def _prepare_music_audio(self) -> None:
        """
        If multiple music tracks exist under the music directory,
        concatenate them into a single file for muxing.
        """
        if self.config.disable_music:
            return
        if self.config.background_music and self.config.background_music.exists():
            return

        music_dir = self.config.music_dir
        if not music_dir.exists() or not music_dir.is_dir():
            return

        tracks = sorted(music_dir.rglob("*.mp3"))
        if not tracks:
            return

        if len(tracks) == 1:
            self.config.background_music = tracks[0]
            return

        concat_list = self.config.output_dir / "music_concat.txt"
        with open(concat_list, "w") as f:
            for track in tracks:
                f.write(f"file '{track.resolve()}'\n")

        combined = self.config.output_dir / "music_combined.mp3"
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(combined)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg concat stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg music concat failed: {result.stderr}")

        self.config.background_music = combined

    def _get_media_duration(self, media_path: Path) -> Optional[float]:
        """Return duration in seconds for a media file via ffprobe."""
        if not media_path or not media_path.exists():
            return None
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffprobe stderr: {result.stderr}")
            raise RuntimeError(f"ffprobe failed for {media_path}")
        try:
            return float(result.stdout.strip())
        except ValueError as e:
            raise RuntimeError(f"Invalid ffprobe duration for {media_path}: {result.stdout}") from e

    def _validate_music_length(self) -> None:
        """
        Ensure background music covers the narration length when music is enabled.
        """
        if self.config.disable_music:
            return

        narration_len = self._get_media_duration(self.config.input_audio)
        music_path = self.config.background_music
        if not music_path or not music_path.exists():
            return
        music_len = self._get_media_duration(music_path)
        if narration_len is None or music_len is None:
            return
        if music_len + 0.5 < narration_len:
            raise RuntimeError(
                f"Background music is too short ({music_len:.1f}s) for narration "
                f"({narration_len:.1f}s). Add more tracks to {self.config.music_dir} "
                "or rerun with --no-music."
            )
    
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
        ImageUpscalerStep.clear_model_cache()

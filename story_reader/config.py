"""
Configuration dataclass for the Story Reader pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import os


@dataclass
class PipelineConfig:
    """
    Configuration for the Story Reader pipeline.
    
    All settings can be overridden via CLI arguments or by passing
    values directly when instantiating.
    """
    
    # Input/Output paths
    input_audio: Path = field(default_factory=lambda: Path("narration.wav"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    background_music: Optional[Path] = None
    
    # Model settings
    whisper_model: str = "tiny"  # tiny/base/small/medium/large
    sd_model: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"  # cpu/cuda
    hf_token: Optional[str] = None
    
    # Image generation settings
    image_size: Tuple[int, int] = (384, 384)
    prompt_style: str = "photojournalistic"
    prompt_template: str = "{style}, documentary style, candid moment, natural lighting, real scene depicting: {text}, 35mm film, high detail, authentic"
    negative_prompt: str = "text, words, letters, writing, watermark, signature, logo, blurry, low quality, cartoon, anime, illustration, drawing, painting, artificial, posed, staged"
    
    # Paragraph segmentation settings
    max_paragraph_duration: float = 15.0
    min_silence: float = 1.5
    max_sentences: int = 3
    
    # Video settings
    fps: int = 25
    ken_burns_zoom_speed: float = 0.0004
    ken_burns_max_zoom: float = 1.08
    
    # Caching options
    use_cache: bool = True
    clear_cache: bool = False
    
    # Output options
    skip_audio_mux: bool = False
    music_volume: float = 0.3
    narration_volume: float = 1.0
    audio_bitrate: str = "192k"
    
    def __post_init__(self):
        """Convert string paths to Path objects and resolve them."""
        if isinstance(self.input_audio, str):
            self.input_audio = Path(self.input_audio)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.background_music, str):
            self.background_music = Path(self.background_music)
        
        # Resolve paths to absolute
        self.input_audio = self.input_audio.resolve()
        self.output_dir = self.output_dir.resolve()
        if self.background_music:
            self.background_music = self.background_music.resolve()
        
        # Load HF token from environment if not provided
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")
    
    @property
    def jobs_file(self) -> Path:
        """Path to the jobs tracking file."""
        return self.output_dir / "jobs.json"
    
    @property
    def cache_dir(self) -> Path:
        """Path to the cache directory."""
        return self.output_dir / ".cache"
    
    @property
    def paragraphs_file(self) -> Path:
        """Path to the paragraphs JSON file."""
        return self.output_dir / "paragraphs.json"
    
    def get_image_prompt(self, text: str) -> str:
        """Generate an image prompt from paragraph text."""
        return self.prompt_template.format(
            style=self.prompt_style,
            text=text
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (for serialization)."""
        return {
            "input_audio": str(self.input_audio),
            "output_dir": str(self.output_dir),
            "background_music": str(self.background_music) if self.background_music else None,
            "whisper_model": self.whisper_model,
            "sd_model": self.sd_model,
            "device": self.device,
            "image_size": self.image_size,
            "prompt_style": self.prompt_style,
            "max_paragraph_duration": self.max_paragraph_duration,
            "min_silence": self.min_silence,
            "max_sentences": self.max_sentences,
            "fps": self.fps,
            "ken_burns_zoom_speed": self.ken_burns_zoom_speed,
            "use_cache": self.use_cache,
            "skip_audio_mux": self.skip_audio_mux,
            "music_volume": self.music_volume,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create config from dictionary."""
        return cls(**data)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()

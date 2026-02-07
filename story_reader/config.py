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
    music_dir: Path = field(default_factory=lambda: Path("music"))
    narration_dir: Path = field(default_factory=lambda: Path("narration"))
    video_title: Optional[str] = None
    
    # Model settings
    whisper_model: str = "tiny"  # tiny/base/small/medium/large
    sd_model: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"  # cpu/cuda
    hf_token: Optional[str] = None
    pexels_api_key: Optional[str] = None
    legnext_api_key: Optional[str] = None
    
    # Image generation settings
    image_size: Tuple[int, int] = (384, 384)
    prompt_style: str = "photojournalistic"
    prompt_template: str = "{style}, documentary style, candid moment, natural lighting, real scene depicting: {text}, 35mm film, high detail, authentic"
    negative_prompt: str = "text, words, letters, writing, watermark, signature, logo, blurry, low quality, cartoon, anime, illustration, drawing, painting, artificial, posed, staged"
    legnext_prompt_template: str = "{text}"
    legnext_tone_keywords: Optional[str] = None
    
    # Paragraph segmentation settings
    max_paragraph_duration: float = 15.0
    min_silence: float = 1.5
    max_sentences: int = 3
    
    # Video settings
    fps: int = 25
    ken_burns_zoom_speed: float = 0.0004
    ken_burns_max_zoom: float = 1.08
    video_padding_sec: float = 0.5
    
    # Caching options
    use_cache: bool = True
    clear_cache: bool = False
    
    # Upscaling options
    upscale_images: bool = False
    upscale_factor: float = 2.0
    upscale_method: str = "pil_lanczos"  # pil_lanczos, pil_bicubic, opencv_cubic, real_esrgan
    upscale_sharpness: float = 1.2
    upscale_contrast: float = 1.0
    
    # Output options
    skip_audio_mux: bool = False
    music_volume: float = 0.15
    music_dir_volume: float = 0.3
    narration_volume: float = 1.0
    audio_bitrate: str = "192k"
    disable_music: bool = False
    keep_scenes: bool = False
    normalize_narration: bool = True

    # Pexels options
    use_pexels: bool = False
    pexels_fallback_to_sd: bool = False
    pexels_per_page: int = 15
    pexels_orientation: str = "landscape"
    pexels_min_width: int = 1280
    pexels_min_height: int = 720

    # Legnext options
    use_legnext: bool = False
    legnext_poll_interval_sec: float = 2.0
    legnext_timeout_sec: float = 180.0
    
    
    # LLM keyword extraction settings
    llm_keyword_extractor: bool = True
    llm_model_name: str = "microsoft/phi-2"
    llm_quantization: bool = True
    llm_min_keywords: int = 3
    llm_max_keywords: int = 5
    use_paragraphs_file: bool = False
    
    def __post_init__(self):
        """Convert string paths to Path objects and resolve them."""
        if isinstance(self.input_audio, str):
            self.input_audio = Path(self.input_audio)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.background_music, str):
            self.background_music = Path(self.background_music)
        if isinstance(self.music_dir, str):
            self.music_dir = Path(self.music_dir)
        if isinstance(self.narration_dir, str):
            self.narration_dir = Path(self.narration_dir)
        
        # Resolve paths to absolute
        self.input_audio = self.input_audio.resolve()
        self.output_dir = self.output_dir.resolve()
        if self.background_music:
            self.background_music = self.background_music.resolve()
        self.music_dir = self.music_dir.resolve()
        self.narration_dir = self.narration_dir.resolve()
        
        # Load HF token from environment if not provided
        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")

        if self.pexels_api_key is None:
            self.pexels_api_key = os.environ.get("PEXELS_API_KEY")

        if self.legnext_api_key is None:
            self.legnext_api_key = os.environ.get("LEGNEXT_API_KEY")
    
    @property
    def jobs_file(self) -> Path:
        """Path to the jobs tracking file."""
        return self.output_dir / "jobs.json"
    
    @property
    def cache_dir(self) -> Path:
        """Path to the cache directory."""
        return self.output_dir / ".cache"

    @property
    def images_dir(self) -> Path:
        """Path to the images directory."""
        return self.output_dir / "images"

    @property
    def scenes_dir(self) -> Path:
        """Path to the scenes directory."""
        return self.output_dir / "scenes"
    
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

    def get_legnext_prompt(self, text: str) -> str:
        """Generate a Legnext prompt from paragraph text, without style biasing."""
        prompt = self.legnext_prompt_template.format(text=text)
        if self.legnext_tone_keywords:
            prompt = f"{prompt} | tone: {self.legnext_tone_keywords}"
        return prompt
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (for serialization)."""
        return {
            "input_audio": str(self.input_audio),
            "output_dir": str(self.output_dir),
            "background_music": str(self.background_music) if self.background_music else None,
            "music_dir": str(self.music_dir),
            "music_dir_volume": self.music_dir_volume,
            "narration_dir": str(self.narration_dir),
            "video_title": self.video_title,
            "whisper_model": self.whisper_model,
            "sd_model": self.sd_model,
            "device": self.device,
            "image_size": self.image_size,
            "prompt_style": self.prompt_style,
            "legnext_prompt_template": self.legnext_prompt_template,
            "legnext_tone_keywords": self.legnext_tone_keywords,
            "max_paragraph_duration": self.max_paragraph_duration,
            "min_silence": self.min_silence,
            "max_sentences": self.max_sentences,
            "fps": self.fps,
            "ken_burns_zoom_speed": self.ken_burns_zoom_speed,
            "use_cache": self.use_cache,
            "skip_audio_mux": self.skip_audio_mux,
            "music_volume": self.music_volume,
            "disable_music": self.disable_music,
            "keep_scenes": self.keep_scenes,
            "normalize_narration": self.normalize_narration,
            "llm_keyword_extractor": self.llm_keyword_extractor,
            "llm_model_name": self.llm_model_name,
            "llm_quantization": self.llm_quantization,
            "use_paragraphs_file": self.use_paragraphs_file,
            "use_pexels": self.use_pexels,
            "pexels_fallback_to_sd": self.pexels_fallback_to_sd,
            "pexels_per_page": self.pexels_per_page,
            "pexels_orientation": self.pexels_orientation,
            "pexels_min_width": self.pexels_min_width,
            "pexels_min_height": self.pexels_min_height,
            "use_legnext": self.use_legnext,
            "legnext_poll_interval_sec": self.legnext_poll_interval_sec,
            "legnext_timeout_sec": self.legnext_timeout_sec,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create config from dictionary."""
        return cls(**data)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()

"""
Command-line interface for the Story Reader pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import re

from .config import PipelineConfig


def check_dependencies() -> bool:
    """Check that required external tools are available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: ffmpeg is not working correctly!")
            return False
    except FileNotFoundError:
        print("ERROR: ffmpeg is not installed! Please install it with your package manager.")
        return False
    return True


def validate_audio_file(audio_path: Path) -> bool:
    """Validate that the input audio file exists and is valid."""
    if not audio_path.exists():
        print(f"ERROR: Audio file '{audio_path}' does not exist!")
        return False
    
    if not audio_path.is_file():
        print(f"ERROR: '{audio_path}' is not a file!")
        return False
    
    valid_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    if audio_path.suffix.lower() not in valid_extensions:
        print(f"WARNING: Audio file has unusual extension '{audio_path.suffix}'. "
              f"Supported formats: {', '.join(valid_extensions)}")
    
    if audio_path.stat().st_size == 0:
        print(f"ERROR: Audio file '{audio_path}' is empty!")
        return False
    
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="story-reader",
        description="Story Reader Pipeline: Convert audio narration to visual video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m story_reader -i narration.wav -o output/
  python -m story_reader --input my_story.mp3 --output results/
  python -m story_reader -i narration.wav -m background.mp3 -o output/
  python -m story_reader --clear-cache  # Clear cached data

For more information, see: https://github.com/your-repo/story-reader
        """
    )
    
    # Input/Output
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("narration.wav"),
        help="Path to input audio file (default: narration.wav)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)"
    )
    
    # Audio options
    parser.add_argument(
        "-m", "--music",
        type=Path,
        default=None,
        help="Optional background music file"
    )
    parser.add_argument(
        "--music-volume",
        type=float,
        default=0.3,
        help="Background music volume (0.0-1.0, default: 0.3)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio muxing (output visuals only)"
    )
    
    
    # LLM keyword extraction options
    parser.add_argument(
        "--llm-keywords",
        action="store_true",
        default=True,
        help="Use LLM for keyword extraction (default: enabled)"
    )
    parser.add_argument(
        "--no-llm-keywords",
        action="store_false",
        dest="llm_keywords",
        help="Disable LLM keyword extraction (use simple extraction)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="microsoft/phi-2",
        help="LLM model for keyword extraction (default: microsoft/phi-2)"
    )
    parser.add_argument(
        "--llm-quantization",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for LLM (default: enabled)"
    )
    parser.add_argument(
        "--no-llm-quantization",
        action="store_false",
        dest="llm_quantization",
        help="Disable LLM quantization (uses more memory)"
    )
    
    # Model options
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: tiny)"
    )
    parser.add_argument(
        "--sd-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model name (default: runwayml/stable-diffusion-v1-5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)"
    )
    
    # Image options
    parser.add_argument(
        "--image-size",
        type=str,
        default="384x384",
        help="Generated image size as WxH (default: 384x384)"
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="photojournalistic",
        help="Style prefix for image prompts (default: photojournalistic)"
    )

    # Image source options (mutually exclusive)
    image_source_group = parser.add_mutually_exclusive_group()
    image_source_group.add_argument(
        "--use-pexels",
        action="store_true",
        help="Fetch images from Pexels instead of generating locally"
    )
    image_source_group.add_argument(
        "--use-legnext",
        action="store_true",
        help="Fetch images from Legnext instead of generating locally"
    )

    # Pexels options
    parser.add_argument(
        "--sd-fallback",
        action="store_true",
        help="If Pexels fails, fall back to Stable Diffusion generation"
    )
    parser.add_argument(
        "--pexels-per-page",
        type=int,
        default=15,
        help="Pexels search results per query (default: 15)"
    )
    parser.add_argument(
        "--pexels-min-width",
        type=int,
        default=1280,
        help="Minimum width for Pexels images (default: 1280)"
    )
    parser.add_argument(
        "--pexels-min-height",
        type=int,
        default=720,
        help="Minimum height for Pexels images (default: 720)"
    )

    # Legnext options
    parser.add_argument(
        "--legnext-poll-interval",
        type=float,
        default=2.0,
        help="Legnext polling interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--legnext-timeout",
        type=float,
        default=180.0,
        help="Legnext job timeout in seconds (default: 180.0)"
    )
    
    # Upscaling options
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Enable image upscaling for higher resolution"
    )
    parser.add_argument(
        "--upscale-factor",
        type=float,
        default=2.0,
        help="Upscale multiplier (default: 2.0 = 2x resolution)"
    )
    parser.add_argument(
        "--upscale-method",
        type=str,
        default="pil_lanczos",
        choices=["pil_lanczos", "pil_bicubic", "opencv_cubic", "real_esrgan"],
        help="Upscaling method (default: pil_lanczos)"
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=1.2,
        help="Sharpness enhancement after upscaling (1.0 = no change, default: 1.2)"
    )
    
    # Cache options
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached data before running"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (regenerate everything)"
    )
    
    # Misc options
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0"
    )
    
    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> PipelineConfig:
    """Convert parsed arguments to PipelineConfig."""
    # Parse image size
    if "x" in args.image_size:
        w, h = args.image_size.split("x")
        image_size = (int(w), int(h))
    else:
        image_size = (384, 384)
    
    return PipelineConfig(
        input_audio=args.input,
        output_dir=args.output,
        background_music=args.music,
        music_volume=args.music_volume,
        skip_audio_mux=args.no_audio,
        whisper_model=args.whisper_model,
        sd_model=args.sd_model,
        device=args.device,
        image_size=image_size,
        prompt_style=args.prompt_style,
        upscale_images=args.upscale,
        upscale_factor=args.upscale_factor,
        upscale_method=args.upscale_method,
        upscale_sharpness=args.sharpen,
        clear_cache=args.clear_cache,
        use_cache=not args.no_cache,
        llm_keyword_extractor=args.llm_keywords,
        llm_model_name=args.llm_model,
        llm_quantization=args.llm_quantization,
        use_pexels=args.use_pexels,
        use_legnext=args.use_legnext,
        pexels_fallback_to_sd=args.sd_fallback,
        pexels_per_page=args.pexels_per_page,
        pexels_min_width=args.pexels_min_width,
        pexels_min_height=args.pexels_min_height,
        legnext_poll_interval_sec=args.legnext_poll_interval,
        legnext_timeout_sec=args.legnext_timeout,
        narration_dir=Path("narration"),
    )


def main() -> int:
    """Main entry point for CLI."""
    args = parse_args()

    # Prompt for video title on invocation
    title = input("Video title: ").strip()
    if not title:
        title = "untitled"

    # If user didn't override output dir, place outputs under a title folder
    if args.output == Path("output"):
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "-", title).strip("-").lower()
        if not safe_title:
            safe_title = "untitled"
        args.output = args.output / safe_title
    
    # Validate dependencies
    if not check_dependencies():
        return 1
    
    # Validate input file unless narration takes exist
    narration_dir = Path("narration")
    narration_takes = list(narration_dir.glob("*.wav")) if narration_dir.exists() else []
    if not narration_takes:
        if not validate_audio_file(args.input):
            return 1
    else:
        print(f"Found {len(narration_takes)} narration take(s) in {narration_dir}/; will concatenate.")
    
    # Create config from args
    config = args_to_config(args)
    config.video_title = title
    
    # Import pipeline here to avoid slow imports on --help
    from .pipeline import StoryReaderPipeline
    
    try:
        pipeline = StoryReaderPipeline(config)
        result = pipeline.run()
        return 0
    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

import os
import sys
import json
import hashlib
import argparse
import fcntl
from pathlib import Path
import subprocess
import whisper
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime

# ----------------------------
# CONFIGURATION
# ----------------------------
DEFAULT_AUDIO_FILE = "narration.wav"
DEFAULT_OUTPUT_DIR = Path("output")
HF_TOKEN = "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"
DEVICE = "cpu"

WHISPER_MODEL = "tiny"  # small/base for 4GB VRAM
#SD_MODEL = "OFA-Sys/small-stable-diffusion-v0"
SD_MODEL = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = (384, 384)
MAX_PARAGRAPH_DURATION = 15.0
MIN_SILENCE = 1.5
MAX_SENTENCES = 3
KEN_BURNS_ZOOM_SPEED = 0.0004
FPS = 25


num_cores = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)

torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
# ----------------------------
# GLOBAL CACHES
# ----------------------------
_whisper_model_cache = {}
_sd_pipeline_cache = {}

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for cache invalidation."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_cached_whisper_model(model_name: str):
    """Load Whisper model with caching."""
    global _whisper_model_cache
    if model_name not in _whisper_model_cache:
        print(f"Loading Whisper model '{model_name}'...")
        _whisper_model_cache[model_name] = whisper.load_model(model_name)
    else:
        print(f"Using cached Whisper model '{model_name}'")
    return _whisper_model_cache[model_name]


def get_cached_sd_pipeline(model_name: str, device: str):
    """Load Stable Diffusion pipeline with caching."""
    global _sd_pipeline_cache
    cache_key = f"{model_name}_{device}"
    if cache_key not in _sd_pipeline_cache:
        print(f"Loading Stable Diffusion model '{model_name}'...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float32, use_auth_token=HF_TOKEN, 
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()  # low VRAM mode
        _sd_pipeline_cache[cache_key] = pipe
    else:
        print(f"Using cached Stable Diffusion pipeline")
    return _sd_pipeline_cache[cache_key]


# ----------------------------
# JOB TRACKING (with file locking)
# ----------------------------
class Job:
    def __init__(self, name: str, tracker_file: Path):
        self.name = name
        self.tracker_file = tracker_file
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.save_status()

    def start(self):
        self.status = "running"
        self.start_time = datetime.now().isoformat()
        self.save_status()
        print(f"[{self.name}] started...")

    def complete(self):
        self.status = "completed"
        self.end_time = datetime.now().isoformat()
        self.save_status()
        print(f"[{self.name}] completed.")

    def fail(self, error_msg: str):
        self.status = f"failed: {error_msg}"
        self.end_time = datetime.now().isoformat()
        self.save_status()
        print(f"[{self.name}] FAILED: {error_msg}")

    def save_status(self):
        """Save status with file locking to prevent race conditions."""
        lock_file = self.tracker_file.with_suffix(".lock")
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(lock_file, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                all_jobs = {}
                if self.tracker_file.exists():
                    with open(self.tracker_file, "r") as f:
                        all_jobs = json.load(f)
                all_jobs[self.name] = {
                    "status": self.status,
                    "start_time": self.start_time,
                    "end_time": self.end_time
                }
                with open(self.tracker_file, "w") as f:
                    json.dump(all_jobs, f, indent=2)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


# ----------------------------
# CACHE MANAGER
# ----------------------------
class CacheManager:
    """Manages caching for transcription and image generation."""
    
    def __init__(self, output_dir: Path):
        self.cache_dir = output_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> dict:
        if self.cache_index_file.exists():
            with open(self.cache_index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache_index(self):
        with open(self.cache_index_file, "w") as f:
            json.dump(self.cache_index, f, indent=2)

    def get_transcription_cache_key(self, audio_path: Path) -> str:
        file_hash = compute_file_hash(audio_path)
        return f"transcription_{file_hash}"

    def get_cached_transcription(self, audio_path: Path):
        cache_key = self.get_transcription_cache_key(audio_path)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                print(f"Using cached transcription for '{audio_path.name}'")
                with open(cache_file, "r") as f:
                    return json.load(f)
        return None

    def save_transcription(self, audio_path: Path, segments: list):
        cache_key = self.get_transcription_cache_key(audio_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(segments, f, indent=2)
        self.cache_index[cache_key] = {
            "audio_file": str(audio_path),
            "created": datetime.now().isoformat()
        }
        self._save_cache_index()

    def get_image_cache_key(self, paragraph_text: str, idx: int) -> str:
        text_hash = hashlib.md5(paragraph_text.encode()).hexdigest()[:12]
        return f"image_{idx:03d}_{text_hash}"

    def get_cached_image(self, paragraph_text: str, idx: int, output_dir: Path) -> Path | None:
        cache_key = self.get_image_cache_key(paragraph_text, idx)
        if cache_key in self.cache_index:
            cached_path = Path(self.cache_index[cache_key].get("path", ""))
            if cached_path.exists():
                # Copy to expected output location if different
                expected_path = output_dir / f"{idx:03d}.png"
                if cached_path != expected_path:
                    import shutil
                    shutil.copy(cached_path, expected_path)
                print(f"Using cached image for paragraph {idx}")
                return expected_path
        return None

    def save_image_cache(self, paragraph_text: str, idx: int, image_path: Path):
        cache_key = self.get_image_cache_key(paragraph_text, idx)
        self.cache_index[cache_key] = {
            "path": str(image_path),
            "text_preview": paragraph_text[:100],
            "created": datetime.now().isoformat()
        }
        self._save_cache_index()


# ----------------------------
# INPUT VALIDATION
# ----------------------------
def validate_inputs(audio_path: Path) -> bool:
    """Validate that required input files exist."""
    if not audio_path.exists():
        print(f"ERROR: Audio file '{audio_path}' does not exist!")
        return False
    
    if not audio_path.is_file():
        print(f"ERROR: '{audio_path}' is not a file!")
        return False
    
    # Check file extension
    valid_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    if audio_path.suffix.lower() not in valid_extensions:
        print(f"WARNING: Audio file has unusual extension '{audio_path.suffix}'. "
              f"Supported formats: {', '.join(valid_extensions)}")
    
    # Check file is not empty
    if audio_path.stat().st_size == 0:
        print(f"ERROR: Audio file '{audio_path}' is empty!")
        return False
    
    return True


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


# ----------------------------
# STEP 1: Transcribe Audio
# ----------------------------
def transcribe_audio(audio_path: Path, output_dir: Path, cache_manager: CacheManager):
    job = Job("transcription", output_dir / "jobs.json")
    try:
        job.start()
        
        # Check cache first
        cached = cache_manager.get_cached_transcription(audio_path)
        if cached is not None:
            job.complete()
            return cached
        
        model = get_cached_whisper_model(WHISPER_MODEL)
        print("Transcribing audio...")
        result = model.transcribe(str(audio_path))
        segments = result["segments"]
        
        # Save to cache
        cache_manager.save_transcription(audio_path, segments)
        
        job.complete()
        return segments
    except Exception as e:
        job.fail(str(e))
        raise


# ----------------------------
# STEP 2: Merge Segments into Paragraphs
# ----------------------------
def segments_to_paragraphs(segments: list, output_dir: Path):
    job = Job("paragraph_segmentation", output_dir / "jobs.json")
    try:
        job.start()
        paragraphs = []
        current = []
        paragraph_start = None

        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"].strip()

            if paragraph_start is None:
                paragraph_start = seg_start

            prev_end = current[-1]["end"] if current else seg_start
            gap = seg_start - prev_end
            paragraph_duration = seg_end - paragraph_start
            num_sentences = len(current) + 1

            if gap >= MIN_SILENCE or paragraph_duration >= MAX_PARAGRAPH_DURATION or num_sentences > MAX_SENTENCES:
                if current:
                    para_text = " ".join([s["text"] for s in current])
                    paragraphs.append({
                        "text": para_text,
                        "start": paragraph_start,
                        "end": current[-1]["end"],
                        "duration": current[-1]["end"] - paragraph_start
                    })
                current = []
                paragraph_start = seg_start

            current.append(seg)

        if current:
            para_text = " ".join([s["text"] for s in current])
            paragraphs.append({
                "text": para_text,
                "start": paragraph_start,
                "end": current[-1]["end"],
                "duration": current[-1]["end"] - paragraph_start
            })

        Path(output_dir / "paragraphs.json").write_text(json.dumps(paragraphs, indent=2))
        job.complete()
        return paragraphs
    except Exception as e:
        job.fail(str(e))
        raise


# ----------------------------
# STEP 3: Generate Images (Stable Diffusion)
# ----------------------------
def generate_images(paragraphs: list, output_dir: Path, cache_manager: CacheManager):
    job = Job("image_generation", output_dir / "jobs.json")
    try:
        job.start()
        device = DEVICE
        
        images_paths = []
        pipe = None  # Lazy load only if needed

        for idx, para in enumerate(paragraphs):
            # Check cache first
            cached_path = cache_manager.get_cached_image(para['text'], idx, output_dir)
            if cached_path is not None:
                images_paths.append(cached_path)
                continue
            
            # Load pipeline only when needed (first uncached image)
            if pipe is None:
                pipe = get_cached_sd_pipeline(SD_MODEL, device)
            
            prompt = f"photojournalistic photography, documentary style, candid moment, natural lighting, real scene depicting: {para['text']}, 35mm film, high detail, authentic"
            negative_prompt = "text, words, letters, writing, watermark, signature, logo, blurry, low quality, cartoon, anime, illustration, drawing, painting, artificial, posed, staged"
            print(f"Generating image {idx+1}/{len(paragraphs)}...")
            image = pipe(
                prompt, 
                negative_prompt=negative_prompt,
                height=IMAGE_SIZE[1], 
                width=IMAGE_SIZE[0]
            ).images[0]
            img_path = output_dir / f"{idx:03d}.png"
            image.save(img_path)
            images_paths.append(img_path)
            
            # Save to cache
            cache_manager.save_image_cache(para['text'], idx, img_path)

        job.complete()
        return images_paths
    except Exception as e:
        job.fail(str(e))
        raise


# ----------------------------
# STEP 4: Create Ken Burns Video
# ----------------------------
def create_ken_burns_video(images_paths: list, paragraphs: list, output_dir: Path) -> Path:
    job = Job("ken_burns_video", output_dir / "jobs.json")
    try:
        job.start()
        video_clips = []

        for idx, (img_path, para) in enumerate(zip(images_paths, paragraphs)):
            clip_path = output_dir / f"scene_{idx:03d}.mp4"
            duration = para["duration"]
            zoompan_filter = f"zoompan=z='min(zoom+{KEN_BURNS_ZOOM_SPEED},1.08)':d={int(duration*FPS)}"
            cmd = [
                "ffmpeg",
                "-y",
                "-loop", "1",
                "-i", str(img_path),
                "-vf", zoompan_filter,
                "-t", str(duration),
                "-r", str(FPS),
                "-pix_fmt", "yuv420p",
                str(clip_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed for scene {idx}: {result.stderr}")
            video_clips.append(clip_path)

        concat_file = output_dir / "scenes.txt"
        with open(concat_file, "w") as f:
            for clip in video_clips:
                f.write(f"file '{clip}'\n")

        final_video = output_dir / "visuals.mp4"
        result = subprocess.run([
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(final_video)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        job.complete()
        return final_video
    except Exception as e:
        job.fail(str(e))
        raise


# ----------------------------
# STEP 5: Mux Audio into Video
# ----------------------------
def mux_audio(video_path: Path, audio_path: Path, output_dir: Path, background_music: Path = None) -> Path:
    """Combine the visual video with narration audio (and optional background music)."""
    job = Job("audio_mux", output_dir / "jobs.json")
    try:
        job.start()
        
        final_output = output_dir / "final_video.mp4"
        
        if background_music and background_music.exists():
            # Mix narration with background music (narration louder)
            print(f"Muxing video with narration and background music...")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-i", str(background_music),
                "-filter_complex",
                "[1:a]volume=1.0[narration];[2:a]volume=0.3[music];[narration][music]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(final_output)
            ]
        else:
            # Just mux narration audio
            print(f"Muxing video with narration audio...")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(final_output)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg audio mux failed: {result.stderr}")
        
        job.complete()
        return final_output
    except Exception as e:
        job.fail(str(e))
        raise


# ----------------------------
# CLI ARGUMENT PARSING
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Story Reader Pipeline: Convert audio narration to visual video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i narration.wav -o output/
  python main.py --input my_story.mp3 --output results/
  python main.py --clear-cache  # Clear all cached data
        """
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path(DEFAULT_AUDIO_FILE),
        help=f"Path to input audio file (default: {DEFAULT_AUDIO_FILE})"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached transcriptions and images before running"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (regenerate everything)"
    )
    parser.add_argument(
        "-m", "--music",
        type=Path,
        default=None,
        help="Optional background music file to mix with narration"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio muxing (output visuals only)"
    )
    return parser.parse_args()


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    args = parse_args()
    
    audio_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not validate_inputs(audio_path):
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Initialize cache manager
    cache_manager = CacheManager(output_dir)
    
    # Clear cache if requested
    if args.clear_cache:
        import shutil
        cache_dir = output_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("Cache cleared.")
        cache_manager = CacheManager(output_dir)  # Reinitialize
    
    # Disable caching if requested (use a dummy cache manager)
    if args.no_cache:
        class NoCacheManager:
            def get_cached_transcription(self, *args): return None
            def save_transcription(self, *args): pass
            def get_cached_image(self, *args, **kwargs): return None
            def save_image_cache(self, *args): pass
        cache_manager = NoCacheManager()
    
    print(f"Input audio: {audio_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Run pipeline
    segments = transcribe_audio(audio_path, output_dir, cache_manager)
    paragraphs = segments_to_paragraphs(segments, output_dir)
    images = generate_images(paragraphs, output_dir, cache_manager)
    visuals_video = create_ken_burns_video(images, paragraphs, output_dir)
    
    # Mux audio into final video
    if not args.no_audio:
        final_video = mux_audio(visuals_video, audio_path, output_dir, args.music)
        print("-" * 50)
        print(f"Pipeline complete! Final video with audio: {final_video}")
    else:
        final_video = visuals_video
        print("-" * 50)
        print(f"Pipeline complete! Visual video (no audio): {final_video}")
        print("Use ffmpeg to manually mux audio if needed.")


if __name__ == "__main__":
    main()

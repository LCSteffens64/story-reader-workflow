"""
Video composition step using FFmpeg with Ken Burns effect.
"""

import random
import subprocess
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Tuple

from PIL import Image

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class VideoComposerStep(PipelineStep[Tuple[List[Path], List[Dict[str, Any]]], Path]):
    """
    Creates video with Ken Burns (zoom/pan) effect from images.
    
    Input: Tuple of (image paths, paragraphs with timing)
    Output: Path to the composed video file (without audio)
    """
    
    name = "ken_burns_video"
    description = "Create Ken Burns effect video from images"
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
    
    def run(self, input_data: Tuple[List[Path], List[Dict[str, Any]]]) -> Path:
        """
        Create video with Ken Burns effect from images.
        
        Args:
            input_data: Tuple containing:
                - List of image paths
                - List of paragraph dictionaries with duration info
            
        Returns:
            Path to the final composed video (visuals.mp4)
        """
        image_paths, paragraphs = input_data
        video_clips = []
        
        # Create individual clips with Ken Burns effect
        last_idx = len(image_paths) - 1
        target_size = None
        for idx, (img_path, para) in enumerate(zip(image_paths, paragraphs)):
            img_path = self._ensure_tile_image(img_path, idx)
            img_path = self._prefer_upscaled(img_path)
            if target_size is None:
                target_size = self._get_target_frame_size(img_path)
            duration = para["duration"]
            if idx == last_idx and self.config.video_padding_sec > 0:
                duration += self.config.video_padding_sec
            print(f"Creating Ken Burns scene {idx+1}/{len(image_paths)} from {img_path.name}...")
            clip_path = self._create_ken_burns_clip(img_path, duration, idx, target_size)
            video_clips.append(clip_path)
        
        # Concatenate all clips
        final_video = self._concatenate_clips(video_clips)
        
        print(f"Created video with {len(video_clips)} scenes: {final_video}")
        return final_video
    
    def _create_ken_burns_clip(
        self,
        image_path: Path,
        duration: float,
        idx: int,
        target_size: tuple,
    ) -> Path:
        """
        Create a single Ken Burns clip from an image.
        
        Args:
            image_path: Path to the source image
            duration: Duration of the clip in seconds
            idx: Index for output filename
            
        Returns:
            Path to the created clip
        """
        clip_path = self.config.scenes_dir / f"scene_{idx:03d}.mp4"
        if self.config.keep_scenes and clip_path.exists():
            print(f"Reusing existing scene {idx+1}: {clip_path.name}")
            return clip_path
        
        # Build zoompan filter
        max_zoom = self.config.ken_burns_max_zoom
        num_frames = max(int(duration * self.config.fps), 1)

        # Randomize a center-biased focus point for smoother pan paths.
        rng = random.Random(idx)
        focus_margin = 0.15
        x_focus = rng.uniform(0.5 - focus_margin, 0.5 + focus_margin)
        y_focus = rng.uniform(0.5 - focus_margin, 0.5 + focus_margin)

        target_w, target_h = target_size
        motion_w = target_w * 2
        motion_h = target_h * 2
        # Fill the full frame before zooming to avoid pan snap from padded bars.
        scale_fill = (
            f"scale={motion_w}:{motion_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={motion_w}:{motion_h}"
        )
        progress = f"on/{max(num_frames - 1, 1)}"
        zoompan_filter = (
            f"{scale_fill},"
            "zoompan="
            f"z='1+({max_zoom}-1)*(0.5-0.5*cos(PI*{progress}))':"
            f"x='clip((iw-iw/zoom)*{x_focus},0,iw-iw/zoom)':"
            f"y='clip((ih-ih/zoom)*{y_focus},0,ih-ih/zoom)':"
            f"d={num_frames}:s={motion_w}x{motion_h}:fps={self.config.fps},"
            f"scale={target_w}:{target_h}:flags=lanczos,"
            "vignette=PI/5"
        )
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-loop", "1",
            "-i", str(image_path),
            "-vf", zoompan_filter,
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            str(clip_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed for scene {idx}: {result.stderr}")
        
        return clip_path

    def _get_target_frame_size(self, image_path: Path) -> tuple:
        """
        Compute a consistent target frame size using a 16:9 canvas,
        padding left/right (pillarbox) as needed.
        """
        img = Image.open(image_path)
        width, height = img.size

        target_h = height
        target_w = int(round(target_h * 16 / 9))

        if width > target_w:
            target_w = width

        # Ensure even dimensions for H.264 compatibility
        if target_w % 2 != 0:
            target_w += 1
        if target_h % 2 != 0:
            target_h += 1

        return (target_w, target_h)

    def _ensure_tile_image(self, image_path: Path, idx: int) -> Path:
        """
        If image is a stitched 2x2 grid, crop a random tile and save as a normal frame.
        """
        name = image_path.name
        if not (name.endswith("-stitched.png") or name.endswith("_stitched.png")):
            return image_path

        stitched_bytes = image_path.read_bytes()
        try:
            img = Image.open(BytesIO(stitched_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open stitched image for scene {idx}: {e}") from e

        width, height = img.size
        if width < 2 or height < 2:
            raise RuntimeError(f"Stitched image too small to crop: {img.size}")

        tile_w = width // 2
        tile_h = height // 2
        tiles = [
            (0, 0, tile_w, tile_h),
            (tile_w, 0, tile_w * 2, tile_h),
            (0, tile_h, tile_w, tile_h * 2),
            (tile_w, tile_h, tile_w * 2, tile_h * 2),
        ]
        box = self._pick_nonempty_tile(img, tiles)
        tile = img.crop(box)

        out_path = self.config.images_dir / f"{idx:03d}.png"
        tile.save(out_path, format="PNG")
        return out_path

    def _pick_nonempty_tile(self, img: Image.Image, tiles: List[tuple]) -> tuple:
        """Prefer tiles that are not mostly empty/black."""
        best_box = None
        best_score = -1.0

        for box in tiles:
            tile = img.crop(box)
            pixels = tile.getdata()
            if not pixels:
                continue
            nonblack = 0
            for r, g, b in pixels:
                if r > 8 or g > 8 or b > 8:
                    nonblack += 1
            score = nonblack / len(pixels)
            if score > best_score:
                best_score = score
                best_box = box

        if best_box is None:
            return random.choice(tiles)

        if best_score < 0.05:
            return random.choice(tiles)

        top_tiles = []
        for box in tiles:
            tile = img.crop(box)
            pixels = tile.getdata()
            if not pixels:
                continue
            nonblack = 0
            for r, g, b in pixels:
                if r > 8 or g > 8 or b > 8:
                    nonblack += 1
            score = nonblack / len(pixels)
            if score >= best_score * 0.9:
                top_tiles.append(box)

        return random.choice(top_tiles) if top_tiles else best_box

    def _prefer_upscaled(self, image_path: Path) -> Path:
        """Prefer an existing *_upscaled image if available."""
        upscaled_path = image_path.parent / f"{image_path.stem}_upscaled{image_path.suffix}"
        if upscaled_path.exists():
            return upscaled_path
        return image_path
    
    def _concatenate_clips(self, video_clips: List[Path]) -> Path:
        """
        Concatenate video clips into a single video.
        
        Args:
            video_clips: List of paths to video clips
            
        Returns:
            Path to the concatenated video
        """
        # Create concat file list
        concat_file = self.config.scenes_dir / "scenes.txt"
        with open(concat_file, "w") as f:
            for clip in video_clips:
                f.write(f"file '{clip}'\n")
        
        # Concatenate using FFmpeg
        final_video = self.config.output_dir / "visuals.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(final_video)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")
        
        return final_video
    
    def cleanup_temp_clips(self) -> None:
        """Remove temporary scene clips to save disk space."""
        for clip in self.config.scenes_dir.glob("scene_*.mp4"):
            clip.unlink()
        
        concat_file = self.config.scenes_dir / "scenes.txt"
        if concat_file.exists():
            concat_file.unlink()

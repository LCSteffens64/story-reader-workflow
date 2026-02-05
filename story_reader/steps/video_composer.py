"""
Video composition step using FFmpeg with Ken Burns effect.
"""

import random
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
        for idx, (img_path, para) in enumerate(zip(image_paths, paragraphs)):
            duration = para["duration"]
            if idx == last_idx and self.config.video_padding_sec > 0:
                duration += self.config.video_padding_sec
            clip_path = self._create_ken_burns_clip(img_path, duration, idx)
            video_clips.append(clip_path)
        
        # Concatenate all clips
        final_video = self._concatenate_clips(video_clips)
        
        print(f"Created video with {len(video_clips)} scenes: {final_video}")
        return final_video
    
    def _create_ken_burns_clip(self, image_path: Path, duration: float, idx: int) -> Path:
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
        
        # Build zoompan filter
        max_zoom = self.config.ken_burns_max_zoom
        num_frames = max(int(duration * self.config.fps), 1)

        # Compute speed so we reach max_zoom at the end of the clip
        zoom_speed = (max_zoom - 1.0) / max(num_frames - 1, 1)

        # Randomize the fixed point for the zoom to avoid awkward crops
        rng = random.Random(idx)
        x_focus = rng.uniform(0.3, 0.7)
        y_focus = rng.uniform(0.3, 0.7)

        zoompan_filter = (
            "zoompan="
            f"z='min(zoom+{zoom_speed},{max_zoom})':"
            f"x='iw*{x_focus}-(iw/zoom)/2':"
            f"y='ih*{y_focus}-(ih/zoom)/2':"
            f"d={num_frames},"
            "vignette=PI/5"
        )
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-loop", "1",
            "-i", str(image_path),
            "-vf", zoompan_filter,
            "-t", str(duration),
            "-r", str(self.config.fps),
            "-pix_fmt", "yuv420p",
            str(clip_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed for scene {idx}: {result.stderr}")
        
        return clip_path
    
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

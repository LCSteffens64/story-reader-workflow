"""
Audio mixing step - combines video with narration and optional background music.
"""

import subprocess
from pathlib import Path
from typing import Optional

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class AudioMixerStep(PipelineStep[Path, Path]):
    """
    Mixes audio tracks into the video.
    
    Input: Path to the visual video (without audio)
    Output: Path to the final video with audio
    
    Combines:
    - Narration audio (from original input)
    - Optional background music (at configurable volume)
    """
    
    name = "audio_mux"
    description = "Mix narration and background music into video"
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
    
    def run(self, video_path: Path) -> Path:
        """
        Mix audio into the video.
        
        Args:
            video_path: Path to the video file (visuals.mp4)
            
        Returns:
            Path to the final video with audio (final_video.mp4)
        """
        final_output = self.config.output_dir / "final_video.mp4"
        audio_path = self.config.input_audio
        music_path = self.config.background_music
        
        if music_path and music_path.exists():
            self._mux_with_music(video_path, audio_path, music_path, final_output, self.config.music_volume)
        else:
            music_path = self._select_music_from_dir()
            if music_path:
                self._mux_with_music(
                    video_path,
                    audio_path,
                    music_path,
                    final_output,
                    self.config.music_dir_volume,
                )
            else:
                self._mux_narration_only(video_path, audio_path, final_output)
        
        print(f"Final video created: {final_output}")
        return final_output
    
    def _mux_narration_only(self, video_path: Path, audio_path: Path, output_path: Path) -> None:
        """Mux just the narration audio into the video."""
        print(f"Muxing video with narration audio...")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", self.config.audio_bitrate,
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg audio mux failed: {result.stderr}")
    
    def _mux_with_music(
        self, 
        video_path: Path, 
        audio_path: Path, 
        music_path: Path, 
        output_path: Path,
        music_volume: float
    ) -> None:
        """Mux narration and background music into the video."""
        print(f"Muxing video with narration and background music...")
        
        narration_vol = self.config.narration_volume
        music_vol = music_volume
        
        # FFmpeg filter complex to mix audio tracks
        filter_complex = (
            f"[1:a]volume={narration_vol}[narration];"
            f"[2:a]volume={music_vol}[music];"
            f"[narration][music]amix=inputs=2:duration=first[aout]"
        )
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-i", str(music_path),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", self.config.audio_bitrate,
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg audio mux with music failed: {result.stderr}")

    def _select_music_from_dir(self) -> Optional[Path]:
        """Pick a music track from the music directory (sorted)."""
        music_dir = self.config.music_dir
        if not music_dir.exists() or not music_dir.is_dir():
            print(f"Music dir not found: {music_dir}")
            return None
        tracks = sorted(music_dir.rglob("*.mp3"))
        if not tracks:
            print(f"No .mp3 files found under: {music_dir}")
            return None
        print(f"Using background music from {tracks[0].name}")
        return tracks[0]

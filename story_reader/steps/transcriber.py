"""
Audio transcription step using OpenAI Whisper.
"""

from pathlib import Path
from typing import List, Dict, Any
import re

import whisper

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class TranscriberStep(PipelineStep[Path, List[Dict[str, Any]]]):
    """
    Transcribes audio to text using OpenAI Whisper.
    
    Input: Path to audio file
    Output: List of segment dictionaries with text and timestamps
    """
    
    name = "transcription"
    description = "Transcribe audio to text using Whisper"
    
    # Class-level model cache to avoid reloading
    _model_cache: Dict[str, Any] = {}
    _curse_words = {
        "fuck", "fucking", "fucked", "fucker", "motherfucker",
        "shit", "shitty", "bullshit",
        "bitch", "bitches",
        "asshole",
        "bastard",
        "cunt",
        "dick", "dicks",
        "piss",
        "prick",
        "slut",
        "whore",
        "damn",
        "goddamn",
        "hell",
    }
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
    
    def run(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Transcribe the audio file.
        
        Args:
            audio_path: Path to the audio file to transcribe
            
        Returns:
            List of segment dictionaries containing:
            - id: Segment index
            - start: Start time in seconds
            - end: End time in seconds
            - text: Transcribed text
        """
        # Check cache first
        cached = self.cache.get_cached_transcription(audio_path)
        if cached is not None:
            return cached
        
        # Load model (with class-level caching)
        model = self._get_model()
        
        print(f"Transcribing audio: {audio_path.name}")
        result = model.transcribe(str(audio_path))
        segments = result["segments"]
        segments = self._censor_segments(segments)
        
        # Save to cache
        self.cache.save_transcription(audio_path, segments)
        
        print(f"Transcribed {len(segments)} segments")
        return segments

    def _censor_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Censor curse words in transcription segments."""
        if not segments:
            return segments
        pattern = re.compile(
            r"\\b(" + "|".join(re.escape(w) for w in sorted(self._curse_words, key=len, reverse=True)) + r")\\b",
            re.IGNORECASE,
        )
        for seg in segments:
            text = seg.get("text", "")
            if text:
                seg["text"] = pattern.sub(lambda m: "*" * len(m.group(0)), text)
        return segments
    
    def _get_model(self):
        """Get or load the Whisper model with caching."""
        model_name = self.config.whisper_model
        
        if model_name not in self._model_cache:
            print(f"Loading Whisper model '{model_name}'...")
            self._model_cache[model_name] = whisper.load_model(model_name)
        else:
            print(f"Using cached Whisper model '{model_name}'")
        
        return self._model_cache[model_name]
    
    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear the cached Whisper models to free memory."""
        cls._model_cache.clear()

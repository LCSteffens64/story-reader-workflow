"""
Cache management for transcriptions and generated images.
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for cache invalidation."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash of text content."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


class CacheManager:
    """
    Manages caching for transcription and image generation results.
    
    Caches are stored in a .cache directory within the output directory,
    with a JSON index tracking all cached items.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the cache manager.
        
        Args:
            output_dir: Base output directory (cache will be in output_dir/.cache)
        """
        self.output_dir = output_dir
        self.cache_dir = output_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> dict:
        """Load the cache index from disk."""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        with open(self.cache_index_file, "w") as f:
            json.dump(self.cache_index, f, indent=2)
    
    # --- Transcription caching ---
    
    def get_transcription_cache_key(self, audio_path: Path) -> str:
        """Generate cache key for a transcription based on audio file hash."""
        file_hash = compute_file_hash(audio_path)
        return f"transcription_{file_hash}"
    
    def get_cached_transcription(self, audio_path: Path) -> Optional[List[dict]]:
        """
        Retrieve cached transcription if available.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of segment dictionaries if cached, None otherwise
        """
        cache_key = self.get_transcription_cache_key(audio_path)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                print(f"Using cached transcription for '{audio_path.name}'")
                with open(cache_file, "r") as f:
                    return json.load(f)
        return None
    
    def save_transcription(self, audio_path: Path, segments: List[dict]) -> None:
        """
        Save transcription to cache.
        
        Args:
            audio_path: Path to the source audio file
            segments: List of transcribed segments
        """
        cache_key = self.get_transcription_cache_key(audio_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, "w") as f:
            json.dump(segments, f, indent=2)
        
        self.cache_index[cache_key] = {
            "type": "transcription",
            "audio_file": str(audio_path),
            "audio_hash": compute_file_hash(audio_path),
            "segment_count": len(segments),
            "created": datetime.now().isoformat(),
        }
        self._save_cache_index()
    
    # --- Image caching ---
    
    def get_image_cache_key(self, paragraph_text: str, idx: int) -> str:
        """Generate cache key for an image based on paragraph text hash."""
        text_hash = compute_text_hash(paragraph_text)
        return f"image_{idx:03d}_{text_hash}"
    
    def get_cached_image(self, paragraph_text: str, idx: int, output_dir: Path) -> Optional[Path]:
        """
        Retrieve cached image if available.
        
        Args:
            paragraph_text: The paragraph text used to generate the image
            idx: Index of the paragraph/image
            output_dir: Directory where the image should be placed
            
        Returns:
            Path to the image if cached, None otherwise
        """
        expected_path = output_dir / f"{idx:03d}.png"
        if expected_path.exists():
            print(f"Using existing image for paragraph {idx}")
            return expected_path
        stitched_path = output_dir / f"{idx:03d}-stitched.png"
        if stitched_path.exists():
            print(f"Using existing stitched image for paragraph {idx}")
            return stitched_path

        cache_key = self.get_image_cache_key(paragraph_text, idx)
        if cache_key in self.cache_index:
            cached_path = Path(self.cache_index[cache_key].get("path", ""))
            if cached_path.exists():
                if cached_path != expected_path:
                    shutil.copy(cached_path, expected_path)
                print(f"Using cached image for paragraph {idx}")
                return expected_path
        return None
    
    def get_image_cache_entry(self, paragraph_text: str, idx: int) -> Optional[dict]:
        """
        Retrieve the raw cache entry for an image if available.

        Args:
            paragraph_text: The paragraph text used to generate the image
            idx: Index of the paragraph/image

        Returns:
            Cache entry dict if present, else None
        """
        cache_key = self.get_image_cache_key(paragraph_text, idx)
        return self.cache_index.get(cache_key)

    def save_image_cache(
        self,
        paragraph_text: str,
        idx: int,
        image_path: Path,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Save image path to cache index.
        
        Args:
            paragraph_text: The paragraph text used to generate the image
            idx: Index of the paragraph/image
            image_path: Path where the image was saved
        """
        cache_key = self.get_image_cache_key(paragraph_text, idx)
        entry = {
            "type": "image",
            "path": str(image_path),
            "text_preview": paragraph_text[:100],
            "text_hash": compute_text_hash(paragraph_text),
            "created": datetime.now().isoformat(),
        }
        if metadata:
            entry.update(metadata)
        self.cache_index[cache_key] = entry
        self._save_cache_index()
    
    
    # --- Cache management ---
    
    def clear(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = {}
        self._save_cache_index()
        print("Cache cleared.")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        transcriptions = sum(1 for v in self.cache_index.values() if v.get("type") == "transcription")
        images = sum(1 for v in self.cache_index.values() if v.get("type") == "image")
        
        # Calculate cache size
        total_size = 0
        for item in self.cache_dir.iterdir():
            if item.is_file():
                total_size += item.stat().st_size
        
        return {
            "transcriptions": transcriptions,
            "images": images,
            "total_items": len(self.cache_index),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"CacheManager(items={stats['total_items']}, size={stats['size_mb']}MB)"


class NullCacheManager:
    """
    A cache manager that doesn't cache anything.
    
    Used when caching is disabled via --no-cache flag.
    """
    
    def get_cached_transcription(self, audio_path: Path) -> None:
        return None
    
    def save_transcription(self, audio_path: Path, segments: List[dict]) -> None:
        pass
    
    def get_cached_image(self, paragraph_text: str, idx: int, output_dir: Path) -> None:
        return None
    
    def save_image_cache(self, paragraph_text: str, idx: int, image_path: Path) -> None:
        pass
    
    def clear(self) -> None:
        pass
    
    def get_stats(self) -> dict:
        return {"transcriptions": 0, "images": 0, "total_items": 0, "size_bytes": 0, "size_mb": 0}
    
    def __repr__(self) -> str:
        return "NullCacheManager(disabled)"

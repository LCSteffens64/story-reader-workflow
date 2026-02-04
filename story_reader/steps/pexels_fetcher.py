"""
Pexels image fetching step for the Story Reader pipeline.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from PIL import Image

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager
from ..utils.llm_nlp import LLMKeywordExtractor


class PexelsImageFetcherStep(PipelineStep[List[Dict[str, Any]], List[Path]]):
    """
    Fetches images from Pexels API for each paragraph.
    
    Input: List of paragraph dictionaries
    Output: List of paths to downloaded images (or None for fallback)
    """
    
    name = "pexels_fetcher"
    description = "Fetch images from Pexels API"
    
    # Pexels API endpoints
    SEARCH_URL = "https://api.pexels.com/v1/search"
    PHOTO_URL = "https://api.pexels.com/v1/photos/{id}"
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
        self.api_key = self._get_api_key()
        
        if not self.api_key:
            raise ValueError(
                "Pexels API key required. Set via --pexels-api-key or PEXELS_API_KEY environment variable"
            )
        
        self.headers = {
            "Authorization": self.api_key,
            "User-Agent": "StoryReader/1.0"
        }
    
    def _get_api_key(self) -> Optional[str]:
        """Get Pexels API key from config or environment."""
        if self.config.pexels_api_key:
            return self.config.pexels_api_key
        return os.environ.get("PEXELS_API_KEY")
    
    def run(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        """
        Fetch images from Pexels for each paragraph.
        
        Args:
            paragraphs: List of paragraph dictionaries with text
            
        Returns:
            List of paths to downloaded images, or None for paragraphs where Pexels failed
        """
        image_paths = []
        
        for idx, para in enumerate(paragraphs):
            # Check cache first
            cached_path = self.cache.get_cached_pexels_image(
                para['text'], idx, self.config.output_dir
            )
            if cached_path is not None:
                image_paths.append(cached_path)
                continue
            
            # Fetch from Pexels
            print(f"Fetching Pexels image {idx+1}/{len(paragraphs)}...")
            image_path = self._fetch_image_for_paragraph(para, idx)
            
            if image_path:
                # Cache the result
                self.cache.save_pexels_cache(para['text'], idx, image_path)
                image_paths.append(image_path)
            else:
                # Pexels failed, return None for fallback
                image_paths.append(None)
        
        success_count = len([p for p in image_paths if p is not None])
        print(f"Pexels fetch complete: {success_count}/{len(paragraphs)} images successful")
        return image_paths
    
    def _fetch_image_for_paragraph(self, paragraph: Dict[str, Any], idx: int) -> Optional[Path]:
        """Fetch a single image for a paragraph from Pexels."""
        query = self._prepare_search_query(paragraph['text'])
        
        try:
            # Search for photos
            photos = self._search_photos(query)
            if not photos:
                print(f"  No photos found for query: '{query}'")
                return None
            
            # Select best photo
            selected_photo = self._select_best_photo(photos, paragraph['text'])
            if not selected_photo:
                return None
            
            # Download the photo
            image_path = self.config.output_dir / f"pexels_{idx:03d}.jpg"
            if self._download_photo(selected_photo, image_path):
                # Process image to match requirements
                processed_path = self._process_image(image_path, idx)
                return processed_path
            
        except Exception as e:
            print(f"  Error fetching Pexels image: {e}")
        
        return None
    
    def _prepare_search_query(self, text: str) -> str:
        """Prepare search query from paragraph text using LLM keyword extraction."""
        # Generate Stable Diffusion prompt for better keyword extraction
        sd_prompt = self.config.get_image_prompt(text)
        
        # Use LLM for keyword extraction if enabled
        if self.config.llm_keyword_extractor:
            extractor = LLMKeywordExtractor(
                model_name=self.config.llm_model_name,
                device=self.config.device,
                quantize=self.config.llm_quantization
            )
            keywords = extractor.extract_keywords(text, sd_prompt, max_keywords=3)
        else:
            # Fallback to simple keyword extraction
            import re
            from collections import Counter
            
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            common_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'this', 'that', 'these', 'those', 'have', 'has', 'had', 'was', 'were', 'be',
                'beautiful', 'amazing', 'stunning', 'gorgeous', 'nice', 'good', 'bad', 'old',
                'new', 'big', 'small', 'large', 'little', 'many', 'much', 'some', 'any'
            }
            filtered_words = [word for word in words if word not in common_words]
            word_counts = Counter(filtered_words)
            keywords = [word for word, count in word_counts.most_common(3)]
        
        # Create search query from keywords
        search_query = " ".join(keywords)
        
        print(f"  Generated search query: '{search_query}'")
        return search_query
    
    def _search_photos(self, query: str) -> List[Dict[str, Any]]:
        """Search for photos on Pexels."""
        params = {
            'query': query,
            'per_page': self.config.pexels_max_results,
            'orientation': 'landscape',
            'size': 'large',  # medium, large, original
        }
        
        response = requests.get(
            self.SEARCH_URL,
            headers=self.headers,
            params=params,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  Pexels API error: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        return data.get('photos', [])
    
    def _select_best_photo(self, photos: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        """Select the best photo from search results."""
        if not photos:
            return None
        
        # Filter by size requirements
        filtered_photos = []
        for photo in photos:
            width = photo.get('width', 0)
            height = photo.get('height', 0)
            
            if width >= self.config.pexels_min_width and height >= self.config.pexels_min_height:
                filtered_photos.append(photo)
        
        if not filtered_photos:
            # If no photos meet size requirements, use original list
            filtered_photos = photos
        
        # Select the first (most relevant) photo
        return filtered_photos[0]
    
    def _download_photo(self, photo: Dict[str, Any], save_path: Path) -> bool:
        """Download a photo from Pexels."""
        # Use the original size for best quality
        download_url = photo.get('src', {}).get('original') or photo.get('src', {}).get('large')
        
        if not download_url:
            return False
        
        try:
            response = requests.get(download_url, headers=self.headers, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"  Download error: {e}")
            return False
    
    def _process_image(self, image_path: Path, idx: int) -> Path:
        """Process downloaded image to match pipeline requirements."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to match pipeline requirements
                target_size = self.config.image_size
                img = img.resize(target_size, Image.LANCZOS)
                
                # Save processed image
                processed_path = self.config.output_dir / f"pexels_processed_{idx:03d}.png"
                img.save(processed_path, 'PNG', optimize=True)
                
                # Remove original downloaded file
                image_path.unlink(missing_ok=True)
                
                return processed_path
                
        except Exception as e:
            print(f"  Image processing error: {e}")
            return None


class PexelsCacheManager:
    """Enhanced cache manager with Pexels-specific methods."""
    
    @staticmethod
    def get_cached_pexels_image(text: str, idx: int, output_dir: Path) -> Optional[Path]:
        """Get cached Pexels image path."""
        cache_dir = output_dir / ".cache"
        if not cache_dir.exists():
            return None
        
        # Create cache key from text and index
        cache_key = hashlib.md5(f"{text}_{idx}".encode()).hexdigest()
        cache_file = cache_dir / f"pexels_cache_{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            image_path = Path(cache_data.get('image_path', ''))
            if image_path.exists():
                return image_path
        except:
            pass
        
        return None
    
    @staticmethod
    def save_pexels_cache(text: str, idx: int, image_path: Path) -> None:
        """Save Pexels image to cache."""
        cache_dir = image_path.parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        
        cache_key = hashlib.md5(f"{text}_{idx}".encode()).hexdigest()
        cache_file = cache_dir / f"pexels_cache_{cache_key}.json"
        
        cache_data = {
            'text': text,
            'idx': idx,
            'image_path': str(image_path),
            'timestamp': time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
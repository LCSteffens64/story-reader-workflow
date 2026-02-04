"""
Image generation step using Stable Diffusion.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from diffusers import StableDiffusionPipeline

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class ImageGeneratorStep(PipelineStep[List[Dict[str, Any]], List[Path]]):
    """
    Generates images for each paragraph using Stable Diffusion.
    
    Input: List of paragraph dictionaries
    Output: List of paths to generated images
    """
    
    name = "image_generation"
    description = "Generate images using Stable Diffusion"
    
    # Class-level pipeline cache to avoid reloading
    _pipeline_cache: Dict[str, Any] = {}
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
        self._pipe: Optional[StableDiffusionPipeline] = None
    
    def run(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        """
        Generate images for each paragraph.
        
        Args:
            paragraphs: List of paragraph dictionaries with text
            
        Returns:
            List of paths to generated PNG images
        """
        image_paths = []
        
        for idx, para in enumerate(paragraphs):
            # Check cache first
            cached_path = self.cache.get_cached_image(
                para['text'], idx, self.config.output_dir
            )
            if cached_path is not None:
                image_paths.append(cached_path)
                continue
            
            # Load pipeline only when needed (lazy loading)
            if self._pipe is None:
                self._pipe = self._get_pipeline()
            
            # Generate image
            prompt = self.config.get_image_prompt(para['text'])
            print(f"Generating image {idx+1}/{len(paragraphs)}...")
            
            image = self._pipe(
                prompt,
                negative_prompt=self.config.negative_prompt,
                height=self.config.image_size[1],
                width=self.config.image_size[0],
            ).images[0]
            
            # Save image
            img_path = self.config.output_dir / f"{idx:03d}.png"
            image.save(img_path)
            image_paths.append(img_path)
            
            # Cache the result
            self.cache.save_image_cache(para['text'], idx, img_path)
        
        print(f"Generated {len(image_paths)} images")
        return image_paths
    
    def _get_pipeline(self) -> StableDiffusionPipeline:
        """Get or load the Stable Diffusion pipeline with caching."""
        cache_key = f"{self.config.sd_model}_{self.config.device}"
        
        if cache_key not in self._pipeline_cache:
            print(f"Loading Stable Diffusion model '{self.config.sd_model}'...")
            
            # Determine dtype based on device
            dtype = torch.float16 if self.config.device == "cuda" else torch.float32
            
            # Load pipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd_model,
                torch_dtype=dtype,
                use_auth_token=self.config.hf_token,
            )
            pipe = pipe.to(self.config.device)
            pipe.enable_attention_slicing()  # Memory optimization
            
            self._pipeline_cache[cache_key] = pipe
        else:
            print(f"Using cached Stable Diffusion pipeline")
        
        return self._pipeline_cache[cache_key]
    
    @classmethod
    def clear_pipeline_cache(cls) -> None:
        """Clear the cached pipelines to free VRAM."""
        cls._pipeline_cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

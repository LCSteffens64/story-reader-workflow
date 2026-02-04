"""
Hybrid image generation step that can use Pexels, Stable Diffusion, or both.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .base import PipelineStep
from .image_generator import ImageGeneratorStep
from .pexels_fetcher import PexelsImageFetcherStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class HybridImageGeneratorStep(PipelineStep[List[Dict[str, Any]], List[Path]]):
    """
    Generates images using Pexels, Stable Diffusion, or a combination.
    
    Input: List of paragraph dictionaries
    Output: List of paths to generated images
    """
    
    name = "hybrid_image_generation"
    description = "Generate images using Pexels and/or Stable Diffusion"
    
    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)
        
        # Initialize both image generators
        self.pexels_step = None
        self.sd_step = None
        
        if config.use_pexels:
            self.pexels_step = PexelsImageFetcherStep(config, cache)
        
        # Always initialize SD step for fallback or pure SD mode
        self.sd_step = ImageGeneratorStep(config, cache)
    
    def run(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        """
        Generate images using the configured strategy.
        
        Args:
            paragraphs: List of paragraph dictionaries with text
            
        Returns:
            List of paths to generated images
        """
        image_paths = []
        
        if self.config.use_pexels:
            # Use Pexels with optional SD fallback
            image_paths = self._run_pexels_with_fallback(paragraphs)
        else:
            # Use pure Stable Diffusion
            image_paths = self._run_stable_diffusion(paragraphs)
        
        return image_paths
    
    def _run_pexels_with_fallback(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        """Run Pexels fetching with Stable Diffusion fallback."""
        print("Using Pexels with Stable Diffusion fallback")
        
        # Fetch from Pexels
        pexels_results = self.pexels_step.run(paragraphs)
        
        image_paths = []
        for idx, (para, pexels_result) in enumerate(zip(paragraphs, pexels_results)):
            if pexels_result is not None:
                # Pexels succeeded
                image_paths.append(pexels_result)
                print(f"  ✓ Pexels image {idx+1}: {pexels_result.name}")
            elif self.config.pexels_fallback:
                # Pexels failed, fall back to Stable Diffusion
                print(f"  ⚠ Pexels failed for paragraph {idx+1}, using Stable Diffusion")
                sd_result = self._generate_single_sd_image(para, idx)
                image_paths.append(sd_result)
            else:
                # No fallback available, raise error
                raise RuntimeError(f"Pexels failed for paragraph {idx+1} and no fallback configured")
        
        return image_paths
    
    def _run_stable_diffusion(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        """Run pure Stable Diffusion generation."""
        print("Using Stable Diffusion only")
        return self.sd_step.run(paragraphs)
    
    def _generate_single_sd_image(self, paragraph: Dict[str, Any], idx: int) -> Path:
        """Generate a single image using Stable Diffusion."""
        # Check cache first
        cached_path = self.cache.get_cached_image(
            paragraph['text'], idx, self.config.output_dir
        )
        if cached_path is not None:
            return cached_path
        
        # Generate image
        prompt = self.config.get_image_prompt(paragraph['text'])
        print(f"  Generating SD image {idx+1}...")
        
        if self.sd_step._pipe is None:
            self.sd_step._pipe = self.sd_step._get_pipeline()
        
        image = self.sd_step._pipe(
            prompt,
            negative_prompt=self.config.negative_prompt,
            height=self.config.image_size[1],
            width=self.config.image_size[0],
        ).images[0]
        
        # Save image
        img_path = self.config.output_dir / f"{idx:03d}.png"
        image.save(img_path)
        
        # Cache the result
        self.cache.save_image_cache(paragraph['text'], idx, img_path)
        
        return img_path
    
    @classmethod
    def clear_pipeline_cache(cls) -> None:
        """Clear all cached pipelines."""
        ImageGeneratorStep.clear_pipeline_cache()
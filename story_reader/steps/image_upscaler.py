"""
Image upscaling and post-processing step.

Supports multiple upscaling backends:
- Real-ESRGAN (high quality, requires GPU)
- PIL/Lanczos (fast, CPU-based)
- OpenCV (moderate quality, CPU-based)
"""

from pathlib import Path
from typing import List, Optional
from enum import Enum

from PIL import Image

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class UpscaleMethod(Enum):
    """Available upscaling methods."""
    PIL_LANCZOS = "pil_lanczos"
    PIL_BICUBIC = "pil_bicubic"
    OPENCV_CUBIC = "opencv_cubic"
    REAL_ESRGAN = "real_esrgan"


class ImageUpscalerStep(PipelineStep[List[Path], List[Path]]):
    """
    Upscales and enhances generated images.
    
    Input: List of image paths
    Output: List of paths to upscaled images
    """
    
    name = "image_upscaling"
    description = "Upscale images for higher resolution output"
    
    # Real-ESRGAN model cache
    _esrgan_model = None
    
    def __init__(
        self, 
        config: PipelineConfig, 
        cache: CacheManager,
        scale_factor: float = 2.0,
        method: UpscaleMethod = UpscaleMethod.PIL_LANCZOS,
        target_size: Optional[tuple] = None,
        enhance_sharpness: float = 1.0,
        enhance_contrast: float = 1.0,
    ):
        """
        Initialize the upscaler.
        
        Args:
            config: Pipeline configuration
            cache: Cache manager
            scale_factor: Upscale multiplier (e.g., 2.0 = 2x resolution)
            method: Upscaling method to use
            target_size: Exact output size (overrides scale_factor if set)
            enhance_sharpness: Sharpness enhancement (1.0 = no change)
            enhance_contrast: Contrast enhancement (1.0 = no change)
        """
        super().__init__(config, cache)
        self.scale_factor = scale_factor
        self.method = method
        self.target_size = target_size
        self.enhance_sharpness = enhance_sharpness
        self.enhance_contrast = enhance_contrast
    
    def run(self, image_paths: List[Path]) -> List[Path]:
        """
        Upscale all images.
        
        Args:
            image_paths: List of paths to images to upscale
            
        Returns:
            List of paths to upscaled images
        """
        upscaled_paths = []
        
        for idx, img_path in enumerate(image_paths):
            output_path = self._get_upscaled_path(img_path)
            
            print(f"Upscaling image {idx+1}/{len(image_paths)}...")
            
            if self.method == UpscaleMethod.REAL_ESRGAN:
                self._upscale_esrgan(img_path, output_path)
            elif self.method == UpscaleMethod.OPENCV_CUBIC:
                self._upscale_opencv(img_path, output_path)
            else:
                self._upscale_pil(img_path, output_path)
            
            # Apply enhancements
            if self.enhance_sharpness != 1.0 or self.enhance_contrast != 1.0:
                self._apply_enhancements(output_path)
            
            upscaled_paths.append(output_path)
        
        print(f"Upscaled {len(upscaled_paths)} images")
        return upscaled_paths
    
    def _get_upscaled_path(self, original_path: Path) -> Path:
        """Generate output path for upscaled image."""
        stem = original_path.stem
        suffix = original_path.suffix
        return original_path.parent / f"{stem}_upscaled{suffix}"
    
    def _upscale_pil(self, input_path: Path, output_path: Path) -> None:
        """Upscale using PIL (Pillow)."""
        img = Image.open(input_path)
        
        # Calculate new size
        if self.target_size:
            new_size = self.target_size
        else:
            new_size = (
                int(img.width * self.scale_factor),
                int(img.height * self.scale_factor)
            )
        
        # Choose resampling filter
        if self.method == UpscaleMethod.PIL_BICUBIC:
            resample = Image.BICUBIC
        else:
            resample = Image.LANCZOS
        
        # Upscale
        upscaled = img.resize(new_size, resample=resample)
        upscaled.save(output_path, quality=95)
    
    def _upscale_opencv(self, input_path: Path, output_path: Path) -> None:
        """Upscale using OpenCV."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("OpenCV not available, falling back to PIL")
            self._upscale_pil(input_path, output_path)
            return
        
        img = cv2.imread(str(input_path))
        
        # Calculate new size
        if self.target_size:
            new_size = self.target_size
        else:
            new_size = (
                int(img.shape[1] * self.scale_factor),
                int(img.shape[0] * self.scale_factor)
            )
        
        # Upscale with cubic interpolation
        upscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(output_path), upscaled)
    
    def _upscale_esrgan(self, input_path: Path, output_path: Path) -> None:
        """Upscale using Real-ESRGAN (best quality)."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            import cv2
            import numpy as np
        except ImportError:
            print("Real-ESRGAN not available, falling back to PIL Lanczos")
            self._upscale_pil(input_path, output_path)
            return
        
        # Load model (cached)
        if self._esrgan_model is None:
            print("Loading Real-ESRGAN model...")
            model = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23, 
                num_grow_ch=32, 
                scale=4
            )
            self._esrgan_model = RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,  # Use float32 for CPU
            )
        
        # Read and upscale
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        output, _ = self._esrgan_model.enhance(img, outscale=self.scale_factor)
        cv2.imwrite(str(output_path), output)
    
    def _apply_enhancements(self, image_path: Path) -> None:
        """Apply sharpness and contrast enhancements."""
        from PIL import ImageEnhance
        
        img = Image.open(image_path)
        
        # Apply sharpness
        if self.enhance_sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.enhance_sharpness)
        
        # Apply contrast
        if self.enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.enhance_contrast)
        
        img.save(image_path, quality=95)
    
    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear Real-ESRGAN model cache."""
        cls._esrgan_model = None


def upscale_images_simple(
    image_paths: List[Path], 
    scale: float = 2.0,
    sharpen: float = 1.2
) -> List[Path]:
    """
    Simple utility function to upscale images without pipeline context.
    
    Args:
        image_paths: List of image paths
        scale: Scale factor
        sharpen: Sharpness enhancement
        
    Returns:
        List of upscaled image paths
    """
    from PIL import Image, ImageEnhance
    
    output_paths = []
    for img_path in image_paths:
        img = Image.open(img_path)
        new_size = (int(img.width * scale), int(img.height * scale))
        upscaled = img.resize(new_size, resample=Image.LANCZOS)
        
        if sharpen != 1.0:
            enhancer = ImageEnhance.Sharpness(upscaled)
            upscaled = enhancer.enhance(sharpen)
        
        output_path = img_path.parent / f"{img_path.stem}_upscaled{img_path.suffix}"
        upscaled.save(output_path, quality=95)
        output_paths.append(output_path)
    
    return output_paths

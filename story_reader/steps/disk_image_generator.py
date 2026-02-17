"""
Disk-backed image selection step.
"""

from pathlib import Path
import random
import shutil
from typing import Any, Dict, List

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class DiskImageGeneratorStep(PipelineStep[List[Dict[str, Any]], List[Path]]):
    """
    Load images placed by the user from the output images directory.

    Input: List of paragraph dictionaries
    Output: List of image paths in paragraph order
    """

    name = "disk_image_selection"
    description = "Use images provided by the user on disk"

    _VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)

    def run(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        required_images = len(paragraphs)
        self.config.images_dir.mkdir(parents=True, exist_ok=True)

        candidates = self._collect_images(self.config.images_dir)

        if not candidates and required_images > 0:
            print(f"Image directory initialized: {self.config.images_dir}")
            print(f"Required images: {required_images}")
            print(f"Current images found: {len(candidates)}")
            print(
                "Place at least one image in this directory, then re-run with --use-disk."
            )
            raise RuntimeError("Not enough images found for --use-disk.")

        selected = self._select_images(candidates, required_images)
        if len(candidates) >= required_images:
            print(
                f"Found {len(candidates)} images; using a random order for {required_images} segments."
            )
        elif required_images > 0:
            print(
                f"Found {len(candidates)} images for {required_images} segments; reusing images in random order."
            )

        output_paths: List[Path] = []
        for idx, src in enumerate(selected):
            dest = self.config.images_dir / f"{idx:03d}{src.suffix.lower()}"
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            output_paths.append(dest)

        print(f"Using {len(output_paths)} disk images from {self.config.images_dir}")
        return output_paths

    def _select_images(self, candidates: List[Path], required_images: int) -> List[Path]:
        if required_images <= 0:
            return []
        if len(candidates) == 1:
            return [candidates[0]] * required_images

        target_counts = self._build_target_counts(candidates, required_images)
        return self._arrange_without_adjacent(target_counts, required_images)

    def _build_target_counts(self, candidates: List[Path], required_images: int) -> Dict[Path, int]:
        n = len(candidates)
        counts: Dict[Path, int] = {img: 0 for img in candidates}

        if required_images <= n:
            for img in random.sample(candidates, required_images):
                counts[img] = 1
            return counts

        if required_images <= 2 * n:
            for img in candidates:
                counts[img] = 1
            for img in random.sample(candidates, required_images - n):
                counts[img] += 1
            return counts

        for img in candidates:
            counts[img] = 2

        extra = required_images - (2 * n)
        while extra > 0:
            shuffled = candidates[:]
            random.shuffle(shuffled)
            for img in shuffled:
                if extra <= 0:
                    break
                counts[img] += 1
                extra -= 1
        return counts

    def _arrange_without_adjacent(self, counts: Dict[Path, int], total: int) -> List[Path]:
        selected: List[Path] = []
        remaining = counts.copy()

        while len(selected) < total:
            last = selected[-1] if selected else None
            choices = [
                img for img, cnt in remaining.items()
                if cnt > 0 and img != last
            ]
            if not choices:
                # Unavoidable (for example, only one image left with remaining count).
                choices = [img for img, cnt in remaining.items() if cnt > 0]

            weights = [remaining[img] for img in choices]
            picked = random.choices(choices, weights=weights, k=1)[0]
            selected.append(picked)
            remaining[picked] -= 1

        return selected

    def _collect_images(self, directory: Path) -> List[Path]:
        paths = [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in self._VALID_EXTS
        ]
        return sorted(paths, key=self._sort_key)

    @staticmethod
    def _sort_key(path: Path):
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem.lower())

"""
Image generation step using Legnext (remote diffusion).
"""

from __future__ import annotations

import base64
import random
from io import BytesIO
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class LegnextImageGeneratorStep(PipelineStep[List[Dict[str, Any]], List[Path]]):
    """
    Generates images for each paragraph using Legnext's hosted diffusion API.

    Input: List of paragraph dictionaries
    Output: List of paths to generated images
    """

    name = "legnext_image_generation"
    description = "Generate images using Legnext API"

    _DIFFUSION_URL = "https://api.legnext.ai/api/v1/diffusion"
    _JOB_URL = "https://api.legnext.ai/api/v1/job/{job_id}"

    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)

    def run(self, paragraphs: List[Dict[str, Any]]) -> List[Path]:
        image_paths: List[Path] = []

        for idx, para in enumerate(paragraphs):
            img_path = self.generate_for_paragraph(para, idx, total=len(paragraphs))
            image_paths.append(img_path)

        print(f"Generated {len(image_paths)} images (Legnext)")
        return image_paths

    def generate_for_paragraph(self, para: Dict[str, Any], idx: int, total: int = 1) -> Path:
        cache_entry = self.cache.get_image_cache_entry(para["text"], idx)
        if cache_entry:
            cached_path = Path(cache_entry.get("path", ""))
            if cached_path.exists() and cache_entry.get("stitched"):
                stitched_bytes = cached_path.read_bytes()
                tile_bytes = self._crop_random_tile(stitched_bytes)
                img_path = self.config.images_dir / f"{idx:03d}.png"
                img_path.write_bytes(tile_bytes)
                print(f"Using cached stitched image for paragraph {idx}")
                return img_path
            if cached_path.exists():
                img_path = self.config.images_dir / f"{idx:03d}.png"
                if cached_path != img_path:
                    shutil.copy(cached_path, img_path)
                print(f"Using cached image for paragraph {idx}")
                return img_path

        cached_path = self.cache.get_cached_image(
            para["text"], idx, self.config.images_dir
        )
        if cached_path is not None:
            if cached_path.name.endswith("-stitched.png") or cached_path.name.endswith("_stitched.png"):
                stitched_bytes = cached_path.read_bytes()
                tile_bytes = self._crop_random_tile(stitched_bytes)
                img_path = self.config.images_dir / f"{idx:03d}.png"
                img_path.write_bytes(tile_bytes)
                print(f"Using cached stitched image for paragraph {idx}")
                return img_path
            return cached_path

        api_key = self.config.legnext_api_key
        if not api_key:
            raise RuntimeError(
                "LEGNEXT_API_KEY is not set. Add it to the environment or use setup_env.py."
            )

        prompt = self.config.get_legnext_prompt(para["text"])
        print(f"Generating image {idx+1}/{total} with Legnext...")

        job_id = self._submit_job(api_key, prompt)
        result = self._poll_job(api_key, job_id)

        stitched = self._find_stitched_image(result)
        if stitched is not None:
            stitched_bytes = self._load_image_from_string(stitched)
            stitched_path = self._write_stitched_cache(para["text"], idx, stitched_bytes)
            tile_bytes = self._crop_random_tile(stitched_bytes)
            img_path = self.config.images_dir / f"{idx:03d}.png"
            img_path.write_bytes(tile_bytes)
            self.cache.save_image_cache(
                para["text"],
                idx,
                stitched_path,
                metadata={"stitched": True},
            )
            return img_path

        image_bytes = self._extract_image_bytes(result)
        img_path = self.config.images_dir / f"{idx:03d}.png"
        img_path.write_bytes(image_bytes)

        self.cache.save_image_cache(para["text"], idx, img_path)
        return img_path

    def _submit_job(self, api_key: str, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }
        payload = {"text": prompt}
        resp = requests.post(self._DIFFUSION_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Legnext diffusion failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        job_id = data.get("task_id") or data.get("job_id") or data.get("id")
        if not job_id:
            raise RuntimeError(f"Legnext diffusion response missing job id: {data}")
        return job_id

    def _poll_job(self, api_key: str, job_id: str) -> Dict[str, Any]:
        headers = {"x-api-key": api_key}
        timeout_sec = max(float(self.config.legnext_timeout_sec), 1.0)
        poll_interval = max(float(self.config.legnext_poll_interval_sec), 0.5)

        start = time.time()
        last_status: Optional[str] = None

        while True:
            if time.time() - start > timeout_sec:
                raise RuntimeError(
                    f"Legnext job timed out after {timeout_sec:.0f}s (last status: {last_status})"
                )

            url = self._JOB_URL.format(job_id=job_id)
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Legnext job fetch failed ({resp.status_code}): {resp.text}")

            data = resp.json()
            status = (data.get("status") or data.get("state") or "").lower()
            last_status = status or "unknown"

            if status in {"succeeded", "completed", "finished", "done"}:
                return data
            if status in {"failed", "error", "canceled", "cancelled"}:
                raise RuntimeError(f"Legnext job failed (status: {status}): {data}")

            time.sleep(poll_interval)

    def _extract_image_bytes(self, payload: Dict[str, Any]) -> bytes:
        value = self._find_image_value(payload)
        if value is None:
            raise RuntimeError(f"Legnext response missing image data: {payload}")

        if isinstance(value, str):
            return self._load_image_from_string(value)

        raise RuntimeError(f"Unsupported Legnext image payload type: {type(value)}")

    def _find_stitched_image(self, payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        output = payload.get("output")
        if isinstance(output, dict) and isinstance(output.get("image_url"), str):
            return output["image_url"]
        if isinstance(payload.get("image_url"), str):
            return payload["image_url"]
        return None

    def _find_image_value(self, payload: Any) -> Optional[Any]:
        if payload is None:
            return None

        if isinstance(payload, str):
            return payload

        if isinstance(payload, list) and payload:
            return self._find_image_value(random.choice(payload))

        if isinstance(payload, dict):
            for key in ("output", "image", "images", "result", "data"):
                if key in payload:
                    return self._find_image_value(payload[key])
            if "image_urls" in payload:
                return self._find_image_value(payload["image_urls"])
            if "image_url" in payload:
                return payload["image_url"]
            if "url" in payload:
                return payload["url"]

        return None

    def _load_image_from_string(self, value: str) -> bytes:
        if value.startswith("http://") or value.startswith("https://"):
            resp = requests.get(value, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"Legnext image download failed ({resp.status_code})")
            return resp.content

        if "base64," in value:
            _, b64 = value.split("base64,", 1)
            return base64.b64decode(b64)

        # Assume raw base64 string
        return base64.b64decode(value)

    def _write_stitched_cache(self, text: str, idx: int, stitched_bytes: bytes) -> Path:
        cache_key = self.cache.get_image_cache_key(text, idx)
        stitched_path = self.config.cache_dir / f"{cache_key}_stitched.png"
        stitched_path.write_bytes(stitched_bytes)
        return stitched_path

    def _crop_random_tile(self, stitched_bytes: bytes) -> bytes:
        try:
            img = Image.open(BytesIO(stitched_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open Legnext stitched image: {e}") from e

        width, height = img.size
        if width < 2 or height < 2:
            raise RuntimeError(f"Legnext stitched image too small to crop: {img.size}")

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

        out = BytesIO()
        tile.save(out, format="PNG")
        return out.getvalue()

    def _pick_nonempty_tile(self, img: Image.Image, tiles: List[tuple]) -> tuple:
        """
        Prefer tiles that are not mostly empty/black.
        Returns a tile box; falls back to random if all look empty.
        """
        best_box = None
        best_score = -1.0

        for box in tiles:
            tile = img.crop(box)
            # Compute non-black ratio
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

        # If best is still mostly empty, randomize among all tiles anyway
        if best_score < 0.05:
            return random.choice(tiles)

        # If multiple tiles are similarly good, pick randomly among them
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

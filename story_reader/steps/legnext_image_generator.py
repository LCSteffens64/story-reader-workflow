"""
Image generation step using Legnext (remote diffusion).
"""

from __future__ import annotations

import base64
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

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
        cached_path = self.cache.get_cached_image(
            para["text"], idx, self.config.images_dir
        )
        if cached_path is not None:
            return cached_path

        api_key = self.config.legnext_api_key
        if not api_key:
            raise RuntimeError(
                "LEGNEXT_API_KEY is not set. Add it to the environment or use setup_env.py."
            )

        prompt = self.config.get_image_prompt(para["text"])
        print(f"Generating image {idx+1}/{total} with Legnext...")

        job_id = self._submit_job(api_key, prompt)
        result = self._poll_job(api_key, job_id)

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
            if "image_url" in payload:
                return payload["image_url"]
            if "image_urls" in payload:
                return self._find_image_value(payload["image_urls"])
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

"""
Transcript verification step - signs paragraphs.json and overlays QR on final video.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from PIL import Image

from .base import PipelineStep
from ..config import PipelineConfig
from ..core.cache import CacheManager


class TranscriptVerifierStep(PipelineStep[Path, Path]):
    """
    Creates a detached PGP signature for paragraphs.json and overlays QR onto final_video.mp4.

    QR visibility:
    - First N frames (trust flash)
    - Last M seconds (scan window)
    """

    name = "transcript_verification_qr"
    description = "Sign paragraphs.json and embed QR verification payload into final video"

    def __init__(self, config: PipelineConfig, cache: CacheManager):
        super().__init__(config, cache)

    def run(self, final_video: Path) -> Path:
        paragraphs_path = self.config.paragraphs_file
        if not paragraphs_path.exists():
            raise RuntimeError(f"paragraphs.json not found: {paragraphs_path}")

        if not self.config.signing_key_fingerprint:
            raise RuntimeError(
                "Signing key is not configured. Set STORY_READER_SIGNING_KEY "
                "or config.signing_key_fingerprint."
            )
        if not self.config.signer_pubkey_url:
            raise RuntimeError(
                "Public key URL is not configured. Set STORY_READER_PUBKEY_URL "
                "or config.signer_pubkey_url."
            )

        signature_path = paragraphs_path.with_suffix(paragraphs_path.suffix + ".asc")
        payload_path = self.config.output_dir / "verification_qr_payload.txt"
        qr_path = self.config.output_dir / "verification_qr.png"
        verified_tmp = self.config.output_dir / "final_video.verified.tmp.mp4"

        self._sign_paragraphs(paragraphs_path, signature_path, self.config.signing_key_fingerprint)
        key_fpr = self._resolve_key_fingerprint(self.config.signing_key_fingerprint)

        signature_text = signature_path.read_text(encoding="utf-8")
        paragraphs_hash = self._sha256_file(paragraphs_path)
        payload_text = self._build_payload_text(
            key_fpr=key_fpr,
            pubkey_url=self.config.signer_pubkey_url,
            paragraphs_hash=paragraphs_hash,
            signature_text=signature_text,
        )
        payload_path.write_text(payload_text, encoding="utf-8")

        video_meta = self._probe_video(final_video)
        self._build_qr(payload_text, qr_path, video_meta["width"])
        self._overlay_qr(final_video, qr_path, verified_tmp, video_meta["duration"])

        verified_tmp.replace(final_video)
        print(f"Verification QR embedded into final video: {final_video}")
        return final_video

    def _sign_paragraphs(self, paragraphs_path: Path, signature_path: Path, key: str) -> None:
        cmd = [
            "gpg",
            "--batch",
            "--yes",
            "--armor",
            "--local-user",
            key,
            "--output",
            str(signature_path),
            "--detach-sign",
            str(paragraphs_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"GPG signing failed: {result.stderr.strip()}")

    def _resolve_key_fingerprint(self, key: str) -> str:
        cmd = ["gpg", "--batch", "--with-colons", "--fingerprint", key]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"GPG fingerprint lookup failed: {result.stderr.strip()}")

        for line in result.stdout.splitlines():
            if line.startswith("fpr:"):
                parts = line.split(":")
                if len(parts) > 9 and parts[9]:
                    return parts[9]
        raise RuntimeError(f"No fingerprint resolved for key: {key}")

    def _sha256_file(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _build_payload_text(
        self,
        key_fpr: str,
        pubkey_url: str,
        paragraphs_hash: str,
        signature_text: str,
    ) -> str:
        sig_clean = signature_text.rstrip("\n")
        lines = [
            "v=1",
            "type=openpgp-detached-signature",
            f"created_utc={datetime.now(timezone.utc).isoformat()}",
            f"signing_key_fpr={key_fpr}",
            f"pubkey_url={pubkey_url}",
            f"paragraphs_file={self.config.paragraphs_file.name}",
            f"paragraphs_sha256={paragraphs_hash}",
            "signature_armored_begin",
            sig_clean,
            "signature_armored_end",
            "",
        ]
        return "\n".join(lines)

    def _probe_video(self, video_path: Path) -> Dict[str, float]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError(f"No video streams found in: {video_path}")
        width = int(streams[0]["width"])
        duration = float(data["format"]["duration"])
        return {"width": width, "duration": duration}

    def _build_qr(self, payload_json: str, qr_path: Path, video_width: int) -> None:
        try:
            import qrcode
            from qrcode.exceptions import DataOverflowError
        except ImportError as e:
            raise RuntimeError("qrcode package is required for transcript verification.") from e

        try:
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(payload_json)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        except DataOverflowError as e:
            raise RuntimeError(
                "QR payload is too large to encode as a single QR code. "
                "Use a smaller payload or switch to URL-based payload mode."
            ) from e

        qr_target = max(256, int(video_width * self.config.qr_width_ratio))
        img = img.resize((qr_target, qr_target), resample=Image.Resampling.NEAREST)
        img.save(qr_path)

    def _overlay_qr(self, video_path: Path, qr_path: Path, output_path: Path, duration: float) -> None:
        intro_end_frame = max(0, self.config.qr_intro_frames - 1)
        outro_start = max(0.0, duration - self.config.qr_outro_seconds)
        # Use arithmetic OR for broad FFmpeg expression compatibility.
        enable_expr = f"between(n\\,0\\,{intro_end_frame})+gte(t\\,{outro_start:.3f})"
        filter_complex = (
            f"[0:v][1:v]overlay="
            f"x=W-w-{self.config.qr_margin_px}:"
            f"y=H-h-{self.config.qr_margin_px}:"
            f"enable='{enable_expr}'"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(qr_path),
            "-filter_complex",
            filter_complex,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg QR overlay failed: {result.stderr.strip()}")

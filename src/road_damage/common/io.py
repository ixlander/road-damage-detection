"""Input/output utility functions."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from road_damage.common.constants import ALLOWED_IMAGE_CONTENT_TYPES, DEFAULT_MAX_UPLOAD_BYTES


def ensure_allowed_image_upload(content_type: str | None, payload_size: int) -> None:
    """Validate upload constraints for API safety."""
    if payload_size > DEFAULT_MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large: {payload_size} bytes (max={DEFAULT_MAX_UPLOAD_BYTES})")
    if content_type and content_type.lower() not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}")


def load_rgb_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into RGB ndarray."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file") from exc
    return np.asarray(img)


def ensure_existing_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p

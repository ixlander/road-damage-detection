"""Project-wide constants and invariants."""

from __future__ import annotations

from dataclasses import dataclass

CLASS_ID_TO_NAME: dict[int, str] = {
    0: "crack",
    1: "pothole",
}
CLASS_NAME_TO_ID: dict[str, int] = {v: k for k, v in CLASS_ID_TO_NAME.items()}
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),
    1: (0, 165, 255),
}

DEFAULT_MODEL_ID = "default"
DEFAULT_MODEL_PATH = "runs/detect/smoke_2ep/weights/best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi"}


@dataclass(frozen=True)
class Invariant:
    """Codified class-map invariant."""

    crack_id: int = 0
    pothole_id: int = 1


def assert_class_map_invariant() -> None:
    """Raise if class mapping invariant is broken."""
    if CLASS_ID_TO_NAME.get(0) != "crack" or CLASS_ID_TO_NAME.get(1) != "pothole":
        msg = "Class mapping invariant violated: expected 0->crack and 1->pothole"
        raise ValueError(msg)

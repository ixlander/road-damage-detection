"""Shared inference service used by API/demo/CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from road_damage.common.config import ModelRegistry
from road_damage.common.constants import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MODEL_ID
from road_damage.inference.postprocess import annotate_image, serialize_yolo_result


def _default_model_loader(path: Path) -> Any:
    from ultralytics import YOLO  # local import to keep tests lightweight

    return YOLO(str(path))


@dataclass
class InferenceService:
    """Model registry + prediction facade with explicit cache."""

    model_registry: ModelRegistry
    model_loader: Callable[[Path], Any] = _default_model_loader

    def __post_init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get_model(self, model_id: str = DEFAULT_MODEL_ID) -> Any:
        if model_id not in self._cache:
            model_path = self.model_registry.resolve(model_id)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found for id '{model_id}': {model_path}")
            self._cache[model_id] = self.model_loader(model_path)
        return self._cache[model_id]

    def predict(
        self,
        image_rgb: np.ndarray,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        model = self.get_model(model_id=model_id)
        results = model.predict(source=image_rgb, conf=conf, iou=iou, verbose=False)
        detections = serialize_yolo_result(results[0])
        annotated_rgb = annotate_image(image_rgb=image_rgb, detections=detections)
        return detections, annotated_rgb

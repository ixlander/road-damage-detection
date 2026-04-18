"""Inference result serialization and annotation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from road_damage.common.constants import CLASS_COLORS_BGR, CLASS_ID_TO_NAME


def serialize_yolo_result(result: Any) -> list[dict[str, Any]]:
    """Serialize Ultralytics result to API-friendly list."""
    out: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out

    for b in boxes:
        cls_id = int(b.cls.item())
        score = float(b.conf.item())
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        out.append(
            {
                "class_id": cls_id,
                "class_name": CLASS_ID_TO_NAME.get(cls_id, str(cls_id)),
                "confidence": round(score, 4),
                "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return out


def annotate_image(image_rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw boxes and labels on RGB image."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        cls_id = int(det["class_id"])
        color = CLASS_COLORS_BGR.get(cls_id, (0, 255, 0))
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, label, (x1, max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

from pathlib import Path

import numpy as np

from road_damage.common.config import ModelRegistry
from road_damage.inference.service import InferenceService


class _Box:
    def __init__(self):
        self.cls = np.array([1.0])
        self.conf = np.array([0.8])
        self.xyxy = np.array([[1.0, 2.0, 3.0, 4.0]])


class _Result:
    boxes = [_Box()]


class _Model:
    def predict(self, source, conf, iou, verbose):
        _ = (source, conf, iou, verbose)
        return [_Result()]



def test_inference_service_basic_flow(tmp_path: Path) -> None:
    model_file = tmp_path / "model.pt"
    model_file.write_text("x", encoding="utf-8")

    svc = InferenceService(
        model_registry=ModelRegistry(models={"default": str(model_file)}),
        model_loader=lambda _: _Model(),
    )
    img = np.zeros((16, 16, 3), dtype=np.uint8)

    detections, annotated = svc.predict(img)
    assert len(detections) == 1
    assert detections[0]["class_id"] == 1
    assert annotated.shape == img.shape

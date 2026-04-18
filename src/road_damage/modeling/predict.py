"""Prediction CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from road_damage.common.config import get_model_registry
from road_damage.inference.service import InferenceService


def run_smoke() -> int:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    class FakeModel:
        def predict(self, source, conf, iou, verbose):
            _ = (source, conf, iou, verbose)

            class Box:
                def __init__(self):
                    import numpy as _np

                    self.cls = _np.array([0.0])
                    self.conf = _np.array([0.9])
                    self.xyxy = _np.array([[5.0, 5.0, 25.0, 25.0]])

            class Result:
                boxes = [Box()]

            return [Result()]

    svc = InferenceService(model_registry=get_model_registry(), model_loader=lambda _: FakeModel())
    dets, _annotated = svc.predict(image_rgb=image)
    print({"n_detections": len(dets), "detections": dets})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference smoke helper")
    parser.add_argument("--smoke", action="store_true")
    _ = parser.parse_args()
    return run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import apps.api.main as api_main
from apps.api.main import APP


class _FakeService:
    def predict(self, image_rgb, model_id, conf, iou):
        _ = (image_rgb, model_id, conf, iou)
        return (
            [
                {
                    "class_id": 0,
                    "class_name": "crack",
                    "confidence": 0.9,
                    "xyxy": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            np.zeros((8, 8, 3), dtype=np.uint8),
        )


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint() -> None:
    client = TestClient(APP)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint_schema(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "SERVICE", _FakeService())
    client = TestClient(APP)

    resp = client.post(
        "/predict_image",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_version"] == "v1"
    assert body["n_detections"] == 1
    assert set(body["detections"][0].keys()) == {"class_id", "class_name", "confidence", "xyxy"}

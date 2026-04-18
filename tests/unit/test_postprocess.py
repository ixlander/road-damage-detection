import numpy as np

from road_damage.inference.postprocess import serialize_yolo_result


class _Box:
    def __init__(self, cls_id: float, conf: float, xyxy: list[float]) -> None:
        self.cls = np.array([cls_id])
        self.conf = np.array([conf])
        self.xyxy = np.array([xyxy])


class _Result:
    def __init__(self) -> None:
        self.boxes = [_Box(0.0, 0.95, [10.0, 11.0, 20.0, 21.0])]



def test_prediction_serialization_schema() -> None:
    out = serialize_yolo_result(_Result())
    assert out == [
        {
            "class_id": 0,
            "class_name": "crack",
            "confidence": 0.95,
            "xyxy": [10.0, 11.0, 20.0, 21.0],
        }
    ]

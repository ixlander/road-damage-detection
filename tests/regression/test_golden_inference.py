import numpy as np

from road_damage.inference.postprocess import serialize_yolo_result


class _Box:
    def __init__(self):
        self.cls = np.array([1.0])
        self.conf = np.array([0.81234])
        self.xyxy = np.array([[12.345, 6.789, 45.123, 27.555]])


class _Result:
    boxes = [_Box()]



def test_golden_json_output() -> None:
    golden = [
        {
            "class_id": 1,
            "class_name": "pothole",
            "confidence": 0.8123,
            "xyxy": [12.3, 6.8, 45.1, 27.6],
        }
    ]
    assert serialize_yolo_result(_Result()) == golden

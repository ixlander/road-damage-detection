from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_damage.common.config import get_model_registry
from road_damage.common.constants import (
    CLASS_ID_TO_NAME,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_MODEL_ID,
)
from road_damage.inference.service import InferenceService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model", default=None, help="Deprecated compatibility argument")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--every_n", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Video not found: {source}")

    svc = InferenceService(get_model_registry())

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = out_dir / f"{source.stem}_pred.mp4"
    out_csv = out_dir / f"{source.stem}_pred.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        cw = csv.writer(fp)
        cw.writerow(
            ["frame_id", "timestamp_sec", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"]
        )

        frame_id = -1
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_id += 1

            if frame_id % args.every_n != 0:
                writer.write(frame_bgr)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections, annotated_rgb = svc.predict(
                frame_rgb,
                model_id=args.model_id,
                conf=args.conf,
                iou=args.iou,
            )

            for det in detections:
                x1, y1, x2, y2 = det["xyxy"]
                cls_id = int(det["class_id"])
                cw.writerow(
                    [
                        frame_id,
                        round(frame_id / fps, 3),
                        cls_id,
                        CLASS_ID_TO_NAME.get(cls_id, str(cls_id)),
                        det["confidence"],
                        round(x1, 1),
                        round(y1, 1),
                        round(x2, 1),
                        round(y2, 1),
                    ]
                )

            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            if frame_id % 300 == 0:
                print(f"[{frame_id}/{total}] processed")

    cap.release()
    writer.release()
    print(f"Saved video: {out_video}")
    print(f"Saved csv: {out_csv}")


if __name__ == "__main__":
    main()

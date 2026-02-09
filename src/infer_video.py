from __future__ import annotations

import argparse
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

CLASSES = {0: "crack", 1: "pothole"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="path to YOLO .pt weights")
    ap.add_argument("--source", type=str, required=True, help="path to input video")
    ap.add_argument("--out_dir", type=str, default="outputs", help="output directory")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--every_n", type=int, default=1, help="process every n-th frame")
    args = ap.parse_args()

    model_path = Path(args.model)
    video_path = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = out_dir / f"{video_path.stem}_pred.mp4"
    out_csv = out_dir / f"{video_path.stem}_pred.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame_id", "timestamp_sec",
            "class_id", "class_name",
            "conf",
            "x1", "y1", "x2", "y2"
        ])

        frame_id = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            if frame_id % args.every_n != 0:
                writer.write(frame)
                continue

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )
            r = results[0]
            boxes = r.boxes

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls.item())
                    conf = float(b.conf.item())
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{CLASSES.get(cls_id, str(cls_id))} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), max(20, int(y1))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    csv_writer.writerow([
                        frame_id, frame_id / fps,
                        cls_id, CLASSES.get(cls_id, str(cls_id)),
                        f"{conf:.4f}",
                        f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"
                    ])

            writer.write(frame)

            if frame_id % 300 == 0:
                print(f"[{frame_id}/{total_frames}] processed")

    cap.release()
    writer.release()
    print(f"Saved video: {out_video}")
    print(f"Saved csv:   {out_csv}")

if __name__ == "__main__":
    main()
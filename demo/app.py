from __future__ import annotations

import io
import csv
from pathlib import Path
import tempfile

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

CLASSES = {0: "crack", 1: "pothole"}

st.set_page_config(page_title="Road Damage Detection Demo", layout="wide")

st.title("Road Damage Detection — YOLOv8 Demo")
st.caption("Upload an image/video and get detections (crack / pothole).")

@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Weights not found: {p}")
    return YOLO(str(p))

def yolo_predict_image(model: YOLO, img_rgb: np.ndarray, conf: float, iou: float):
    results = model.predict(source=img_rgb, conf=conf, iou=iou, verbose=False)
    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            dets.append({
                "class_id": cls_id,
                "class_name": CLASSES.get(cls_id, str(cls_id)),
                "confidence": round(score, 4),
                "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

    annotated_bgr = r.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return dets, annotated_rgb

def run_video_inference(
    model: YOLO,
    video_path: Path,
    conf: float,
    iou: float,
    every_n: int,
    out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = out_dir / f"{video_path.stem}_pred.mp4"
    out_csv = out_dir / f"{video_path.stem}_pred.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["frame_id", "timestamp_sec", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"])

        frame_id = -1
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_id += 1

            if frame_id % every_n != 0:
                writer.write(frame_bgr)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = model.predict(source=frame_rgb, conf=conf, iou=iou, verbose=False)
            r = results[0]

            annotated_bgr = r.plot()  # already BGR

            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    score = float(b.conf.item())
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    cw.writerow([
                        frame_id, round(frame_id / fps, 3),
                        cls_id, CLASSES.get(cls_id, str(cls_id)),
                        round(score, 4),
                        round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)
                    ])

            writer.write(annotated_bgr)

            if frame_id % 300 == 0:
                st.write(f"Processed {frame_id}/{total} frames...")

    cap.release()
    writer.release()
    return out_video, out_csv

st.sidebar.header("Settings")

default_weights = "runs/detect/smoke_2ep/weights/best.pt"
weights_path = st.sidebar.text_input("Weights path (.pt)", value=default_weights)

st.sidebar.markdown("### Quick presets")
preset = st.sidebar.selectbox(
    "Choose a preset",
    [
        "Balanced (recommended)",
        "More detections (higher recall)",
        "Fewer false positives (higher precision)",
        "Demo / Smoke model (very low conf)",
    ],
    index=0
)

if preset == "Balanced (recommended)":
    conf_default, iou_default = 0.15, 0.45
elif preset == "More detections (higher recall)":
    conf_default, iou_default = 0.05, 0.45
elif preset == "Fewer false positives (higher precision)":
    conf_default, iou_default = 0.35, 0.50
else:  # Demo / Smoke model
    conf_default, iou_default = 0.01, 0.45

st.sidebar.markdown("### Confidence threshold (conf)")
st.sidebar.caption(
    "Filters out low-confidence detections.\n"
    "- Lower = more boxes (higher recall, more false positives)\n"
    "- Higher = fewer boxes (higher precision, may miss objects)\n\n"
    "**Typical range:** 0.05–0.35"
)
conf = st.sidebar.slider(
    "conf",
    min_value=0.0,
    max_value=1.0,
    value=float(conf_default),
    step=0.01
)

st.sidebar.markdown("### NMS IoU (iou)")
st.sidebar.caption(
    "Controls how aggressively overlapping boxes are merged by Non-Maximum Suppression (NMS).\n"
    "- Lower IoU = stricter merging → fewer duplicate boxes\n"
    "- Higher IoU = keeps more overlapping boxes → can show duplicates\n\n"
    "**Typical range:** 0.40–0.60"
)
iou = st.sidebar.slider(
    "iou",
    min_value=0.0,
    max_value=1.0,
    value=float(iou_default),
    step=0.01
)

if conf < 0.03:
    st.sidebar.warning("Very low conf: expect many boxes/noise. Useful for smoke models or debugging.")
elif conf > 0.50:
    st.sidebar.warning("Very high conf: model may return 0 detections on many images.")

if iou < 0.25:
    st.sidebar.info("Low IoU: duplicates will be aggressively removed (might suppress nearby objects).")
elif iou > 0.75:
    st.sidebar.info("High IoU: NMS will keep many overlapping boxes (more duplicates).")

try:
    model = load_model(weights_path)
    st.sidebar.success("Model loaded")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

tab_img, tab_vid = st.tabs(["🖼 Image", "🎥 Video"])

with tab_img:
    st.subheader("Image inference")
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if up is not None:
        img = Image.open(io.BytesIO(up.read())).convert("RGB")
        img_np = np.array(img)

        dets, annotated = yolo_predict_image(model, img_np, conf=conf, iou=iou)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_np, caption="Input", use_container_width=True)
        with c2:
            st.image(annotated, caption="Annotated", use_container_width=True)

        st.write(f"Detections: {len(dets)}")
        if len(dets) == 0:
            st.info("No detections. Try lowering `conf` (e.g., 0.05) or using the 'More detections' preset.")
        elif len(dets) > 30:
            st.info("Many detections. Try increasing `conf` (e.g., 0.25+) or using the 'Fewer false positives' preset.")
        st.json({"detections": dets})

with tab_vid:
    st.subheader("Video inference")
    st.caption("Tip: if your video is long, increase `every_n` to process fewer frames.")

    every_n = st.number_input("Process every N-th frame", min_value=1, max_value=30, value=3, step=1)
    upv = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if upv is not None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_path = td / "input_video.mp4"
            in_path.write_bytes(upv.read())

            out_dir = td / "out"
            st.info("Running video inference...")

            out_video, out_csv = run_video_inference(
                model=model,
                video_path=in_path,
                conf=conf,
                iou=iou,
                every_n=int(every_n),
                out_dir=out_dir
            )

            st.success("Done!")
            st.video(out_video.read_bytes())
            st.download_button(
                "Download annotated video (mp4)",
                data=out_video.read_bytes(),
                file_name=out_video.name,
                mime="video/mp4"
            )
            st.download_button(
                "Download predictions (csv)",
                data=out_csv.read_bytes(),
                file_name=out_csv.name,
                mime="text/csv"
            )
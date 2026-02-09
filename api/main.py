from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import io

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from ultralytics import YOLO
from fastapi.responses import Response
import cv2


APP = FastAPI(title="Road Damage Detection API", version="0.1.0")

CLASSES = {0: "crack", 1: "pothole"}

_model: Optional[YOLO] = None

def get_model(model_path: str) -> YOLO:
    global _model
    if _model is None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        _model = YOLO(str(p))
    return _model

@APP.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@APP.post("/predict_image")
async def predict_image(
    file: UploadFile = File(...),
    model_path: str = Query(
        default="runs/detect/smoke_2ep/weights/best.pt",
        description="Path to YOLO weights (.pt)"
    ),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    iou: float = Query(default=0.45, ge=0.0, le=1.0),
) -> JSONResponse:
    """
    Upload an image and get detections as JSON.
    """
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    img_np = np.array(img)

    try:
        model = get_model(model_path)
        results = model.predict(source=img_np, conf=conf, iou=iou, verbose=False)
    except FileNotFoundError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Inference failed: {e}"})

    r = results[0]
    out: List[Dict[str, Any]] = []

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append({
                "class_id": cls_id,
                "class_name": CLASSES.get(cls_id, str(cls_id)),
                "confidence": round(score, 4),
                "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

    return JSONResponse(content={
        "filename": file.filename,
        "n_detections": len(out),
        "detections": out,
        "conf": conf,
        "iou": iou,
        "model_path": model_path,
    })
    
@APP.post("/predict_image_annotated")
async def predict_image_annotated(
    file: UploadFile = File(...),
    model_path: str = Query(default="runs/detect/smoke_2ep/weights/best.pt"),
    conf: float = Query(default=0.05, ge=0.0, le=1.0),
    iou: float = Query(default=0.45, ge=0.0, le=1.0),
) -> Response:
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(img)

    model = get_model(model_path)
    results = model.predict(source=img_np, conf=conf, iou=iou, verbose=False)
    r = results[0]

    annotated = img_np.copy()

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{CLASSES.get(cls_id, str(cls_id))} {score:.2f}"
            cv2.putText(annotated, label, (x1, max(20, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if not ok:
        return Response(content=b"", media_type="image/png")

    return Response(content=buf.tobytes(), media_type="image/png")

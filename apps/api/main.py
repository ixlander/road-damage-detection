from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import cv2
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from road_damage.common.config import get_model_registry
from road_damage.common.constants import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MODEL_ID
from road_damage.common.io import ensure_allowed_image_upload, load_rgb_image_from_bytes
from road_damage.inference.api_schemas import Detection, PredictResponseV1
from road_damage.inference.service import InferenceService

APP = FastAPI(title="Road Damage Detection API", version="1.0.0")
LOGGER = logging.getLogger("road_damage.api")

SERVICE = InferenceService(model_registry=get_model_registry())


@APP.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@APP.post("/v1/predict_image", response_model=PredictResponseV1)
async def predict_image_v1(
    file: UploadFile = File(...),
    model_id: str = Query(default=DEFAULT_MODEL_ID),
    model_path: str | None = Query(default=None, description="Deprecated: use model_id"),
    conf: float = Query(default=DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Query(default=DEFAULT_IOU, ge=0.0, le=1.0),
) -> PredictResponseV1 | JSONResponse:
    data = await file.read()
    try:
        ensure_allowed_image_upload(file.content_type, len(data))
        image_rgb = load_rgb_image_from_bytes(data)

        # compatibility handling: model_path accepted only if it matches a whitelisted model path
        if model_path:
            matched_id = None
            for mid, mpath in get_model_registry().models.items():
                if model_path == mpath:
                    matched_id = mid
                    break
            if matched_id is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Invalid model_path: path must be registered in ROAD_DAMAGE_MODELS.",
                        "api_version": "v1",
                    },
                )
            model_id = matched_id

        detections, _annotated = SERVICE.predict(
            image_rgb=image_rgb, model_id=model_id, conf=conf, iou=iou
        )
        return PredictResponseV1(
            filename=file.filename,
            n_detections=len(detections),
            detections=[Detection(**d) for d in detections],
            conf=conf,
            iou=iou,
            model_id=model_id,
            api_version="v1",
        )
    except ValueError as exc:
        LOGGER.warning("Invalid request payload: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request payload", "api_version": "v1"},
        )
    except FileNotFoundError as exc:
        LOGGER.warning("Requested model unavailable: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": "Requested model unavailable", "api_version": "v1"},
        )
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("Unhandled inference failure")
        return JSONResponse(
            status_code=500, content={"error": "Inference failed", "api_version": "v1"}
        )


@APP.post("/predict_image", response_model=PredictResponseV1)
async def predict_image_compat(
    file: UploadFile = File(...),
    model_id: str = Query(default=DEFAULT_MODEL_ID),
    model_path: str | None = Query(default=None),
    conf: float = Query(default=DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Query(default=DEFAULT_IOU, ge=0.0, le=1.0),
) -> PredictResponseV1 | JSONResponse:
    """Backward-compatible alias for v1 predict endpoint."""
    return await predict_image_v1(
        file=file, model_id=model_id, model_path=model_path, conf=conf, iou=iou
    )


@APP.post("/predict_image_annotated")
async def predict_image_annotated(
    file: UploadFile = File(...),
    model_id: str = Query(default=DEFAULT_MODEL_ID),
    conf: float = Query(default=0.05, ge=0.0, le=1.0),
    iou: float = Query(default=DEFAULT_IOU, ge=0.0, le=1.0),
) -> Response:
    data = await file.read()
    try:
        ensure_allowed_image_upload(file.content_type, len(data))
        image_rgb = load_rgb_image_from_bytes(data)
        _detections, annotated_rgb = SERVICE.predict(
            image_rgb=image_rgb, model_id=model_id, conf=conf, iou=iou
        )
        ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return Response(content=b"", media_type="image/png")
        return Response(content=buf.tobytes(), media_type="image/png")
    except ValueError as exc:
        LOGGER.warning("Invalid annotated request payload: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request payload", "api_version": "v1"},
        )
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("Unhandled annotated inference failure")
        return JSONResponse(
            status_code=500, content={"error": "Inference failed", "api_version": "v1"}
        )

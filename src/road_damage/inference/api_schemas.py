"""API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    xyxy: list[float] = Field(min_length=4, max_length=4)


class PredictResponseV1(BaseModel):
    filename: str | None
    n_detections: int
    detections: list[Detection]
    conf: float = Field(ge=0.0, le=1.0)
    iou: float = Field(ge=0.0, le=1.0)
    model_id: str
    api_version: str = "v1"

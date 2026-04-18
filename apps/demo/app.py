from __future__ import annotations

import io
from pathlib import Path
import sys

import streamlit as st
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_damage.common.config import get_model_registry
from road_damage.common.constants import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MODEL_ID
from road_damage.inference.service import InferenceService


@st.cache_resource
def get_service() -> InferenceService:
    return InferenceService(model_registry=get_model_registry())


def main() -> None:
    st.set_page_config(page_title="Road Damage Detection Demo", layout="wide")
    st.title("Road Damage Detection — YOLOv8 Demo")

    conf = st.sidebar.slider("conf", 0.0, 1.0, float(DEFAULT_CONF), 0.01)
    iou = st.sidebar.slider("iou", 0.0, 1.0, float(DEFAULT_IOU), 0.01)
    model_id = st.sidebar.text_input("Model ID", value=DEFAULT_MODEL_ID)

    up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if up is None:
        return

    img = Image.open(io.BytesIO(up.read())).convert("RGB")
    img_np = np.asarray(img)
    service = get_service()

    try:
        detections, annotated = service.predict(img_np, model_id=model_id, conf=conf, iou=iou)
        c1, c2 = st.columns(2)
        c1.image(img_np, caption="Input", use_container_width=True)
        c2.image(annotated, caption="Annotated", use_container_width=True)
        st.json({"detections": detections})
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()

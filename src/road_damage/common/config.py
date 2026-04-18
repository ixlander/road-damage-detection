"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import yaml

from road_damage.common.constants import DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH


@dataclass(frozen=True)
class ModelRegistry:
    """Allowed model IDs and paths."""

    models: dict[str, str]

    def resolve(self, model_id: str) -> Path:
        if model_id not in self.models:
            raise KeyError(f"Unknown model_id '{model_id}'. Allowed: {sorted(self.models)}")
        return Path(self.models[model_id])


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return data


def get_model_registry() -> ModelRegistry:
    """Load model registry from env or defaults.

    Env format: ROAD_DAMAGE_MODELS='default=runs/.../best.pt,small=models/x.pt'
    """
    raw = os.getenv("ROAD_DAMAGE_MODELS", "").strip()
    models: dict[str, str] = {}
    if raw:
        for chunk in raw.split(","):
            item = chunk.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError("ROAD_DAMAGE_MODELS entries must be id=path")
            model_id, model_path = item.split("=", 1)
            models[model_id.strip()] = model_path.strip()
    if not models:
        models = {DEFAULT_MODEL_ID: DEFAULT_MODEL_PATH}
    return ModelRegistry(models=models)

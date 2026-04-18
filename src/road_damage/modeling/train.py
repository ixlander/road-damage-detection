"""Reproducible training wrapper around Ultralytics."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from road_damage.common.config import load_yaml
from road_damage.data.contracts import create_manifest


@dataclass
class TrainMetadata:
    commit_sha: str
    seed: int
    config_path: str
    config_snapshot: dict[str, Any]
    dataset_manifest: dict[str, Any]
    created_at_utc: str


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_train(config_path: Path, output_dir: Path) -> int:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    _set_seed(seed)

    dataset_root = Path(config.get("dataset_root", "data/rdd2class_yolo"))
    manifest = asdict(
        create_manifest(dataset_root=dataset_root, source="RDD2022", version="2class")
    )

    metadata = TrainMetadata(
        commit_sha=_get_git_sha(),
        seed=seed,
        config_path=str(config_path),
        config_snapshot=config,
        dataset_manifest=manifest,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2), encoding="utf-8"
    )

    # Optional train execution for local full runs.
    if config.get("run_ultralytics", False):
        from ultralytics import YOLO

        model = YOLO(str(config.get("model", "yolov8n.pt")))
        model.train(
            data=str(config.get("data", "configs/data/rdd2class.yaml")),
            epochs=int(config.get("epochs", 50)),
            imgsz=int(config.get("imgsz", 640)),
            batch=int(config.get("batch", 8)),
            seed=seed,
            device=config.get("device", "cpu"),
            name=str(config.get("name", "baseline")),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible training wrapper")
    parser.add_argument("--config", default="configs/train/baseline.yaml")
    parser.add_argument("--out", default="outputs/train")
    args = parser.parse_args()
    return run_train(config_path=Path(args.config), output_dir=Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())

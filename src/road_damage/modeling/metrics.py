"""Metrics helpers for evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json


@dataclass
class EvalSummary:
    map50: float
    precision: float
    recall: float
    per_class_ap: dict[str, float]



def write_eval_outputs(summary: EvalSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    with (out_dir / "per_class_ap.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["class_name", "ap"])
        for class_name, ap in summary.per_class_ap.items():
            writer.writerow([class_name, ap])

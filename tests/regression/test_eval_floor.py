import json
from pathlib import Path

from road_damage.modeling.evaluate import run_evaluate


def test_tiny_eval_subset_metric_floors(tmp_path: Path) -> None:
    baseline = json.loads(
        Path("tests/regression/fixtures/baseline_metrics.json").read_text(encoding="utf-8")
    )

    config = tmp_path / "eval.yaml"
    config.write_text(
        "\n".join(
            [
                "smoke_map50: 0.56",
                f"baseline_floor_map50: {baseline['map50_floor']}",
                f"baseline_floor_crack_ap: {baseline['crack_ap_floor']}",
                f"baseline_floor_pothole_ap: {baseline['pothole_ap_floor']}",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    run_evaluate(config_path=config, out_dir=out_dir, smoke=True)

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["map50"] >= baseline["map50_floor"]
    assert metrics["per_class_ap"]["crack"] >= baseline["crack_ap_floor"]
    assert metrics["per_class_ap"]["pothole"] >= baseline["pothole_ap_floor"]

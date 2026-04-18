from pathlib import Path

from PIL import Image

from road_damage.data.validate import validate_dataset



def _touch_img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(path)



def test_validator_happy_path(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for split in ("train", "val", "test"):
        _touch_img(root / "images" / split / f"{split}_1.jpg")
        lbl = root / "labels" / split / f"{split}_1.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = validate_dataset(root)
    assert report.ok is True
    assert report.summary == {}



def test_validator_detects_class_and_bbox_errors(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _touch_img(root / "images" / "train" / "sample.jpg")
    lbl = root / "labels" / "train" / "sample.txt"
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text("3 1.5 0.5 0.0 0.2\n", encoding="utf-8")

    for split in ("val", "test"):
        _touch_img(root / "images" / split / f"{split}.jpg")
        p = root / "labels" / split / f"{split}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = validate_dataset(root)
    assert report.ok is False
    assert report.summary.get("class_id_oob", 0) >= 1
    assert report.summary.get("bbox_oob", 0) >= 1
    assert report.summary.get("bbox_non_positive", 0) >= 1

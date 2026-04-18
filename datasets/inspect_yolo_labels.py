import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_damage.data.validate import validate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/rdd2class_yolo")
    args = parser.parse_args()

    report = validate_dataset(Path(args.root))
    print({"ok": report.ok, "summary": report.summary, "issues": len(report.issues)})
    raise SystemExit(0 if report.ok else 1)

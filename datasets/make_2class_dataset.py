import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_damage.data.convert_rdd import convert_dataset

if __name__ == "__main__":
    convert_dataset(Path("data/raw/RDD_SPLIT"), Path("data/rdd2class_yolo"))

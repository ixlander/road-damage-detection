import random
from pathlib import Path
import cv2

ROOT = Path("data/raw/RDD_SPLIT/train")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"

OUT_DIR = Path("data/class_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

def load_labels(lbl_path: Path):
    boxes = []
    txt = lbl_path.read_text(encoding="utf-8").strip()
    if not txt:
        return boxes
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        boxes.append((cls, xc, yc, w, h))
    return boxes

def find_image(stem: str):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = IMG_DIR / (stem + ext)
        if p.exists():
            return p
    return None

per_class = {i: [] for i in range(5)}
for lp in LBL_DIR.glob("*.txt"):
    img_path = find_image(lp.stem)
    if img_path is None:
        continue
    labs = load_labels(lp)
    for cls, xc, yc, w, h in labs:
        if cls in per_class:
            per_class[cls].append((img_path, (cls, xc, yc, w, h)))

for cls in range(5):
    items = per_class.get(cls, [])
    if not items:
        print(f"class {cls}: no boxes")
        continue

    sample = random.sample(items, k=min(40, len(items)))
    out_cls_dir = OUT_DIR / f"class_{cls}"
    out_cls_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_path, (_, xc, yc, w, h)) in enumerate(sample):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        x1 = int((xc - w/2) * W)
        y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W)
        y2 = int((yc + h/2) * H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        h_c, w_c = crop.shape[:2]
        scale = 256 / max(h_c, w_c)
        if scale > 1.0:
            crop = cv2.resize(crop, (int(w_c*scale), int(h_c*scale)), interpolation=cv2.INTER_LINEAR)

        cv2.putText(crop, f"class {cls}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imwrite(str(out_cls_dir / f"{i:03d}.jpg"), crop)

print(f"Saved crops to: {OUT_DIR.resolve()}")

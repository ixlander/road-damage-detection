import random
from pathlib import Path
import cv2

ROOT = Path("data/raw/RDD_SPLIT/train")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"

OUT_DIR = Path("data/class_preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

def load_labels(lbl_path):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes

files_by_class = {i: [] for i in range(5)}
label_files = list(LBL_DIR.glob("*.txt"))

for lp in label_files:
    labs = load_labels(lp)
    present = set(cls for cls, *_ in labs)
    for c in present:
        if c in files_by_class:
            files_by_class[c].append(lp)

for c in range(5):
    if not files_by_class[c]:
        print(f"class {c}: no files found")
        continue

    sample = random.sample(files_by_class[c], k=min(8, len(files_by_class[c])))
    for i, lp in enumerate(sample):
        img_path = IMG_DIR / (lp.stem + ".jpg")
        if not img_path.exists():
            img_path = IMG_DIR / (lp.stem + ".png")
            if not img_path.exists():
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        labs = load_labels(lp)

        for cls, xc, yc, w, h in labs:
            if cls != c:
                continue
            x1 = int((xc - w/2) * W)
            y1 = int((yc - h/2) * H)
            x2 = int((xc + w/2) * W)
            y2 = int((yc + h/2) * H)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"class {c}", (max(0,x1), max(20,y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        out_path = OUT_DIR / f"class_{c}_{i}_{lp.stem}.jpg"
        cv2.imwrite(str(out_path), img)

print(f"Saved previews to: {OUT_DIR}")

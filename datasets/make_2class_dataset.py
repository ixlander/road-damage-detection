from pathlib import Path
import shutil

SRC_ROOT = Path("data/raw/RDD_SPLIT")
DST_ROOT = Path("data/rdd2class_yolo")
OLD_TO_NEW = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
}

IMG_EXTS = [".jpg", ".jpeg", ".png"]

def find_image(split_img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = split_img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def convert_label_file(src_lbl: Path) -> str:
    """
    Return converted label file content in YOLO format.
    Unknown class ids are dropped.
    """
    lines_out = []
    txt = src_lbl.read_text(encoding="utf-8").strip()
    if not txt:
        return ""
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        old_cls = int(parts[0])
        if old_cls not in OLD_TO_NEW:
            continue
        new_cls = OLD_TO_NEW[old_cls]
        # keep bbox as-is
        xc, yc, w, h = parts[1:5]
        lines_out.append(f"{new_cls} {xc} {yc} {w} {h}")
    return "\n".join(lines_out) + ("\n" if lines_out else "")

def process_split(split: str):
    src_img_dir = SRC_ROOT / split / "images"
    src_lbl_dir = SRC_ROOT / split / "labels"

    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(src_lbl_dir.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {src_lbl_dir}")

    copied = 0
    for src_lbl in label_files:
        stem = src_lbl.stem
        img_path = find_image(src_img_dir, stem)
        if img_path is None:
            continue

        converted = convert_label_file(src_lbl)
        (dst_lbl_dir / src_lbl.name).write_text(converted, encoding="utf-8")

        shutil.copy2(img_path, dst_img_dir / img_path.name)
        copied += 1

    print(f"{split}: processed {copied} samples")

def main():
    if DST_ROOT.exists():
        print(f"Removing existing {DST_ROOT} ...")
        shutil.rmtree(DST_ROOT)

    for split in ["train", "val", "test"]:
        process_split(split)

    print(f"Done. 2-class dataset at: {DST_ROOT}")
    print("Classes: 0=crack, 1=pothole")

if __name__ == "__main__":
    main()
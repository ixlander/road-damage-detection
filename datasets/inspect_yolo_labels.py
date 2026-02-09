from pathlib import Path
from collections import Counter
import argparse

def count_labels(labels_dir: Path) -> Counter:
    c = Counter()
    for p in labels_dir.rglob("*.txt"):
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            parts = line.split()
            if not parts:
                continue
            c[int(parts[0])] += 1
    return c

def find_non_allowed(labels_dir: Path, allowed: set[int], max_files: int = 10):
    bad = []
    for p in labels_dir.rglob("*.txt"):
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        for line in txt.splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls not in allowed:
                bad.append((p, cls))
                break
        if len(bad) >= max_files:
            break
    return bad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/rdd2class_yolo",
        help="YOLO dataset root with images/ and labels/ subfolders",
    )
    parser.add_argument(
        "--allowed",
        type=str,
        default="0,1",
        help="Comma-separated allowed class ids (e.g. 0,1)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    allowed = set(int(x.strip()) for x in args.allowed.split(",") if x.strip() != "")

    print(f"Dataset root: {root.resolve()}")
    print(f"Allowed class ids: {sorted(allowed)}")

    for split in ["train", "val", "test"]:
        labels_dir = root / "labels" / split
        if not labels_dir.exists():
            print(f"\n{split}: labels dir not found: {labels_dir}")
            continue

        c = count_labels(labels_dir)
        total = sum(c.values())
        uniq = sorted(c.keys())

        print(f"\n{split}: total_boxes={total}")
        print(f"  unique_class_ids={uniq}")
        for k, v in c.most_common():
            print(f"  class_id {k}: {v}")

        bad_files = find_non_allowed(labels_dir, allowed=allowed, max_files=10)
        if bad_files:
            print("  WARNING: found non-allowed class ids in files:")
            for p, cls in bad_files:
                print(f"    {p}  (class_id={cls})")
        else:
            print("  OK: all class ids are within allowed set")

if __name__ == "__main__":
    main()

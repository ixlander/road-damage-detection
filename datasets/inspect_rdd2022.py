from pathlib import Path
from collections import Counter

root = Path("data/raw")
exts = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())

print("Top file extensions:")
for k, v in exts.most_common(30):
    print(f"{k or '<no_ext>'}: {v}")

for cand in [".xml", ".json", ".csv", ".txt"]:
    print(f"Found {cand}: {exts.get(cand, 0)} files")
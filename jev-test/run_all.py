import re
import subprocess
import sys
from pathlib import Path

FOLDER = Path(r"C:\Users\kirkmanz\OneDrive - Milwaukee School of Engineering\Fall 2026\Innovation Lab\csi-vial-fill-dataset\vial_fill\images")
LIMIT = None  # max number of NEW images to process this run; set to None for all

here = Path(__file__).resolve().parent
script = here / "label_one.py"
boxed_dir = here / "boxed"  # must match the folder label_one.py saves into

def boxed_path(image_path):
    # Same name rule as label_one.py
    safe_stem = re.sub(r"[^\w.\-]", "_", image_path.stem)
    return boxed_dir / f"{safe_stem}_boxed.png"

images = sorted(p for p in FOLDER.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

todo = [p for p in images if not boxed_path(p).exists()]
print(f"{len(images)} images found, {len(images) - len(todo)} already boxed, {len(todo)} to do")

if LIMIT:
    todo = todo[:LIMIT]

for i, path in enumerate(todo, 1):
    print(f"\n[{i}/{len(todo)}] {path.name}")
    subprocess.run([sys.executable, str(script), str(path)])
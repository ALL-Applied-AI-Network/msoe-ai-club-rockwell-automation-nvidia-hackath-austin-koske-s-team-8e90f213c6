import re
import sys
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
from jev_lite import Box, Image, Noul, TypeSafe

path = Path(sys.argv[1])
w, h = PILImage.open(path).size

# Edit these descriptions freely. Each one becomes its own Box question.
prompts = {
    "height": "The full height of the vial in `photo`, including the part below the holder that contains the liquid",
}
colors = {"height": "blue"}

questions = {name: Box(instructions=text) for name, text in prompts.items()}

with TypeSafe(model="jev-1.14.0") as client:
    result = client.system_one.create(
        state={"photo": Image(path)},
        questions=questions,
    )

img = PILImage.open(path).convert("RGB")
draw = ImageDraw.Draw(img)

for name in prompts:
    b = result.answers[name]
    e = b.box
    x1, y1 = e.left * w,  e.top * h
    x2, y2 = e.right * w, e.bottom * h
    print(f"{name:7s} {colors[name]:6s} conf {b.confidence:.2f}  "
          f"({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
    draw.rectangle([x1, y1, x2, y2], outline=colors[name], width=3)

safe_stem = re.sub(r"[^\w.\-]", "_", path.stem)
out_dir = Path(__file__).resolve().parent / "boxed"
out_dir.mkdir(exist_ok=True)
out = out_dir / f"{safe_stem}_boxed.png"
print("saving to", repr(str(out)))
img.save(out)
print("saved", out)
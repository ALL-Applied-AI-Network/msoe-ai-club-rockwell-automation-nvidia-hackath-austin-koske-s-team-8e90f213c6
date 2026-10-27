"""Create an obvious, non-sensitive image for testing inference plumbing."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
image = Image.new('RGB', (320, 240), 'white')
draw = ImageDraw.Draw(image)
draw.rectangle((35, 70, 125, 200), fill=(240, 120, 20), outline='black', width=3)
draw.rectangle((195, 70, 285, 200), fill=(20, 90, 225), outline='black', width=3)
args.out.parent.mkdir(parents=True, exist_ok=True)
image.save(args.out)
print(args.out)

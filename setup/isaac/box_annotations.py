"""Renderer-independent annotation contract. Boxes use half-open XYXY pixels."""
import hashlib
import json
import math
from pathlib import Path

CLASSES = {"vial": 0, "liquid": 1, "stream": 2}
COLORS = {"vial": (80, 230, 120), "liquid": (255, 185, 40), "stream": (70, 200, 255)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_box(mask):
    import numpy as np
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def lens_mm(usd_values, meters_per_unit):
    """USD camera focal/aperture units are tenths of a stage length unit."""
    import numpy as np
    return np.asarray(usd_values, dtype=float) * float(meters_per_unit) * 100


def instance_mask(data, prim_path):
    """Accept only IDs whose runtime prim path equals or descends from the target."""
    import numpy as np
    pixels = np.asarray(data["data"])
    if pixels.ndim != 2:
        raise ValueError(f"Expected uint instance plane, got {pixels.shape}")
    mapping = {int(token): path for token, path in data["info"]["idToLabels"].items()}
    for token in np.unique(pixels):
        if token >= 2:
            path = mapping.get(int(token))
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError("Renderer instance pixels have an empty prim-path mapping or invalid/missing path; reject ambiguous labels")
    ids = []
    for token, path in data["info"]["idToLabels"].items():
        if not isinstance(path, str):
            raise ValueError("Unexpected Replicator instance ID mapping")
        if path == prim_path or path.startswith(prim_path + "/"):
            ids.append(int(token))
    return np.isin(pixels, ids)


def semantic_mask(data, object_tag):
    """Unique scene-authored class tags identify objects without a prim-path LUT."""
    import numpy as np
    pixels = np.asarray(data["data"])
    if pixels.ndim != 2:
        raise ValueError(f"Expected uint semantic plane, got {pixels.shape}")
    mapping = {int(token): labels for token, labels in data["info"]["idToLabels"].items()}
    ids = []
    for token in np.unique(pixels):
        if token < 2:
            continue
        labels = mapping.get(int(token))
        if not isinstance(labels, dict) or not isinstance(labels.get("class"), str) or not labels["class"]:
            raise ValueError("Renderer semantic pixels lack an explicit object tag; reject ambiguous labels")
        if labels["class"] == object_tag:
            ids.append(int(token))
    return np.isin(pixels, ids)


def yolo_line(class_name, box, width=640, height=480):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid source-image box: {box}")
    return f"{CLASSES[class_name]} {(x0+x1)/(2*width):.8f} {(y0+y1)/(2*height):.8f} {(x1-x0)/width:.8f} {(y1-y0)/height:.8f}"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_fixture(fixture):
    if fixture["schema_version"] != 1 or fixture["station_frame"] != 686:
        raise ValueError("This reviewed camera/geometry boundary requires fixture v1, frozen frame 686")
    seen = set()
    for row in fixture["cases"]:
        name = row["id"]
        if name in seen or not name or any(c not in "0123456789abcdefghijklmnopqrstuvwxyz-" for c in name):
            raise ValueError("Case IDs must be unique safe file stems")
        seen.add(name)
        # Keep preflight standard-library-only. Importing NumPy before Kit
        # changes native library loading on the pinned ARM Isaac runtime.
        rgb = row["rgb"]
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3 or not all(
                isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in rgb):
            raise ValueError("Invalid linear diffuse RGB")
        if not 0 <= row["level"] <= 1 or not 0 < row["opacity"] <= 1:
            raise ValueError("Invalid level/opacity")
        if row["stream"] not in ("off", "short", "continuous"):
            raise ValueError("Unknown stream mode")
    if not seen:
        raise ValueError("Empty fixture")

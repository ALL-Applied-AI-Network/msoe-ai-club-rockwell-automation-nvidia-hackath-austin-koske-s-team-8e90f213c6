"""Reusable YOLO/temporal inference with the existing measurement QA."""
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from integration.measurement_qa import MeasurementQA

VISION = Path(__file__).resolve().parents[1] / "setup" / "vision"
sys.path.insert(0, str(VISION))
from temporal_pool import CHECKPOINT_SHA256, TemporalPool

VIAL_SHA256 = "0bfac07a256eaf190416b632ad1f094a6b534309df1bb11a8c175e9d811b56bc"


class ModelRuntime:
    def __init__(self, assets, device="cpu"):
        import torch
        from ultralytics import YOLO
        assets = Path(assets)
        vial = assets / "models" / "vial-best.pt"
        if not vial.is_file() or sha256(vial.read_bytes()).hexdigest() != VIAL_SHA256:
            raise ValueError("Use the released, hash-verified vial checkpoint")
        torch.set_num_threads(4)
        self.device = device
        self.vial = YOLO(str(vial))
        self.liquid = TemporalPool(assets / "models" / "liquid-best.pt", device)

    def new_measurement(self):
        return MeasurementQA()

    def infer(self, rgb, reference, qa):
        prediction = self.vial.predict(
            Image.fromarray(rgb), imgsz=640, rect=False, conf=.35, iou=.45,
            classes=[0], device=self.device, verbose=False, save=False,
        )[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        result = self.liquid.predict(rgb, reference)
        # QA owns fill-position selection, track stability and box containment.
        measurement = qa.measure(
            rgb, boxes, scores, result["proposal_box"], result["proposal_mask"],
            result["presence"], result["presence_gate_passed"],
        )
        return result, measurement, boxes

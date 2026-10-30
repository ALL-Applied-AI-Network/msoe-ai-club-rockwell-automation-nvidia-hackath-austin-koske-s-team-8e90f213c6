"""Reusable YOLO/temporal inference with the existing measurement QA."""
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import cv2
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

    @staticmethod
    def _profile_box(box):
        if box is None:
            return None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Station fill_vial_box must be [x0, y0, x1, y1]")
        values = [float(value) for value in box]
        x0, y0, x1, y1 = values
        if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480):
            raise ValueError("Station fill_vial_box must fit the native 640x480 camera profile")
        return values

    @staticmethod
    def render_overlay(rgb, result, measured, boxes):
        """Operator preview only: model proposal green, QA sampling mask blue, geometry boxes."""
        from replay.geometry import decode_mask
        rendered = rgb.copy()
        proposal = np.asarray(result["proposal_mask"], dtype=bool)
        rendered[proposal] = (.55 * rendered[proposal] + .45 * np.array([30, 220, 130])).astype(np.uint8)
        geometry = measured.get("geometry") or {}
        encoded = geometry.get("mask")
        sampling = decode_mask(encoded).astype(bool) if encoded else np.zeros(proposal.shape, dtype=bool)
        rendered[sampling] = (.4 * rendered[sampling] + .6 * np.array([40, 160, 255])).astype(np.uint8)
        for box in boxes:
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(rendered, (x0, y0), (x1, y1), (80, 140, 255), 2)
        source = result.get("vial_geometry_source", "yolo")
        cv2.putText(rendered, f"GREEN: PROPOSAL  BLUE: QA SAMPLE  BOX: {source.upper()}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(rendered, f"YOLO VIALS: {result.get('yolo_vial_boxes', len(boxes))}",
                    (10, 43), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 2, cv2.LINE_AA)
        return rendered

    def infer(self, rgb, reference, qa, fallback_vial_box=None):
        prediction = self.vial.predict(
            Image.fromarray(rgb), imgsz=640, rect=False, conf=.35, iou=.45,
            classes=[0], device=self.device, verbose=False, save=False,
        )[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        yolo_vial_boxes = int(len(boxes))
        geometry_source = "yolo"
        profile_box = self._profile_box(fallback_vial_box)
        if yolo_vial_boxes == 0 and profile_box is not None:
            boxes = np.asarray([profile_box], dtype=np.float32)
            scores = np.asarray([1.0], dtype=np.float32)
            geometry_source = "station_profile_candidate"
        result = self.liquid.predict(rgb, reference)
        result["yolo_vial_boxes"] = yolo_vial_boxes
        result["vial_geometry_source"] = geometry_source
        # QA owns fill-position selection, track stability and box containment.
        measurement = qa.measure(
            rgb, boxes, scores, result["proposal_box"], result["proposal_mask"],
            result["presence"], result["presence_gate_passed"],
        )
        return result, measurement, boxes

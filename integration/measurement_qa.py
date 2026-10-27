import copy
import hashlib
from replay.config import load_config
from replay.geometry import encode_mask
from replay.measurement import Extractor


class ProposalAdapter:
    def set(self, rgb, predictions, proposals):
        self.digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        self.predictions = copy.deepcopy(predictions)
        self.proposals = copy.deepcopy(proposals)

    def check(self, rgb):
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest != self.digest:
            raise ValueError("QA proposals belong to a different frame")

    def detect(self, rgb):
        self.check(rgb)
        return copy.deepcopy(self.predictions)

    def propose(self, rgb, vial):
        self.check(rgb)
        return copy.deepcopy(self.proposals)


class MeasurementQA:
    def __init__(self):
        self.adapter = ProposalAdapter()
        self.extractor = Extractor(
            load_config(),
            detector=self.adapter
        )
        self.extractor.geometry.segmenter = self.adapter

    def measure(
        self,
        rgb,
        vial_boxes,
        vial_scores,
        proposal_box,
        proposal_mask,
        presence,
        association_passed
    ):
        predictions = []

        for box, score in zip(vial_boxes, vial_scores):
            predictions.append({
                "class_id": 0,
                "box": [int(v) for v in box],
                "confidence": float(score)
            })

        proposals = []

        if association_passed:
            liquid = {
                "class_id": 1,
                "box": [int(v) for v in proposal_box],
                "confidence": float(presence),
                "candidate_mask": encode_mask(proposal_mask)
            }

            proposals.append(liquid)
            predictions.append({
                key: value
                for key, value in liquid.items()
                if key != "candidate_mask"
            })

        self.adapter.set(rgb, predictions, proposals)
        return self.extractor.measure(rgb)

"""Long-lived model adapter. Unqualified reference/recipe evidence stays explicit."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import time

from models.inspection import OperatorDecision, OperatorStatus, PipelineMeasurement
from pipelines.base import FrameContext, PipelineResult

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def decode_image(content):
    import numpy as np
    from PIL import Image
    with Image.open(BytesIO(content)) as image:
        if image.size != (640, 480):
            raise ValueError("Camera profile requires original 640x480 pixels; no automatic resizing")
        return np.array(image.convert("RGB"), copy=True)


class TemporalInspectionPipeline:
    mode = "model"

    def __init__(self, assets=None, reference=None, device="cpu"):
        self.assets = assets
        self.reference = Path(reference) if reference else None
        self.device = device
        self.runtime = None
        self._lock = asyncio.Lock()
        self._states = OrderedDict()
        self._references = {}

    async def install_reference(self, session_id, content, evidence_id):
        rgb = await asyncio.to_thread(decode_image, content)
        async with self._lock:
            self._references[str(session_id)] = (rgb, sha256(content).hexdigest(), evidence_id)
            self._states.pop(str(session_id), None)
        return {"status": "unreviewed_not_certified_empty", "sha256": sha256(content).hexdigest(),
                "evidence_id": evidence_id, "recipe_qualified": False}

    async def analyze(self, image_bytes, session, frame_number, simulate=None, context=None):
        if simulate:
            raise ValueError("Simulation controls require explicit mock mode")
        async with self._lock:
            return await asyncio.to_thread(self._analyze, image_bytes, session, context)

    def _analyze(self, image_bytes, session, context: FrameContext | None):
        import numpy as np
        from PIL import Image, ImageDraw
        from replay.geometry import decode_mask
        from integration.runtime import CHECKPOINT_SHA256, ModelRuntime, VIAL_SHA256
        started = time.perf_counter()
        rgb = decode_image(image_bytes)
        if context and (context.width, context.height) != (640, 480):
            raise ValueError("Declared dimensions disagree with model camera profile")
        if not self.assets:
            raise ValueError("Set APP__MODEL_ASSETS to the extracted temporal-cnn-assets-v1 directory")
        if self.runtime is None:
            self.runtime = ModelRuntime(self.assets, self.device)
        key = str(session.session_id)
        reference = self._references.get(key)
        if reference is None and self.reference:
            content = self.reference.read_bytes()
            reference = (decode_image(content), sha256(content).hexdigest(), "configured-preview-reference")
        if reference is None:
            raise ValueError("Choose an earlier aligned reference image before processing; it remains unreviewed")
        state = self._states.get(key)
        source = context.source_id if context else "unknown"
        stamp = context.source_timestamp_ms if context else None
        # Stills are independent; never infer a physical sequence from list order.
        reset = (state is None or state["source"] != source or stamp is None
                 or (state["time"] is not None and stamp <= state["time"]))
        if reset:
            state = {"qa": self.runtime.new_measurement(), "source": source, "time": None}
        result, measured, boxes = self.runtime.infer(rgb, reference[0], state["qa"])
        state["time"] = stamp
        self._states[key] = state
        self._states.move_to_end(key)
        while len(self._states) > 128:
            old, _ = self._states.popitem(last=False)
            self._references.pop(old, None)
        model_mask = result["proposal_mask"].astype(bool)
        sampling_mask = decode_mask(measured["geometry"]["mask"]).astype(bool)
        rendered = rgb.copy()
        rendered[model_mask] = (.55 * rendered[model_mask] + .45 * np.array([30, 220, 130])).astype(np.uint8)
        rendered[sampling_mask] = (.4 * rendered[sampling_mask] + .6 * np.array([40, 160, 255])).astype(np.uint8)
        image = Image.fromarray(rendered)
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(tuple(map(int, box)), outline=(80, 140, 255), width=2)
        draw.text((12, 12), "YOLO + TEMPORAL U-NET | GREEN: PROPOSAL  BLUE: SAMPLE", fill="white", stroke_width=1, stroke_fill="black")
        rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        rgba[model_mask] = [30, 220, 130, 160]
        def png(value):
            output = BytesIO()
            (value if isinstance(value, Image.Image) else Image.fromarray(value)).save(output, format="PNG")
            return output.getvalue()
        qa_pass = bool(measured["measurement_valid"])
        flags = measured["quality_flags"]
        reasons = list(flags) + ["temporal_reference_unreviewed", "recipe_endpoint_reference_missing",
                                "fill_completion_and_physical_identity_unconfirmed"]
        if not context or not context.recipe_matched:
            reasons.append("recipe_context_unmatched")
        decision = OperatorDecision(
            status=OperatorStatus.WARNING if qa_pass else OperatorStatus.UNAVAILABLE,
            headline="Liquid measurement available · review context" if qa_pass else "Measurement unavailable",
            detail=("RGB/Lab sampled by measurement QA. Recipe agreement is not established."
                    if qa_pass else "Sampling checks: " + ", ".join(flags)),
            recommended_action="Review the mask, reference and intended recipe; this is not product approval",
            model_state="YOLO11n + temporal U-Net + unchanged measurement QA",
            confidence=None,
            measurements=[
                PipelineMeasurement(label="Liquid proposal", value="Visible" if model_mask.any() else "Not proposed", state="neutral"),
                PipelineMeasurement(label="Measurement QA", value="Available" if qa_pass else "Unavailable", state="good" if qa_pass else "warning"),
                PipelineMeasurement(label="Recipe", value="Unqualified", state="warning"),
            ],
            evidence={"measurement_valid": qa_pass, "quality_flags": flags,
                      "rgb": measured["rgb"], "lab": measured["lab"],
                      "presence": result["presence"], "proposal_pixels": int(model_mask.sum()),
                      "sampling_pixels": int(sampling_mask.sum()), "measurement": measured,
                      "reference": {"sha256": reference[1], "evidence_id": reference[2], "status": "unreviewed_not_certified_empty"},
                      "source": {"rgb_sha256": sha256(rgb.tobytes()).hexdigest(), "clock": context.source_clock if context else "unknown",
                                 "timestamp_ms": stamp, "source_id": source},
                      "models": {"vial_sha256": VIAL_SHA256, "liquid_sha256": CHECKPOINT_SHA256},
                      "reset": reset, "recipe_comparison_ready": False, "blocked_reasons": reasons,
                      "quality_decision": None, "hardware_command": None,
                      "processing_ms": (time.perf_counter() - started) * 1000},
        )
        return PipelineResult(decision=decision, rendered_frame=png(image),
                              rendered_frame_mime_type="image/png", mask=png(rgba))

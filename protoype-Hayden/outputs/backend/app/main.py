from __future__ import annotations

import base64
import io
import json
import os
import threading
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Annotated, Any, Literal, Protocol

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

app = FastAPI(
    title="Inline Vial Inspection Backend",
    version="0.1.0",
    description=(
        "Camera frame ingest and live inspection updates for the changeover-aware vial inspection website."
    ),
)
origins = [value.strip() for value in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrontendClients:
    """WebSocket subscribers keyed by camera, so the UI receives live frame events."""

    def __init__(self):
        self.clients: dict[str, set[WebSocket]] = defaultdict(set)
        self.lock = threading.Lock()

    async def connect(self, camera_id: str, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.clients[camera_id].add(websocket)

    def disconnect(self, camera_id: str, websocket: WebSocket):
        with self.lock:
            self.clients[camera_id].discard(websocket)
            if not self.clients[camera_id]:
                self.clients.pop(camera_id, None)

    async def publish(self, camera_id: str, event: dict[str, Any]) -> int:
        with self.lock:
            listeners = list(self.clients.get(camera_id, set()))
        stale = []
        for listener in listeners:
            try:
                await listener.send_json(event)
            except Exception:
                stale.append(listener)
        for listener in stale:
            self.disconnect(camera_id, listener)
        return len(listeners) - len(stale)


frontend_clients = FrontendClients()
WEB_INDEX = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")


class Box(BaseModel):
    """Pixel box with xyxy-exclusive right and bottom edges."""

    xyxy: tuple[float, float, float, float]
    class_id: int
    label: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str | None = None


class FrameAnalysis(BaseModel):
    source_id: str = Field(min_length=1, max_length=256)
    pts_seconds: float = Field(ge=0)
    detections: list[Box] = Field(default_factory=list)
    # Detector frame dimensions, as recorded by the source pipeline.
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pool_present: bool | None = None
    sample_usable: bool | None = None
    unusable_reasons: list[str] = Field(default_factory=list)
    current_recipe_id: str | None = None
    previous_recipe_id: str | None = None
    vial_index: int | None = Field(default=None, ge=1)
    is_first_vial_after_change: bool | None = None
    component_name: str | None = None
    # Linear or sRGB values in 0..255 as computed by the edge detector/camera pipeline.
    observed_color_rgb: tuple[int, int, int] | None = None
    expected_color_rgb: tuple[int, int, int] | None = None
    color_roi_xyxy: tuple[int, int, int, int] | None = None


class AnalysisRequest(BaseModel):
    stream_id: str = Field(default="default", min_length=1, max_length=128)
    frame: FrameAnalysis


class MaskEngine(Protocol):
    def predict(self, image_rgb: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> tuple[np.ndarray, float | None]: ...


class UnconfiguredEngine:
    def predict(self, image_rgb: np.ndarray, box_xyxy: tuple[int, int, int, int]):
        raise RuntimeError("SAM2 is not configured; set SAM2_CONFIG and SAM2_CHECKPOINT")


class Sam2ImageEngine:
    """Lazy SAM2 wrapper: one independent prompt per liquid detection."""

    def __init__(self, config: str, checkpoint: str, device: str):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)
        self.lock = threading.Lock()

    def predict(self, image_rgb: np.ndarray, box_xyxy: tuple[int, int, int, int]):
        # Predictor keeps mutable image embeddings, so protect each set-image/predict pair.
        with self.lock:
            self.predictor.set_image(image_rgb)
            # SAM uses inclusive xyxy coordinates; API contract uses exclusive edges.
            x1, y1, x2, y2 = box_xyxy
            masks, scores, _ = self.predictor.predict(
                box=np.array([x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)], dtype=np.float32),
                multimask_output=False,
            )
        return masks[0].astype(bool), float(scores[0]) if len(scores) else None


def create_engine() -> MaskEngine:
    config = os.getenv("SAM2_CONFIG", "").strip()
    checkpoint = os.getenv("SAM2_CHECKPOINT", "").strip()
    if not config or not checkpoint:
        return UnconfiguredEngine()
    try:
        return Sam2ImageEngine(config, checkpoint, os.getenv("SAM2_DEVICE", "cuda"))
    except Exception as exc:
        # Keep health and integration endpoints up; analysis reports availability.
        return UnconfiguredEngine()


engine: MaskEngine = create_engine()
MASK_THRESHOLD = float(os.getenv("MASK_THRESHOLD", "0"))
POOL_MIN_AREA_RATIO = float(os.getenv("POOL_MIN_AREA_RATIO", "0.0005"))
POOL_MAX_AREA_RATIO = float(os.getenv("POOL_MAX_AREA_RATIO", "0.70"))
SEQUENCE_WINDOW = max(2, int(os.getenv("SEQUENCE_WINDOW", "5")))
history: dict[str, deque[tuple[float, bool, float]]] = defaultdict(
    lambda: deque(maxlen=SEQUENCE_WINDOW)
)


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Image could not be decoded") from exc


def png_base64(mask: np.ndarray) -> str:
    out = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def clipped_box(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box.xyxy
    # Pixel coordinates are rounded to nearest integer, then clamped to frame.
    x1, y1 = max(0, min(width, round(x1))), max(0, min(height, round(y1)))
    x2, y2 = max(0, min(width, round(x2))), max(0, min(height, round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert sRGB D65 to CIE Lab for a comparable camera color reading."""
    values = [component / 255 for component in rgb]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
    r, g, b = linear
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    delta = 6 / 29
    f = lambda t: t ** (1 / 3) if t > delta ** 3 else t / (3 * delta ** 2) + 4 / 29
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e_2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """CIEDE2000 color difference (kL=kC=kH=1)."""
    import math
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0
    elif dh > 180:
        dhp = dh - 360
    elif dh < -180:
        dhp = dh + 360
    else:
        dhp = dh
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lbarp, Cbarp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbarp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbarp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbarp = (h1p + h2p + 360) / 2
    else:
        hbarp = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbarp - 30)) + 0.24 * math.cos(math.radians(2 * hbarp))
         + 0.32 * math.cos(math.radians(3 * hbarp + 6)) - 0.20 * math.cos(math.radians(4 * hbarp - 63)))
    dtheta = 30 * math.exp(-((hbarp - 275) / 25) ** 2)
    Rc = 2 * math.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbarp - 50) ** 2 / math.sqrt(20 + (Lbarp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbarp, 1 + 0.015 * Cbarp * T
    Rt = -math.sin(math.radians(2 * dtheta)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def color_decision(metadata: FrameAnalysis) -> dict[str, Any]:
    if metadata.observed_color_rgb is None or metadata.expected_color_rgb is None:
        return {"status": "unavailable", "reason": "observed_and_expected_rgb_required"}
    observed_lab = rgb_to_lab(metadata.observed_color_rgb)
    expected_lab = rgb_to_lab(metadata.expected_color_rgb)
    distance = delta_e_2000(observed_lab, expected_lab)
    alert = distance > 2.3
    transition = metadata.previous_recipe_id is not None and metadata.current_recipe_id is not None and metadata.previous_recipe_id != metadata.current_recipe_id
    cause = None
    if alert and transition and metadata.is_first_vial_after_change:
        cause = {
            "type": "previous_recipe_residue",
            "likely_previous_recipe_id": metadata.previous_recipe_id,
            "explanation": "The first vial after a recipe change differs from the clean current-recipe reference; residual blend from the preceding recipe is a plausible cause.",
            "residual_fraction_alpha": {"lower": 0.0, "upper": 1.0, "status": "not_estimated_process_variable"},
        }
    elif alert:
        cause = {"type": "undetermined", "explanation": "Color differs from the clean reference; the supplied frame context does not establish a changeover carryover cause."}
    return {
        "status": "alert" if alert else "within_threshold",
        "delta_e_2000": round(distance, 3),
        "threshold": 2.3,
        "alert": alert,
        "current_recipe_id": metadata.current_recipe_id,
        "previous_recipe_id": metadata.previous_recipe_id,
        "vial_index": metadata.vial_index,
        "component_name": metadata.component_name,
        "observed_rgb": list(metadata.observed_color_rgb),
        "expected_clean_rgb": list(metadata.expected_color_rgb),
        "observed_lab_d65": [round(x, 2) for x in observed_lab],
        "expected_lab_d65": [round(x, 2) for x in expected_lab],
        "likely_cause": cause,
        "reference_note": "Requires camera/color calibration and an approved clean-vial reference under matched lighting.",
    }


def sequence_decision(stream_id: str, pts: float, present: bool, area_ratio: float) -> dict[str, Any]:
    """Deterministic evidence summary; it does not claim onset or reliability labels."""
    window = history[stream_id]
    if window and pts <= window[-1][0]:
        # Out-of-order frames must not corrupt temporal state.
        return {"state": "unavailable", "reason": "non_increasing_pts", "trend": "unavailable"}
    window.append((pts, present, area_ratio))
    recent = list(window)
    if len(recent) < 2:
        state = "warming_up"
        trend = "insufficient_history"
    elif not present:
        state = "no_pool_proposal"
        trend = "falling" if recent[-2][1] else "stable_absent"
    else:
        trend = "rising" if area_ratio > recent[-2][2] * 1.10 else (
            "falling" if area_ratio < recent[-2][2] * 0.90 else "stable"
        )
        state = "transition" if trend in {"rising", "falling"} else "pool_proposed"
    return {"state": state, "trend": trend, "frames_in_window": len(recent)}


def run_analysis(image_bytes: bytes, metadata: FrameAnalysis, stream_id: str) -> dict[str, Any]:
    image = decode_image(image_bytes)
    height, width = image.shape[:2]
    if (width, height) != (metadata.width, metadata.height):
        raise HTTPException(
            status_code=422,
            detail=f"Image size {width}x{height} does not match frame contract {metadata.width}x{metadata.height}",
        )
    if metadata.observed_color_rgb is None and metadata.color_roi_xyxy is not None:
        x1, y1, x2, y2 = metadata.color_roi_xyxy
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise HTTPException(status_code=422, detail="color_roi_xyxy must be a non-empty region inside the frame")
        roi = image[y1:y2, x1:x2]
        measured = tuple(int(v) for v in np.median(roi.reshape(-1, 3), axis=0))
        metadata = metadata.model_copy(update={"observed_color_rgb": measured})

    proposals: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    merged = np.zeros((height, width), dtype=bool)
    liquid_boxes = [d for d in metadata.detections if d.class_id == 1]
    vial_count = sum(d.class_id == 0 for d in metadata.detections)
    engine_error: str | None = None
    for index, detection in enumerate(liquid_boxes):
        box = clipped_box(detection, width, height)
        common = {
            "detection_index": index,
            "class_id": detection.class_id,
            "label": detection.label or "liquid",
            "confidence": detection.confidence,
            "detector_source": detection.source,
        }
        if box is None:
            rejected.append({**common, "reason": "box_outside_frame_or_empty"})
            continue
        try:
            mask, score = engine.predict(image, box)
        except Exception as exc:
            engine_error = str(exc)
            break
        if mask.shape != (height, width):
            rejected.append({**common, "reason": "mask_shape_mismatch"})
            continue
        binary = np.asarray(mask, dtype=bool)
        # Drop SAM's lowest-quality selected mask only when caller configured a threshold.
        if score is not None and score < MASK_THRESHOLD:
            rejected.append({**common, "reason": "below_mask_score_threshold", "mask_score": score})
            continue
        merged |= binary
        proposals.append({
            **common,
            "prompt_box_xyxy_exclusive": list(box),
            "mask_score": score,
            "mask_png_base64": png_base64(binary),
            "mask_encoding": "PNG grayscale binary 0/255",
            "mask_width": width,
            "mask_height": height,
            "coordinate_space": "original_source_frame",
        })

    if engine_error:
        proposals.clear()
        merged.fill(False)
        status: Literal["ok", "no_liquid_proposal", "segmentation_unavailable"] = "segmentation_unavailable"
        absent_reason = "sam2_unavailable"
    elif not liquid_boxes:
        status = "no_liquid_proposal"
        absent_reason = "no_class_1_liquid_detection"
    elif not proposals:
        status = "no_liquid_proposal"
        absent_reason = "all_liquid_boxes_rejected"
    else:
        status = "ok"
        absent_reason = None

    area_ratio = float(merged.sum() / (width * height))
    if status == "ok" and area_ratio > POOL_MAX_AREA_RATIO:
        quality = {"usable": False, "reasons": ["mask_area_exceeds_configured_maximum"]}
    elif status == "ok" and area_ratio < POOL_MIN_AREA_RATIO:
        quality = {"usable": False, "reasons": ["mask_area_below_configured_minimum"]}
    elif metadata.sample_usable is False:
        quality = {"usable": False, "reasons": metadata.unusable_reasons or ["marked_unusable_by_upstream"]}
    else:
        quality = {"usable": metadata.sample_usable, "reasons": metadata.unusable_reasons}
    decision = sequence_decision(stream_id, metadata.pts_seconds, status == "ok", area_ratio)
    return {
        "status": status,
        "source_id": metadata.source_id,
        "stream_id": stream_id,
        "pts_seconds": metadata.pts_seconds,
        "frame": {"width": width, "height": height, "color_space": "RGB"},
        "decision": {
            "pool_present": status == "ok",
            "pool_presence_source": "mask_proposal_from_class_1_detections" if status == "ok" else "no_proposal",
            "pool_mask_area_ratio": area_ratio if status == "ok" else None,
            "sequence": decision,
            "sample_usability": quality,
            "color_inspection": color_decision(metadata),
        },
        "proposals": proposals,
        "combined_mask_png_base64": png_base64(merged) if status == "ok" else None,
        "rejected_detections": rejected,
        "excluded_vial_detection_count": vial_count,
        "unavailable_reason": absent_reason,
        "provenance": {
            "segmentation_model": "SAM2.1 Hiera Tiny" if status != "segmentation_unavailable" else None,
            "prompt_policy": "independent same-frame class-1 liquid boxes; class-0 vial boxes excluded",
            "box_coordinate_convention": "pixel xyxy-exclusive",
            "mask_coordinate_convention": "original-frame coordinates",
            "mask_values": "binary 0/255 PNG",
            "evaluation_split_warning": "Run 4 is development data; draft masks are exploratory references, not approved ground truth",
        },
    }


@app.get("/health")
def health():
    with frontend_clients.lock:
        connected_frontends = sum(len(value) for value in frontend_clients.clients.values())
    return {"status": "ok", "service": "vial-inspection-backend", "sam2_configured": isinstance(engine, Sam2ImageEngine), "connected_frontends": connected_frontends}


@app.get("/", include_in_schema=False)
def website():
    return FileResponse(WEB_INDEX)


@app.websocket("/ws/frontend/{camera_id}")
async def frontend_stream(websocket: WebSocket, camera_id: str):
    """Frontend subscribes here and receives one JSON event for each accepted camera frame."""
    await frontend_clients.connect(camera_id, websocket)
    try:
        while True:
            # Keep the connection alive; the client can send ping/heartbeat messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        frontend_clients.disconnect(camera_id, websocket)


@app.post("/v1/cameras/{camera_id}/frames")
async def receive_camera_frame(
    camera_id: str,
    image: Annotated[UploadFile, File()],
    source_id: Annotated[str, Form()],
    pts_seconds: Annotated[float, Form(ge=0)],
    width: Annotated[int, Form(gt=0)],
    height: Annotated[int, Form(gt=0)],
    detections_json: Annotated[str, Form()] = "[]",
    sample_usable: Annotated[bool | None, Form()] = None,
    current_recipe_id: Annotated[str | None, Form()] = None,
    previous_recipe_id: Annotated[str | None, Form()] = None,
    vial_index: Annotated[int | None, Form(ge=1)] = None,
    is_first_vial_after_change: Annotated[bool | None, Form()] = None,
    component_name: Annotated[str | None, Form()] = None,
    observed_color_rgb_json: Annotated[str | None, Form()] = None,
    expected_color_rgb_json: Annotated[str | None, Form()] = None,
    color_roi_xyxy_json: Annotated[str | None, Form()] = None,
):
    """Camera/edge client posts a JPEG frame; processed frame and decision are pushed to the website."""
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Image must be JPEG, PNG, or WebP")
    try:
        detections_data = json.loads(detections_json)
        if not isinstance(detections_data, list):
            raise ValueError("Expected array")
        detections = [Box.model_validate(item) for item in detections_data]
        observed_rgb = tuple(json.loads(observed_color_rgb_json)) if observed_color_rgb_json else None
        expected_rgb = tuple(json.loads(expected_color_rgb_json)) if expected_color_rgb_json else None
        color_roi = tuple(json.loads(color_roi_xyxy_json)) if color_roi_xyxy_json else None
    except Exception as exc:
        raise HTTPException(status_code=422, detail="detections and color fields must be valid JSON") from exc
    image_bytes = await image.read(20 * 1024 * 1024 + 1)
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")
    metadata = FrameAnalysis(
        source_id=source_id,
        pts_seconds=pts_seconds,
        detections=detections,
        width=width,
        height=height,
        sample_usable=sample_usable,
        current_recipe_id=current_recipe_id,
        previous_recipe_id=previous_recipe_id,
        vial_index=vial_index,
        is_first_vial_after_change=is_first_vial_after_change,
        component_name=component_name,
        observed_color_rgb=observed_rgb,
        expected_color_rgb=expected_rgb,
        color_roi_xyxy=color_roi,
    )
    result = await run_in_threadpool(run_analysis, image_bytes, metadata, camera_id)
    event = {
        "type": "camera.frame",
        "camera_id": camera_id,
        "frame_id": source_id,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "image": {
            "mime_type": image.content_type,
            "base64": base64.b64encode(image_bytes).decode("ascii"),
            "width": width,
            "height": height,
        },
        "inspection": result,
    }
    frontend_subscribers = await frontend_clients.publish(camera_id, event)
    return {"accepted": True, "camera_id": camera_id, "frame_id": source_id, "frontend_subscribers": frontend_subscribers, "inspection": result}


@app.post("/v1/analyze")
async def analyze(request_json: Annotated[str, Form()], image: Annotated[UploadFile, File()]):
    try:
        request = AnalysisRequest.model_validate_json(request_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="request_json must match the documented analysis contract") from exc
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Image must be JPEG, PNG, or WebP")
    data = await image.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")
    return JSONResponse(run_analysis(data, request.frame, request.stream_id))


@app.post("/v1/analyze-frame")
async def analyze_frame(
    image: Annotated[UploadFile, File()],
    source_id: Annotated[str, Form()],
    pts_seconds: Annotated[float, Form(ge=0)],
    width: Annotated[int, Form(gt=0)],
    height: Annotated[int, Form(gt=0)],
    detections_json: Annotated[str, Form()] = "[]",
    stream_id: Annotated[str, Form()] = "default",
    pool_present: Annotated[bool | None, Form()] = None,
    sample_usable: Annotated[bool | None, Form()] = None,
):
    """Multipart-friendly bridge for browser/camera clients."""
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Image must be JPEG, PNG, or WebP")
    try:
        detections = [Box.model_validate(item) for item in json.loads(detections_json)]
    except Exception as exc:
        raise HTTPException(status_code=422, detail="detections_json must be a JSON array of detections") from exc
    metadata = FrameAnalysis(
        source_id=source_id, pts_seconds=pts_seconds, detections=detections,
        width=width, height=height, pool_present=pool_present, sample_usable=sample_usable,
    )
    data = await image.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit")
    return JSONResponse(run_analysis(data, metadata, stream_id))


@app.post("/v1/streams/{stream_id}/reset")
def reset_stream(stream_id: str):
    history.pop(stream_id, None)
    return {"status": "ok", "stream_id": stream_id, "sequence_state": "reset"}

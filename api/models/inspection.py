from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class OperatorStatus(str, Enum):
    STARTING = "starting"
    PROCESSING = "processing"
    RUNNING_OK = "running_ok"
    WARNING = "warning"
    FAILURE = "failure"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"


class RecipeContext(BaseModel):
    current_recipe_id: str = Field(min_length=1, max_length=80)
    current_recipe_name: str = Field(min_length=1, max_length=120)
    previous_recipe_id: str | None = Field(default=None, max_length=80)
    vial_index: int = Field(default=1, ge=1)
    is_first_vial_after_change: bool = False
    expected_color_name: str | None = Field(default=None, max_length=120)
    expected_color_hex: str | None = Field(
        default=None, pattern=r"^#[0-9a-fA-F]{6}$"
    )
    t100_ml: float | None = Field(default=None, ge=0)
    t200_ml: float | None = Field(default=None, ge=0)
    t300_ml: float | None = Field(default=None, ge=0)
    t400_ml: float | None = Field(default=None, ge=0)
    total_ml: float | None = Field(default=None, ge=0)
    dispense_order: str | None = Field(default=None, max_length=120)
    color_purpose: str | None = Field(default=None, max_length=240)


SourceMode = Literal["live_camera", "images", "recording"]


class RecipeSegmentCreate(BaseModel):
    start_ms: int = Field(default=0, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    recipe: RecipeContext


class RecipeSegment(RecipeSegmentCreate):
    segment_id: UUID
    session_id: UUID
    created_at: datetime


class ReplaceRecipePlanRequest(BaseModel):
    recipes: list[RecipeContext] = Field(min_length=1, max_length=200)


class RecipePlanItem(BaseModel):
    plan_item_id: UUID
    session_id: UUID
    position: int = Field(ge=1)
    recipe: RecipeContext
    created_at: datetime


class CreateSessionRequest(BaseModel):
    camera_id: str = Field(default="fillcam-1", min_length=1, max_length=80)
    station_name: str = Field(default="Fluid Mixing Station 3", min_length=1, max_length=120)
    run_name: str = Field(default="Inspection run", min_length=1, max_length=160)
    source_mode: SourceMode = "live_camera"
    source_name: str | None = Field(default=None, max_length=260)
    recipe: RecipeContext


class InspectionSession(BaseModel):
    session_id: UUID
    camera_id: str
    station_name: str
    run_name: str
    source_mode: SourceMode
    source_name: str | None = None
    recipe: RecipeContext
    status: OperatorStatus
    pipeline_mode: Literal["mock", "model"]
    created_at: datetime
    updated_at: datetime
    frames_processed: int = 0


class FrameImage(BaseModel):
    mime_type: str
    base64: str | None = None
    url: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)


ArtifactKind = Literal["source", "rendered", "mask", "recording"]


class ArtifactRecord(BaseModel):
    artifact_id: UUID
    session_id: UUID
    event_id: UUID | None = None
    kind: ArtifactKind
    url: str
    mime_type: str
    original_filename: str | None = None
    size_bytes: int = Field(ge=0)
    sha256: str
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    source_timestamp_ms: int | None = Field(default=None, ge=0)
    created_at: datetime


class PipelineVisuals(BaseModel):
    """Optional images produced after inference, separate from the source frame."""

    rendered_frame: FrameImage | None = None
    mask: FrameImage | None = None
    status: Literal["available", "not_configured"] = "not_configured"


class PipelineMeasurement(BaseModel):
    label: str
    value: str
    state: Literal["neutral", "good", "warning", "failure"] = "neutral"


class OperatorDecision(BaseModel):
    status: OperatorStatus
    headline: str
    detail: str
    recommended_action: str
    recipe_progress_percent: int = Field(ge=0, le=100)
    model_state: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    measurements: list[PipelineMeasurement] = Field(default_factory=list)


class InspectionEvent(BaseModel):
    type: Literal["inspection.frame"] = "inspection.frame"
    event_id: UUID
    session_id: UUID
    frame_id: str
    frame_number: int = Field(ge=1)
    source_timestamp_ms: int | None = Field(default=None, ge=0)
    original_filename: str | None = None
    recipe: RecipeContext
    created_at: datetime
    pipeline_mode: Literal["mock", "model"]
    image: FrameImage
    visuals: PipelineVisuals = Field(default_factory=PipelineVisuals)
    decision: OperatorDecision
    acknowledged_at: datetime | None = None


class AcknowledgeEventResponse(BaseModel):
    event_id: UUID
    acknowledged_at: datetime


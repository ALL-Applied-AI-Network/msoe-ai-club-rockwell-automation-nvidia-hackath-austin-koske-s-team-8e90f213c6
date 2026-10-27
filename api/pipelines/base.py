from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from models.inspection import InspectionSession, OperatorDecision


@dataclass(slots=True)
class FrameContext:
    frame_id: str
    source_timestamp_ms: int | None
    source_clock: str
    source_id: str
    recipe_matched: bool
    width: int
    height: int


@dataclass(slots=True)
class PipelineResult:
    decision: OperatorDecision
    rendered_frame: bytes | None = None
    rendered_frame_mime_type: str = "image/jpeg"
    mask: bytes | None = None
    mask_mime_type: str = "image/png"


class InspectionPipeline(Protocol):
    mode: str

    async def analyze(
        self,
        image_bytes: bytes,
        session: InspectionSession,
        frame_number: int,
        simulate: str | None = None,
        context: FrameContext | None = None,
    ) -> PipelineResult: ...


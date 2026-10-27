from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID, uuid4

from models.inspection import ArtifactKind, ArtifactRecord


@dataclass(slots=True)
class StoredArtifact:
    record: ArtifactRecord
    relative_path: str


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(value: str | None, fallback: str) -> str:
        name = Path(value or fallback).name
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
        return cleaned[:180] or fallback

    def _target(self, session_id: UUID, kind: ArtifactKind, filename: str | None) -> tuple[UUID, Path]:
        artifact_id = uuid4()
        safe = self._safe_name(filename, f"{kind}.bin")
        target = self.root / str(session_id) / kind / f"{artifact_id}-{safe}"
        target.parent.mkdir(parents=True, exist_ok=True)
        return artifact_id, target

    async def save_bytes(
        self,
        *,
        session_id: UUID,
        event_id: UUID | None,
        kind: ArtifactKind,
        content: bytes,
        mime_type: str,
        filename: str | None,
        width: int | None = None,
        height: int | None = None,
        source_timestamp_ms: int | None = None,
    ) -> StoredArtifact:
        artifact_id, target = self._target(session_id, kind, filename)
        target.write_bytes(content)
        return self._record(
            artifact_id, target, session_id, event_id, kind, mime_type, filename,
            len(content), hashlib.sha256(content).hexdigest(), width, height,
            source_timestamp_ms,
        )

    async def save_stream(
        self,
        *,
        session_id: UUID,
        kind: ArtifactKind,
        chunks: AsyncIterator[bytes],
        mime_type: str,
        filename: str,
        max_bytes: int,
    ) -> StoredArtifact:
        artifact_id, target = self._target(session_id, kind, filename)
        digest = hashlib.sha256()
        size = 0
        try:
            with target.open("wb") as output:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("Recording exceeds configured upload limit")
                    digest.update(chunk)
                    output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return self._record(
            artifact_id, target, session_id, None, kind, mime_type, filename,
            size, digest.hexdigest(), None, None, None,
        )

    def _record(
        self,
        artifact_id: UUID,
        target: Path,
        session_id: UUID,
        event_id: UUID | None,
        kind: ArtifactKind,
        mime_type: str,
        filename: str | None,
        size: int,
        digest: str,
        width: int | None,
        height: int | None,
        source_timestamp_ms: int | None,
    ) -> StoredArtifact:
        relative = target.relative_to(self.root).as_posix()
        record = ArtifactRecord(
            artifact_id=artifact_id,
            session_id=session_id,
            event_id=event_id,
            kind=kind,
            url=f"/api/v1/artifacts/{artifact_id}",
            mime_type=mime_type,
            original_filename=filename,
            size_bytes=size,
            sha256=digest,
            width=width,
            height=height,
            source_timestamp_ms=source_timestamp_ms,
            created_at=datetime.now(timezone.utc),
        )
        return StoredArtifact(record=record, relative_path=relative)

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if self.root not in candidate.parents:
            raise ValueError("Artifact path escapes storage root")
        return candidate

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from models.inspection import (
    ArtifactRecord,
    CreateSessionRequest,
    FrameImage,
    InspectionEvent,
    InspectionSession,
    OperatorDecision,
    OperatorStatus,
    PipelineVisuals,
    RecipeContext,
    RecipeSegment,
    RecipeSegmentCreate,
)
from services.artifacts import StoredArtifact


def _iso(value: datetime) -> str:
    return value.isoformat()


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteInspectionBridge:
    """Durable metadata store. Binary media remains in ArtifactStore."""

    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = asyncio.Lock()
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspection_runs (
                session_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                station_name TEXT NOT NULL,
                run_name TEXT NOT NULL,
                source_mode TEXT NOT NULL,
                source_name TEXT,
                status TEXT NOT NULL,
                pipeline_mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                frames_processed INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS recipe_segments (
                segment_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES inspection_runs(session_id) ON DELETE CASCADE,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER,
                recipe_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_recipe_segments_run_time
                ON recipe_segments(session_id, start_ms);
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES inspection_runs(session_id) ON DELETE CASCADE,
                event_id TEXT,
                kind TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                original_filename TEXT,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                source_timestamp_ms INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(session_id, kind);
            CREATE TABLE IF NOT EXISTS inspection_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES inspection_runs(session_id) ON DELETE CASCADE,
                frame_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                source_timestamp_ms INTEGER,
                original_filename TEXT,
                created_at TEXT NOT NULL,
                pipeline_mode TEXT NOT NULL,
                recipe_json TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                acknowledged_at TEXT,
                source_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                rendered_artifact_id TEXT REFERENCES artifacts(artifact_id),
                mask_artifact_id TEXT REFERENCES artifacts(artifact_id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_run_number
                ON inspection_events(session_id, frame_number DESC);
            """
        )
        self._connection.commit()

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    async def create_session(
        self, request: CreateSessionRequest, pipeline_mode: str
    ) -> InspectionSession:
        session_id = uuid4()
        now = datetime.now(timezone.utc)
        async with self._lock:
            self._connection.execute(
                """INSERT INTO inspection_runs
                (session_id, camera_id, station_name, run_name, source_mode, source_name,
                 status, pipeline_mode, created_at, updated_at, frames_processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    str(session_id), request.camera_id, request.station_name,
                    request.run_name, request.source_mode, request.source_name,
                    OperatorStatus.STARTING.value, pipeline_mode, _iso(now), _iso(now),
                ),
            )
            self._insert_segment(
                session_id,
                RecipeSegmentCreate(start_ms=0, recipe=request.recipe),
                now,
            )
            self._connection.commit()
        return InspectionSession(
            session_id=session_id,
            camera_id=request.camera_id,
            station_name=request.station_name,
            run_name=request.run_name,
            source_mode=request.source_mode,
            source_name=request.source_name,
            recipe=request.recipe,
            status=OperatorStatus.STARTING,
            pipeline_mode=pipeline_mode,
            created_at=now,
            updated_at=now,
        )

    def _insert_segment(
        self,
        session_id: UUID,
        request: RecipeSegmentCreate,
        created_at: datetime | None = None,
    ) -> RecipeSegment:
        segment = RecipeSegment(
            segment_id=uuid4(),
            session_id=session_id,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
            recipe=request.recipe,
            created_at=created_at or datetime.now(timezone.utc),
        )
        self._connection.execute(
            """INSERT INTO recipe_segments
            (segment_id, session_id, start_ms, end_ms, recipe_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(segment.segment_id), str(session_id), segment.start_ms,
                segment.end_ms, segment.recipe.model_dump_json(), _iso(segment.created_at),
            ),
        )
        return segment

    async def add_recipe_segment(
        self, session_id: UUID, request: RecipeSegmentCreate
    ) -> RecipeSegment | None:
        async with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM inspection_runs WHERE session_id = ?", (str(session_id),)
            ).fetchone()
            if not exists:
                return None
            segment = self._insert_segment(session_id, request)
            self._connection.execute(
                "UPDATE inspection_runs SET updated_at = ? WHERE session_id = ?",
                (_iso(segment.created_at), str(session_id)),
            )
            self._connection.commit()
        return segment

    async def list_recipe_segments(self, session_id: UUID) -> list[RecipeSegment] | None:
        async with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM inspection_runs WHERE session_id = ?", (str(session_id),)
            ).fetchone()
            if not exists:
                return None
            rows = self._connection.execute(
                "SELECT * FROM recipe_segments WHERE session_id = ? ORDER BY start_ms",
                (str(session_id),),
            ).fetchall()
        return [self._segment(row) for row in rows]

    async def resolve_recipe(
        self, session_id: UUID, source_timestamp_ms: int | None
    ) -> RecipeContext | None:
        async with self._lock:
            if source_timestamp_ms is None:
                row = self._connection.execute(
                    """SELECT recipe_json FROM recipe_segments
                    WHERE session_id = ? ORDER BY start_ms DESC, created_at DESC LIMIT 1""",
                    (str(session_id),),
                ).fetchone()
            else:
                row = self._connection.execute(
                    """SELECT recipe_json FROM recipe_segments
                    WHERE session_id = ? AND start_ms <= ?
                      AND (end_ms IS NULL OR end_ms >= ?)
                    ORDER BY start_ms DESC, created_at DESC LIMIT 1""",
                    (str(session_id), source_timestamp_ms, source_timestamp_ms),
                ).fetchone()
                if row is None:
                    row = self._connection.execute(
                        """SELECT recipe_json FROM recipe_segments
                        WHERE session_id = ? AND start_ms <= ?
                        ORDER BY start_ms DESC, created_at DESC LIMIT 1""",
                        (str(session_id), source_timestamp_ms),
                    ).fetchone()
            return RecipeContext.model_validate_json(row["recipe_json"]) if row else None

    async def get_session(self, session_id: UUID) -> InspectionSession | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM inspection_runs WHERE session_id = ?", (str(session_id),)
            ).fetchone()
            if row is None:
                return None
            recipe_row = self._connection.execute(
                """SELECT recipe_json FROM recipe_segments
                WHERE session_id = ? ORDER BY start_ms DESC, created_at DESC LIMIT 1""",
                (str(session_id),),
            ).fetchone()
        return self._session(row, RecipeContext.model_validate_json(recipe_row["recipe_json"]))

    async def list_sessions(self, limit: int = 100) -> list[InspectionSession]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM inspection_runs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            result: list[InspectionSession] = []
            for row in rows:
                recipe_row = self._connection.execute(
                    """SELECT recipe_json FROM recipe_segments
                    WHERE session_id = ? ORDER BY start_ms DESC, created_at DESC LIMIT 1""",
                    (row["session_id"],),
                ).fetchone()
                result.append(
                    self._session(row, RecipeContext.model_validate_json(recipe_row["recipe_json"]))
                )
        return result

    async def set_source_name(self, session_id: UUID, source_name: str) -> None:
        async with self._lock:
            self._connection.execute(
                """UPDATE inspection_runs SET source_name = ?
                WHERE session_id = ? AND (source_name IS NULL OR source_name = '')""",
                (source_name[:260], str(session_id)),
            )
            self._connection.commit()

    def _session(self, row: sqlite3.Row, recipe: RecipeContext) -> InspectionSession:
        return InspectionSession(
            session_id=UUID(row["session_id"]),
            camera_id=row["camera_id"],
            station_name=row["station_name"],
            run_name=row["run_name"],
            source_mode=row["source_mode"],
            source_name=row["source_name"],
            recipe=recipe,
            status=row["status"],
            pipeline_mode=row["pipeline_mode"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            frames_processed=row["frames_processed"],
        )

    @staticmethod
    def _segment(row: sqlite3.Row) -> RecipeSegment:
        return RecipeSegment(
            segment_id=UUID(row["segment_id"]),
            session_id=UUID(row["session_id"]),
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            recipe=RecipeContext.model_validate_json(row["recipe_json"]),
            created_at=_dt(row["created_at"]),
        )

    async def add_artifact(self, artifact: StoredArtifact) -> ArtifactRecord:
        record = artifact.record
        async with self._lock:
            self._connection.execute(
                """INSERT INTO artifacts
                (artifact_id, session_id, event_id, kind, relative_path, mime_type,
                 original_filename, size_bytes, sha256, width, height,
                 source_timestamp_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(record.artifact_id), str(record.session_id),
                    str(record.event_id) if record.event_id else None, record.kind,
                    artifact.relative_path, record.mime_type, record.original_filename,
                    record.size_bytes, record.sha256, record.width, record.height,
                    record.source_timestamp_ms, _iso(record.created_at),
                ),
            )
            self._connection.commit()
        return record

    async def get_artifact(self, artifact_id: UUID) -> tuple[ArtifactRecord, str] | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (str(artifact_id),)
            ).fetchone()
        if row is None:
            return None
        return self._artifact(row), row["relative_path"]

    @staticmethod
    def _artifact(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=UUID(row["artifact_id"]),
            session_id=UUID(row["session_id"]),
            event_id=UUID(row["event_id"]) if row["event_id"] else None,
            kind=row["kind"],
            url=f"/api/v1/artifacts/{row['artifact_id']}",
            mime_type=row["mime_type"],
            original_filename=row["original_filename"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            width=row["width"],
            height=row["height"],
            source_timestamp_ms=row["source_timestamp_ms"],
            created_at=_dt(row["created_at"]),
        )

    async def add_event(
        self,
        event: InspectionEvent,
        source_artifact_id: UUID,
        rendered_artifact_id: UUID | None,
        mask_artifact_id: UUID | None,
    ) -> InspectionSession:
        async with self._lock:
            self._connection.execute(
                """INSERT INTO inspection_events
                (event_id, session_id, frame_id, frame_number, source_timestamp_ms,
                 original_filename, created_at, pipeline_mode, recipe_json,
                 decision_json, acknowledged_at, source_artifact_id,
                 rendered_artifact_id, mask_artifact_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(event.event_id), str(event.session_id), event.frame_id,
                    event.frame_number, event.source_timestamp_ms,
                    event.original_filename, _iso(event.created_at), event.pipeline_mode,
                    event.recipe.model_dump_json(), event.decision.model_dump_json(),
                    None, str(source_artifact_id),
                    str(rendered_artifact_id) if rendered_artifact_id else None,
                    str(mask_artifact_id) if mask_artifact_id else None,
                ),
            )
            self._connection.execute(
                """UPDATE inspection_runs SET status = ?, updated_at = ?, frames_processed = ?
                WHERE session_id = ?""",
                (
                    event.decision.status.value, _iso(event.created_at),
                    event.frame_number, str(event.session_id),
                ),
            )
            self._connection.commit()
        session = await self.get_session(event.session_id)
        if session is None:
            raise RuntimeError("Run disappeared while saving event")
        return session

    async def list_events(
        self, session_id: UUID, limit: int = 25
    ) -> list[InspectionEvent] | None:
        async with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM inspection_runs WHERE session_id = ?", (str(session_id),)
            ).fetchone()
            if not exists:
                return None
            rows = self._connection.execute(
                """SELECT e.*,
                    s.mime_type AS s_mime, s.width AS s_width, s.height AS s_height,
                    r.mime_type AS r_mime, r.width AS r_width, r.height AS r_height,
                    m.mime_type AS m_mime, m.width AS m_width, m.height AS m_height
                FROM inspection_events e
                JOIN artifacts s ON s.artifact_id = e.source_artifact_id
                LEFT JOIN artifacts r ON r.artifact_id = e.rendered_artifact_id
                LEFT JOIN artifacts m ON m.artifact_id = e.mask_artifact_id
                WHERE e.session_id = ? ORDER BY e.frame_number DESC LIMIT ?""",
                (str(session_id), limit),
            ).fetchall()
        return [self._event(row) for row in rows]

    @staticmethod
    def _image(row: sqlite3.Row, prefix: str, artifact_column: str) -> FrameImage | None:
        artifact_id = row[artifact_column]
        if not artifact_id:
            return None
        return FrameImage(
            mime_type=row[f"{prefix}_mime"],
            url=f"/api/v1/artifacts/{artifact_id}",
            width=row[f"{prefix}_width"],
            height=row[f"{prefix}_height"],
        )

    def _event(self, row: sqlite3.Row) -> InspectionEvent:
        rendered = self._image(row, "r", "rendered_artifact_id")
        mask = self._image(row, "m", "mask_artifact_id")
        return InspectionEvent(
            event_id=UUID(row["event_id"]),
            session_id=UUID(row["session_id"]),
            frame_id=row["frame_id"],
            frame_number=row["frame_number"],
            source_timestamp_ms=row["source_timestamp_ms"],
            original_filename=row["original_filename"],
            created_at=_dt(row["created_at"]),
            pipeline_mode=row["pipeline_mode"],
            recipe=RecipeContext.model_validate_json(row["recipe_json"]),
            image=self._image(row, "s", "source_artifact_id"),
            visuals=PipelineVisuals(
                rendered_frame=rendered,
                mask=mask,
                status="available" if rendered or mask else "not_configured",
            ),
            decision=OperatorDecision.model_validate_json(row["decision_json"]),
            acknowledged_at=_dt(row["acknowledged_at"]) if row["acknowledged_at"] else None,
        )

    async def acknowledge_event(
        self, session_id: UUID, event_id: UUID
    ) -> InspectionEvent | None:
        now = datetime.now(timezone.utc)
        async with self._lock:
            cursor = self._connection.execute(
                """UPDATE inspection_events SET acknowledged_at = ?
                WHERE session_id = ? AND event_id = ?""",
                (_iso(now), str(session_id), str(event_id)),
            )
            self._connection.commit()
        if cursor.rowcount == 0:
            return None
        events = await self.list_events(session_id, 100)
        return next((event for event in events or [] if event.event_id == event_id), None)

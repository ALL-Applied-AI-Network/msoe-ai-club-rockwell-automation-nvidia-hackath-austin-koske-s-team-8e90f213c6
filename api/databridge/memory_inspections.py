from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from uuid import UUID, uuid4

from models.inspection import (
    CreateSessionRequest,
    InspectionEvent,
    InspectionSession,
    OperatorStatus,
)


class MemoryInspectionBridge:
    """Process-local session/event store.

    This keeps the first integration slice dependency-free. It deliberately sits
    behind a databridge so SQLite can replace it without changing HTTP or UI code.
    """

    def __init__(self, max_events_per_session: int = 100):
        self._sessions: dict[UUID, InspectionSession] = {}
        self._events: dict[UUID, deque[InspectionEvent]] = defaultdict(
            lambda: deque(maxlen=max_events_per_session)
        )
        self._lock = asyncio.Lock()

    async def create_session(
        self, request: CreateSessionRequest, pipeline_mode: str
    ) -> InspectionSession:
        now = datetime.now(timezone.utc)
        session = InspectionSession(
            session_id=uuid4(),
            camera_id=request.camera_id,
            station_name=request.station_name,
            recipe=request.recipe,
            status=OperatorStatus.STARTING,
            pipeline_mode=pipeline_mode,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            self._sessions[session.session_id] = session
        return session.model_copy(deep=True)

    async def get_session(self, session_id: UUID) -> InspectionSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            return session.model_copy(deep=True) if session else None

    async def add_event(self, event: InspectionEvent) -> InspectionSession:
        async with self._lock:
            session = self._sessions[event.session_id]
            updated = session.model_copy(
                update={
                    "status": event.decision.status,
                    "updated_at": event.created_at,
                    "frames_processed": event.frame_number,
                }
            )
            self._sessions[event.session_id] = updated
            self._events[event.session_id].appendleft(event)
            return updated.model_copy(deep=True)

    async def list_events(
        self, session_id: UUID, limit: int = 25
    ) -> list[InspectionEvent] | None:
        async with self._lock:
            if session_id not in self._sessions:
                return None
            return [event.model_copy(deep=True) for event in list(self._events[session_id])[:limit]]

    async def acknowledge_event(
        self, session_id: UUID, event_id: UUID
    ) -> InspectionEvent | None:
        now = datetime.now(timezone.utc)
        async with self._lock:
            events = self._events.get(session_id)
            if not events:
                return None
            for index, event in enumerate(events):
                if event.event_id == event_id:
                    acknowledged = event.model_copy(update={"acknowledged_at": now})
                    events[index] = acknowledged
                    return acknowledged.model_copy(deep=True)
        return None


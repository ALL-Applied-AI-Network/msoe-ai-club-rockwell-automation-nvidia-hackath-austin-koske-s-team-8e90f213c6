from datetime import datetime, timezone
from threading import RLock
from uuid import UUID, uuid4

from models.widget import Widget


class MemoryWidgetBridge:
    """Disposable example data, shared within one API process only."""

    def __init__(self):
        self._rows: dict[UUID, Widget] = {}
        self._lock = RLock()

    async def list(self, limit: int, offset: int) -> list[Widget]:
        with self._lock:
            rows = sorted(self._rows.values(), key=lambda row: row.created_at, reverse=True)
            return [row.model_copy(deep=True) for row in rows[offset:offset + limit]]

    async def get(self, widget_id: UUID) -> Widget | None:
        with self._lock:
            row = self._rows.get(widget_id)
            return row.model_copy(deep=True) if row else None

    async def create(self, name: str, description: str | None) -> Widget:
        now = datetime.now(timezone.utc)
        row = Widget(widget_id=uuid4(), name=name, description=description,
                     created_at=now, updated_at=now)
        with self._lock:
            self._rows[row.widget_id] = row
        return row.model_copy(deep=True)

    async def update(self, widget_id: UUID, changes: dict) -> Widget | None:
        with self._lock:
            row = self._rows.get(widget_id)
            if row is None:
                return None
            updated = row.model_copy(update={**changes, "updated_at": datetime.now(timezone.utc)})
            self._rows[widget_id] = updated
            return updated.model_copy(deep=True)

    async def delete(self, widget_id: UUID) -> bool:
        with self._lock:
            return self._rows.pop(widget_id, None) is not None

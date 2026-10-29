from __future__ import annotations

import asyncio
from collections import defaultdict
from uuid import UUID

from fastapi import WebSocket

from models.inspection import InspectionEvent


class EventBroker:
    def __init__(self):
        self._clients: dict[UUID, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, session_id: UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients[session_id].add(websocket)

    async def disconnect(self, session_id: UUID, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients[session_id].discard(websocket)
            if not self._clients[session_id]:
                self._clients.pop(session_id, None)

    async def publish(self, event: InspectionEvent) -> int:
        async with self._lock:
            clients = list(self._clients.get(event.session_id, set()))
        stale: list[WebSocket] = []
        for client in clients:
            try:
                await asyncio.wait_for(client.send_json(event.model_dump(mode="json")), timeout=1)
            except Exception:
                stale.append(client)
        for client in stale:
            await self.disconnect(event.session_id, client)
        return len(clients) - len(stale)

    async def connected_count(self) -> int:
        async with self._lock:
            return sum(len(clients) for clients in self._clients.values())


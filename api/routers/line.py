import asyncio
import json
import threading
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from dependencies import get_inspection_service, get_line_runs
from services.line import MAX_SESSIONS, SPEEDS, SessionClock

router = APIRouter(prefix="/line", tags=["line"])
_active = {"sessions": 0}


@router.get("/runs")
async def list_runs(runs=Depends(get_line_runs)):
    return runs.list()


async def recipe_plan(service, order: str):
    """The recipe plan of inspection run `order` as a list of recipe dicts."""
    try:
        items = await service.databridge.list_recipe_plan(UUID(order))
    except ValueError as error:
        raise KeyError(order) from error
    if items is None:
        raise KeyError(order)
    return [item.recipe.model_dump() for item in items]


@router.get("/orders")
async def recipe_orders(service=Depends(get_inspection_service)):
    """Inspection runs that have a recipe plan (the order copied from CSI's machine), newest first."""
    out = []
    for session in await service.list_sessions(50):
        plan = await service.databridge.list_recipe_plan(session.session_id) or []
        if plan:
            out.append({"session_id": str(session.session_id), "run_name": session.run_name,
                        "created_at": session.created_at, "vials": len(plan),
                        "recipes": [item.recipe.current_recipe_name for item in plan][:40]})
    return out


@router.get("/runs/{run_id}/events")
async def review_events(run_id: str, order: str | None = None, runs=Depends(get_line_runs),
                        service=Depends(get_inspection_service)):
    """All events of a recorded run at once, for scrubbing. Same engine and events as live.
    With order=<inspection run id>, the recipe order comes from that run's recipe plan."""
    try:
        plan = await recipe_plan(service, order) if order else None
        return {"events": await asyncio.to_thread(runs.review, run_id, plan)}
    except KeyError as error:
        raise HTTPException(404, "Line run not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.get("/runs/{run_id}/video/{camera}")
async def video(run_id: str, camera: str, runs=Depends(get_line_runs)):
    if camera not in ("fill", "tray"):
        raise HTTPException(404, "Unknown camera")
    try:
        return FileResponse(runs.video(run_id, camera), media_type="video/mp4")
    except KeyError as error:
        raise HTTPException(404, "Video not found") from error


@router.websocket("/runs/{run_id}/live")
async def live(websocket: WebSocket, run_id: str, runs=Depends(get_line_runs),
               service=Depends(get_inspection_service)):
    """Paced live session. Query: source=recorded|video, speed, start (seconds, recorded only),
    order (an inspection run id whose recipe plan is followed, re-read while running).
    The client may send {"speed": N} or {"paused": bool}."""
    await websocket.accept()
    params = websocket.query_params
    try:
        manifest = runs.manifest(run_id)
        source = params.get("source", "recorded")
        speed = float(params.get("speed", 1))
        start = float(params.get("start", 0))
        if source not in runs.sources(manifest):
            raise ValueError(f"Source {source} is not available for this run")
        if speed not in SPEEDS or start < 0 or (source == "video" and start):
            raise ValueError("Unsupported speed or start")
        if _active["sessions"] >= MAX_SESSIONS:
            raise ValueError("Too many live sessions")
        order_id = params.get("order") or None
        if order_id:
            await recipe_plan(service, order_id)
    except (KeyError, ValueError) as error:
        await websocket.send_json({"type": "error", "message": str(error) if isinstance(error, ValueError) else "Line run not found"})
        await websocket.close(code=4400)
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    clock = SessionClock(start, speed)

    def send(message):
        loop.call_soon_threadsafe(queue.put_nowait, message)

    def order():
        return asyncio.run_coroutine_threadsafe(recipe_plan(service, order_id), loop).result(timeout=5)

    def worker():
        try:
            runs.session(run_id, source, clock, send, stop, order if order_id else None)
        except Exception as error:  # report to the client; the session ends either way
            send({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            send(None)

    async def pump():
        while (message := await queue.get()) is not None:
            await websocket.send_text(json.dumps(message, default=str))

    async def control():
        while True:
            data = await websocket.receive_json()
            speed = data.get("speed")
            clock.set(speed=speed if speed in SPEEDS else None,
                      paused=data["paused"] if isinstance(data.get("paused"), bool) else None)

    _active["sessions"] += 1
    thread = threading.Thread(target=worker, name=f"line-{run_id}", daemon=True)
    thread.start()
    pumping, controlling = asyncio.create_task(pump()), asyncio.create_task(control())
    try:
        await asyncio.wait({pumping, controlling}, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        controlling.cancel()
        if not pumping.done():
            pumping.cancel()
        _active["sessions"] -= 1
        try:
            await websocket.close()
        except RuntimeError:
            pass

import asyncio
import json
import threading

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from dependencies import get_line_runs
from services.line import MAX_SESSIONS, SPEEDS, SessionClock

router = APIRouter(prefix="/line", tags=["line"])
_active = {"sessions": 0}


@router.get("/runs")
async def list_runs(runs=Depends(get_line_runs)):
    return runs.list()


@router.get("/runs/{run_id}/events")
async def review_events(run_id: str, runs=Depends(get_line_runs)):
    """All events of a recorded run at once, for scrubbing. Same engine and events as live."""
    try:
        return {"events": await asyncio.to_thread(runs.review, run_id)}
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
async def live(websocket: WebSocket, run_id: str, runs=Depends(get_line_runs)):
    """Paced live session. Query: source=recorded|video, speed, start (seconds, recorded only).
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

    def worker():
        try:
            runs.session(run_id, source, clock, send, stop)
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

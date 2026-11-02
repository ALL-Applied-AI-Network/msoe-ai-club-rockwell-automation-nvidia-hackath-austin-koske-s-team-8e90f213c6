import asyncio
import json
import threading
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from dependencies import get_explainer, get_inspection_service, get_line_runs
from services.explain import packet
from services.line import LIVE, MAX_SESSIONS, MOCK, SPEEDS, SessionClock

router = APIRouter(prefix="/line", tags=["line"])
_active = {"sessions": 0}


@router.get("/runs")
async def list_runs(runs=Depends(get_line_runs)):
    return runs.list()


class NewRecipe(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    t100_ml: float = Field(default=0, ge=0, le=1000)
    t200_ml: float = Field(default=0, ge=0, le=1000)
    t400_ml: float = Field(default=0, ge=0, le=1000)


class Capture(BaseModel):
    recipe_id: str | None = None


@router.get("/recipes")
async def recipes(runs=Depends(get_line_runs)):
    """The live station's recipe library, with recipes created at the line, and CREATE mode."""
    try:
        return runs.recipes()
    except KeyError as error:
        raise HTTPException(404, "No live station configured") from error


@router.post("/recipes", status_code=201)
async def create_recipe(body: NewRecipe, runs=Depends(get_line_runs)):
    try:
        return runs.add_recipe(body.name.strip(), {"T100": body.t100_ml, "T200": body.t200_ml, "T400": body.t400_ml})
    except KeyError as error:
        raise HTTPException(404, "No live station configured") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@router.put("/recipes/capture")
async def set_capture(body: Capture, runs=Depends(get_line_runs)):
    """CREATE mode: the next vial a camera session decides sets this recipe's reference (null: off)."""
    try:
        return {"capture": runs.arm(body.recipe_id)}
    except KeyError as error:
        raise HTTPException(404, "Unknown recipe or no live station") from error


class OrderCheck(BaseModel):
    recipes: list[dict] = Field(min_length=1, max_length=500)
    f: float | None = Field(default=None, ge=0, le=0.95)
    model: Literal["mix", "lag"] | None = None
    max_shift: int | None = Field(default=None, ge=-1, le=20)     # -1: move runs freely
    line_lab: list[float] | None = Field(default=None, min_length=3, max_length=3)


@router.get("/recipe-order/settings")
async def recipe_order_settings(runs=Depends(get_line_runs)):
    """The station's line model for the recipe-order planner (f is assumed, not measured)."""
    return runs.order_settings()


@router.post("/recipe-order/suggest")
async def suggest_recipe_order(body: OrderCheck, runs=Depends(get_line_runs)):
    """Forecast an order (recipe-plan recipes, one per vial) and propose a run order with flushes.
    Advice only: the order changes only if the operator saves it."""
    try:
        return await asyncio.to_thread(runs.suggest_order, body.recipes, body.f, body.model,
                                       body.max_shift, body.line_lab)
    except KeyError as error:
        raise HTTPException(404, "No recipe library configured") from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


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
    if run_id == LIVE:
        # The camera session running now (or the last one): what it has decided so far.
        return runs.live_events()
    try:
        plan = await recipe_plan(service, order) if order else None
        return {"events": await asyncio.to_thread(runs.review, run_id, plan)}
    except KeyError as error:
        raise HTTPException(404, "Line run not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


class ExplainRequest(BaseModel):
    run_id: str
    vial: dict
    actions: list[dict] = []
    decided: list[dict] = []
    next_recipe: str | None = None


@router.get("/explain")
async def explain_status(explainer=Depends(get_explainer)):
    """Which model explains vials, and whether this server can call it (a key is configured)."""
    return explainer.available()


@router.post("/explain")
async def explain(body: ExplainRequest, runs=Depends(get_line_runs), explainer=Depends(get_explainer)):
    """Advisory AI explanation of one decided vial; never changes its verdict or actions."""
    if "seq" not in body.vial or "recipe_id" not in body.vial:
        raise HTTPException(400, "vial needs seq and recipe_id")
    try:
        library = runs.manifest(body.run_id).get("library")
    except KeyError:
        library = None
    recorded = Path(library).parent / "explanations.json" if library else None
    evidence = packet(body.vial, body.actions, body.decided, body.next_recipe)
    try:
        return {**await asyncio.to_thread(explainer.explain, evidence, recorded), "packet": evidence}
    except LookupError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/runs/{run_id}/video/{camera}")
async def video(run_id: str, camera: str, runs=Depends(get_line_runs)):
    if camera not in ("fill", "tray"):
        raise HTTPException(404, "Unknown camera")
    try:
        return FileResponse(runs.video(run_id, camera), media_type="video/mp4")
    except KeyError as error:
        raise HTTPException(404, "Video not found") from error


@router.get("/runs/{run_id}/frame/{camera}")
async def frame(run_id: str, camera: str, t: float, runs=Depends(get_line_runs)):
    """One JPEG frame of a run's recorded video at time t (seconds), e.g. a vial at the end of its fill."""
    if camera not in ("fill", "tray"):
        raise HTTPException(404, "Unknown camera")
    try:
        jpeg = await asyncio.to_thread(runs.frame, run_id, camera, t)
    except KeyError as error:
        raise HTTPException(404, "Frame not found") from error
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@router.get("/preview/{key}/{camera}")
async def preview(key: str, camera: str, runs=Depends(get_line_runs)):
    """MJPEG stream of a live session's camera (the newest frame, as it arrives)."""
    if key not in runs.previews or camera not in ("fill", "fill_mask", "tray", "tray_detection"):
        raise HTTPException(404, "No such live preview")

    async def frames():
        last = None
        while key in runs.previews:
            jpeg = runs.previews.get(key, {}).get(camera)
            if jpeg is not None and jpeg is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.03)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.websocket("/runs/{run_id}/live")
async def live(websocket: WebSocket, run_id: str, runs=Depends(get_line_runs),
               service=Depends(get_inspection_service)):
    """Live session. Query: source=recorded|video|camera, speed, start (seconds), order (an
    inspection run id whose recipe plan is followed, re-read while running). Source camera is
    run "live": the station cameras in real time, or with simulate=<run id> that run's videos
    played at real time through the camera input. The client may send {"speed": N} or
    {"paused": bool} (recorded and video sources), and {"stop": true} to end the session and
    receive "saved" (camera sessions) and "end"."""
    await websocket.accept()
    params = websocket.query_params
    try:
        manifest = runs.manifest(run_id)
        source = params.get("source", "recorded")
        speed = float(params.get("speed", 1))
        start = float(params.get("start", 0))
        simulate = params.get("simulate") or None
        if source == "camera":
            if run_id != LIVE:
                raise ValueError("Cameras are the run 'live'")
            if simulate and simulate != MOCK:
                runs.manifest(simulate)
            speed = 1
        elif source not in runs.sources(manifest):
            raise ValueError(f"Source {source} is not available for this run")
        if speed not in SPEEDS or start < 0 or (source in ("video", "camera") and start and not simulate):
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
            if source == "camera":
                runs.camera_session(clock, send, stop, order if order_id else None, simulate, start)
            else:
                runs.session(run_id, source, clock, send, stop, order if order_id else None)
        except Exception as error:  # report to the client; the session ends either way
            send({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            send(None)

    async def pump():
        try:
            while (message := await queue.get()) is not None:
                await websocket.send_text(json.dumps(message, default=str))
        except (WebSocketDisconnect, RuntimeError):
            stop.set()      # the client went away; the session stops and still saves

    async def control():
        while True:
            data = await websocket.receive_json()
            if data.get("stop"):
                stop.set()      # finish: the session saves, sends "saved"/"end", then closes
                continue
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
        except (RuntimeError, WebSocketDisconnect):
            pass        # already closed by the client

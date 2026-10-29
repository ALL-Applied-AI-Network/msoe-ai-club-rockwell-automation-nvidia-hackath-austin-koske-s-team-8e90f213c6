from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dependencies import get_inspection_service
from routers.inspection import artifact_router, router as inspections
from routers.investigation import router as investigation
from routers.line import router as line
from settings import config

@asynccontextmanager
async def lifespan(_: FastAPI):
    service = get_inspection_service()
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(
    title="Linewatch Inspection API", version="0.2.0",
    description="Operator-in-the-loop camera and model-pipeline integration API.",
    docs_url=None if config.environment.name == "production" else "/docs",
    redoc_url=None if config.environment.name == "production" else "/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)
app.include_router(inspections, prefix="/api/v1")
app.include_router(artifact_router, prefix="/api/v1")
app.include_router(investigation, prefix="/api/v1")
app.include_router(line, prefix="/api/v1")


@app.get("/health")
async def health_check(service=Depends(get_inspection_service)):
    return {
        "status": "healthy",
        "service": "linewatch-inspection-api",
        "pipeline_mode": service.pipeline.mode,
        "pipeline_status": getattr(service.pipeline, "status", "ready"),
        "connected_frontends": await service.broker.connected_count(),
        "queue_depth": service.queue_depth,
        "persistence": "sqlite+filesystem",
    }


if config.client_dist is not None:
    # Same-origin UI for native GB10 runs; Docker uses Nginx instead.
    client_root = config.client_dist.resolve()
    if not (client_root / "index.html").is_file():
        raise RuntimeError(f"APP__CLIENT_DIST has no index.html: {client_root}")

    @app.get("/{path:path}", include_in_schema=False)
    async def client_app(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        candidate = (client_root / path).resolve()
        if path and candidate.is_relative_to(client_root) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(client_root / "index.html")

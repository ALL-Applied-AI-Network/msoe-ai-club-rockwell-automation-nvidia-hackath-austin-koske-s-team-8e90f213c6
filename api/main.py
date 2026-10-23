from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.widget import router as widgets
from settings import config

app = FastAPI(
    title="Team App API", version="0.1.0",
    description="Local frontend/API starter with disposable process-local example data.",
    docs_url=None if config.environment.name == "production" else "/docs",
    redoc_url=None if config.environment.name == "production" else "/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)
app.include_router(widgets, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

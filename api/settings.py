from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentConfig(BaseModel):
    name: str = "dev"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP__", env_nested_delimiter="__", extra="ignore",
        env_file=Path(__file__).parent / ".env",
    )
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data_dir: Path = Field(default=Path(__file__).parent / "data")
    model_workers: int = Field(default=1, ge=1, le=8)
    pipeline_mode: Literal["model", "mock"] = "model"
    model_assets: Path | None = None
    reference_image: Path | None = None
    device: str = "cpu"
    # Native (non-Docker) deployments serve the prebuilt React UI from this folder.
    client_dist: Path | None = None


config = AppConfig()

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentConfig(BaseModel):
    name: str = "dev"


class AppConfig(BaseSettings):
    # Keep local runtime selection stable regardless of the server's launch directory.
    model_config = SettingsConfigDict(
        env_prefix="APP__", env_nested_delimiter="__", extra="ignore",
        env_file=Path(__file__).parent / ".env",
    )
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data_dir: Path = Field(default=Path(__file__).parent / "data")
    model_workers: int = Field(default=1, ge=1, le=8)
    pipeline_mode: Literal["mock", "model"] = "mock"
    yolo_checkpoint: Path = Field(
        default=Path(__file__).parent.parent / "integration" / "models" / "yolo_best.pt"
    )
    temporal_checkpoint: Path = Field(
        default=Path(__file__).parent.parent / "integration" / "models" / "liquid-best.pt"
    )
    temporal_reference_image: Path | None = None
    temporal_reference_qualified: bool = False


config = AppConfig()

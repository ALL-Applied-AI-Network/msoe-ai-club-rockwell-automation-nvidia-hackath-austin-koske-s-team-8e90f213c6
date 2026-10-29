from pathlib import Path

from databridge.memory_widgets import MemoryWidgetBridge
from databridge.sqlite_inspections import SQLiteInspectionBridge
from pipelines.base import InspectionPipeline
from pipelines.mock import MockInspectionPipeline
from pipelines.temporal import TemporalInspectionPipeline
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker
from services.inspection import InspectionService
from services.widget import WidgetService
from settings import config

_widgets = WidgetService(MemoryWidgetBridge())
_api_root = Path(__file__).resolve().parent
_data_dir = (
    config.data_dir.resolve()
    if config.data_dir.is_absolute()
    else (_api_root / config.data_dir).resolve()
)


def build_inspection_pipeline() -> InspectionPipeline:
    if config.pipeline_mode == "model" and config.model_assets is not None:
        return TemporalInspectionPipeline(config.model_assets, config.reference_image, config.device)
    if config.pipeline_mode == "model":
        from pipelines.temporal_cnn_bridge import TemporalCnnInspectionBridge

        return TemporalCnnInspectionBridge(
            yolo_checkpoint=config.yolo_checkpoint,
            temporal_checkpoint=config.temporal_checkpoint,
            reference_image=config.temporal_reference_image,
            reference_qualified=config.temporal_reference_qualified,
        )
    return MockInspectionPipeline()


_inspection = InspectionService(
    SQLiteInspectionBridge(_data_dir / "linewatch.sqlite3"),
    build_inspection_pipeline(),
    EventBroker(),
    ArtifactStore(_data_dir / "artifacts"),
    worker_count=config.model_workers,
)


def get_widget_service() -> WidgetService:
    return _widgets


def get_inspection_service() -> InspectionService:
    return _inspection

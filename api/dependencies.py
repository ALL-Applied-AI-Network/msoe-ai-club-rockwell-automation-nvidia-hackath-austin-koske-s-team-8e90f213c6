from pathlib import Path

from databridge.sqlite_inspections import SQLiteInspectionBridge
from pipelines.base import InspectionPipeline
from pipelines.mock import MockInspectionPipeline
from pipelines.temporal import TemporalInspectionPipeline
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker
from services.explain import Explainer
from services.inspection import InspectionService
from services.line import LineRuns
from settings import config

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

_line_runs = LineRuns(config.line_runs, config.model_assets, config.device, config.line_station,
                      config.line_recordings or _data_dir / "line-recordings", _data_dir / "site-recipes.json")



def get_inspection_service() -> InspectionService:
    return _inspection


def get_line_runs() -> LineRuns:
    return _line_runs


_explainer = Explainer(config.explain_url, config.explain_model, config.explain_key_file)


def get_explainer() -> Explainer:
    return _explainer

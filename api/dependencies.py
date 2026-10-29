from databridge.memory_widgets import MemoryWidgetBridge
from databridge.sqlite_inspections import SQLiteInspectionBridge
from pipelines.mock import MockInspectionPipeline
from pipelines.temporal import TemporalInspectionPipeline
from services.artifacts import ArtifactStore
from services.event_broker import EventBroker
from services.inspection import InspectionService
from services.line import LineRuns
from services.widget import WidgetService
from settings import config

_widgets = WidgetService(MemoryWidgetBridge())
_data_dir = config.data_dir.resolve()
_inspection = InspectionService(
    SQLiteInspectionBridge(_data_dir / "linewatch.sqlite3"),
    (MockInspectionPipeline() if config.pipeline_mode == "mock" else
     TemporalInspectionPipeline(config.model_assets, config.reference_image, config.device)),
    EventBroker(),
    ArtifactStore(_data_dir / "artifacts"),
    worker_count=config.model_workers,
)

_line_runs = LineRuns(config.line_runs, config.model_assets, config.device)


def get_widget_service() -> WidgetService:
    return _widgets


def get_inspection_service() -> InspectionService:
    return _inspection


def get_line_runs() -> LineRuns:
    return _line_runs

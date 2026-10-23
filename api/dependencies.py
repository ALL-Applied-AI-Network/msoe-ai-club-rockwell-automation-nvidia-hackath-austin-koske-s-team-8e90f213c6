from databridge.memory_widgets import MemoryWidgetBridge
from services.widget import WidgetService

# One process-local example store. Restarting the API clears it.
_widgets = WidgetService(MemoryWidgetBridge())


def get_widget_service() -> WidgetService:
    return _widgets

from fastapi.testclient import TestClient
import pytest

from databridge.memory_widgets import MemoryWidgetBridge
from dependencies import get_widget_service
from main import app
from services.widget import WidgetService


@pytest.fixture
def client():
    service = WidgetService(MemoryWidgetBridge())
    app.dependency_overrides[get_widget_service] = lambda: service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_health_and_validation_without_database_or_credentials(client):
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/api/v1/widgets").json() == []
    assert client.post("/api/v1/widgets", json={"name": ""}).status_code == 422


def test_frontend_resource_round_trip(client):
    created = client.post("/api/v1/widgets", json={"name": "Example", "description": "Demo only"})
    assert created.status_code == 201
    resource = "/api/v1/widgets/" + created.json()["widget_id"]
    assert client.get(resource).json()["name"] == "Example"
    assert len(client.get("/api/v1/widgets").json()) == 1
    updated = client.put(resource, json={"name": "Updated", "description": None})
    assert updated.json()["name"] == "Updated"
    assert updated.json()["description"] is None
    assert client.delete(resource).status_code == 204
    assert client.get(resource).status_code == 404
    assert client.delete(resource).status_code == 404

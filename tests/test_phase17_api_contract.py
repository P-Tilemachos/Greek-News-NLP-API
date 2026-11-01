"""Phase 17: API contract smoke tests without loading ML model weights."""

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api.main import app


@pytest.fixture(scope="module")
def client():
    # TestClient runs the actual ASGI router, not an external deployed server.
    with TestClient(app) as test_client:
        yield test_client


def test_health_smoke(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_openapi_document_is_available(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    body = response.json()
    assert body["openapi"].startswith("3.")
    assert isinstance(body.get("paths"), dict)


@pytest.mark.parametrize(
    "path",
    ["/health", "/classify", "/search", "/ask", "/rag/ask", "/generate"],
)
def test_documented_endpoints_exist_in_schema(client, path):
    paths = client.get("/openapi.json").json()["paths"]
    assert path in paths, f"Documented route missing from OpenAPI schema: {path}"
    method = "get" if path == "/health" else "post"
    assert method in paths[path], f"Expected {method.upper()} for {path}"


@pytest.mark.parametrize("path", ["/classify", "/search", "/ask", "/rag/ask", "/generate"])
def test_empty_json_is_rejected_without_running_inference(client, path):
    # Assumes all documented POST endpoints require at least one request field.
    response = client.post(path, json={})
    assert response.status_code == 422, f"Unexpected status from {path}: {response.status_code}"

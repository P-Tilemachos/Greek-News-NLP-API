"""Phase 17.2 retrieval API regression checks with isolated mocked dependencies."""

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main


class SearchSpy:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def search(self, query, k=5):
        self.calls.append((query, k))
        return [{"id": self.name, "score": 0.8, "label": "news", "snippet": "Δείγμα"}]


@pytest.mark.parametrize("mode", ["dense", "hybrid", "rerank"])
def test_retrieval_routes_only_selected_mode(monkeypatch, mode):
    engines = {key: SearchSpy(key) for key in ("dense", "hybrid", "rerank")}
    monkeypatch.setitem(main.app.dependency_overrides, main.get_retriever, lambda: engines["dense"])
    monkeypatch.setattr(main, "get_hybrid_retriever", lambda: engines["hybrid"])
    monkeypatch.setattr(main, "get_reranking_retriever", lambda: engines["rerank"])

    response = TestClient(main.app).post(
        "/search", json={"query": "εκλογικά αποτελέσματα", "k": 2, "mode": mode}
    )
    assert response.status_code == 200
    assert response.json()["mode"] == mode
    assert response.json()["results"][0]["id"] == mode
    for key, engine in engines.items():
        assert engine.calls == ([("εκλογικά αποτελέσματα", 2)] if key == mode else [])


def test_retrieval_rejects_unrecognized_mode_before_search(monkeypatch):
    engine = SearchSpy("dense")
    monkeypatch.setitem(main.app.dependency_overrides, main.get_retriever, lambda: engine)

    response = TestClient(main.app).post("/search", json={"query": "εκλογές", "mode": "unknown"})
    assert response.status_code == 422
    assert engine.calls == []

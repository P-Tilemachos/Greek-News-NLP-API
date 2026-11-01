"""Phase 17.2 extractive QA regression checks with isolated mocked dependencies."""

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main


class QASpy:
    def __init__(self):
        self.calls = []

    def ask(self, question, k=3):
        self.calls.append((question, k))
        return {
            "mode": "extractive",
            "evidence": [
                {"doc_id": "article-1", "sentence": "Η συνεδρίαση έγινε χθες.", "score": 0.7}
            ],
            "sources": [
                {"id": "article-1", "score": 0.9, "label": "politics", "snippet": "Η συνεδρίαση"}
            ],
        }


def test_extractive_qa_returns_traceable_evidence(monkeypatch):
    fake = QASpy()
    monkeypatch.setitem(main.app.dependency_overrides, main.get_qa, lambda: fake)

    response = TestClient(main.app).post(
        "/ask", json={"question": "Πότε έγινε η συνεδρίαση;", "k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "extractive"
    assert body["evidence"][0]["doc_id"] == body["sources"][0]["id"]
    assert fake.calls == [("Πότε έγινε η συνεδρίαση;", 2)]


@pytest.mark.parametrize("k", [0, 11])
def test_extractive_qa_rejects_invalid_k(monkeypatch, k):
    monkeypatch.setitem(main.app.dependency_overrides, main.get_qa, QASpy)

    response = TestClient(main.app).post(
        "/ask", json={"question": "Πότε έγινε η συνεδρίαση;", "k": k}
    )
    assert response.status_code == 422

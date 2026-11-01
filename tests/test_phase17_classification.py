"""Phase 17.2 classification API regression checks with isolated mocked dependencies."""

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main


class ClassifierSpy:
    def __init__(self):
        self.calls = []

    def predict(self, text, top_k=3):
        self.calls.append((text, top_k))
        return [{"label": "politics", "score": 0.91}][:top_k]


def test_classify_passes_inputs_and_preserves_scores(monkeypatch):
    fake = ClassifierSpy()
    monkeypatch.setitem(main.app.dependency_overrides, main.get_classifier, lambda: fake)

    response = TestClient(main.app).post(
        "/classify", json={"text": "Η κυβέρνηση ανακοίνωσε νέο σχέδιο", "top_k": 1}
    )
    assert response.status_code == 200
    assert response.json() == {"predictions": [{"label": "politics", "score": 0.91}]}
    assert fake.calls == [("Η κυβέρνηση ανακοίνωσε νέο σχέδιο", 1)]


@pytest.mark.parametrize("top_k", [0, 18, -1])
def test_classify_rejects_invalid_top_k_without_calling_model(monkeypatch, top_k):
    fake = ClassifierSpy()
    monkeypatch.setitem(main.app.dependency_overrides, main.get_classifier, lambda: fake)

    response = TestClient(main.app).post(
        "/classify", json={"text": "Η κυβέρνηση ανακοίνωσε νέο σχέδιο", "top_k": top_k}
    )
    assert response.status_code == 422
    assert fake.calls == []

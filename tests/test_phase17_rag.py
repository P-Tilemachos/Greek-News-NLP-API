"""Phase 17.2 endpoint regression checks with isolated mocked dependencies."""

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main


class RAGSpy:
    def __init__(self, abstained):
        self.abstained = abstained
        self.calls = []

    def ask(self, question, k, max_new_tokens):
        self.calls.append((question, k, max_new_tokens))
        return {
            "mode": "grounded_rag",
            "answer": None if self.abstained else "Η συνεδρίαση έγινε χθες. [1]",
            "abstained": self.abstained,
            "reason": "insufficient_evidence" if self.abstained else None,
            "model": "fake/model",
            "sources": []
            if self.abstained
            else [
                {
                    "reference": "[1]",
                    "doc_id": "article-1",
                    "label": "politics",
                    "excerpt": "Η συνεδρίαση έγινε χθες.",
                    "retrieval_score": 0.8,
                }
            ],
            "faithfulness_verified": False,
        }


@pytest.mark.parametrize("abstained", [False, True])
def test_rag_answer_and_abstention_response_contract(monkeypatch, abstained):
    fake = RAGSpy(abstained)
    monkeypatch.setattr(main, "get_grounded_rag", lambda: fake)
    response = TestClient(main.app).post(
        "/rag/ask", json={"question": "Πότε έγινε η συνεδρίαση;", "k": 2, "max_new_tokens": 32}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "grounded_rag"
    assert result["abstained"] is abstained
    assert result["faithfulness_verified"] is False
    if abstained:
        assert result["answer"] is None
        assert result["reason"]
    else:
        assert "[1]" in result["answer"]
        assert result["sources"][0]["reference"] == "[1]"
    assert fake.calls == [("Πότε έγινε η συνεδρίαση;", 2, 32)]


@pytest.mark.parametrize("error", [RuntimeError("index unavailable"), ValueError("bad data")])
def test_rag_backend_failure_is_sanitized(monkeypatch, error):
    class BrokenRAG:
        def ask(self, *args):
            raise error

    monkeypatch.setattr(main, "get_grounded_rag", lambda: BrokenRAG())
    response = TestClient(main.app).post("/rag/ask", json={"question": "Πότε έγινε η συνεδρίαση;"})
    assert response.status_code == 503
    assert str(error) not in response.text

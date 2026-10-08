from fastapi.testclient import TestClient

from greek_nlp.api.main import app, get_qa


class FakeQA:
    def ask(self, question: str, k: int = 3) -> dict:
        return {
            "mode": "extractive",
            "evidence": [{"doc_id": "a", "sentence": "Η Τρίπολη έχασε με 0-3.", "score": 0.9}],
            "sources": [{"id": "a", "score": 0.9, "label": "sport", "snippet": "Αθλητική είδηση"}],
        }


app.dependency_overrides[get_qa] = FakeQA
client = TestClient(app)


def test_ask_returns_evidence_and_sources():
    response = client.post("/ask", json={"question": "Ποιος κέρδισε τον αγώνα;", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "extractive"
    assert body["evidence"][0]["doc_id"] == "a"
    assert body["sources"][0]["label"] == "sport"


def test_ask_rejects_short_question():
    response = client.post("/ask", json={"question": "ab"})
    assert response.status_code == 422

from fastapi.testclient import TestClient

from greek_nlp.api.main import app, get_retriever


class FakeRetriever:
    def search(self, query: str, k: int = 5) -> list[dict]:
        hits = [
            {"id": "a", "score": 0.9, "label": "sport", "snippet": "Αθλητική είδηση"},
            {"id": "b", "score": 0.8, "label": "politics", "snippet": "Πολιτική είδηση"},
        ]
        return hits[:k]


app.dependency_overrides[get_retriever] = FakeRetriever
client = TestClient(app)


def test_search_returns_results():
    response = client.post("/search", json={"query": "ποδόσφαιρο", "k": 1})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_search_rejects_short_query():
    response = client.post("/search", json={"query": "ab"})
    assert response.status_code == 422

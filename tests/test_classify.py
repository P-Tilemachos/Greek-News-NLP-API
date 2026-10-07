from fastapi.testclient import TestClient

from greek_nlp.api.main import app, get_classifier


class FakeClassifier:
    def predict(self, text: str, top_k: int = 3) -> list[dict]:
        return [{"label": "sport", "score": 0.9}, {"label": "politics", "score": 0.05}][:top_k]


app.dependency_overrides[get_classifier] = FakeClassifier
client = TestClient(app)


def test_classify_returns_predictions():
    payload = {"text": "Ο Ολυμπιακός νίκησε στο ντέρμπι", "top_k": 2}
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert predictions[0] == {"label": "sport", "score": 0.9}
    assert len(predictions) == 2


def test_classify_rejects_short_text():
    response = client.post("/classify", json={"text": "hi"})
    assert response.status_code == 422

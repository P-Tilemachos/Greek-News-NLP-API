"""Phase 17.2 endpoint regression checks with isolated mocked dependencies."""

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main


class GeneratorSpy:
    def __init__(self, label):
        self.label = label
        self.calls = []

    def generate(self, prompt, max_new_tokens):
        self.calls.append((prompt, max_new_tokens))
        return {
            "mode": "local_llm",
            "model": self.label,
            "device": "cpu",
            "generated_text": "Δοκιμαστική απάντηση",
            "input_tokens": 5,
            "output_tokens": 4,
        }


@pytest.mark.parametrize("backend", ["torch", "llamacpp"])
def test_generation_backend_selection(monkeypatch, backend):
    pytorch = GeneratorSpy("mock/torch")
    llama = GeneratorSpy("mock/llamacpp")
    monkeypatch.setenv("LLM_BACKEND", backend)
    monkeypatch.setattr(main, "get_generator", lambda: pytorch)
    monkeypatch.setattr(main, "_load_llamacpp_generator", lambda: llama)
    response = TestClient(main.app).post(
        "/generate", json={"prompt": "Γράψε περίληψη για την είδηση", "max_new_tokens": 16}
    )
    assert response.status_code == 200
    assert response.json()["model"] == f"mock/{backend}"
    assert (llama if backend == "llamacpp" else pytorch).calls == [
        ("Γράψε περίληψη για την είδηση", 16)
    ]
    assert (pytorch if backend == "llamacpp" else llama).calls == []


def test_default_backend_is_torch(monkeypatch):
    fake = GeneratorSpy("mock/torch")
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setattr(main, "get_generator", lambda: fake)
    response = TestClient(main.app).post("/generate", json={"prompt": "Γράψε ελληνικό κείμενο"})
    assert response.status_code == 200
    assert fake.calls == [("Γράψε ελληνικό κείμενο", 64)]


def test_generation_unknown_backend_returns_sanitized_503(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "unknown")
    response = TestClient(main.app).post("/generate", json={"prompt": "Γράψε ελληνικό κείμενο"})
    assert response.status_code == 503
    assert "LLM_BACKEND" not in response.text

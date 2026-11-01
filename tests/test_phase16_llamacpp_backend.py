"""Offline tests: never load model weights, start servers, or make real HTTP calls."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main
from greek_nlp.llm import llamacpp_backend as backend

MODEL = "ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M"


def fake_upstream(url, *, timeout, payload=None):
    if url.endswith("/v1/models"):
        return {"data": [{"id": MODEL}]}
    assert url.endswith("/v1/chat/completions")
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    return {
        "model": MODEL,
        "choices": [{"message": {"content": " Η μηχανική μάθηση μαθαίνει από δεδομένα. "}}],
        "usage": {"prompt_tokens": 70, "completion_tokens": 15},
    }


def test_llamacpp_generator_schema_and_prompt(monkeypatch):
    monkeypatch.setattr(backend, "_request_json", fake_upstream)
    result = backend.LlamaCppGenerator().generate("Τι είναι η μηχανική μάθηση;", 48)
    assert result == {
        "mode": "local_llm",
        "model": MODEL,
        "device": "cpu",
        "generated_text": "Η μηχανική μάθηση μαθαίνει από δεδομένα.",
        "input_tokens": 70,
        "output_tokens": 15,
    }


def test_constructor_does_not_contact_server(monkeypatch):
    monkeypatch.setattr(backend, "_request_json", lambda *a, **kw: pytest.fail("unexpected I/O"))
    assert backend.LlamaCppGenerator().model_match == "Krikri"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:9931",
        "https://127.0.0.1:9931",
        "http://127.0.0.1:9931/admin",
        "http://user:pass@127.0.0.1:9931",
        "http://127.0.0.1:9931?redirect=1",
        "http://127.0.0.1",
    ],
)
def test_rejects_non_loopback_or_unsafe_urls(monkeypatch, url):
    monkeypatch.setenv("LLAMACPP_BASE_URL", url)
    with pytest.raises(ValueError, match="LLAMACPP_BASE_URL"):
        backend.LlamaCppGenerator()


def test_rejects_wrong_server_model(monkeypatch):
    monkeypatch.setattr(
        backend, "_request_json", lambda *a, **kw: {"data": [{"id": "Qwen/Qwen2.5"}]}
    )
    with pytest.raises(backend.LlamaCppUnavailable, match="model"):
        backend.LlamaCppGenerator().generate("Some prompt", 48)


@pytest.mark.parametrize(
    "broken",
    [
        {"choices": [], "usage": {"prompt_tokens": 70, "completion_tokens": 15}},
        {"choices": [{"message": {"content": "ok"}}]},
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 15},
        },
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 70, "completion_tokens": 999},
        },
    ],
)
def test_rejects_broken_upstream_replies(monkeypatch, broken):
    def fake(url, *, timeout, payload=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": MODEL}]}
        return broken

    monkeypatch.setattr(backend, "_request_json", fake)
    with pytest.raises(backend.LlamaCppUnavailable):
        backend.LlamaCppGenerator().generate("Some prompt", 48)


def test_default_backend_remains_torch(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    fake = SimpleNamespace(generate=lambda prompt, cap: {"prompt": prompt, "cap": cap})
    monkeypatch.setattr(main, "get_generator", lambda: fake)
    assert main.get_generation_backend() is fake


def test_optin_backend_does_not_change_grounded_rag_factory(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "llamacpp")
    gguf = object()
    monkeypatch.setattr(main, "_load_llamacpp_generator", lambda: gguf)
    assert main.get_generation_backend() is gguf
    # Grounded RAG must still receive main.get_generator, NOT get_generation_backend.
    source = inspect.getsource(main.get_grounded_rag)
    assert "get_generator" in source
    assert "get_generation_backend" not in source


def test_optin_generate_endpoint_no_real_network(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "llamacpp")
    monkeypatch.setattr(backend, "_request_json", fake_upstream)
    main._load_llamacpp_generator.cache_clear()
    try:
        reply = TestClient(main.app).post(
            "/generate", json={"prompt": "Τι είναι η μηχανική μάθηση;", "max_new_tokens": 48}
        )
        assert reply.status_code == 200
        assert reply.json()["model"] == MODEL
        assert reply.json()["device"] == "cpu"
        assert reply.json()["output_tokens"] == 15
    finally:
        main._load_llamacpp_generator.cache_clear()


def test_upstream_errors_return_sanitized_503(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "llamacpp")

    def unavailable(*args, **kwargs):
        raise backend.LlamaCppUnavailable("private upstream path")

    monkeypatch.setattr(backend, "_request_json", unavailable)
    main._load_llamacpp_generator.cache_clear()
    try:
        reply = TestClient(main.app).post(
            "/generate", json={"prompt": "Τι είναι η μηχανική μάθηση;"}
        )
        assert reply.status_code == 503
        assert "private upstream path" not in reply.text
    finally:
        main._load_llamacpp_generator.cache_clear()


def test_unknown_backend_returns_503(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "something_else")
    reply = TestClient(main.app).post("/generate", json={"prompt": "Τι είναι η μηχανική μάθηση;"})
    assert reply.status_code == 503


def test_no_network_redirects():
    assert backend._NoRedirects().redirect_request(None, None, 302, "Moved", {}, "http://x") is None

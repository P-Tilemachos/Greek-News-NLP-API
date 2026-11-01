"""Offline unit checks for the Phase 16 development-only matched benchmark."""

import json

import pytest

from scripts import phase16_compare_backends as bench


def test_local_url_restriction():
    assert bench.localhost("http://127.0.0.1:9931/") == "http://127.0.0.1:9931"
    for url in ("https://example.com", "http://example.com", "http://localhost:9931/x"):
        with pytest.raises(ValueError):
            bench.localhost(url)


def test_gguf_response_scrubbed():
    reply = {
        "model": "test.gguf",
        "choices": [{"message": {"content": "PRIVATE_GENERATED_TEXT"}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 5},
        "timings": {"predicted_per_second": 14.8},
    }
    record = bench.summarise_response("gguf", reply, 2000.0, 1)
    assert record["output_tokens"] == 5
    assert record["input_tokens"] == 40
    assert record["generated_tokens_per_end_to_end_second"] == 2.5
    assert "PRIVATE_GENERATED_TEXT" not in json.dumps(record)


def test_torch_response_scrubbed():
    reply = {
        "generated_text": "PRIVATE_GENERATED_TEXT",
        "input_tokens": 96,
        "output_tokens": 8,
        "model": "qwen",
        "device": "cpu",
    }
    record = bench.summarise_response("torch", reply, 4000.0, 1)
    assert record["output_tokens"] == 8
    assert record["server_timings"] is None
    assert "PRIVATE_GENERATED_TEXT" not in json.dumps(record)


def test_matching_system_prompt_for_gguf():
    body = bench.build_request("gguf", "system instructions", "question", 48, "model")
    assert body["messages"] == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "question"},
    ]
    assert body["max_tokens"] == 48
    assert body["temperature"] == 0

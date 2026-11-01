"""Offline unit tests for Phase 16 GGUF SSE parsing and URL restrictions."""

from __future__ import annotations

import pytest

from scripts.phase16_streaming_probe import (
    content_from_event,
    parse_data_line,
    safe_base_url,
)


def test_sse_parser_delta() -> None:
    line = b'data: {"choices": [{"delta": {"content": "Hello"}}]}'
    event = parse_data_line(line)
    assert isinstance(event, dict)
    assert content_from_event(event) == "Hello"


def test_sse_done_and_metadata() -> None:
    assert parse_data_line(b"data: [DONE]") == "DONE"
    assert parse_data_line(b":keepalive") is None
    assert content_from_event({"choices": [{"delta": {"role": "assistant"}}]}) == ""


def test_localhost_only() -> None:
    assert safe_base_url("http://127.0.0.1:9931/") == "http://127.0.0.1:9931"
    with pytest.raises(ValueError):
        safe_base_url("https://example.com")
    with pytest.raises(ValueError):
        safe_base_url("http://127.0.0.1:9931/other")


def test_invalid_sse_json_rejected() -> None:
    with pytest.raises(ValueError):
        parse_data_line(b"data: [1,2,3]")

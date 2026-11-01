"""Unit tests for the offline-only parts of Phase 16 baseline benchmarking."""

import json

import pytest

from scripts import phase16_baseline as bm


def test_load_dev_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "status": "DEVELOPMENT_ONLY",
                "cases": [{"id": "a", "endpoint": "/generate", "request": {"prompt": "hi"}}],
            }
        ),
        encoding="utf-8",
    )
    assert bm.load_development_cases(path)[0]["id"] == "a"


def test_reject_non_development_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "status": "FROZEN_EVALUATION",
                "cases": [{"id": "a", "endpoint": "/generate", "request": {"prompt": "hi"}}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DEVELOPMENT_ONLY"):
        bm.load_development_cases(path)


def test_url_restricts_to_localhost():
    bm.check_local_url("http://127.0.0.1:8000")
    with pytest.raises(ValueError):
        bm.check_local_url("https://example.com")


def test_no_leak_of_raw_answer():
    result = bm.summarise(
        {
            "id": "x",
            "endpoint": "/generate",
            "repeat_index": 1,
            "http_status": 200,
            "elapsed_ms": 10,
            "response": {"generated_text": "SECRET_TEXT", "output_tokens": 2},
        }
    )
    assert result["answer_chars"] == 11
    assert "SECRET_TEXT" not in json.dumps(result)

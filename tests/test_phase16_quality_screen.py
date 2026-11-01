"""Offline tests for the Phase 16 author-review quality screen."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "phase16_quality_screen.py"
SPEC = importlib.util.spec_from_file_location("phase16_quality_screen", MODULE)
assert SPEC is not None and SPEC.loader is not None
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def test_localhost_only():
    assert quality.local_base("http://127.0.0.1:9931") == "http://127.0.0.1:9931"
    with pytest.raises(ValueError):
        quality.local_base("https://example.org")
    with pytest.raises(ValueError):
        quality.local_base("http://localhost:9931/other")


def test_safe_tags():
    assert quality.valid_tag("krikri8b_q4km") == "krikri8b_q4km"
    with pytest.raises(ValueError):
        quality.valid_tag("../../escape")


def test_load_all_development_cases():
    cases = quality.load_cases()
    assert len(cases) == 8
    assert len({case["id"] for case in cases}) == 8


def test_response_record_does_not_auto_grade():
    row = {
        "id": "demo",
        "category": "quality",
        "question": "Test?",
        "review_criterion": "Review manually",
    }
    response = {
        "choices": [{"message": {"content": "Παράδειγμα"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    record = quality.response_record(row, response, 123.0)
    assert record["answer"] == "Παράδειγμα"
    assert record["human_review"] is None
    assert record["output_tokens"] == 3

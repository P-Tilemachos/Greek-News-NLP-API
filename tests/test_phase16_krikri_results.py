"""Offline regression tests for Phase 16 model-labeled benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import phase16_compare_backends as matched
from scripts import phase16_memory_profile as memory


@pytest.mark.parametrize("module", [matched, memory])
def test_tag_rejects_unsafe_paths(module):
    for tag in ("../private", "bad/tag", "-leading", "has space", "éλληνικα", "x" * 41, ""):
        with pytest.raises(ValueError, match="--tag"):
            module.validate_tag(tag)
    assert module.validate_tag("krikri8b_q4km") == "krikri8b_q4km"
    assert module.validate_tag(None) is None


def test_krikri_memory_dry_run_creates_no_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(memory, "OUT_DIR", tmp_path)
    args = argparse.Namespace(
        backend="gguf",
        interval=0.1,
        warmup=1,
        repeat=3,
        execute=False,
        pid=None,
        tag="krikri8b_q4km",
        model_match="Krikri",
    )
    memory.run(args)
    out = capsys.readouterr().out
    assert "phase16_memory_gguf_krikri8b_q4km_DEV_ONLY.json" in out
    assert "phase16_memory_matched_gguf_krikri8b_q4km_DEV_ONLY.json" in out
    assert not list(tmp_path.iterdir())


def test_model_identity_guard_stops_before_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(matched, "OUT_DIR", tmp_path)
    monkeypatch.setattr(matched, "load_workload", lambda: ("system", "user", 48))
    monkeypatch.setattr(
        matched,
        "get_json",
        lambda url, _timeout: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"data": [{"id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"}]}
        ),
    )
    monkeypatch.setattr(
        matched,
        "post_json",
        lambda *_: pytest.fail("Inference must not run after model mismatch"),
    )
    args = argparse.Namespace(
        backend="gguf",
        url="http://127.0.0.1:9931",
        tag="krikri8b_q4km",
        model_match="Krikri",
        output=None,
        warmup=1,
        repeat=3,
        timeout=600,
    )
    with pytest.raises(ValueError, match="Unexpected model"):
        matched.run(args)
    assert not list(tmp_path.iterdir())


def test_compare_two_distinct_gguf_models(tmp_path: Path, capsys):
    base = {
        "status": "COMPLETE",
        "backend": "gguf",
        "user_prompt_sha256": "same-user",
        "system_prompt_sha256": "same-system",
        "max_new_tokens": 48,
        "records": [
            {"elapsed_ms": 3000, "output_tokens": 48, "generated_tokens_per_end_to_end_second": 16}
        ],
    }
    paths = []
    for tag, model in (("qwen1p5b", "Qwen-1.5B"), ("krikri8b", "Llama-Krikri-8B")):
        path = tmp_path / f"{tag}.json"
        path.write_text(
            json.dumps({**base, "model_tag": tag, "model_id_reported": model}),
            encoding="utf-8",
        )
        paths.append(path)
    matched.compare(paths, allow_gguf_compare=True)
    output = capsys.readouterr().out
    assert "qwen1p5b" in output
    assert "krikri8b" in output


def test_compare_rejects_same_model_id(tmp_path: Path):
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "backend": "gguf",
                    "model_id_reported": "same-model",
                    "user_prompt_sha256": "same-user",
                    "system_prompt_sha256": "same-system",
                    "max_new_tokens": 48,
                    "records": [{"elapsed_ms": 1000}],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    with pytest.raises(ValueError, match="distinct GGUF"):
        matched.compare(paths, allow_gguf_compare=True)


def test_compare_two_gguf_disallowed_by_default(tmp_path: Path):
    paths = []
    for i in range(2):
        path = tmp_path / f"model{i}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "backend": "gguf",
                    "model_id_reported": f"model_{i}",
                    "user_prompt_sha256": "same-user",
                    "system_prompt_sha256": "same-system",
                    "max_new_tokens": 48,
                    "records": [{"elapsed_ms": 1000}],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    with pytest.raises(ValueError, match="--allow-gguf-compare"):
        matched.compare(paths)

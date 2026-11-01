"""Phase 17.4: validate the release smoke checker without network/model downloads."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def smoke():
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase17_release_validate.py"
    spec = importlib.util.spec_from_file_location("phase17_release_validate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_target_is_restricted_to_loopback(smoke):
    assert smoke.validate_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    with pytest.raises(ValueError, match="Non-local"):
        smoke.validate_base_url("https://example.org")
    with pytest.raises(ValueError, match="credentials"):
        smoke.validate_base_url("http://user:password@127.0.0.1:8000")


def test_release_smoke_checks_openapi_version_and_routes(smoke):
    paths = {path: {method: {}} for path, method in smoke.ROUTES.items()}
    calls = []

    def fake_requester(base_url, method, path, payload, timeout):
        calls.append((method, path, payload))
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/openapi.json":
            return 200, {"info": {"version": "0.1.0"}, "paths": paths}
        return 422, {"detail": "validation error"}

    results = smoke.run_checks("http://127.0.0.1:8000", "0.1.0", [], 3, fake_requester)
    assert len(results) == 4
    assert all(item["passed"] for item in results)
    assert ("POST", "/generate", {}) in calls
    assert ("POST", "/rag/ask", {}) in calls


def test_release_smoke_rejects_version_mismatch(smoke):
    paths = {path: {method: {}} for path, method in smoke.ROUTES.items()}

    def fake_requester(base_url, method, path, payload, timeout):
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/openapi.json":
            return 200, {"info": {"version": "wrong"}, "paths": paths}
        return 422, {}

    result = smoke.run_checks("http://127.0.0.1:8000", "0.1.0", [], 3, fake_requester)
    assert next(item for item in result if item["check"] == "openapi")["passed"] is False


@pytest.mark.parametrize("mode", ["dense", "hybrid", "rerank"])
def test_retrieval_modes_are_validated(smoke, mode):
    smoke.verify_probe(
        f"search-{mode}",
        200,
        {"mode": mode, "results": [{"id": "example-1", "score": 0.5}]},
    )
    with pytest.raises(AssertionError, match="wrong retrieval mode"):
        smoke.verify_probe(f"search-{mode}", 200, {"mode": "invalid", "results": []})


def test_rag_abstention_is_valid_contract_not_quality_proof(smoke):
    smoke.verify_probe(
        "rag",
        200,
        {
            "mode": "grounded_rag",
            "abstained": True,
            "answer": None,
            "faithfulness_verified": False,
            "sources": [],
        },
    )
    with pytest.raises(AssertionError, match="null answer"):
        smoke.verify_probe(
            "rag",
            200,
            {
                "mode": "grounded_rag",
                "abstained": True,
                "answer": "fabricated",
                "faithfulness_verified": False,
                "sources": [],
            },
        )


def test_generation_probe_requires_schema(smoke):
    smoke.verify_probe(
        "generate",
        200,
        {
            "mode": "local_llm",
            "generated_text": "Καλημέρα",
            "model": "test/mock",
            "output_tokens": 1,
        },
    )
    with pytest.raises(AssertionError):
        smoke.verify_probe("generate", 503, {"detail": "unavailable"})

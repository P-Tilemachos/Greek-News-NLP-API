"""Phase 18: deterministic checks that do not load actual models or use network."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def validation():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        target = scripts_dir / "phase18_real_model_validation.py"
        spec = importlib.util.spec_from_file_location("phase18_validation", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts_dir))


def test_classify_detects_real_probabilities(validation):
    result = validation.assess_probe(
        "classify",
        200,
        {"predictions": [{"label": "sport", "score": 0.8}, {"label": "politics", "score": 0.2}]},
    )
    assert result["prediction_count"] == 2
    with pytest.raises(AssertionError, match="labels or scores"):
        validation.assess_probe("classify", 200, {"predictions": [{"label": "x", "score": 1.5}]})


@pytest.mark.parametrize("mode", ["dense", "hybrid", "rerank"])
def test_retrieval_sanity(validation, mode):
    body = {
        "mode": mode,
        "results": [{"id": "article-1", "label": "sport", "score": -0.2, "snippet": "Δοκιμή"}],
    }
    assert validation.assess_probe(f"search-{mode}", 200, body)["retrieval_count"] == 1
    with pytest.raises(AssertionError, match="no retrieved articles"):
        validation.assess_probe(f"search-{mode}", 200, {"mode": mode, "results": []})


def test_extract_qa_empty_evidence_is_disclosed(validation):
    result = validation.assess_probe("ask", 200, {"evidence": [], "sources": []})
    assert result["empty_evidence"] is True


def test_generation_requires_content(validation):
    body = {
        "mode": "local_llm",
        "generated_text": "Δοκιμή",
        "output_tokens": 2,
        "model": "test/mock",
        "device": "cpu",
        "input_tokens": 4,
    }
    assert validation.assess_probe("generate", 200, body)["output_tokens"] == 2
    body["generated_text"] = " "
    with pytest.raises(AssertionError, match="empty text"):
        validation.assess_probe("generate", 200, body)


def test_rag_abstention_blocks_quality_not_contract(validation):
    body = {
        "mode": "grounded_rag",
        "abstained": True,
        "answer": None,
        "faithfulness_verified": False,
        "sources": [],
    }
    result = validation.assess_probe("rag", 200, body)
    assert result["technical_pass"] is True
    assert result["quality_status"] == "BLOCKED_BY_ABSTENTION"
    gate = validation.evaluate_status([{"passed": True}], [result])
    assert gate["technical_pass"] is True
    assert gate["release_approved"] is False
    assert gate["rag_abstention_observed"] is True


def test_rag_answer_requires_sources(validation):
    body = {
        "mode": "grounded_rag",
        "abstained": False,
        "answer": "Απάντηση",
        "faithfulness_verified": False,
        "sources": [],
    }
    with pytest.raises(AssertionError, match="citations"):
        validation.assess_probe("rag", 200, body)


def test_execute_probes_redacts_response_contents(validation):
    def fake(base_url, method, path, payload, timeout):
        return 200, {
            "mode": "local_llm",
            "generated_text": "PRIVATE GENERATED TEXT",
            "output_tokens": 2,
            "model": "test/model",
            "device": "cpu",
            "input_tokens": 3,
        }

    result = validation.execute_probes("http://127.0.0.1:8000", ["generate"], 5, fake)
    assert result[0]["technical_pass"] is True
    assert "PRIVATE" not in str(result)


def test_transport_failure_is_reported_without_credentials(validation):
    def broken(base_url, method, path, payload, timeout):
        raise OSError("SECRET_TOKEN")

    result = validation.execute_probes("http://127.0.0.1:8000", ["classify"], 5, broken)
    assert result[0]["technical_pass"] is False
    assert result[0]["error_type"] == "OSError"
    assert "SECRET_TOKEN" not in str(result)


def test_empty_probe_set_cannot_pass(validation):
    assert validation.evaluate_status([{"passed": True}], [])["technical_pass"] is False


def test_runtime_validation_stays_local_by_default(validation):
    with pytest.raises(ValueError, match="Non-local"):
        validation.validate_base_url("https://example.com")

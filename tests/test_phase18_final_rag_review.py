"""Offline regression tests for Phase 18 final review tool; no network/models."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def gate():
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase18_final_rag_review.py"
    spec = importlib.util.spec_from_file_location("final_gate", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(answer=None, abstained=True, reason="model_abstained", sources=None):
    return {
        "mode": "grounded_rag",
        "answer": answer,
        "abstained": abstained,
        "reason": reason,
        "sources": [] if sources is None else sources,
        "faithfulness_verified": False,
    }


def source():
    return {
        "doc_id": "mandra",
        "excerpt": "Σε σειρά διαφορετικών φορέων του δημοσίου εντοπίζει ευθύνες.",
    }


def test_local_only_gate(gate):
    assert gate.validate_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    with pytest.raises(ValueError):
        gate.validate_base_url("https://example.com")
    with pytest.raises(ValueError):
        gate.validate_base_url("http://user:pass@127.0.0.1:8000")


def test_answerable_abstention_blocks_quality(gate):
    record = gate.assess_case(gate.CASES[0], 200, response())
    assert record["technical_pass"]
    assert record["quality_status"] == "KNOWN_ANSWERABLE_ABSTENTION"


def test_real_verbatim_answer_needs_human_review(gate):
    record = gate.assess_case(
        gate.CASES[0],
        200,
        response(
            answer="Σε σειρά διαφορετικών φορέων του δημοσίου [1]",
            abstained=False,
            reason=None,
            sources=[source()],
        ),
    )
    assert record["technical_pass"]
    assert record["has_citation"]
    assert record["verbatim_span_in_source"]
    assert record["quality_status"] == "ANSWER_REQUIRES_HUMAN_REVIEW"


def test_missing_citation_is_rejected(gate):
    record = gate.assess_case(
        gate.CASES[0],
        200,
        response(answer="Μερικούς φορείς.", abstained=False, sources=[source()]),
    )
    assert not record["technical_pass"]
    assert record["quality_status"] == "STRUCTURAL_ANSWER_FAILURE"


def test_out_of_range_citation_is_rejected(gate):
    record = gate.assess_case(
        gate.CASES[0],
        200,
        response(answer="φορέων του δημοσίου [3]", abstained=False, sources=[source()]),
    )
    assert not record["technical_pass"]


def test_question_echo_is_rejected(gate):
    record = gate.assess_case(
        gate.CASES[0],
        200,
        response(answer=gate.CASES[0]["question"] + " [1]", abstained=False, sources=[source()]),
    )
    assert record["question_echo_detected"]
    assert not record["technical_pass"]


def test_abstention_must_not_have_an_answer(gate):
    record = gate.assess_case(gate.CASES[0], 200, response(answer="bad"))
    assert not record["technical_pass"]


def test_successful_structural_cases_never_approve_release(gate):
    records = [
        gate.assess_case(
            gate.CASES[0],
            200,
            response(answer="φορέων του δημοσίου [1]", abstained=False, sources=[source()]),
        ),
        gate.assess_case(gate.CASES[1], 200, response()),
    ]
    status = gate.evaluate_gate(True, records)
    assert status["technical_pass"]
    assert status["known_answerable_produced_answer"]
    assert status["release_approved"] is False


def test_report_does_not_contain_model_answer(gate):
    secret = "PRIVATE RESPONSE TEXT"

    def fake_requester(base_url, method, path, payload, timeout):
        if path == "/health":
            return 200, {"status": "ok"}
        return 200, response(answer=secret + " [1]", abstained=False, sources=[source()])

    health, records = gate.run_review("http://127.0.0.1:8000", 5, fake_requester)
    assert health
    assert len(records) == 2
    assert secret not in str(records)


def test_broad_case_not_treated_as_gold_negative(gate):
    assert gate.CASES[1]["type"] == "exploratory_not_labeled_unanswerable"

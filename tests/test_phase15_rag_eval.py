"""Phase 15 scorer/collector regression tests: synthetic fixtures, no network or models."""

import csv
import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from greek_nlp.evaluation.rag_eval import (
    SCHEMA,
    evaluate,
    export_review_csv,
    import_review_csv,
    load_cases,
    markdown_report,
)


def case(key="a", answerable=None, answer="2017 [1]", abstained=False, sources=None):
    if sources is None:
        sources = [{"reference": "[1]", "doc_id": "d1", "excerpt": "2017"}]
    return {
        "id": key,
        "question": f"Synthetic question {key}?",
        "answerable": answerable,
        "gold_doc_ids": [],
        "prediction": {
            "answer": None if abstained else answer,
            "abstained": abstained,
            "reason": "model_abstained" if abstained else None,
            "sources": sources,
            "faithfulness_verified": False,
        },
        "human_review": None,
    }


def save(tmp_path, rows):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema_version": SCHEMA, "cases": rows}), encoding="utf-8")
    return path


def test_empty_dataset_not_fabricated(tmp_path):
    assert load_cases(save(tmp_path, [])) == []
    metrics = evaluate([])
    assert metrics["counts"]["human_reviewed"] == 0
    assert metrics["automatic"]["response_coverage"]["value"] is None
    assert "N/A" in markdown_report(metrics)


def test_missing_prediction_not_counted():
    missing = case("pending")
    missing["prediction"] = None
    result = evaluate([case("done"), missing])
    assert result["counts"]["predictions_missing"] == 1
    assert result["counts"]["predictions_present"] == 1
    assert result["automatic"]["response_coverage"]["value"] == 1


def test_citation_syntax_is_not_faithfulness():
    row = case(answer="Unsupported claim [1]")
    scored = evaluate([row])
    assert scored["automatic"]["citation_reference_valid_on_answered"]["value"] == 1
    assert scored["human_review"]["faithfulness_fully_supported"]["value"] is None
    assert scored["counts"]["human_reviewed"] == 0


def test_invalid_citation_and_missing_citation():
    invalid = case("bad", answer="Date [2]")
    absent = case("absent", answer="No source citation")
    result = evaluate([invalid, absent])
    assert result["automatic"]["citation_presence_on_answered"]["value"] == 0.5
    assert result["automatic"]["citation_reference_valid_on_answered"]["value"] == 0


def test_labeled_answerability_separates_unknown():
    rows = [
        case("answerable", answerable=True),
        case("unknown"),
        case("unanswerable", answerable=False, abstained=True),
    ]
    report = evaluate(rows)
    assert report["counts"]["answerability_labeled"] == 2
    assert report["counts"]["answerability_unlabeled"] == 1
    assert report["answerability"]["answerable_answer_rate"]["value"] == 1
    assert report["answerability"]["unanswerable_abstention_rate"]["value"] == 1
    assert report["answerability"]["unanswerable_false_answer_rate"]["value"] == 0


def test_gold_doc_id_overlap_not_claimed_faithful():
    row = case()
    row["gold_doc_ids"] = ["d1"]
    report = evaluate([row])
    assert report["automatic"]["gold_document_in_cited_sources"]["value"] == 1
    assert report["human_review"]["faithfulness_fully_supported"]["value"] is None


def test_abstention_skips_citation_denominator():
    report = evaluate([case("a", abstained=True)])
    assert report["automatic"]["citation_reference_valid_on_answered"]["value"] is None
    assert report["abstention_reasons"] == {"model_abstained": 1}


def test_actual_review_with_provenance_scored(tmp_path):
    row = case()
    row["human_review"] = {
        "reviewer": "human-1",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "faithfulness": "unsupported",
        "correctness": "incorrect",
        "hallucination": True,
        "notes": "A review in this synthetic test fixture.",
    }
    load_cases(save(tmp_path, [row]))
    result = evaluate([row])
    assert result["human_review"]["faithfulness_unsupported"]["value"] == 1
    assert result["human_review"]["hallucination_flagged"]["value"] == 1
    assert result["human_review"]["answer_correct"]["value"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"reviewer": ""},
        {"reviewed_at": "invalid"},
        {"reviewed_at": "2026-10-17T12:00:00"},
        {"faithfulness": "AI-confirmed"},
        {"hallucination": "no"},
    ],
)
def test_fake_or_invalid_reviews_rejected(tmp_path, change):
    row = case()
    row["human_review"] = {
        "reviewer": "reviewer",
        "reviewed_at": "2026-10-17T12:00:00+03:00",
        "faithfulness": "supported",
        "correctness": "correct",
        "hallucination": False,
    }
    row["human_review"].update(change)
    with pytest.raises(ValueError):
        load_cases(save(tmp_path, [row]))


def test_duplicate_or_missing_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(save(tmp_path, [case("d"), case("d")]))
    row = case()
    del row["id"]
    with pytest.raises(ValueError):
        load_cases(save(tmp_path, [row]))


def test_abstention_cannot_have_answer(tmp_path):
    row = case(abstained=True)
    row["prediction"]["answer"] = "invented"
    with pytest.raises(ValueError):
        load_cases(save(tmp_path, [row]))


def test_worksheet_has_blank_judgments_and_roundtrip(tmp_path):
    rows = [case()]
    path = tmp_path / "review.csv"
    export_review_csv(rows, path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        record = list(csv.DictReader(stream))[0]
    assert record["reviewer"] == ""
    assert record["faithfulness"] == ""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=record.keys())
        writer.writeheader()
        record.update(
            {
                "reviewer": "reviewer1",
                "reviewed_at": "2026-10-17T12:00:00+03:00",
                "faithfulness": "supported",
                "correctness": "correct",
                "hallucination": "false",
                "notes": "reviewer checked source",
            }
        )
        writer.writerow(record)
    merged = import_review_csv(deepcopy(rows), path)
    assert merged[0]["human_review"]["hallucination"] is False
    assert rows[0]["human_review"] is None
    with pytest.raises(ValueError, match="refusing overwrite"):
        import_review_csv(merged, path)


def test_csv_tampered_answer_rejected(tmp_path):
    path = tmp_path / "review.csv"
    export_review_csv([case()], path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        record = list(csv.DictReader(stream))[0]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=record.keys())
        writer.writeheader()
        record["answer"] = "tampered"
        writer.writerow(record)
    with pytest.raises(ValueError, match="changed prediction"):
        import_review_csv([case()], path)


def test_empty_csv_keeps_unreviewed(tmp_path):
    path = tmp_path / "review.csv"
    export_review_csv([case()], path)
    merged = import_review_csv([case()], path)
    assert merged[0]["human_review"] is None


def test_collector_resume_and_skip(monkeypatch, tmp_path):
    from scripts.phase15_collect_predictions import collect, merge_existing

    old = case("saved")
    new = case("new")
    new["prediction"] = None
    requested = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(case("fresh")["prediction"]).encode("utf-8")

    def fake_urlopen(request, timeout):
        requested.append((request.data, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = tmp_path / "saved.json"
    merged = merge_existing([deepcopy(old), deepcopy(new)], [old])
    assert collect(merged, output, "http://127.0.0.1:8000/rag/ask", 5, 600) == 1
    assert len(requested) == 1
    assert load_cases(output)[1]["prediction"]["answer"] == "2017 [1]"
    assert collect(merged, output, "http://127.0.0.1:8000/rag/ask", 5, 600) == 0
    assert len(requested) == 1


def test_collector_refuses_changed_question():
    from scripts.phase15_collect_predictions import merge_existing

    previous = case()
    altered = case()
    altered["question"] = "Changed"
    with pytest.raises(ValueError, match="Question changed"):
        merge_existing([altered], [previous])

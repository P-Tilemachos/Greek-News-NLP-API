"""Offline smoke checks for the Phase 15 completion protocol (no inference)."""

import json

import pytest

from scripts import phase15_complete_workflow as wf


def _rows():
    positive = [
        {
            "id": f"p15_a_{n:03d}",
            "question": f"positive {n}?",
            "answerable": True,
            "gold_doc_ids": [f"doc{n}"],
            "prediction": None,
            "human_review": None,
        }
        for n in range(1, 41)
    ]
    negatives = [
        {
            "id": f"p15_u_{n:03d}",
            "question": f"negative {n}?",
            "answerable": None,
            "gold_doc_ids": [],
            "prediction": None,
            "human_review": None,
        }
        for n in range(1, 41)
    ]
    return positive, negatives


def _save(path, rows):
    path.write_text(
        json.dumps(
            {"schema_version": "phase15.rag_eval.v1", "cases": rows}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name in ("HOME",):
        monkeypatch.setattr(wf, name, tmp_path)
    paths = {
        "FILE_STAGED": tmp_path / "phase15_80_CANDIDATES_NOT_FROZEN.json",
        "FILE_PILOT_LABELS": tmp_path / "phase15_80_PILOT_AUTHOR_LABELS_NOT_GOLD.json",
        "FILE_PILOT2": tmp_path / "phase15_pilot2_predictions.json",
        "FILE_INPUT78": tmp_path / "phase15_eval78_NOT_FROZEN.json",
        "FILE_PRED78": tmp_path / "phase15_eval78_predictions.json",
        "FILE_PRED80": tmp_path / "phase15_pilot80_predictions_NOT_GOLD.json",
        "FILE_MANIFEST": tmp_path / "phase15_completion_manifest.json",
        "FILE_REPORT": tmp_path / "phase15_completion_summary_NOT_GOLD.md",
        "FILE_REVIEW_HTML": tmp_path / "phase15_prediction_author_review.html",
    }
    for name, path in paths.items():
        monkeypatch.setattr(wf, name, path)
    positive, negative = _rows()
    _save(paths["FILE_STAGED"], positive + negative)
    _save(paths["FILE_PILOT_LABELS"], positive + [{**r, "answerable": False} for r in negative])
    two = [positive[0].copy(), negative[0].copy()]
    for r in two:
        r["prediction"] = {"abstained": True, "answer": None, "reason": "test_only", "sources": []}
    _save(paths["FILE_PILOT2"], two)
    return paths


def test_prepare_exact_78_and_idempotent(sandbox):
    wf.command_prepare()
    wf.command_prepare()
    rows = wf.load(sandbox["FILE_INPUT78"])
    assert len(rows) == 78
    assert "p15_a_001" not in wf.by_id(rows)
    assert "p15_u_001" not in wf.by_id(rows)


def test_changes_to_inputs_refused(sandbox):
    wf.command_prepare()
    rows = wf.load(sandbox["FILE_INPUT78"])
    rows[0]["question"] = "modified?"
    _save(sandbox["FILE_INPUT78"], rows)
    with pytest.raises(ValueError, match="Input changed"):
        wf.load_predictions()


def test_reports_correct_partial_denominators(sandbox):
    wf.command_prepare()
    wf.write_reports()
    report = json.loads(
        (sandbox["FILE_INPUT78"].parent / "phase15_final_summary_NOT_GOLD.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["retrieved_inference"]["finished_78"] == 0
    assert report["eval78"]["counts"]["predictions_missing"] == 78
    assert report["status"] == "INCOMPLETE_PREDICTION_COLLECTION"


def test_refuse_modified_review_predictions(sandbox):
    wf.command_prepare()
    input_rows = wf.load(sandbox["FILE_INPUT78"])
    for row in input_rows:
        row["prediction"] = {
            "abstained": True,
            "answer": None,
            "reason": "test_only",
            "sources": [],
        }
    _save(sandbox["FILE_PRED78"], input_rows)
    _, pilot, done = wf.load_predictions(strict_complete=True)
    assert done == 78 and len(pilot) == 80
    pilot[0]["question"] = "changed after inference"
    reviewed = sandbox["FILE_INPUT78"].parent / "bad_review.json"
    _save(reviewed, pilot)
    with pytest.raises(ValueError, match="Incompatible review"):
        wf.reviewed_report(reviewed)

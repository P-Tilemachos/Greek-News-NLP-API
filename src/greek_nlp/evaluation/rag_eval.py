"""Phase 15: offline, no-model evaluation of saved grounded-RAG outputs.

Automatic citation checks are structural, never semantic faithfulness judgments.
Human review is counted only when an explicit reviewer and timestamp are supplied.
No model weights, network access, or source-document corpus are required.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

SCHEMA = "phase15.rag_eval.v1"
FAITHFULNESS = {"supported", "partial", "unsupported", "not_applicable"}
CORRECTNESS = {"correct", "partial", "incorrect", "not_applicable"}
CITATION = re.compile(r"\[(\d+)\]")
REVIEW_COLS = (
    "id",
    "question",
    "answerable",
    "answer",
    "abstained",
    "sources_json",
    "reviewer",
    "reviewed_at",
    "faithfulness",
    "correctness",
    "hallucination",
    "notes",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_cases(path: str | Path) -> list[dict]:
    """Load validated benchmark cases; never infer or fabricate missing labels."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict) and payload.get("schema_version") == SCHEMA,
        "Bad schema_version",
    )
    cases = payload.get("cases")
    _require(isinstance(cases, list), "cases must be an array")
    seen = set()
    for row in cases:
        _require(isinstance(row, dict), "case must be an object")
        key = row.get("id")
        _require(isinstance(key, str) and key.strip(), "nonempty id is required")
        _require(key not in seen, f"duplicate case id: {key}")
        seen.add(key)
        _require(
            isinstance(row.get("question"), str) and row["question"].strip(),
            f"bad question: {key}",
        )
        _require(row.get("answerable") in (None, True, False), f"bad answerability: {key}")
        _require(
            not isinstance(row.get("answerable"), int) or isinstance(row["answerable"], bool),
            f"answerable must be bool or null: {key}",
        )
        ids = row.get("gold_doc_ids", [])
        _require(
            isinstance(ids, list) and all(isinstance(x, str) for x in ids),
            f"bad gold_doc_ids: {key}",
        )
        pred = row.get("prediction")
        if pred is not None:
            _require(isinstance(pred, dict), f"bad prediction: {key}")
            _require(type(pred.get("abstained")) is bool, f"missing abstained boolean: {key}")
            if pred["abstained"]:
                _require(pred.get("answer") is None, f"abstention with non-null answer: {key}")
            else:
                _require(
                    isinstance(pred.get("answer"), str) and pred["answer"].strip(),
                    f"answered prediction without text: {key}",
                )
            sources = pred.get("sources", [])
            _require(
                isinstance(sources, list) and all(isinstance(s, dict) for s in sources),
                f"bad sources: {key}",
            )
            for source in sources:
                _require(isinstance(source.get("doc_id"), str), f"missing source doc_id: {key}")
                _require(
                    isinstance(source.get("reference"), str),
                    f"missing source reference: {key}",
                )
        review = row.get("human_review")
        if review is not None:
            _validate_review(review, key)
    return cases


def _validate_review(review: dict, key: str) -> None:
    _require(isinstance(review, dict), f"review not object: {key}")
    reviewer = review.get("reviewer")
    timestamp = review.get("reviewed_at")
    _require(isinstance(reviewer, str) and reviewer.strip(), f"reviewer missing: {key}")
    _require(isinstance(timestamp, str) and timestamp.strip(), f"reviewed_at missing: {key}")
    try:
        date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid review timestamp: {key}") from exc
    _require(date.tzinfo is not None, f"review timestamp requires timezone: {key}")
    _require(review.get("faithfulness") in FAITHFULNESS, f"invalid faithfulness: {key}")
    _require(review.get("correctness") in CORRECTNESS, f"invalid correctness: {key}")
    _require(
        review.get("hallucination") is None or type(review["hallucination"]) is bool,
        f"invalid hallucination: {key}",
    )


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 4) if denominator else None,
    }


def evaluate(cases: list[dict]) -> dict:
    """Score saved outputs; report denominator and missing coverage explicitly."""
    completed = [row for row in cases if row.get("prediction") is not None]
    answered = [r for r in completed if not r["prediction"]["abstained"]]
    abstained = [r for r in completed if r["prediction"]["abstained"]]
    answerable = [r for r in completed if r.get("answerable") is True]
    unanswerable = [r for r in completed if r.get("answerable") is False]
    gold = [r for r in completed if r.get("gold_doc_ids")]
    reviewed = [r for r in completed if r.get("human_review") is not None]
    assessed_answered = [r for r in answered if r.get("human_review") is not None]
    faith_judged = [
        r for r in assessed_answered if r["human_review"]["faithfulness"] != "not_applicable"
    ]
    correct_judged = [
        r for r in assessed_answered if r["human_review"]["correctness"] != "not_applicable"
    ]
    hallucination_judged = [
        r for r in assessed_answered if r["human_review"]["hallucination"] is not None
    ]
    citation_stats = []
    for row in answered:
        pred = row["prediction"]
        cited = CITATION.findall(pred["answer"])
        valid_refs = {s["reference"] for s in pred.get("sources", [])}
        structurally_valid = bool(cited) and all(f"[{n}]" in valid_refs for n in cited)
        citation_stats.append((bool(cited), structurally_valid))
    out = {
        "schema_version": SCHEMA,
        "evaluation_type": "offline_saved_predictions_only",
        "counts": {
            "total_cases": len(cases),
            "predictions_present": len(completed),
            "predictions_missing": len(cases) - len(completed),
            "answered": len(answered),
            "abstained": len(abstained),
            "answerability_labeled": len(answerable) + len(unanswerable),
            "answerability_unlabeled": len(completed) - len(answerable) - len(unanswerable),
            "gold_source_labeled": len(gold),
            "human_reviewed": len(reviewed),
            "human_reviewed_answered": len(assessed_answered),
        },
        "automatic": {
            "response_coverage": _rate(len(answered), len(completed)),
            "abstention_rate": _rate(len(abstained), len(completed)),
            "citation_presence_on_answered": _rate(
                sum(p for p, _ in citation_stats), len(answered)
            ),
            "citation_reference_valid_on_answered": _rate(
                sum(v for _, v in citation_stats), len(answered)
            ),
            "gold_document_in_cited_sources": _rate(
                sum(
                    bool(
                        set(r["gold_doc_ids"])
                        & {s["doc_id"] for s in r["prediction"].get("sources", [])}
                    )
                    for r in gold
                ),
                len(gold),
            ),
        },
        "answerability": {
            "answerable_answer_rate": _rate(
                sum(not r["prediction"]["abstained"] for r in answerable), len(answerable)
            ),
            "unanswerable_abstention_rate": _rate(
                sum(r["prediction"]["abstained"] for r in unanswerable), len(unanswerable)
            ),
            "unanswerable_false_answer_rate": _rate(
                sum(not r["prediction"]["abstained"] for r in unanswerable), len(unanswerable)
            ),
        },
        "human_review": {
            "faithfulness_fully_supported": _rate(
                sum(r["human_review"]["faithfulness"] == "supported" for r in faith_judged),
                len(faith_judged),
            ),
            "faithfulness_partially_supported": _rate(
                sum(r["human_review"]["faithfulness"] == "partial" for r in faith_judged),
                len(faith_judged),
            ),
            "faithfulness_unsupported": _rate(
                sum(r["human_review"]["faithfulness"] == "unsupported" for r in faith_judged),
                len(faith_judged),
            ),
            "answer_correct": _rate(
                sum(r["human_review"]["correctness"] == "correct" for r in correct_judged),
                len(correct_judged),
            ),
            "hallucination_flagged": _rate(
                sum(r["human_review"]["hallucination"] is True for r in hallucination_judged),
                len(hallucination_judged),
            ),
            "faithfulness_judgments": len(faith_judged),
            "correctness_judgments": len(correct_judged),
            "hallucination_judgments": len(hallucination_judged),
        },
        "abstention_reasons": dict(
            sorted(
                Counter(
                    str(r["prediction"].get("reason") or "unspecified") for r in abstained
                ).items()
            )
        ),
        "notes": [
            "Citation checks validate syntax/references only; they do NOT prove factual support.",
            "Gold-document hit means source-ID overlap, not answer correctness or faithfulness.",
            "Human labels are counted only with explicit reviewer identity and dated review.",
            "Rates are computed only on the named subset; null indicates zero denominator.",
            "Pre-existing smoke-test questions are not independent held-out evaluation.",
            "Do not generalize human-reviewed subset results to unreviewed cases.",
            "API faithfulness_verified=false is not replaced or overruled by this report.",
        ],
    }
    return out


def markdown_report(result: dict) -> str:
    """Create shareable aggregate-only report, without article texts or raw responses."""
    out = [
        "# Phase 15 — Grounded RAG Offline Evaluation",
        "",
        "**Status: aggregate analysis of saved predictions,",
        "not completed human validation.**",
        "",
    ]
    if "input_sha256" in result:
        out.append(f"Saved input SHA-256: `{result['input_sha256']}`")
        out.append("")
    for name in ("counts", "automatic", "answerability", "human_review"):
        out.extend(
            [
                f"## {name.replace('_', ' ').title()}",
                "",
                "| Measure | Value |",
                "|---|---:|",
            ]
        )
        for key, value in result[name].items():
            if isinstance(value, dict):
                fraction = "N/A" if value["value"] is None else f"{100 * value['value']:.2f}%"
                txt = f"{fraction} ({value['numerator']}/{value['denominator']})"
            else:
                txt = str(value)
            out.append(f"| `{key}` | {txt} |")
        out.append("")
    out.extend(["## Abstention Reasons", "", "| Reason | Cases |", "|---|---:|"])
    for reason, count in result["abstention_reasons"].items():
        out.append(f"| `{reason}` | {count} |")
    out.extend(["", "## Interpretation and limitations", ""])
    out.extend(f"- {note}" for note in result["notes"])
    return "\n".join(out) + "\n"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(data, encoding="utf-8")
    temp.replace(path)


def export_review_csv(cases: list[dict], dest: str | Path) -> None:
    """Export local reviewer worksheet; blank judgments must be manually filled."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLS)
        writer.writeheader()
        for row in cases:
            pred = row.get("prediction")
            if pred is None:
                continue
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answerable": (
                        "" if row.get("answerable") is None else str(row["answerable"]).lower()
                    ),
                    "answer": pred.get("answer") or "",
                    "abstained": str(pred["abstained"]).lower(),
                    "sources_json": json.dumps(pred.get("sources", []), ensure_ascii=False),
                    "reviewer": "",
                    "reviewed_at": "",
                    "faithfulness": "",
                    "correctness": "",
                    "hallucination": "",
                    "notes": "",
                }
            )


def import_review_csv(cases: list[dict], csv_path: str | Path) -> list[dict]:
    """Merge only completed, attributable manual judgments; leave others pending."""
    ids = {row["id"]: row for row in cases}
    observed = set()
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        _require(
            reader.fieldnames is not None and set(REVIEW_COLS) <= set(reader.fieldnames),
            "review worksheet is missing columns",
        )
        for line in reader:
            key = line["id"]
            _require(key in ids and key not in observed, f"unknown or duplicate review id: {key}")
            observed.add(key)
            row = ids[key]
            _require(row.get("prediction") is not None, f"review for missing prediction: {key}")
            _require(line["question"] == row["question"], f"changed question: {key}")
            _require(
                line["answer"] == (row["prediction"].get("answer") or ""),
                f"changed prediction: {key}",
            )
            fields = {
                k: (line[k] or "").strip()
                for k in (
                    "reviewer",
                    "reviewed_at",
                    "faithfulness",
                    "correctness",
                    "hallucination",
                    "notes",
                )
            }
            if not any(fields.values()):
                continue
            _require(
                all(fields[k] for k in ("reviewer", "reviewed_at", "faithfulness", "correctness")),
                f"partial review without provenance/judgments: {key}",
            )
            _require(
                fields["hallucination"].lower() in ("", "true", "false"),
                f"invalid hallucination boolean: {key}",
            )
            flag = (
                None if fields["hallucination"] == "" else fields["hallucination"].lower() == "true"
            )
            review = {
                "reviewer": fields["reviewer"],
                "reviewed_at": fields["reviewed_at"],
                "faithfulness": fields["faithfulness"],
                "correctness": fields["correctness"],
                "hallucination": flag,
                "notes": fields["notes"],
            }
            _validate_review(review, key)
            _require(
                row.get("human_review") is None,
                f"review already exists, refusing overwrite: {key}",
            )
            row["human_review"] = review
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 15: evaluate existing /rag/ask predictions")
    commands = parser.add_subparsers(dest="command", required=True)
    for cmd in ("score", "export-review", "import-review"):
        sub = commands.add_parser(cmd)
        sub.add_argument("--input", required=True)
        if cmd == "score":
            sub.add_argument("--output-dir", required=True)
        elif cmd == "export-review":
            sub.add_argument("--csv", required=True)
        else:
            sub.add_argument("--csv", required=True)
            sub.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = load_cases(args.input)
    if args.command == "score":
        report = evaluate(cases)
        report["input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
        dest = Path(args.output_dir)
        _atomic_write(
            dest / "metrics.json",
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write(dest / "report.md", markdown_report(report))
        print(
            f"Scored {report['counts']['predictions_present']}/{len(cases)} saved predictions."
            " No inference performed."
        )
        print(f"Saved: {dest / 'metrics.json'} and {dest / 'report.md'}")
    elif args.command == "export-review":
        export_review_csv(cases, args.csv)
        print("Blank human-review worksheet exported; no judgments were generated.")
    else:
        merged = import_review_csv(cases, args.csv)
        _atomic_write(
            Path(args.output),
            json.dumps({"schema_version": SCHEMA, "cases": merged}, ensure_ascii=False, indent=2)
            + "\n",
        )
        print("Reviewer-attributed judgments merged. Existing labels not overwritten.")


if __name__ == "__main__":
    main()

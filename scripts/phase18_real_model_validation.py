"""Phase 18: opt-in live-model probes; API smoke is not a quality benchmark.

Requires the Phase 17.4 helper alongside this script. Uses only the stdlib.
Each --probe explicitly loads the corresponding real model/index on the running API.
"""

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError

from phase17_release_validate import (
    PROBES,
    expected_project_version,
    http_json,
    run_checks,
    validate_base_url,
    verify_probe,
)

ALL_PROBES = (
    "classify",
    "search-dense",
    "search-hybrid",
    "search-rerank",
    "ask",
    "generate",
    "rag",
)


def _valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def assess_probe(name: str, status: int, body: dict) -> dict:
    """Return bounded metadata only; do not persist questions, answers or excerpts."""
    verify_probe(name, status, body)
    summary = {"probe": name, "technical_pass": True, "manual_review_required": True}
    if name == "classify":
        predictions = body["predictions"]
        assert predictions, "classifier returned zero predictions"
        assert all(
            isinstance(item["label"], str)
            and item["label"].strip()
            and _valid_number(item["score"])
            and 0 <= item["score"] <= 1
            for item in predictions
        ), "invalid classifier labels or scores"
        assert len({item["label"] for item in predictions}) == len(predictions), "duplicate labels"
        scores = [item["score"] for item in predictions]
        assert scores == sorted(scores, reverse=True), "predictions not score-ranked"
        summary["prediction_count"] = len(predictions)
    elif name.startswith("search-"):
        hits = body["results"]
        assert hits, "no retrieved articles"
        assert all(
            isinstance(hit.get("id"), str)
            and hit["id"].strip()
            and _valid_number(hit.get("score"))
            and isinstance(hit.get("label"), str)
            and isinstance(hit.get("snippet"), str)
            for hit in hits
        ), "retrieval metadata is incomplete"
        assert len({hit["id"] for hit in hits}) == len(hits), "duplicate article IDs"
        summary["retrieval_count"] = len(hits)
    elif name == "ask":
        summary["evidence_count"] = len(body["evidence"])
        summary["source_count"] = len(body["sources"])
        assert all(
            isinstance(e.get("doc_id"), str)
            and isinstance(e.get("sentence"), str)
            and _valid_number(e.get("score"))
            for e in body["evidence"]
        ), "invalid evidence records"
        summary["empty_evidence"] = len(body["evidence"]) == 0
    elif name == "generate":
        assert body["generated_text"].strip(), "generator returned empty text"
        assert body["output_tokens"] > 0, "generator returned zero output tokens"
        summary["output_tokens"] = body["output_tokens"]
    elif name == "rag":
        summary["abstained"] = body["abstained"]
        summary["source_count"] = len(body["sources"])
        summary["faithfulness_verified"] = False
        if body["abstained"]:
            summary["quality_status"] = "BLOCKED_BY_ABSTENTION"
        else:
            assert body["answer"].strip(), "non-abstained response has empty answer"
            assert body["sources"], "RAG answer has no citations/sources"
            summary["quality_status"] = "HUMAN_EVIDENCE_REVIEW_REQUIRED"
    else:
        raise ValueError(f"unknown probe: {name}")
    return summary


def execute_probes(
    base_url: str,
    selected: list[str],
    timeout: float,
    requester=http_json,
) -> list[dict]:
    """Execute explicit probes, catching endpoint failures without aborting others."""
    results = []
    for name in dict.fromkeys(selected):
        path, payload = PROBES[name]
        started = time.monotonic()
        try:
            status, body = requester(base_url, "POST", path, payload, timeout)
            summary = assess_probe(name, status, body)
            summary["http_status"] = status
        except (AssertionError, ValueError, KeyError, TypeError, OSError, URLError) as exc:
            summary = {
                "probe": name,
                "technical_pass": False,
                "manual_review_required": False,
                "error_type": type(exc).__name__,
                # Deliberately do not include exception string or response content.
            }
        summary["duration_s"] = round(time.monotonic() - started, 3)
        results.append(summary)
    return results


def evaluate_status(preflight: list[dict], probes: list[dict]) -> dict:
    """A technical PASS never automatically certifies production/release readiness."""
    technical_pass = bool(preflight) and all(x["passed"] for x in preflight)
    technical_pass = technical_pass and bool(probes) and all(x["technical_pass"] for x in probes)
    rag = next((x for x in probes if x["probe"] == "rag"), None)
    return {
        "technical_pass": technical_pass,
        "rag_abstention_observed": rag.get("abstained") if rag and rag["technical_pass"] else None,
        "human_quality_review_complete": False,
        "release_approved": False,
        "decision": "PENDING_HUMAN_REVIEW_AND_RELEASE_GATES",
        "notice": (
            "This is a live-model interface/sanity screen, not an accuracy, "
            "citation-faithfulness, security or production certification. "
            "Prior 80/80 RAG abstentions remain a documented limitation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--probe",
        action="append",
        choices=ALL_PROBES,
        required=True,
        help="Explicit model/index probe; repeat for additional components",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--expected-version", default=None)
    parser.add_argument("--report", default="phase18-real-model-report.json")
    args = parser.parse_args(argv)
    try:
        base_url = validate_base_url(args.base_url, args.allow_remote)
        if not 0 < args.timeout <= 3600:
            raise ValueError("--timeout must be between 0 and 3600 seconds")
        expected_version = args.expected_version or expected_project_version()
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))

    preflight = run_checks(base_url, expected_version, [], min(args.timeout, 30))
    if all(x["passed"] for x in preflight):
        probes = execute_probes(base_url, args.probe, args.timeout)
    else:
        probes = []  # Never initialize expensive models against a failed preflight.
    gate = evaluate_status(preflight, probes)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "version_checked": expected_version,
        "probes_requested": list(dict.fromkeys(args.probe)),
        "preflight": preflight,
        "probes": probes,
        "gate": gate,
        # No raw queries, documents, model text, response bodies or credentials.
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for check in preflight:
        print(f"{'PASS' if check['passed'] else 'FAIL'} preflight {check['check']}")
    for result in probes:
        outcome = "PASS" if result["technical_pass"] else "FAIL"
        print(f"{outcome} {result['probe']} ({result['duration_s']}s)")
        if result.get("quality_status") == "BLOCKED_BY_ABSTENTION":
            print("  NOTE: RAG abstained; answer-quality gate remains BLOCKED")
    print(f"Technical checks: {'PASS' if gate['technical_pass'] else 'FAIL'}")
    print("Release decision: PENDING manual review, model/data licenses and security checks")
    print(f"Report: {report_path}")
    return 0 if gate["technical_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

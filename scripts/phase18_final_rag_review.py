"""Phase 18 final live RAG review (local-only; NOT automated faithfulness certification).

Run against an already-started FastAPI server. This script stores only technical
metadata in the JSON report. Raw answers can optionally be printed to the terminal
for a HUMAN reviewer, but are not stored in the report.
"""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

CASES = (
    {
        "name": "mandra_answerable",
        "question": (
            "Σε ποιους επιρρίπτει πειθαρχικές ευθύνες το πόρισμα για τις πλημμύρες στη Μάνδρα;"
        ),
        "k": 5,
        "max_new_tokens": 128,
        "type": "known_answerable",
    },
    {
        "name": "broad_flood_exploratory",
        "question": "Τι ανακοινώθηκε για τις πλημμύρες;",
        "k": 5,
        "max_new_tokens": 128,
        "type": "exploratory_not_labeled_unanswerable",
    },
)


def validate_base_url(url: str) -> str:
    """Never send diagnostic corpus questions to remote or authenticated URLs."""
    parts = urlsplit(url)
    if (
        parts.scheme != "http"
        or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path not in {"", "/"}
        or parts.port is None
    ):
        raise ValueError("--base-url must be plain HTTP on localhost with an explicit port")
    return url.rstrip("/")


def http_json(base_url: str, method: str, path: str, payload: dict | None, timeout: float):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        # Do not expose response bodies from failed requests in stored reports.
        return error.code, {}


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split()).strip(" ?!.,;:·")


def assess_case(case: dict, status: int, body: dict) -> dict:
    """Structural validation; do not infer correctness from citation presence."""
    result = {
        "case": case["name"],
        "case_type": case["type"],
        "http_status": status,
        "technical_pass": False,
        "manual_review_required": True,
        "quality_status": "NOT_EVALUATED",
        "abstained": None,
        "reason": None,
        "source_count": 0,
        "has_citation": False,
        "question_echo_detected": False,
        "prompt_leak_detected": False,
        "verbatim_span_in_source": False,
    }
    if status != 200 or not isinstance(body, dict):
        result["quality_status"] = "HTTP_OR_RESPONSE_ERROR"
        return result
    sources = body.get("sources")
    answer = body.get("answer")
    abstained = body.get("abstained")
    if body.get("mode") != "grounded_rag" or not isinstance(sources, list):
        result["quality_status"] = "INVALID_RAG_RESPONSE"
        return result
    if not isinstance(abstained, bool):
        result["quality_status"] = "INVALID_ABSTENTION_FIELD"
        return result
    if any(
        not isinstance(source, dict)
        or not isinstance(source.get("doc_id"), str)
        or not isinstance(source.get("excerpt"), str)
        for source in sources
    ):
        result["quality_status"] = "INVALID_SOURCES"
        return result
    result["abstained"] = abstained
    result["reason"] = body.get("reason") if isinstance(body.get("reason"), str) else None
    result["source_count"] = len(sources)

    if abstained:
        if answer is not None:
            result["quality_status"] = "INVALID_ABSTAINED_RESPONSE"
            return result
        result["technical_pass"] = True
        result["quality_status"] = (
            "KNOWN_ANSWERABLE_ABSTENTION"
            if case["type"] == "known_answerable"
            else "ABSTENTION_NEEDS_CASE_REVIEW"
        )
        return result

    if not isinstance(answer, str) or not answer.strip() or not sources:
        result["quality_status"] = "INVALID_ANSWER_OR_SOURCES"
        return result
    citations = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    valid_citations = [n for n in citations if 1 <= n <= len(sources)]
    invalid_citations = [n for n in citations if n < 1 or n > len(sources)]
    result["has_citation"] = bool(valid_citations) and not invalid_citations
    result["question_echo_detected"] = _normalize(case["question"]) in _normalize(answer)
    result["prompt_leak_detected"] = bool(
        re.search(r"(?:^|\n)\s*(?:τεκμήρια|ερώτηση|evidence|question)\s*:", answer, re.I)
    )
    answer_without_refs = _normalize(re.sub(r"\[\d+\]", "", answer))
    result["verbatim_span_in_source"] = len(answer_without_refs) >= 4 and any(
        answer_without_refs in _normalize(s["excerpt"]) for s in sources
    )
    if (
        not result["has_citation"]
        or result["question_echo_detected"]
        or result["prompt_leak_detected"]
    ):
        result["quality_status"] = "STRUCTURAL_ANSWER_FAILURE"
        return result
    result["technical_pass"] = True
    result["quality_status"] = "ANSWER_REQUIRES_HUMAN_REVIEW"
    return result


def evaluate_gate(health_ok: bool, records: list[dict]) -> dict:
    technical_ok = (
        health_ok
        and len(records) == len(CASES)
        and all(record["technical_pass"] for record in records)
    )
    known = next((record for record in records if record["case"] == "mandra_answerable"), None)
    known_answered = bool(known and known["technical_pass"] and known["abstained"] is False)
    return {
        "technical_pass": technical_ok,
        "known_answerable_produced_answer": known_answered,
        "human_quality_review_complete": False,
        "release_approved": False,
        "decision": "PENDING_MANUAL_QUALITY_LICENSE_SECURITY_AND_RELEASE_REVIEW",
        "notice": (
            "A 200 response and a citation do not prove an answer is faithful. "
            "The broad flood question is not a validated negative example. "
            "Do not approve a release from this report alone."
        ),
    }


def run_review(base_url: str, timeout: float, requester=http_json, show_answers: bool = False):
    status, health = requester(base_url, "GET", "/health", None, min(timeout, 30))
    health_ok = status == 200 and health.get("status") == "ok"
    records = []
    if health_ok:
        for case in CASES:
            start = time.monotonic()
            try:
                status, body = requester(
                    base_url,
                    "POST",
                    "/rag/ask",
                    {key: case[key] for key in ("question", "k", "max_new_tokens")},
                    timeout,
                )
                record = assess_case(case, status, body)
                if show_answers:
                    print(f"\n{case['name']}\nQuestion: {case['question']}")
                    print(f"Answer: {body.get('answer') if isinstance(body, dict) else None}")
                    visible_sources = body.get("sources", []) if isinstance(body, dict) else []
                    for i, source in enumerate(visible_sources, 1):
                        print(f"Evidence {i}: {source.get('doc_id')}: {source.get('excerpt')}")
            except (OSError, ValueError, TimeoutError, URLError, json.JSONDecodeError) as error:
                record = {
                    "case": case["name"],
                    "case_type": case["type"],
                    "technical_pass": False,
                    "manual_review_required": True,
                    "quality_status": "TRANSPORT_OR_PARSE_ERROR",
                    "error_type": type(error).__name__,
                }
            record["duration_s"] = round(time.monotonic() - start, 3)
            records.append(record)
    return health_ok, records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--report", default="phase18-final-rag-review.json")
    parser.add_argument("--show-answers", action="store_true", help="Print raw answers locally")
    args = parser.parse_args(argv)
    try:
        url = validate_base_url(args.base_url)
        if not (0 < args.timeout <= 3600):
            raise ValueError("timeout must be within (0, 3600]")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        health_ok, records = run_review(url, args.timeout, show_answers=args.show_answers)
    except (OSError, ValueError, URLError, json.JSONDecodeError) as exc:
        print(f"Health check failed: {type(exc).__name__}")
        health_ok, records = False, []
    gate = evaluate_gate(health_ok, records)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "health_ok": health_ok,
        "checks": records,
        "gate": gate,
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Health: {'PASS' if health_ok else 'FAIL'}")
    for item in records:
        print(f"{item['case']}: {item['quality_status']} ({item['duration_s']}s)")
    print(f"Technical checks: {'PASS' if gate['technical_pass'] else 'FAIL'}")
    print("Release: PENDING manual quality, license and security review")
    print(f"Report: {path}")
    return 0 if gate["technical_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Optional, resumable live /rag/ask collection for Phase 15.

NEVER runs by default: explicit --confirm-live is mandatory. No imports of ML models.
Saves each successful response atomically, so interrupted CPU inference is not repeated.
Keep these files private until article/source redistribution has been reviewed.
"""

import argparse
import json
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path

from greek_nlp.evaluation.rag_eval import SCHEMA, _atomic_write, load_cases


def merge_existing(input_cases: list[dict], output_cases: list[dict]) -> list[dict]:
    """Resume saved responses, rejecting altered questions or silently removed ids."""
    originals = {row["id"]: row for row in input_cases}
    saved = {row["id"]: row for row in output_cases}
    extra = set(saved) - set(originals)
    if extra:
        raise ValueError(f"Input dropped existing case ids: {sorted(extra)}")
    merged = deepcopy(input_cases)
    for row in merged:
        prior = saved.get(row["id"])
        if prior is None:
            continue
        if prior["question"] != row["question"]:
            raise ValueError(f"Question changed since saved prediction: {row['id']}")
        if row.get("prediction") is not None and row["prediction"] != prior.get("prediction"):
            raise ValueError(f"Conflicting predictions: {row['id']}")
        row["prediction"] = prior.get("prediction")
        if prior.get("human_review") is not None:
            if row.get("human_review") not in (None, prior["human_review"]):
                raise ValueError(f"Conflicting human reviews: {row['id']}")
            row["human_review"] = prior["human_review"]
    return merged


def collect(cases: list[dict], output: Path, url: str, max_requests: int, timeout: int) -> int:
    """Exactly one POST per missing case; checkpoint immediately after every success."""
    completed = 0
    for row in cases:
        if row.get("prediction") is not None:
            continue
        if completed >= max_requests:
            break
        request = urllib.request.Request(
            url,
            data=json.dumps(
                {"question": row["question"], "k": 3, "max_new_tokens": 96}, ensure_ascii=False
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        print(f"Live inference {completed + 1}/{max_requests}: {row['id']}", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                _status = response.status
                payload = json.loads(response.read().decode("utf-8"))
            if _status != 200 or not isinstance(payload, dict):
                raise ValueError(f"Unexpected API response for {row['id']}: HTTP {_status}")
            if type(payload.get("abstained")) is not bool:
                raise ValueError(f"Malformed API response for {row['id']}")
            row["prediction"] = payload
            _atomic_write(
                output,
                json.dumps({"schema_version": SCHEMA, "cases": cases}, indent=2, ensure_ascii=False)
                + "\n",
            )
            completed += 1
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            print(f"STOPPED at {row['id']}: {exc}. Previous successes remain saved.", flush=True)
            raise
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit opt-in /rag/ask batch collection")
    parser.add_argument("--cases", required=True, help="Author-curated input cases JSON")
    parser.add_argument("--output", required=True, help="Private checkpointed JSON output")
    parser.add_argument("--url", default="http://127.0.0.1:8000/rag/ask")
    parser.add_argument("--max-requests", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--confirm-live", action="store_true", help="Required: calls real model")
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("Live inference is disabled; add --confirm-live after preparing the dataset")
    if args.url != "http://127.0.0.1:8000/rag/ask":
        parser.error("Only the explicit local /rag/ask endpoint is supported")
    if args.max_requests < 1 or args.timeout < 1:
        parser.error("max-requests and timeout must be >= 1")
    cases = load_cases(args.cases)
    _ = Path(args.output)
    if _.resolve() == Path(args.cases).resolve():
        parser.error("--output must differ from --cases: never overwrite the original cases")
    if _.exists():
        cases = merge_existing(cases, load_cases(_))
    pending = sum(c.get("prediction") is None for c in cases)
    if pending == 0:
        print("No missing predictions. No inference performed.")
        return
    print(f"Remaining: {pending}; this run: at most {min(pending, args.max_requests)} POST calls.")
    total = collect(cases, _, args.url, args.max_requests, args.timeout)
    print(f"Saved {total} new responses. Run again to resume remaining cases.")


if __name__ == "__main__":
    main()

"""Private Phase 16 Greek-generation quality screen on a *local* llama.cpp server.

This is an exploratory, author-reviewed development screen, NOT a validated
benchmark. It does not access Phase 15 predictions or held-out cases.

Example (dry run):
    python scripts/phase16_quality_screen.py --tag krikri8b_q4km --model-match Krikri
Live run:
    python scripts/phase16_quality_screen.py --tag krikri8b_q4km --model-match Krikri --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evaluation" / "phase16_greek_quality_dev_cases.json"
OUT_DIR = ROOT / "results" / "phase16_local"
SYSTEM_PROMPT = (
    "Είσαι βοηθός που απαντά σε σωστά νέα ελληνικά. "
    "Ακολούθησε ακριβώς την οδηγία της ερώτησης. "
    "Μην επινοείς πληροφορίες που δεν περιλαμβάνονται στα παρεχόμενα στοιχεία."
)


def local_base(value: str) -> str:
    uri = urlparse(value)
    if uri.scheme != "http" or uri.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only loopback HTTP servers are allowed")
    if uri.username or uri.password or uri.query or uri.fragment or uri.path not in {"", "/"}:
        raise ValueError("Use an unadorned local base URL")
    return value.rstrip("/")


def valid_tag(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", value):
        raise ValueError("--tag must be 1..40 lower-case letters, digits, _ or -")
    return value


def load_cases(path: Path = CASES) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "DEVELOPMENT_ONLY_NOT_GOLD":
        raise ValueError("Refusing non-development quality dataset")
    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases or len(cases) > 20:
        raise ValueError("Expected 1..20 development cases")
    ids = set()
    for row in cases:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k].strip()
            for k in ("id", "category", "question", "review_criterion")
        ):
            raise ValueError("Every case requires id/category/question/review_criterion")
        if row["id"] in ids or len(row["question"]) > 1500:
            raise ValueError("Duplicate id or excessively long question")
        ids.add(row["id"])
    return cases


def get_json(url: str, timeout: int) -> dict:
    with urlopen(Request(url, method="GET"), timeout=timeout) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def invoke(url: str, payload: dict, timeout: int) -> tuple[dict, float]:
    request = Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    start = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status: {response.status}")
        result = json.load(response)
    return result, round((time.perf_counter() - start) * 1000, 2)


def response_record(row: dict, reply: dict, elapsed_ms: float) -> dict:
    choices = reply.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("No response choices")
    choice = choices[0]
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise ValueError("No string assistant content")
    usage = reply.get("usage") or {}
    return {
        "id": row["id"],
        "category": row["category"],
        "question": row["question"],
        "review_criterion": row["review_criterion"],
        "answer": content,
        "finish_reason": choice.get("finish_reason"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "http_elapsed_ms": elapsed_ms,
        "human_review": None,
    }


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-match", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:9931")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    tag = valid_tag(args.tag)
    base = local_base(args.url)
    if not args.model_match.strip():
        parser.error("--model-match must be nonempty")
    if not 1 <= args.max_tokens <= 256 or not 1 <= args.max_cases <= 20:
        parser.error("max-tokens must be 1..256 and max-cases 1..20")
    cases = load_cases()[: args.max_cases]
    output = OUT_DIR / f"phase16_quality_{tag}_DEV_ONLY.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior quality results: {output}")
    print(f"QUALITY SCREEN: {len(cases)} development cases; output={output}")
    if not args.execute:
        print("DRY RUN: no inference. Add --execute to send requests to localhost.")
        return
    health = get_json(base + "/health", 15)
    if health.get("status") != "ok":
        raise RuntimeError("llama.cpp server not ready")
    models = get_json(base + "/v1/models", 15).get("data", [])
    if not models or not isinstance(models[0].get("id"), str):
        raise ValueError("Could not identify serving model")
    model_id = models[0]["id"]
    if args.model_match.casefold() not in model_id.casefold():
        raise RuntimeError(f"Wrong model loaded: {model_id!r}")
    report = {
        "schema_version": "phase16.quality_screen.v1",
        "scope": "DEVELOPMENT_ONLY_NOT_GOLD",
        "status": "PARTIAL",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "model": model_id,
        "model_match": args.model_match,
        "system_prompt": SYSTEM_PROMPT,
        "case_file_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "caveats": [
            "Synthetic and development examples only; not an independent gold benchmark.",
            "Human inspection required: no automatic correctness, Greek fluency "
            "or faithfulness scores.",
            "Raw generated answers stay in ignored local results, not GitHub.",
            "No changes to FastAPI or Phase 15 evaluation assets.",
        ],
        "records": [],
    }
    for index, row in enumerate(cases, start=1):
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        response, ms = invoke(base + "/v1/chat/completions", payload, args.timeout)
        record = response_record(row, response, ms)
        report["records"].append(record)
        atomic_json(output, report)
        print(f"\n[{index}/{len(cases)}] {row['id']} ({ms / 1000:.2f} s)")
        print("ANSWER:", record["answer"][:1500])
        print("FINISH:", record["finish_reason"])
    report["status"] = "COMPLETE"
    atomic_json(output, report)
    print("\nSAVED:", output)
    print("IMPORTANT: author-review each answer; no automatic quality claim.")


if __name__ == "__main__":
    main()

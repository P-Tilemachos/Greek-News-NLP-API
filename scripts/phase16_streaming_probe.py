"""Phase 16 GGUF SSE streaming latency probe (development data only).

Calls the local llama.cpp OpenAI-compatible chat endpoint. It does not
modify the FastAPI project or store generated text. Explicit --execute required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evaluation" / "phase16_development_cases.json"
OUTPUT = ROOT / "results" / "phase16_local" / "phase16_gguf_streaming_DEV_ONLY.json"
CASE_ID = "dev_generate_greeting"


def safe_base_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Only unadorned local HTTP server URLs are allowed")
    return value.rstrip("/")


def load_development_request() -> tuple[str, str, int]:
    data = json.loads(CASES.read_text(encoding="utf-8"))
    if data.get("status") != "DEVELOPMENT_ONLY":
        raise ValueError("Expected DEVELOPMENT_ONLY case file")
    rows = [r for r in data["cases"] if r.get("id") == CASE_ID]
    if len(rows) != 1 or rows[0]["endpoint"] != "/generate":
        raise ValueError("Expected one matching /generate development case")
    request = rows[0]["request"]
    prompt = request["prompt"]
    max_tokens = request["max_new_tokens"]
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 400:
        raise ValueError("Invalid development prompt")
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 48:
        raise ValueError("Unexpected max token count")

    # Importing this constant does not initialize the model.
    from greek_nlp.llm.generator import DEFAULT_SYSTEM_PROMPT

    return DEFAULT_SYSTEM_PROMPT, prompt, max_tokens


def parse_data_line(line: bytes) -> dict | str | None:
    """Read one UTF-8 SSE data line, ignoring keepalive and event metadata."""
    raw = line.strip()
    if not raw.startswith(b"data:"):
        return None
    raw = raw[5:].strip()
    if raw == b"[DONE]":
        return "DONE"
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Unexpected streaming payload")
    return value


def content_from_event(event: dict) -> str:
    """Return visible text, not role/finish-only streaming events."""
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if isinstance(content, str) else ""


def perform_stream(url: str, body: dict, timeout: float) -> dict:
    start = time.perf_counter()
    first_event = None
    first_content = None
    content_chars = 0
    content_events = 0
    done = False
    finish_reason = None
    server_completion_tokens = None
    with requests.post(
        url,
        json=body,
        stream=True,
        timeout=(10, timeout),
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            event = parse_data_line(line)
            if event is None:
                continue
            now = time.perf_counter()
            if event == "DONE":
                done = True
                break
            if first_event is None:
                first_event = now
            if "error" in event:
                raise RuntimeError("Streaming server returned an error event")
            content = content_from_event(event)
            if content:
                if first_content is None:
                    first_content = now
                content_chars += len(content)
                content_events += 1
            usage = event.get("usage") or {}
            if isinstance(usage.get("completion_tokens"), int):
                server_completion_tokens = usage["completion_tokens"]
            for choice in event.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if not done or first_content is None:
        raise RuntimeError("SSE stream incomplete or generated no visible content")
    return {
        "elapsed_ms": round((end - start) * 1000, 2),
        "first_sse_event_ms": round((first_event - start) * 1000, 2),
        "first_content_ms": round((first_content - start) * 1000, 2),
        "content_chunks": content_events,
        "output_chars": content_chars,
        "server_completion_tokens": server_completion_tokens,
        "finish_reason": finish_reason,
        "sse_done_received": done,
    }


def atomic_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    base = safe_base_url(args.url)
    system, prompt, cap = load_development_request()
    health = requests.get(base + "/health", timeout=15).json()
    if health.get("status") != "ok":
        raise RuntimeError("llama.cpp is not ready")
    models = requests.get(base + "/v1/models", timeout=15).json().get("data") or []
    if not models or not isinstance(models[0].get("id"), str):
        raise RuntimeError("No llama.cpp model found")
    model_id = models[0]["id"]
    print("READY:", model_id, "development case:", CASE_ID)
    if not args.execute:
        print("DRY RUN ONLY. Pass --execute to generate streaming responses.")
        return
    if OUTPUT.exists():
        raise FileExistsError(f"Existing results will not be overwritten: {OUTPUT}")

    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": cap,
        "stream": True,
    }
    result = {
        "status": "PARTIAL",
        "scope": "DEVELOPMENT_ONLY_NOT_HELD_OUT",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "backend": "llama.cpp_gguf_Q4_K_M",
        "model": model_id,
        "case_id": CASE_ID,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "max_tokens": cap,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "notes": [
            "First content is SSE arrival time, not necessarily token-internal TTFT.",
            "SSE chunks are not equivalent to token counts.",
            "No raw prompt or generated text stored.",
            "No peak RAM, RAG quality, or production streaming measured.",
        ],
        "records": [],
    }
    for i in range(args.warmup + args.repeat):
        measured = i >= args.warmup
        print("MEASURE" if measured else "WARMUP", i + 1, flush=True)
        record = perform_stream(base + "/v1/chat/completions", body, args.timeout)
        if measured:
            record["repeat_index"] = i - args.warmup + 1
            result["records"].append(record)
            atomic_save(OUTPUT, result)
            print(
                f"SAVED first-content={record['first_content_ms']}ms "
                f"end-to-end={record['elapsed_ms']}ms chunks={record['content_chunks']}"
            )
    result["status"] = "COMPLETE"
    result["median_first_content_ms"] = round(
        statistics.median(r["first_content_ms"] for r in result["records"]), 2
    )
    result["median_elapsed_ms"] = round(
        statistics.median(r["elapsed_ms"] for r in result["records"]), 2
    )
    atomic_save(OUTPUT, result)
    print("SAVED:", OUTPUT)
    print(
        "MEDIAN: first content",
        result["median_first_content_ms"],
        "ms; total",
        result["median_elapsed_ms"],
        "ms",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9931")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.warmup <= 10 or not 1 <= args.repeat <= 20:
        parser.error("Only warmup 0-10 and repeat 1-20 are allowed")
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be between 1 and 600 seconds")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        parser.exit(2, f"PHASE 16 STOP: {exc}\n")


if __name__ == "__main__":
    main()

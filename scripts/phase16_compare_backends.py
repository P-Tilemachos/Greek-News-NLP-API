"""Phase 16 matched-prompt CPU HTTP benchmark (development only).

Compares the unchanged FastAPI /generate route with a separate llama.cpp
/v1/chat/completions route. No model is loaded by this script. Runs on localhost
only, never records raw prompts or model-generated text, and requires --execute.
This is a backend+quantization comparison, not an isolated quantization study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "phase16_local"
CASES = ROOT / "evaluation" / "phase16_development_cases.json"
CASE_ID = "dev_generate_greeting"


def validate_tag(tag: str | None) -> str | None:
    """Allow filename-safe tags only; never accept a path component."""
    if tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", tag):
        raise ValueError(
            "--tag must be 1..40 ASCII letters, digits, '_' or '-' (starts alphanumeric)"
        )
    return tag


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def localhost(base: str) -> str:
    u = urlparse(base)
    if u.scheme != "http" or u.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Refusing non-local benchmark endpoint")
    if u.username or u.password or u.query or u.fragment or u.path not in {"", "/"}:
        raise ValueError("Endpoint must be an unadorned local HTTP base URL")
    return base.rstrip("/")


def get_json(url: str, timeout: float) -> dict:
    with urlopen(Request(url, method="GET"), timeout=timeout) as r:
        result = json.load(r)
    if not isinstance(result, dict):
        raise ValueError("Server returned non-object JSON")
    return result


def post_json(url: str, body: dict, timeout: float) -> tuple[dict, float]:
    req = Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
    )
    begin = time.perf_counter()
    with urlopen(req, timeout=timeout) as r:
        reply = json.load(r)
        if r.status != 200:
            raise RuntimeError(f"Unexpected HTTP status {r.status}")
    elapsed_ms = (time.perf_counter() - begin) * 1000
    if not isinstance(reply, dict):
        raise ValueError("Expected response JSON object")
    return reply, elapsed_ms


def load_workload() -> tuple[str, str, int]:
    obj = json.loads(CASES.read_text(encoding="utf-8"))
    if obj.get("status") != "DEVELOPMENT_ONLY":
        raise ValueError("Only DEVELOPMENT_ONLY cases are allowed")
    matched = [c for c in obj.get("cases", []) if c.get("id") == CASE_ID]
    if len(matched) != 1 or matched[0].get("endpoint") != "/generate":
        raise ValueError("Expected one /generate development case")
    case = matched[0]["request"]
    user_prompt = case["prompt"]
    cap = case["max_new_tokens"]
    if not isinstance(user_prompt, str) or not 1 <= len(user_prompt) <= 400:
        raise ValueError("Invalid bounded development prompt")
    if not isinstance(cap, int) or not 1 <= cap <= 48:
        raise ValueError("Unexpected output-token budget")
    # Import current project prompt without initializing the model.
    from greek_nlp.llm.generator import DEFAULT_SYSTEM_PROMPT

    if not isinstance(DEFAULT_SYSTEM_PROMPT, str) or not DEFAULT_SYSTEM_PROMPT:
        raise ValueError("Could not load the current /generate system prompt")
    return DEFAULT_SYSTEM_PROMPT, user_prompt, cap


def build_request(
    backend: str, system_prompt: str, prompt: str, cap: int, model_id: str | None
) -> dict:
    if backend == "torch":
        # Current FastAPI route inserts DEFAULT_SYSTEM_PROMPT internally.
        return {"prompt": prompt, "max_new_tokens": cap}
    if not model_id:
        raise ValueError("llama.cpp model id required")
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": cap,
        "temperature": 0,
        "stream": False,
    }


def summarise_response(backend: str, resp: dict, elapsed_ms: float, index: int) -> dict:
    if backend == "torch":
        generated = resp.get("generated_text")
        n_input = resp.get("input_tokens")
        n_output = resp.get("output_tokens")
        model_id = resp.get("model")
        device = resp.get("device")
        timings = None
    else:
        candidates = resp.get("choices") or []
        generated = (candidates[0].get("message") or {}).get("content") if candidates else None
        usage = resp.get("usage") or {}
        n_input = usage.get("prompt_tokens")
        n_output = usage.get("completion_tokens")
        model_id = resp.get("model")
        device = "llama.cpp_cpu"
        raw_timings = resp.get("timings") or {}
        timings = {
            key: raw_timings[key]
            for key in (
                "prompt_per_second",
                "predicted_per_second",
                "prompt_n",
                "predicted_n",
                "prompt_ms",
                "predicted_ms",
            )
            if key in raw_timings
        }
    if not isinstance(generated, str):
        raise ValueError("No generated text; this is not a successful generation")
    if not isinstance(n_output, int) or n_output < 1:
        raise ValueError("Missing positive output-token count")
    return {
        "repeat_index": index,
        "elapsed_ms": round(elapsed_ms, 2),
        "input_tokens": n_input,
        "output_tokens": n_output,
        "output_chars": len(generated),
        "output_sha256": digest(generated),
        "generated_tokens_per_end_to_end_second": round(n_output / (elapsed_ms / 1000), 3),
        "model": model_id,
        "device": device,
        "server_timings": timings,
    }


def atomic_save(target: Path, data: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)


def compare(paths: list[Path], *, allow_gguf_compare: bool = False) -> None:
    if len(paths) != 2:
        raise ValueError("--compare requires exactly two JSON files")
    data = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(d.get("status") != "COMPLETE" for d in data):
        raise ValueError("Both comparison runs must be COMPLETE")
    keys = ("user_prompt_sha256", "system_prompt_sha256", "max_new_tokens")
    for k in keys:
        if data[0].get(k) != data[1].get(k):
            raise ValueError(f"Cannot compare: {k} differs")
    backends = {x["backend"] for x in data}
    if backends == {"gguf"} and allow_gguf_compare:
        model_ids = [x.get("model_id_reported") for x in data]
        if not all(isinstance(model, str) and model for model in model_ids):
            raise ValueError("GGUF comparisons require reported model IDs")
        if model_ids[0] == model_ids[1]:
            raise ValueError("Expected distinct GGUF model IDs")
    elif backends != {"torch", "gguf"}:
        raise ValueError(
            "Expected torch versus GGUF; use --allow-gguf-compare "
            "to compare two distinct GGUF models"
        )
    if any(not item.get("records") for item in data):
        raise ValueError("No measured records in comparison")
    print("\nMATCHED DEVELOPMENT PROMPT COMPARISON (not a production quality evaluation)")
    print("Backend | n | median HTTP s | median output tokens | median output tok/E2E s")
    for d in sorted(data, key=lambda x: (x["backend"], str(x.get("model_id_reported")))):
        rows = d["records"]
        label = d.get("model_tag") or d.get("model_id_reported") or d["backend"]
        print(
            f"{d['backend']} ({label}) | {len(rows)} | "
            f"{statistics.median(r['elapsed_ms'] for r in rows) / 1000:.2f} | "
            f"{statistics.median(r['output_tokens'] for r in rows):.1f} | "
            f"{statistics.median(r['generated_tokens_per_end_to_end_second'] for r in rows):.2f}"
        )
    print("CAVEATS: Different backend and weight precision; tokenizers and templates may differ.")
    print("E2E token rate includes processing and request overhead; it is NOT decode TPS.")
    print("This run does not measure TTFT or peak server RAM. Check the generated text separately.")


def run(args: argparse.Namespace) -> None:
    root = localhost(args.url)
    tag = validate_tag(getattr(args, "tag", None))
    model_match = getattr(args, "model_match", None)
    if model_match and args.backend != "gguf":
        raise ValueError("--model-match only applies to GGUF servers")
    system_prompt, prompt, cap = load_workload()
    if args.backend == "gguf":
        health = get_json(root + "/health", 15)
        if health.get("status") != "ok":
            raise RuntimeError(f"llama.cpp health not ready: {health.get('status')}")
        models = get_json(root + "/v1/models", 15).get("data") or []
        if not models or not isinstance(models[0].get("id"), str):
            raise ValueError("llama.cpp did not provide a model id")
        model_id = models[0]["id"]
        if model_match and model_match.casefold() not in model_id.casefold():
            raise ValueError(
                f"Unexpected model {model_id!r}; expected name containing {model_match!r}"
            )
        print("SERVER MODEL:", model_id)
        route = "/v1/chat/completions"
    else:
        get_json(root + "/health", 15)
        model_id = None
        route = "/generate"
    body = build_request(args.backend, system_prompt, prompt, cap, model_id)
    suffix = f"_{tag}" if tag else ""
    target = args.output or OUT_DIR / f"phase16_matched_{args.backend}{suffix}_DEV_ONLY.json"
    if target.exists():
        raise FileExistsError(f"Will not overwrite prior benchmark: {target}")
    data = {
        "status": "PARTIAL",
        "scope": "DEVELOPMENT_ONLY_NOT_HELD_OUT",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "backend": args.backend,
        "base_url": root,
        "model_id_reported": model_id,
        "model_tag": tag,
        "expected_model_match": model_match,
        "source_cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
        "user_prompt_sha256": digest(prompt),
        "system_prompt_sha256": digest(system_prompt),
        "max_new_tokens": cap,
        "generation": (
            "greedy_requested (temperature=0 for GGUF; "
            "/generate uses its configured deterministic decode)"
        ),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "records": [],
    }
    print(
        f"Backend={args.backend} case={CASE_ID} max_tokens={cap}; "
        f"warmup={args.warmup}, measured={args.repeat}"
    )
    for i in range(args.warmup + args.repeat):
        print(f"{'WARMUP' if i < args.warmup else 'MEASURE'} {i + 1}", flush=True)
        resp, elapsed_ms = post_json(root + route, body, args.timeout)
        if i >= args.warmup:
            record = summarise_response(args.backend, resp, elapsed_ms, i - args.warmup + 1)
            data["records"].append(record)
            atomic_save(target, data)
            print(f"SAVED latency={elapsed_ms / 1000:.2f}s output_tokens={record['output_tokens']}")
    data["status"] = "COMPLETE"
    atomic_save(target, data)
    print("BENCHMARK SAVED:", target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "gguf"))
    parser.add_argument("--url", default=None)
    parser.add_argument("--execute", action="store_true", help="Authorize local live inference")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag", help="Safe suffix for model-specific results (e.g. krikri8b_q4km)")
    parser.add_argument(
        "--model-match", help="Require the GGUF server model ID to contain this text"
    )
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument(
        "--allow-gguf-compare",
        action="store_true",
        help="Explicitly allow comparisons of two distinct GGUF model IDs",
    )
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, allow_gguf_compare=args.allow_gguf_compare)
        return
    if not args.backend:
        parser.error("Specify --backend or --compare")
    try:
        validate_tag(args.tag)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= args.warmup <= 2 or not 1 <= args.repeat <= 5:
        parser.error("Allow 0..2 warmups and 1..5 measured repetitions only")
    if not 1 <= args.timeout <= 1200:
        parser.error("--timeout must be 1..1200 seconds")
    if not args.execute:
        print("DRY RUN: no inference. Pass --execute to benchmark a local endpoint.")
        return
    if args.url is None:
        args.url = "http://127.0.0.1:9931" if args.backend == "gguf" else "http://127.0.0.1:8000"
    run(args)


if __name__ == "__main__":
    main()

"""Phase 16 local inference benchmarking: explicit, bounded, DEV-only runs.

No inference unless --execute. Outputs metadata, never prompts or answer text.
Run from repository root. No third-party benchmark dependencies required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "phase16_local"
DEFAULT_CASES = ROOT / "evaluation" / "phase16_development_cases.json"
VERSIONS = ("torch", "transformers", "fastapi", "uvicorn", "psutil", "bitsandbytes")


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def hardware_metadata() -> dict:
    info = {
        "python": sys.version.split()[0],
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "package_versions": {name: package_version(name) for name in VERSIONS},
        "client_env_llm_device": os.getenv("LLM_DEVICE"),
        "client_env_llm_model_id": os.getenv("LLM_MODEL_ID"),
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "src/greek_nlp/llm/generator.py",
                "src/greek_nlp/rag/grounded.py",
                "src/greek_nlp/api/main.py",
            )
        },
    }
    try:
        import psutil

        info["physical_ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
    except ImportError:
        info["physical_ram_total_gb"] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 2**30, 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            info["cuda_devices"] = []
    except (ImportError, OSError) as exc:
        info["cuda_available"] = None
        info["cuda_probe_error_type"] = type(exc).__name__
    return info


def check_local_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Benchmarking is limited to a local HTTP server")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Base URL cannot include credentials, query or fragment")


def get_json(url: str, *, timeout: float) -> dict:
    with urlopen(Request(url, method="GET"), timeout=timeout) as response:
        return json.load(response)


def openapi_fields(base_url: str, *, timeout: float) -> dict:
    schema = get_json(base_url + "/openapi.json", timeout=timeout)
    parts = schema.get("components", {}).get("schemas", {})
    result = {}
    for endpoint in ("/generate", "/rag/ask"):
        method = schema.get("paths", {}).get(endpoint, {}).get("post", {})
        content = method.get("requestBody", {}).get("content", {}).get("application/json", {})
        model = content.get("schema", {})
        ref = model.get("$ref", "")
        model = parts.get(ref.rsplit("/", 1)[-1], model) if ref else model
        result[endpoint] = {
            "found": bool(method),
            "required": model.get("required", []),
            "properties": sorted(model.get("properties", {})),
        }
    return result


def load_development_cases(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("status") != "DEVELOPMENT_ONLY":
        raise ValueError("Refusing to benchmark a set not marked DEVELOPMENT_ONLY")
    cases = obj.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError("Development cases must contain 1..12 records")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Malformed case id")
        if case["id"] in ids:
            raise ValueError("Duplicate case id")
        ids.add(case["id"])
        if case.get("endpoint") not in {"/generate", "/rag/ask"}:
            raise ValueError("Only /generate and /rag/ask are allowed")
        body = case.get("request")
        if not isinstance(body, dict) or not body:
            raise ValueError("Each case must contain a JSON request object")
    return cases


def post_json(url: str, body: dict, timeout: float) -> tuple[int, dict, float]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    before = time.perf_counter()
    with urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
        status = response.status
    milliseconds = (time.perf_counter() - before) * 1000
    return status, payload, milliseconds


def summarise(result: dict) -> dict:
    pred = result["response"]
    answer = pred.get("answer") if result["endpoint"] == "/rag/ask" else pred.get("generated_text")
    return {
        "case_id": result["id"],
        "endpoint": result["endpoint"],
        "repeat_index": result["repeat_index"],
        "http_status": result["http_status"],
        "elapsed_ms": round(result["elapsed_ms"], 2),
        "abstained": pred.get("abstained"),
        "abstention_reason": pred.get("reason"),
        "answer_chars": len(answer) if isinstance(answer, str) else 0,
        "input_tokens": pred.get("input_tokens"),
        "output_tokens": pred.get("output_tokens"),
        "model": pred.get("model"),
        "device": pred.get("device"),
        "retrieved_sources": (
            len(pred["sources"]) if isinstance(pred.get("sources"), list) else None
        ),
    }


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def stats(records: list[dict]) -> dict:
    result = {}
    for endpoint in ("/generate", "/rag/ask"):
        values = [r["elapsed_ms"] for r in records if r["endpoint"] == endpoint]
        if values:
            result[endpoint] = {
                "n": len(values),
                "median_latency_ms": round(statistics.median(values), 2),
                "min_latency_ms": round(min(values), 2),
                "max_latency_ms": round(max(values), 2),
                "p95_latency_ms": None,  # too few development runs for reliable p95
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight", action="store_true", help="Store hardware and dependency inventory"
    )
    parser.add_argument(
        "--inspect-openapi", action="store_true", help="Inspect local request schemas"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Explicitly authorize LIVE inference"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "phase16_baseline_DEV_ONLY.json")
    args = parser.parse_args()
    check_local_url(args.base_url)
    if not 1 <= args.repeat <= 3:
        parser.error("--repeat must be 1..3 for this initial development diagnostic")
    if not 1 <= args.timeout <= 1200:
        parser.error("--timeout must be 1..1200 seconds")
    if args.preflight:
        target = OUT_DIR / "phase16_preflight.json"
        data = {"recorded_utc": datetime.now(UTC).isoformat()}
        data.update(hardware_metadata())
        atomic_json(target, data)
        print("PREFLIGHT SAVED:", target)
    if args.inspect_openapi:
        print(json.dumps(openapi_fields(args.base_url, timeout=10), ensure_ascii=False, indent=2))
    if not args.execute:
        if not args.preflight and not args.inspect_openapi:
            print("No live inference. Use --preflight, --inspect-openapi or --execute.")
        return
    cases = load_development_cases(args.cases)
    if args.output.exists():
        raise FileExistsError(
            f"Baseline output already exists; choose another --output: {args.output}"
        )
    records = []
    log = {
        "status": "PARTIAL",
        "scope": "DEVELOPMENT_ONLY_NOT_HELD_OUT",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "source_cases_sha256": sha256(args.cases),
        "repetitions": args.repeat,
        "base_url": args.base_url,
        "records": records,
    }
    for case in cases:
        for repetition in range(args.repeat):
            print(f"RUNNING: {case['id']} repeat={repetition + 1}", flush=True)
            try:
                status, response, elapsed = post_json(
                    args.base_url + case["endpoint"], case["request"], args.timeout
                )
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                log["error_type"] = type(exc).__name__
                log["error_case_id"] = case["id"]
                atomic_json(args.output, log)
                raise
            if not isinstance(response, dict):
                raise ValueError("API response must be a JSON object")
            records.append(
                summarise(
                    {
                        "id": case["id"],
                        "endpoint": case["endpoint"],
                        "repeat_index": repetition + 1,
                        "http_status": status,
                        "response": response,
                        "elapsed_ms": elapsed,
                    }
                )
            )
            log["summary"] = stats(records)
            atomic_json(args.output, log)
            print(f"SAVED: {case['id']} latency={elapsed:.0f} ms", flush=True)
    log["status"] = "COMPLETE"
    atomic_json(args.output, log)
    print("DEVELOPMENT BASELINE SAVED:", args.output)
    print("NOTE: No cold/warm decomposition, true TTFT, throughput or peak server RAM yet.")


if __name__ == "__main__":
    main()

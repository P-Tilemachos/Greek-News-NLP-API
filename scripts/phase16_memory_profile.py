"""Phase 16 CPU server RAM profiler for the *existing* matched backend benchmark.

This helper does not modify models or API code. It monitors a local serving PID
(and its descendants) while running phase16_compare_backends.py using fresh
output paths. Requires --execute to make requests; all requests remain local.

Windows: psutil RSS is process working set, NOT GPU VRAM and not unique private
allocation. Model file mappings/shared pages can inflate summed RSS.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import psutil

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "phase16_local"
BENCH_SCRIPT = ROOT / "scripts" / "phase16_compare_backends.py"
PORTS = {"gguf": 9931, "torch": 8000}


def validate_tag(tag: str | None) -> str | None:
    """Apply the same safe filename rules as the matched benchmark."""
    if tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", tag):
        raise ValueError(
            "--tag must be 1..40 ASCII letters, digits, '_' or '-' (starts alphanumeric)"
        )
    return tag


def resolve_listener_pid(port: int) -> int:
    """Find a unique TCP listening process; else ask for an explicit --pid."""
    pids = {
        row.pid
        for row in psutil.net_connections(kind="tcp")
        if row.pid is not None
        and row.status == psutil.CONN_LISTEN
        and row.laddr
        and row.laddr.port == port
    }
    if len(pids) != 1:
        raise ValueError(
            f"Cannot identify one process listening on port {port}: {sorted(pids)}. "
            "Supply --pid explicitly (PowerShell: "
            f"(Get-NetTCPConnection -LocalPort {port} -State Listen).OwningProcess)."
        )
    return next(iter(pids))


def memory_snapshot(pid: int) -> tuple[int, int]:
    """Return total RSS bytes and count of accessible processes in PID tree."""
    root = psutil.Process(pid)
    members = [root, *root.children(recursive=True)]
    rss = 0
    count = 0
    for proc in {p.pid: p for p in members}.values():
        try:
            with proc.oneshot():
                rss += proc.memory_info().rss
                count += 1
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    if count == 0:
        raise RuntimeError("Serving PID or all descendants became inaccessible")
    return rss, count


def mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def summarize(values: list[int], baseline: int) -> dict:
    if not values:
        raise ValueError("No memory samples collected")
    peak = max(values)
    return {
        "sample_count": len(values),
        "idle_before_rss_mib": mb(baseline),
        "min_during_rss_mib": mb(min(values)),
        "median_during_rss_mib": mb(int(statistics.median(values))),
        "peak_observed_rss_mib": mb(peak),
        "peak_minus_idle_mib": mb(peak - baseline),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args: argparse.Namespace) -> None:
    if not BENCH_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing matched benchmark script: {BENCH_SCRIPT}")
    if not 0.05 <= args.interval <= 1.0:
        raise ValueError("--interval must be in 0.05..1.0 seconds")
    if not 0 <= args.warmup <= 2 or not 1 <= args.repeat <= 5:
        raise ValueError("Keep warmup 0..2 and repeat 1..5")

    port = PORTS[args.backend]
    tag = validate_tag(getattr(args, "tag", None))
    model_match = getattr(args, "model_match", None)
    if model_match and args.backend != "gguf":
        raise ValueError("--model-match only applies to GGUF servers")
    suffix = f"_{tag}" if tag else ""
    profile = OUT_DIR / f"phase16_memory_{args.backend}{suffix}_DEV_ONLY.json"
    benchmark = OUT_DIR / f"phase16_memory_matched_{args.backend}{suffix}_DEV_ONLY.json"
    command = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--backend",
        args.backend,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--output",
        str(benchmark),
        "--execute",
    ]
    if tag:
        command.extend(["--tag", tag])
    if model_match:
        command.extend(["--model-match", model_match])
    if not args.execute:
        print(f"DRY RUN: backend={args.backend}, server_port={port}; NO inference")
        print("Will monitor server RSS + descendant processes using psutil.")
        print("Benchmark output:", benchmark)
        print("RAM report:", profile)
        print("Use --execute to authorize localhost development benchmark requests.")
        return

    for path in (profile, benchmark):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite prior results: {path}")
    # Check the actual model BEFORE writing reports or running inference.
    model_id = None
    if args.backend == "gguf":
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=15) as reply:
            models = json.load(reply).get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise ValueError("llama.cpp did not provide a valid model ID")
        model_id = models[0]["id"]
        if model_match and model_match.casefold() not in model_id.casefold():
            raise ValueError(
                f"Unexpected model {model_id!r}; expected name containing {model_match!r}"
            )
        print("SERVER MODEL:", model_id, flush=True)
    pid = args.pid if args.pid is not None else resolve_listener_pid(port)
    server = psutil.Process(pid)
    if not server.is_running():
        raise RuntimeError(f"Serving process {pid} is not running")
    base, process_count = memory_snapshot(pid)
    print(
        f"MONITORING {args.backend}: PID={pid} port={port} "
        f"idle_RSS={mb(base)} MiB process_count={process_count}",
        flush=True,
    )
    print("No model changes; calling existing matched development benchmark.", flush=True)
    data = {
        "schema_version": "phase16.memory_profile.v1",
        "scope": "DEVELOPMENT_ONLY_NOT_HELD_OUT",
        "status": "RUNNING",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "backend": args.backend,
        "model_tag": tag,
        "model_id_reported": model_id,
        "expected_model_match": model_match,
        "model_server_port": port,
        "server_pid": pid,
        "sample_interval_seconds": args.interval,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "memory_metric": "psutil process-tree RSS (Windows working-set approximation)",
        "caveats": [
            "RAM does not equal VRAM, private bytes, or exact model allocation.",
            "Includes resident shared/mapped pages; process-tree sums may double count them.",
            "Observed peak depends on sampling interval; brief spikes may be missed.",
            "Measurement includes warm-up and measured inference; not a pure per-request peak.",
            "Do not run GGUF and PyTorch servers concurrently for memory comparisons.",
            "No original 78-case held-out set is used; no answer-quality guarantee.",
        ],
        "matched_benchmark_output": str(benchmark.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(profile, data)
    samples: list[int] = []
    max_count = process_count
    started = time.perf_counter()
    result = None
    try:
        proc = subprocess.Popen(command, cwd=ROOT)
        while proc.poll() is None:
            try:
                value, count = memory_snapshot(pid)
                samples.append(value)
                max_count = max(count, max_count)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                raise RuntimeError(
                    "Server PID changed or became inaccessible during benchmark"
                ) from None
            time.sleep(args.interval)
        result = proc.returncode
        samples.append(memory_snapshot(pid)[0])
    finally:
        data["elapsed_wall_seconds"] = round(time.perf_counter() - started, 2)
        if samples:
            data["rss"] = summarize(samples, base)
            data["max_accessible_process_count"] = max_count
        data["child_benchmark_exit_code"] = result
        data["status"] = "COMPLETE" if result == 0 and samples else "FAILED_OR_PARTIAL"
        write_json(profile, data)
    print("RAM PROFILE:", profile)
    if data["status"] != "COMPLETE":
        raise RuntimeError("Matched benchmark failed or memory sampling was interrupted")
    print("MEMORY:", json.dumps(data["rss"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gguf", "torch"), required=True)
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Optional server PID if automatic port lookup fails",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--tag", help="Safe model-specific filename suffix, e.g. krikri8b_q4km")
    parser.add_argument("--model-match", help="Require the GGUF model ID to contain this text")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow benchmark HTTP requests to localhost",
    )
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, psutil.Error, RuntimeError) as exc:
        parser.exit(2, f"PHASE16 MEMORY STOP: {exc}\n")


if __name__ == "__main__":
    main()

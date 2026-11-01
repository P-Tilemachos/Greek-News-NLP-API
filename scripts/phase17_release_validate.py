"""Phase 17.4 opt-in release checks against a running Greek News NLP API.

By default, performs lightweight checks only: no model weights or HF downloads.
Use --probe explicitly for expensive real-model checks.
"""

import argparse
import json
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROUTES = {
    "/health": "get",
    "/classify": "post",
    "/search": "post",
    "/ask": "post",
    "/generate": "post",
    "/rag/ask": "post",
}

PROBES = {
    "classify": (
        "/classify",
        {"text": "Η κυβέρνηση ανακοίνωσε μέτρα για τους αγρότες μετά τις πλημμύρες.", "top_k": 3},
    ),
    "search-dense": (
        "/search",
        {"query": "πλημμύρες και αποζημιώσεις αγροτών", "k": 3, "mode": "dense"},
    ),
    "search-hybrid": (
        "/search",
        {"query": "πλημμύρες και αποζημιώσεις αγροτών", "k": 3, "mode": "hybrid"},
    ),
    "search-rerank": (
        "/search",
        {"query": "πλημμύρες και αποζημιώσεις αγροτών", "k": 3, "mode": "rerank"},
    ),
    "ask": ("/ask", {"question": "Τι ανακοινώθηκε για τις πλημμύρες;", "k": 3}),
    "generate": (
        "/generate",
        {
            "prompt": "Γράψε μία σύντομη πρόταση στα ελληνικά για τις ειδήσεις.",
            "max_new_tokens": 24,
        },
    ),
    "rag": (
        "/rag/ask",
        {"question": "Τι ανακοινώθηκε για τις πλημμύρες;", "k": 3, "max_new_tokens": 48},
    ),
}


def validate_base_url(base_url: str, allow_remote: bool = False) -> str:
    """Do not accidentally probe a public host with model-loading requests."""
    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--base-url must be an HTTP(S) URL with a hostname")
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("--base-url must not contain credentials, path, query or fragment")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} and not allow_remote:
        raise ValueError("Non-local API blocked; use --allow-remote only if authorized")
    return base_url.rstrip("/")


def expected_project_version() -> str:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def http_json(base_url: str, method: str, path: str, payload: dict | None, timeout: float):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            status, raw = response.status, response.read(2_000_001)
    except HTTPError as exc:
        status, raw = exc.code, exc.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("API response exceeds safety limit of 2 MB")
    return status, json.loads(raw.decode("utf-8"))


def verify_lightweight(name: str, status: int, body: dict, expected_version: str):
    if name == "health":
        assert status == 200 and body == {"status": "ok"}, "unexpected /health result"
    elif name == "openapi":
        assert status == 200 and isinstance(body, dict), "OpenAPI request failed"
        info = body.get("info", {})
        assert info.get("version") == expected_version, "API/project version mismatch"
        paths = body.get("paths", {})
        for path, method in ROUTES.items():
            assert method in paths.get(path, {}), f"missing {method.upper()} {path}"
    elif name in {"reject-invalid-generate", "reject-invalid-rag"}:
        assert status == 422, f"invalid payload must be rejected with 422, got {status}"
    else:
        raise ValueError(f"unknown lightweight check: {name}")


def verify_probe(name: str, status: int, body: dict):
    assert status == 200 and isinstance(body, dict), f"HTTP {status} (expected 200)"
    if name == "classify":
        predictions = body.get("predictions")
        assert isinstance(predictions, list) and predictions, "missing predictions"
        assert all(
            isinstance(p.get("label"), str) and isinstance(p.get("score"), (int, float))
            for p in predictions
        )
    elif name.startswith("search-"):
        assert body.get("mode") == name.removeprefix("search-"), "wrong retrieval mode"
        results = body.get("results")
        assert isinstance(results, list), "missing search results"
        assert all(
            isinstance(hit.get("id"), str) and isinstance(hit.get("score"), (int, float))
            for hit in results
        )
    elif name == "ask":
        assert isinstance(body.get("evidence"), list), "missing evidence list"
        assert isinstance(body.get("sources"), list), "missing source list"
    elif name == "generate":
        assert body.get("mode") == "local_llm", "wrong generation mode"
        assert isinstance(body.get("generated_text"), str), "missing generated text"
        assert isinstance(body.get("model"), str), "missing model ID"
        assert isinstance(body.get("output_tokens"), int), "missing output token count"
    elif name == "rag":
        assert body.get("mode") == "grounded_rag", "wrong RAG mode"
        assert isinstance(body.get("abstained"), bool), "missing abstention status"
        assert body.get("faithfulness_verified") is False, "unexpected RAG verification flag"
        assert isinstance(body.get("sources"), list), "missing source list"
        if body["abstained"]:
            assert body.get("answer") is None, "abstained response must have null answer"
        else:
            assert isinstance(body.get("answer"), str), "missing generated RAG answer"
    else:
        raise ValueError(f"unknown probe: {name}")


def run_checks(
    base_url: str,
    expected_version: str,
    probes: list[str],
    timeout: float,
    requester=http_json,
):
    cases = [
        ("health", "GET", "/health", None),
        ("openapi", "GET", "/openapi.json", None),
        ("reject-invalid-generate", "POST", "/generate", {}),
        ("reject-invalid-rag", "POST", "/rag/ask", {}),
    ]
    cases.extend((name, "POST", *PROBES[name]) for name in probes)
    results = []
    for name, method, path, payload in cases:
        start = time.monotonic()
        try:
            status, body = requester(base_url, method, path, payload, timeout)
            if name in PROBES:
                verify_probe(name, status, body)
            else:
                verify_lightweight(name, status, body, expected_version)
            results.append(
                {
                    "check": name,
                    "passed": True,
                    "duration_s": round(time.monotonic() - start, 3),
                }
            )
        except (AssertionError, ValueError, KeyError, TypeError, OSError, URLError) as exc:
            results.append(
                {
                    "check": name,
                    "passed": False,
                    "duration_s": round(time.monotonic() - start, 3),
                    "reason": str(exc)[:200],
                }
            )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Only for an API you own or are authorized to test",
    )
    parser.add_argument(
        "--probe",
        action="append",
        choices=tuple(PROBES),
        default=[],
        help="Opt-in real-model check; repeat for multiple endpoints",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="Per-request timeout (seconds)"
    )
    parser.add_argument("--report", default="phase17-release-smoke.json")
    parser.add_argument(
        "--expected-version", default=None, help="Default: local pyproject.toml version"
    )
    args = parser.parse_args(argv)
    try:
        base_url = validate_base_url(args.base_url, args.allow_remote)
        if not 0 < args.timeout <= 3600:
            raise ValueError("--timeout must be between 0 and 3600 seconds")
        version = args.expected_version or expected_project_version()
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    results = run_checks(base_url, version, list(dict.fromkeys(args.probe)), args.timeout)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        line = f"{status} {result['check']} ({result['duration_s']}s)"
        if not result["passed"]:
            line += f" — {result['reason']}"
        print(line)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "expected_version": version,
        "base_url": base_url,
        "mode": "real-model-opt-in" if args.probe else "lightweight-no-model-load",
        "checks": results,
        "passed": all(result["passed"] for result in results),
        "note": "Contract checks do not measure model quality, RAG faithfulness or security.",
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

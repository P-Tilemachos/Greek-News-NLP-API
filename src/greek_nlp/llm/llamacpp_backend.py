"""Optional, loopback-only llama.cpp HTTP generator for standalone /generate.

No model downloads, server process creation, streaming or network I/O at import
or construction time. This module is not used by /rag/ask.
"""

from __future__ import annotations

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from greek_nlp.llm.generator import DEFAULT_SYSTEM_PROMPT

MAX_RESPONSE_BYTES = 65_536


class LlamaCppUnavailable(RuntimeError):
    """A sanitized upstream failure; /generate translates this to HTTP 503."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _loopback_base(value: str) -> str:
    """Prevent server-side requests to arbitrary destinations."""
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid LLAMACPP_BASE_URL") from exc
    if (
        parts.scheme != "http"
        or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
        or port is None
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
    ):
        raise ValueError("LLAMACPP_BASE_URL must be a loopback HTTP URL with a port")
    return value.strip().rstrip("/")


def _request_json(url: str, *, timeout: float, payload: dict | None = None) -> dict:
    """Read a bounded JSON response, never following a redirect or using proxies."""
    opener = build_opener(ProxyHandler({}), _NoRedirects())
    request = Request(
        url,
        data=(json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None),
        headers={"Content-Type": "application/json"} if payload else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise LlamaCppUnavailable("Unexpected llama.cpp HTTP status")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise LlamaCppUnavailable("llama.cpp server is unavailable") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise LlamaCppUnavailable("llama.cpp response is too large")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise LlamaCppUnavailable("llama.cpp returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise LlamaCppUnavailable("llama.cpp returned an invalid JSON shape")
    return result


class LlamaCppGenerator:
    """HTTP-only generator preserving the existing /generate response schema."""

    def __init__(self) -> None:
        self.base_url = _loopback_base(os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:9931"))
        self.model_match = os.environ.get("LLAMACPP_MODEL_MATCH", "Krikri").strip()
        if not self.model_match:
            raise ValueError("LLAMACPP_MODEL_MATCH must not be empty")
        try:
            self.timeout = float(os.environ.get("LLAMACPP_TIMEOUT_SECONDS", "300"))
        except ValueError as exc:
            raise ValueError("Invalid LLAMACPP_TIMEOUT_SECONDS") from exc
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 600:
            raise ValueError("LLAMACPP_TIMEOUT_SECONDS must be between 1 and 600")

    def generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        """Greedy single-response inference; no raw prompt or answer is logged."""
        if not isinstance(prompt, str) or not 3 <= len(prompt) <= 1500:
            raise ValueError("Prompt length must be in 3..1500 characters")
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be within 1..128")

        # Check the serving model on EVERY call, even after a server restart.
        models = _request_json(self.base_url + "/v1/models", timeout=min(15, self.timeout))
        models_list = models.get("data")
        if not isinstance(models_list, list) or len(models_list) != 1:
            raise LlamaCppUnavailable("Expected one llama.cpp model")
        item = models_list[0]
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise LlamaCppUnavailable("Invalid llama.cpp model metadata")
        model_id = item["id"]
        if self.model_match.casefold() not in model_id.casefold():
            raise LlamaCppUnavailable("Serving model does not match LLAMACPP_MODEL_MATCH")

        response = _request_json(
            self.base_url + "/v1/chat/completions",
            timeout=self.timeout,
            payload={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_new_tokens,
                "stream": False,
            },
        )
        reported_model = response.get("model")
        if reported_model is not None and reported_model != model_id:
            raise LlamaCppUnavailable("llama.cpp response model changed")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LlamaCppUnavailable("llama.cpp returned no completion")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LlamaCppUnavailable("llama.cpp returned invalid completion content")
        content = message["content"].strip()
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise LlamaCppUnavailable("llama.cpp did not return token usage")
        n_input, n_output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if (
            not isinstance(n_input, int)
            or isinstance(n_input, bool)
            or n_input < 1
            or not isinstance(n_output, int)
            or isinstance(n_output, bool)
            or not 0 <= n_output <= max_new_tokens
            or not content
        ):
            raise LlamaCppUnavailable("llama.cpp returned invalid token usage or empty text")

        return {
            "mode": "local_llm",
            "model": model_id,
            "device": "cpu",  # API compatibility; this option is configured with -ngl 0.
            "generated_text": content,
            "input_tokens": n_input,
            "output_tokens": n_output,
        }

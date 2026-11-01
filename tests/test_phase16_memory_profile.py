from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "phase16_memory_profile.py"
spec = importlib.util.spec_from_file_location("phase16_memory_profile", MODULE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mib_and_peak():
    b = 1024 * 1024
    result = module.summarize([5 * b, 8 * b, 7 * b], 4 * b)
    assert result["peak_observed_rss_mib"] == 8.0
    assert result["peak_minus_idle_mib"] == 4.0
    assert result["sample_count"] == 3


def test_empty_samples_rejected():
    with pytest.raises(ValueError, match="No memory samples"):
        module.summarize([], 0)


def test_unique_listener(monkeypatch):
    row = SimpleNamespace(
        status=module.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=9931), pid=4321
    )
    monkeypatch.setattr(module.psutil, "net_connections", lambda kind: [row])
    assert module.resolve_listener_pid(9931) == 4321


def test_listener_missing(monkeypatch):
    monkeypatch.setattr(module.psutil, "net_connections", lambda kind: [])
    with pytest.raises(ValueError, match="Cannot identify"):
        module.resolve_listener_pid(9931)

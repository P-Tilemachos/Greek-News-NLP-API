# Phase 16.4 — CPU process RAM measurement (development-only)

This utility samples **server process-tree RSS** during the existing matched-prompt
benchmark. It makes no changes to the model, original Phase 15 evaluation or
FastAPI production code. It **will not rerun any 78 held-out questions**.

## First: GGUF

1. Stop the PyTorch/Uvicorn server (Ctrl+C) so its model does not occupy RAM.
2. Keep `llama serve` running on **127.0.0.1:9931**, using Q4_K_M, `-t 8 -ngl 0`.
3. Install `psutil` only if needed (`python -m pip show psutil`).
4. In another Anaconda Prompt, from the repository root:

```bat
python scripts\phase16_memory_profile.py --backend gguf
python -m ruff check .
python -m pytest -q
python scripts\phase16_memory_profile.py --backend gguf --warmup 1 --repeat 3 --execute
```

Two **new**, separately named files will be saved under the ignored
`results/phase16_local/` directory:

- `phase16_memory_gguf_DEV_ONLY.json` (RAM report)
- `phase16_memory_matched_gguf_DEV_ONLY.json` (new matched latency samples)

This does not overwrite `phase16_matched_gguf_DEV_ONLY.json` from the previous benchmark.

If PID detection fails on Windows, discover the listening PID in PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 9931 -State Listen | Select-Object OwningProcess
```

Then add `--pid THE_NUMBER` to the benchmarking command.

## Later: PyTorch

Stop llama.cpp with Ctrl+C. Start FastAPI on port 8000, `LLM_DEVICE=cpu`,
**without `--reload`**. Repeat the same profiler with `--backend torch`.
Keep it separate from GGUF to avoid memory and CPU contention.

## Methodology and limitations

- This measures **resident set size (RSS)** / Windows working-set approximation,
  including server subprocesses. It is **not** isolated model RAM, GPU VRAM or
  private committed bytes.
- Shared/memory-mapped pages can be counted more than once; runtime and OS cache
  behavior can differ by backend.
- Samples every 0.1 s by default. Short memory spikes may be missed.
- The peak includes the warm-up and all measured requests, rather than an
  individual request peak. Measurements are not independent quality evidence.
- No textual generations or prompts are saved by this wrapper. The child matched
  benchmark writes lengths, hashes, tokens and timing only.
- **Important:** On Windows, a process's working set does not represent total
  system RAM occupancy. Inspect Task Manager if assessing system-level pressure.
- Verify real outputs for quality separately; a faster wrong or abstaining answer
  is not an optimization success.

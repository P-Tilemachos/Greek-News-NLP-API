# Phase 16 — Krikri 8B Q4_K_M benchmark (development only)

This kit updates two existing scripts **without changing API/model implementations**:

- `scripts/phase16_compare_backends.py`: supports `--tag`, `--model-match`, and an explicitly enabled GGUF-vs-GGUF report.
- `scripts/phase16_memory_profile.py`: supports the same tags, refuses existing output files, and verifies the server's real model ID before inference.
- `tests/test_phase16_krikri_results.py`: regression tests (offline).

**Scope:** development-only performance experiment on one existing question (48 output tokens); NOT an accuracy/faithfulness benchmark. Keep the original Phase 15 set untouched.

## 1. Unzip and check

Unzip into `Greek-News-NLP-API/`, replacing the **two scripts** in `scripts/` only after keeping a backup if desired. Do not remove or rename anything inside `results/phase16_local/`.

```bat
python -m ruff format scripts/phase16_compare_backends.py scripts/phase16_memory_profile.py tests/test_phase16_krikri_results.py
python -m ruff check .
python -m pytest -q
```

## 2. Server windows

Leave **Krikri 8B GGUF Q4_K_M** serving on port 9931:

```bat
llama serve -hf ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M --host 127.0.0.1 --port 9931 -t 8 -c 2048 -ngl 0
```

Stop the FastAPI server during RAM comparison, to reduce memory contention. Use a **second** Anaconda Prompt for the following steps.

## 3. Dry run (no inference)

```bat
python scripts\phase16_memory_profile.py --backend gguf --tag krikri8b_q4km --model-match Krikri
```

Expected paths:

- `results/phase16_local/phase16_memory_gguf_krikri8b_q4km_DEV_ONLY.json`
- `results/phase16_local/phase16_memory_matched_gguf_krikri8b_q4km_DEV_ONLY.json`

## 4. One measured run (RAM + latency together)

```bat
python scripts\phase16_memory_profile.py --backend gguf --tag krikri8b_q4km --model-match Krikri --warmup 1 --repeat 3 --execute
```

This **one command** samples server RSS *and* invokes the matched prompt benchmark; there is no need to run `phase16_compare_backends.py --execute` separately. The HTTP requests are confined to localhost. It saves metadata and hashes, not raw generated answers.

The explicit `--model-match Krikri` check aborts the run if the port is serving a different model. The script refuses to overwrite previous or new results. If a new run is necessary, use a different safe `--tag`, preserving the original trial.

## 5. Compare Krikri against original Qwen 1.5B GGUF

The previously saved Qwen result was:

`results/phase16_local/phase16_matched_gguf_DEV_ONLY.json`

Compare it with the new, separately saved Krikri run:

```bat
python scripts\phase16_compare_backends.py --allow-gguf-compare --compare results\phase16_local\phase16_matched_gguf_DEV_ONLY.json results\phase16_local\phase16_memory_matched_gguf_krikri8b_q4km_DEV_ONLY.json
```

This comparison checks equality of prompt hashes and max-output-token budget. It accepts two **distinct** GGUF model IDs only with the explicit flag. It compares **different models**, not the isolated effect of quantization. Output-token counts and tokenizers can differ; interpret latency and E2E rate accordingly.

## Interpretation and limitations

- Memory is `psutil` process-tree RSS (Windows working-set approximation), not private bytes, cold-start allocation or GPU VRAM. Existing resident mapped pages and sampling frequency can affect the result.
- Krikri has ~8B parameters versus Qwen 1.5B; compare language quality and memory as well as throughput.
- The development workload is only one prompt. Three earlier Krikri smoke tests are not sufficient for a production quality claim.
- No changes are made to `/generate`, `/rag/ask`, `generator.py`, or the Phase 15 baseline. Never label this as an independently verified answer-quality assessment.
- Before adopting Krikri as an optional API backend, run a separate Greek-language quality set, regression checks, document the Llama-model license, and verify a clean GitHub CI run.

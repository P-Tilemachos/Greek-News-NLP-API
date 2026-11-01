# Phase 16 — Performance optimization: evaluation plan

Status: **baseline preparation**, not optimization completed.

## Objective

Compare CPU/GPU inference backends, model precision, safe streaming and serving configurations using reproducible measurement. Do not trade away faithfulness, citation validation, or answerability for speed.

## Evidence from Phase 15

The original Phase 15 benchmark had 80/80 abstentions, 0/40 positive questions answered, and positive gold-source retrieval at 38/40. The negative labels were provisional; reported judgments were by the dataset author. Preserve the original predictions and their protocol. Do not tune on the 78 evaluated questions; use the two original development cases or a newly authored development set.

## Steps

1. Capture hardware/software/source hashes using `scripts/phase16_baseline.py --preflight`. Keep `results/phase16_local/` private.
2. Inspect live request schema with `scripts/phase16_baseline.py --inspect-openapi`. Confirm the sample JSON payloads match the live API.
3. Make a **small, explicitly authorized** baseline run with `--execute`. The script measures completed request latency, counts answered vs abstained, and stores token counts when provided. It does not store answer text or excerpts.
4. Measure model loading time, cold vs warm latency, real peak server process RSS and GPU VRAM using **separate** instrumentation. Do not mislabel end-to-end latency as time-to-first-token or tokens/second when no token telemetry exists.
5. Implement and compare streaming on `/generate` first. For grounded RAG, **never reveal unverified output tokens** before citation/evidence validation. Stream only validated content or explicitly label a separate unverified preview with a safety review.
6. Evaluate quantization backends **on supported hardware**: CUDA fp16 baseline and compatible 8/4-bit methods, or CPU-appropriate GGUF/llama.cpp alternatives. Confirm model + tokenizer + prompt identity. Measure memory, latency and task quality; do not assume bitsandbytes 4-bit CUDA settings work on Windows CPU.
7. Compare one change at a time with fixed prompts, generation settings, model revisions, index hashes and environment metadata. Include warm-up, repetitions, medians and p95 only when there are enough samples; report OOM/error cases as failures rather than dropping them.
8. Run Ruff, pytest, API integration tests, and CI; document trade-offs and provide a rollback path.

## Baseline metrics and meanings

- End-to-end request latency (ms): network+server+loading+generation on first call; later calls may be warm.
- TTFT (ms): first streamed, user-visible token; **not yet measured** by the provided baseline script.
- Throughput (generated tokens/s): needs accurate generated token counts; not derived from entire HTTP latency without qualification.
- RAM/VRAM: memory footprint and **peak** usage must be measured in the API process, not client process.
- Answer coverage, abstention rate, citation validity and verified answer quality are essential quality gates.

## Commands

Start the local API with its existing pinned model setting in another terminal, then:

```bat
python scripts\phase16_baseline.py --preflight
python scripts\phase16_baseline.py --inspect-openapi
python scripts\phase16_baseline.py --cases evaluation\phase16_development_cases.json
```

The three commands above execute **no model inference**. After validating the request schemas and confirming local API health:

```bat
python scripts\phase16_baseline.py --cases evaluation\phase16_development_cases.json --execute --repeat 1
```

This last command explicitly triggers **three** development-only model calls. Running Qwen on CPU may be slow. Save the run output locally and do not commit it. Run just one of the 3 cases by editing a private copy of the development JSON, if desired.

## Deliverable and next dependencies

To implement proper model-level timing, memory sampling, quantization and streaming we need to inspect the current `src/greek_nlp/llm/generator.py`, `src/greek_nlp/rag/grounded.py`, and the API schema/route definitions. A previous generator copy was Phase 13 and may differ from the committed version; avoid assuming it is identical.

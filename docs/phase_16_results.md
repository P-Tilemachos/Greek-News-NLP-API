# Phase 16 — CPU inference optimization and Greek model screening

**Scope:** Local development/profiling; not independent quality evaluation or a production benchmark.
**Status:** Local implementation and smoke tests complete; GitHub publication and CI verification pending.
**Platform:** Windows, Python 3.11, CPU-only PyTorch, 8 logical processors, ~15.87 GiB RAM.

## What changed

- Retained the default lazy PyTorch generator and existing `/generate` request/response schema.
- Added an opt-in localhost llama.cpp HTTP backend selected via `LLM_BACKEND=llamacpp` for `/generate` only.
- Deliberately left `/rag/ask` on the original PyTorch path; Phase 15 over-abstention is unresolved.
- Added development benchmark, streaming, memory profiling, quality-screen scripts and offline tests.

## Measured development results

The same Greek `/generate` development request and 48-token output cap were used, with one warm-up and three recorded HTTP requests for the matched benchmark. This is not a matched-architecture ablation.

| Config | HTTP median (s) | Median output tokens | Observed peak process RSS (MiB) |
|---|---:|---:|---:|
| PyTorch CPU Qwen2.5-1.5B-Instruct | 25.21 | 48 | 6,002.13 |
| llama.cpp Qwen2.5-1.5B GGUF Q4_K_M | 2.79 | 48 | 1,156.09 |
| llama.cpp Llama-Krikri-8B-Instruct GGUF Q4_K_M | 11.14 | 48 | 4,841.20 |

Qwen GGUF had a ~9.04x lower median HTTP time than PyTorch in this one local matched workload, and ~81% less observed RSS. This improvement combines backend implementation and quantized weight format. The Krikri row is a *different model*, so quality and speed cannot be attributed only to quantization.

### Streaming (Qwen GGUF only)

- Median first visible content: **57.93 ms**.
- Median end-to-end streamed response: **2.37 s**.
- 47 content chunks across each measured run (1 warm-up + 3 measurements).

First visible content does not equal an independently verified internal TTFT. Prompt caching, chunk boundaries and HTTP overhead affect the result.

### Memory measurement limitations

- `psutil` process-tree RSS / Windows working-set approximation, sampled at 0.1-second intervals.
- Measurements start when the model is *already loaded*; they do not capture load-time peak or total system RAM usage.
- RSS may count mapped/shared pages and is not private committed memory, physical model bytes, or GPU VRAM.
- GGUF and PyTorch server measurements were made separately to avoid direct resource competition.

## Small Greek quality screen — Krikri 8B

Eight synthetic development examples tested Greek definition, arithmetic, number/date extraction, summarization, missing salary, conflicting evidence and a known album title. A provisional inspection judged **7/8 acceptable**. The conflicting-evidence case failed: although the model identified disagreement between 09:00 and 10:00, it invented 100% certainty that the actual start fell in that interval. This is a significant factuality limitation. The JSON still marks `human_review` as null; this is an informal screening judgment, not an official annotation-derived accuracy result.

No held-out Phase 15 questions were reused for optimization. Do not report faithfulness or hallucination scores from the synthetic screen, and do not infer that `/rag/ask` has improved.

## Local integration and safety

`/generate` with `LLM_BACKEND=llamacpp` was verified with the Krikri model ID and `HTTP 200`, while the default torch path remains available by default. The integration uses a separate loopback llama.cpp server and preserves the API response schema. Local Ruff and **132 pytest tests** passed following integration. GitHub CI must be verified separately.

Never expose the unauthenticated llama.cpp server publicly. Bind it to `127.0.0.1` for development; review CORS, authentication, rate limits, network topology, and model license terms before deployment.

## Local reproducibility

```bat
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Run the standalone llama.cpp service and FastAPI with environment variables documented in [`phase_16_optional_llamacpp_integration.md`](phase_16_optional_llamacpp_integration.md). The project stores model artifacts and raw phase16 predictions **outside version control**, under the ignored `results/phase16_local/` folder.

## Next steps

After GitHub CI is green: expand independent Greek evaluation, investigate RAG abstentions separately, strengthen regression coverage and release/deployment documentation in Phase 17. Do not promote the Krikri opt-in to the default solely from eight synthetic test cases.

# Phase 13 — Local open-source LLM integration

## Scope

Adds an **opt-in** `POST /generate` endpoint for local, open-weight causal language-model
inference. The endpoint is **not** a RAG endpoint: it does not search the corpus, cite
sources, check factual accuracy, or access the Phase 05 classification QLoRA adapter.
Grounded news RAG is Phase 14. Dense/hybrid/rerank `/search`, extractive `/ask`,
`/classify`, and `/health` remain unchanged.

## Model and requirements

- Default: `Qwen/Qwen2.5-0.5B-Instruct`, a small public Qwen checkpoint suitable for
  a first CPU smoke test. The Greek-language quality of this particular checkpoint
  has **not** been evaluated here; selecting a larger model is optional and requires
  additional RAM/VRAM.
- Override **server-side** using `LLM_MODEL_ID` (for example
  `Qwen/Qwen2.5-1.5B-Instruct`), `LLM_DEVICE=auto|cpu|cuda`, and optionally
  `LLM_MODEL_REVISION=<verified commit sha>` to pin a reproducible model revision.
  Do not allow users to set model IDs through API requests.
- No cloud inference API or paid key; public weights download from the Hugging Face
  Hub on the **first** `/generate` call. Network access is needed the first time.
  Models/weights are not bundled in this repository or in the Phase 13 ZIP.
- CPU uses float32; GPU uses float16 when CUDA is available. CUDA is **not** required.
  The selected model is cached in process memory; avoid multiple Uvicorn workers and
  avoid `--reload` during heavy inference. A lock serializes concurrent generations.
- `torch` and `transformers` are optional runtime dependencies. Installing only
  `[dev]` is sufficient for offline tests and CI.

## Run on Windows (Anaconda Prompt)

From the repository root with the `greek-nlp` environment activated:

```bat
scripts\run_phase13_checks.bat
scripts\install_phase13_llm.bat
set "LLM_DEVICE=cpu"
python -m uvicorn greek_nlp.api.main:app
```

Then open `http://127.0.0.1:8000/docs` and execute `POST /generate`:

```json
{
  "prompt": "Γράψε μια σύντομη παράγραφο στα ελληνικά για την αξία της ενημέρωσης.",
  "max_new_tokens": 32
}
```

Expected HTTP 200 structure:

```json
{
  "mode": "local_llm",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "cpu",
  "generated_text": "...",
  "input_tokens": 42,
  "output_tokens": 32
}
```

The token numbers and text above are **illustrative**, not expected exact values.
The first request may be slow due to checkpoint download and CPU generation.
Use `set "LLM_DEVICE=auto"` if you have CUDA properly installed and want to use it.
An explicit unavailable CUDA device returns HTTP 503 rather than silently
falling back to CPU.

## Controls and failure modes

- Requests: prompt length 3–1500 characters, output budget 1–128 new tokens,
  fixed greedy decoding (`do_sample=False`), chat-template formatting, 512 maximum
  input tokens, and `trust_remote_code=False`.
- The `generated_text` is decoded **only** from newly generated tokens, not the input.
- Dependencies, checkpoint download failures, or insufficient memory return HTTP
  503 without exposing internal file paths. Invalid requests return HTTP 422.
- HTTP 200 means inference worked, **not** that the output is correct, up-to-date,
  grounded, or safe for factual decision-making.
- This basic prototype is **not ready for public unauthenticated production serving**:
  it has no rate limiting, request authentication, isolation between tenants,
  streaming, quantization, source citations or systematic hallucination testing.
- No results/benchmark artifacts from Notebooks 01–12 are modified. CI uses fake
  generators, no weights/downloads; real inference must be tested separately.

## Reproducibility

Record the resolved Hugging Face model revision, torch/transformers versions,
CPU/GPU identity, prompt, generation budget, and observed result after live smoke.
The local model is *not* the classification QLoRA configuration of Notebook 05/07.
Phase 14 will provide retrieval context, citations, and abstention safeguards.

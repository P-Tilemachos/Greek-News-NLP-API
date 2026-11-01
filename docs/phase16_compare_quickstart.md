# Phase 16.2 — Matched-prompt local CPU benchmark

**Status:** development-only diagnostic, not an independent held-out quality evaluation.

The script uses `dev_generate_greeting` from the existing `evaluation/phase16_development_cases.json` file. Despite its ID, this case asks in Greek: “Απάντησε σύντομα στα ελληνικά: Τι είναι η μηχανική μάθηση;”

To match prompting more closely, both backends receive the current `DEFAULT_SYSTEM_PROMPT` defined in `src/greek_nlp/llm/generator.py`, the same Greek user question, max 48 new tokens and deterministic decoding. The FastAPI `/generate` endpoint adds the system message internally; the llama.cpp OpenAI chat endpoint receives it explicitly.

**Important:** The comparison changes both precision **and** inference runtime (PyTorch CPU FP32 versus llama.cpp GGUF Q4_K_M). It does **not** isolate the effect of quantization, cannot measure quality based only on latency, and is not a benchmark of `/rag/ask`.

## Installation

Copy the included files into your project, preserving directories:

- `scripts/phase16_compare_backends.py`
- `tests/test_phase16_compare_backends.py`

Run from the project root in the `greek-nlp` environment:

```bat
python -m ruff format scripts/phase16_compare_backends.py tests/test_phase16_compare_backends.py
python -m ruff check .
python -m pytest -q
```

## Step A — GGUF

Use an existing `llama serve -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --host 127.0.0.1 --port 9931 -t 8 -c 2048 -ngl 0` server. Prefer not to keep the PyTorch/FastAPI model loaded simultaneously when measuring CPU and RAM on a 16 GB machine.

From a second terminal:

```bat
python scripts\phase16_compare_backends.py --backend gguf --warmup 1 --repeat 3 --execute
```

Output (local, ignored by Git): `results/phase16_local/phase16_matched_gguf_DEV_ONLY.json`.

## Step B — PyTorch CPU

Stop the GGUF server gracefully (`Ctrl+C` in its terminal), and start the FastAPI server with the exact baseline configuration (`LLM_DEVICE=cpu`, `LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct`):

```bat
set "LLM_DEVICE=cpu"
set "LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct"
python -m uvicorn greek_nlp.api.main:app
```

In another terminal:

```bat
python scripts\phase16_compare_backends.py --backend torch --warmup 1 --repeat 3 --execute
```

Output: `results/phase16_local/phase16_matched_torch_DEV_ONLY.json`.

## Step C — Compare saved results without inference

```bat
python scripts\phase16_compare_backends.py --compare results\phase16_local\phase16_matched_torch_DEV_ONLY.json results\phase16_local\phase16_matched_gguf_DEV_ONLY.json
```

The comparison checks that user prompt, system prompt and max output tokens have matching SHA-256 fingerprints before displaying medians.

## Reporting cautions

- Exact user and system **content** is matched, but the backends may tokenize, template and handle EOS differently. Inspect the recorded actual input/output-token counts.
- Full HTTP latency includes prompt processing and overhead; do not confuse `output_tokens / HTTP seconds` with pure decoding throughput. llama.cpp may separately expose internal token timings.
- Warm-up calls are discarded; measured runs are bounded to 1–5. Each success is written atomically, and existing result files are never overwritten.
- No raw generated texts or prompts are written to results. `output_sha256` allows checking whether answers match without publishing them; inspect answer quality separately.
- Benchmark scripts make no changes to the Qwen model, RAG pipeline, Phase 15 results, or production API.
- This starter does not measure TTFT, memory peak, power usage or full cold-start latency. Those require separate profiling after initial comparisons.

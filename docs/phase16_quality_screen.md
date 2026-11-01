# Phase 16 — Greek quality screen (development only)

The tool screens a local llama.cpp candidate on eight self-contained Greek prompts, including extraction, missing information, contradiction handling and factual explanation.

**This is not a gold benchmark and not a substitute for independent human evaluation.** The source passages in the cases are synthetic. The existing Phase 15 held-out set is not used.

## Run

From the repository root, while the candidate model is running via `llama serve` on `127.0.0.1:9931`:

```bat
python -m ruff format scripts/phase16_quality_screen.py tests/test_phase16_quality_screen.py
python -m ruff check .
python -m pytest -q
python scripts\phase16_quality_screen.py --tag krikri8b_q4km --model-match Krikri
python scripts\phase16_quality_screen.py --tag krikri8b_q4km --model-match Krikri --execute
```

Outputs are written only to `results/phase16_local/phase16_quality_krikri8b_q4km_DEV_ONLY.json`. Do **not** commit this file. The benchmark refuses to overwrite an existing result.

The report stores the *actual generated responses* for manual inspection. For every case, assess factual correctness against the self-contained text or basic fact, Greek readability, completeness, relevance, hallucinated details, and whether the answer finished naturally. Keep the original response unchanged. Record evaluations separately and attribute them as author judgments; do not silently turn this small exploratory screen into a quantified model-quality benchmark.

A model suitable for optional `/generate` integration should give grounded, meaningful answers to several Greek queries, not merely a correct answer to `2+2`. Do not stream unverified `/rag/ask` tokens to a user: RAG requires a separate validation/abstention design.

If testing another model, use a **different `--tag`** to preserve existing results. Validate the model shown by `/v1/models` with the `--model-match` argument.

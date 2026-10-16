# Phase 12 — Opt-in Cross-Encoder Reranking Integration

## What changed

`POST /search` now accepts three modes, while preserving the Phase 11 response schema:

- `dense` (default): existing multilingual E5 + FAISS.
- `hybrid`: Phase 11 BM25 Full + Dense E5 + RRF.
- `rerank` (opt-in): Phase 11 Hybrid RRF candidates, then the **Top-5** are rescored
  using `BAAI/bge-reranker-v2-m3` (Cross-Encoder). Results after position five
  retain their original Hybrid order.

The `/ask` route remains **unchanged and extractive/dense-only**. Neither the
FAISS index nor frozen Notebook 09 evaluation artifacts are modified. The
Cross-Encoder is instantiated only on the first explicit `mode: "rerank"` call
and reused for subsequent requests within the same process.

## Relation to frozen Notebook 09

The implementation follows the saved reranking setup:

- Model: `BAAI/bge-reranker-v2-m3`.
- Pinned Hugging Face revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- `max_length=512` query+full-document tokens, `batch_size=2`, CPU by default.
- Score each candidate using the **full article text from the deployed `docs.json`**,
  never the API's 300-character `snippet`.
- Sort Top-5 by descending Cross-Encoder score; break ties by original Hybrid
  rank and ID. Preserve Hybrid order for documents beyond the Top-5.
- When `k` is smaller than 5, obtain and rerank five candidates **before**
  returning the first `k`. When `k` is larger than 5, obtain `k` Hybrid
  candidates, rerank only the first five and preserve the rest.

**Important scope:** The Notebook 09 results (480 frozen synthetic, source-document
queries) are **offline** benchmarks, not a new live-API evaluation. The live index,
model dependencies and inputs can differ. The measured historical CPU cost was
**27.62 seconds per query** on average across resumed experiment sessions,
not production API latency and not a promise about your hardware. Expect slow
CPU requests. Reranker outputs are *uncalibrated*, and the returned `score`
for `mode=rerank` is **not comparable** with E5 cosine or RRF scores.

## Install, local checks and smoke test (Windows / Anaconda Prompt)

From the root of your existing `Greek-News-NLP-API` repository, with the
`greek-nlp` Conda environment active:

```bat
scripts\run_phase12_checks.bat
```

The script runs the unit and API tests, lint and formatting checks. Tests inject
**fake Cross-Encoder predictions**, so they do not download the model or require
GPU. They also check that the original `dense`/`hybrid` routes remain available.

For an **optional real-model smoke test** (internet required on first load):

```bat
set "HF_MODEL_REPO=Tilemachos-P/greek-news-classifier"
set "HF_INDEX_REPO=Tilemachos-P/greek-news-index"
set "RERANK_DEVICE=cpu"
python -m uvicorn greek_nlp.api.main:app --reload
```

Open <http://127.0.0.1:8000/docs>, then `POST /search` with:

```json
{"query": "πλημμύρες και αποζημιώσεις αγροτών", "k": 5, "mode": "rerank"}
```

Verify HTTP `200`, exactly 5 or fewer results, `"mode": "rerank"`, and useful
articles. Repeat with `mode="hybrid"` and without `mode` (default `dense`) to
confirm backwards compatibility. The optional
[`12_cross_encoder_integration.ipynb`](../notebooks/12_cross_encoder_integration.ipynb)
contains **one code cell** to issue the three requests against an already
running local API. It does not run Notebook 09 or write benchmark files.

**Do not re-run the 480-query benchmark.** The old evaluation does not need to
be repeated to enable this serving mode. A fresh, independent human-authored
relevance dataset is still needed before claiming real-user quality gains.

## Operational considerations and rollback

- First rerank request downloads/loads the model; this can be substantial.
- CPU mode can be very slow. `RERANK_DEVICE=cuda` may be used **only if** a
  compatible CUDA installation/GPU is available; record hardware in any new
  performance claim. Docker/CI only verify build/health and mocked tests, **not**
  real model loading.
- Cross-Encoder errors (e.g. no network, missing model) are not silently
  downgraded to a different algorithm. Clients can retry `mode="hybrid"` or
  omit `mode` for `dense` search.
- `RERANK_DEVICE` is read on the **first load** and cached for that server
  process; restart the server to change it.
- The reranker does not generate answers or modify the `/ask` endpoint.
- There are no database or storage migrations. The original modes remain
  available for immediate rollback.

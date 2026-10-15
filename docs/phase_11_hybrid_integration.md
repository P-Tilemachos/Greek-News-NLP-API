# Phase 11 — Hybrid Retrieval Integration (BM25 Full + E5 + RRF)

## What is implemented

Production `/search` now accepts a `mode` field:

- `dense` (default): unchanged E5 + FAISS serving behaviour.
- `hybrid` (opt-in): full-text BM25 + existing E5 + FAISS, fused with RRF.

The response keeps its existing `results` payload and adds `mode`. Each returned result
still has only the existing public fields: `id`, `score`, `label`, `snippet`. Internally
the hybrid retriever tracks each candidate's BM25 and dense ranks.

This phase intentionally **does not** change `/ask` (still uses dense retrieval), switch
the `/search` default, perform cross-encoder reranking, or regenerate the frozen benchmark.

## Notebook 08 parity

- Corpus: deployed `docs.json` from the same `rag_index` as `articles.faiss`.
- BM25 Full: entire article text, Greek `casefold` + NFD accent stripping and Unicode tokenization.
- BM25Okapi 0.2.2: `k1=1.5`, `b=0.75`, `epsilon=0.25`.
- Dense retrieval: existing `intfloat/multilingual-e5-small`, `query:` / `passage:` prefixes,
  256-token index, FAISS cosine via normalized inner product.
- Top 50 retrieved separately from each system (or `k`, if larger), RRF `k=60`.
- Fusion scores: `1/(60 + rank_bm25) + 1/(60 + rank_dense)`, omitting missing ranks.
- Deterministic tie-break: fused score descending, best individual rank, document ID.
- Results are *serving integration*, not a new independent retrieval evaluation.
  Historical 480-query metrics apply to Notebook 08's frozen offline protocol only.

## Setup (Windows Anaconda Prompt, repository root)

```bat
python -m pip install -e ".[dev,ml]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Run the existing API as described in the main README, e.g.:

```bat
set "HF_MODEL_REPO=Tilemachos-P/greek-news-classifier"
set "HF_INDEX_REPO=Tilemachos-P/greek-news-index"
python -m uvicorn greek_nlp.api.main:app --reload
```

### Request examples

Dense mode (no change):

```json
{"query": "πλημμύρες και αποζημιώσεις αγροτών", "k": 5}
```

Hybrid mode:

```json
{"query": "πλημμύρες και αποζημιώσεις αγροτών", "k": 5, "mode": "hybrid"}
```

Send these to `POST /search` using `http://127.0.0.1:8000/docs` (Swagger UI) or your HTTP client.
First hybrid request builds an in-memory BM25 index over all articles; later requests reuse it.
It does **not** overwrite `articles.faiss` or require dataset source files on disk.

## Validation and limitations

The offline tests use small synthetic Greek documents and a fake dense model, so they
require neither weights nor external downloads. They check text normalization, BM25
indexing, score fusion, deterministic ranking, missing BM25 matches, duplicate-ID
handling, request routing and backwards compatibility.

A live smoke test with actual published Hugging Face artifacts is still recommended
before enabling hybrid as a default. Exact parity of live rankings with Notebook 08
depends on the content and ordering of the published corpus/index, dependency versions,
and embedding model version; those have not been pinned to a new end-to-end comparison here.
Retrieval quality for human-authored queries remains unverified. Hybrid introduces extra
BM25 indexing time and memory during startup of that mode and is not currently a streaming
or distributed production service.

## Rollback

To retain v1 behavior, omit `mode` or set `"mode": "dense"`. There is no migration
and no data transformation. `/ask` remains extractive and dense-only.

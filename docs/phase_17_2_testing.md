# Phase 17.2 — Endpoint regression tests

These five test modules were created against the supplied `src/greek_nlp/api/main.py` and `schemas.py`.
They check classification parameter forwarding, search-mode routing, extractive QA evidence/source linkage, grounded RAG answer/abstention response contracts and sanitized 503 failures, as well as optional `llamacpp` vs default `torch` selection.

These are mocked API-level tests, **not** real model integration/quality tests. Their assertions do not prove model correctness, retrieval relevance, hallucination rates, or availability of a llama.cpp server.

Local validation of the five new modules in this authoring environment: **19 passed**. Ruff was unavailable here. Run formatting and lint checks in your project's configured environment.

From the project root in your `greek-nlp` environment:

```bash
python -m pytest -q tests/test_phase17_classification.py tests/test_phase17_retrieval.py tests/test_phase17_qa.py tests/test_phase17_rag.py tests/test_phase17_generation.py
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Note that some existing tests can attempt to load optional ML libraries in environments lacking `transformers` or `faiss` due to FastAPI dependency resolution. Your `greek-nlp` environment may have these installed; no existing files are replaced by this patch.

Once all checks pass, commit only the new files. Do not update the roadmap to mark Phase 17 fully complete yet.

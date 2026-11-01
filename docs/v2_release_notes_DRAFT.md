# Draft release notes — v2 candidate (NOT PUBLISHED)

**Status:** Draft only. The current package/API version is `0.1.0` until
explicitly changed and validated. Do not tag or publish a v2 release yet.

## Existing API capabilities documented in the repository

- GreekBERT classification of Greek news into 17 IPTC topics.
- Dense multilingual-e5/FAISS search and opt-in hybrid BM25 + RRF and reranking.
- Extractive question answering and experimental grounded `/rag/ask`.
- Local generation with a PyTorch default and an opt-in localhost llama.cpp
  backend for `/generate`.
- Docker packaging, GitHub Actions regression tests and test coverage reporting.

## Phase 17 verification additions

- API contract and mock-based regression tests (Phase 17.1/17.2).
- CI improvements, test reports and coverage reporting (Phase 17.3).
- Release contract smoke checks and optional real-model probes (Phase 17.4).

## Disclosed limitations

- Phase 15 exploratory RAG evaluation had 80/80 abstentions, so generated-answer
  accuracy, faithfulness and hallucination rates were **not measured**.
- Real-model inference correctness is not established by mocked CI tests or
  a `/health` response.
- Initial local overall code coverage was 75%; some ML modules were not
  exercised by tests because they were mocked.
- Optional llama.cpp is localhost-only and experimental. Independent model
  quality and license review are still needed.
- Classification metrics compare against automatically assigned labels,
  not independently verified human ground truth.

## Before release

Complete the checklist in `docs/phase_17_4_release_validation.md`.
Record the commit SHA, updated version, tag, CI run and real-model evidence.

# Phase 18 — Final release review (candidate only)

## Verified so far

- Local suite: 194 tests passed; Ruff lint and formatting clean (user run, 2 November 2026).
- Live functional checks: GreekBERT, dense/hybrid/reranked search, extractive QA and PyTorch `/generate` passed.
- RAG historically abstained on 80/80 exploratory Phase 15 questions. A later live Mandra example returned an unsupported answer, which the output guard subsequently blocked (`invalid_generated_answer`).
- A contextual evidence and extractive retry refinement has been installed locally, but **its final live result has not yet been reviewed**.
- `faithfulness_verified: false` is part of the current API schema; do not call the generated answers automatically verified.

## Install and run

Place `scripts/phase18_final_rag_review.py` and `tests/test_phase18_final_rag_review.py` in their respective directories. Keep the existing production `grounded.py` and all prior tests. Use the Python 3.11 `greek-nlp` environment.

```bat
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Start only one Uvicorn server (leave it open):

```bat
python -m uvicorn greek_nlp.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bat
python scripts\phase18_final_rag_review.py --show-answers --timeout 600
```

The command runs two live RAG questions (specific Mandra and broad floods) and writes `phase18-final-rag-review.json`, *without raw model text or excerpts*. The `--show-answers` option prints them only to the local console for manual review. Keep generated reports out of Git; `.gitignore` should include `phase18-*.json`.

The script returns a nonzero exit code only for transport/response-structure errors. An abstention on the answerable Mandra example is **not** a successful quality outcome even if the command exit code is zero. An answer with a citation is **not** automatically truthful. Read the `quality_status` fields.

## Manual review (mandatory)

Record the decision for the Mandra example using the human-reviewed source:

> Πειθαρχικές ευθύνες επιρρίπτει το πόρισμα για τις πλημμύρες στη Μάνδρα. Σε σειρά διαφορετικών φορέων του δημοσίου εντοπίζει ευθύνες.

- [ ] Response is on-topic and explicitly answers **who** is held responsible (various public-sector bodies), or validly abstains for a documented reason.
- [ ] The factual answer is supported by the **exact cited passage**; citation marker resolves to that article.
- [ ] No prompt echo, fabricated named entities, unfounded paraphrase or exposed internal prompt text.
- [ ] The broad flood question is labeled *exploratory*; do not treat it as a validated negative/answerable case.
- [ ] Review results from a **larger, independently checked** answerable/unanswerable set (existing Phase 15 set requires label revalidation). Record answer rate, supported answer rate, false answers, and abstention separately.

## Release gate — separate from technical PASS

- [ ] Tests, Ruff, coverage and Docker smoke pass in current GitHub Actions commit.
- [ ] Real-model checks pass on the **current code**, not only older commits.
- [ ] Manual grounded-answer quality review is documented and evidence-based.
- [ ] Model, dataset, and index licenses reviewed (MIT source license does not grant model/data rights).
- [ ] Local-only llama.cpp service is not exposed; authentication/rate limits, data handling and deployment security reviewed before public deployment.
- [ ] README, release notes, limitations, version number, and reproducibility steps match the shipped code.
- [ ] Decide release scope: stable classification/search/QA; experimental RAG clearly labeled or not exposed as a stable feature.

**Do not set `release_approved = true` automatically or create a GitHub release solely because this script exits successfully.**

A project phase can be marked *engineering validation completed with known RAG quality limitations*, while a broader production-ready or v2 claim remains blocked until these reviews are signed off.

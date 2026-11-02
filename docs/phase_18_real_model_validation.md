# Phase 18 — Real-Model Validation & Release Decision

**Status: validation framework prepared; actual real-model results pending.**

This is the final *planned* verification phase before deciding on a first GitHub
release of Greek-News-NLP-API. It does not automatically establish that a
production-quality `v2.0.0` is ready.

## What Phase 18 does

Unlike Phase 17's primarily mocked CI tests, this runs opt-in HTTP requests
against a locally running API that will load actual weights, indexes and
embedding models. It reuses the validated Phase 17.4 request definitions and
checks, and adds sanity checks and a report with **no raw prompts or responses**.

An HTTP 200, a nonempty classification, or a generated answer does **not** prove
accuracy, retrieval relevance, citation support or safety. Independent/manual
evaluation is a separate release gate.

## Before running

- Python 3.11, the project and `scripts/phase17_release_validate.py` from Phase 17.4.
- Local, authorized instance only (`http://127.0.0.1:8000`); do not expose
  experimental model services publicly.
- Public HF repositories for the trained classifier and FAISS index, or your
  verified local artifact cache. Downloads may be large.
- Sufficient free RAM, storage and network; optional LLM is much heavier.
- Ensure required extras are installed. If not already present, from an active
  environment: `python -m pip install -e ".[dev,ml,llm]"` (heavy packages).
- Do **not** put access tokens in source files, CLI commands, screenshots or reports.

### Step A — Start the API (Windows Anaconda Prompt #1)

```bat
set "HF_MODEL_REPO=Tilemachos-P/greek-news-classifier"
set "HF_INDEX_REPO=Tilemachos-P/greek-news-index"
set "LLM_BACKEND=torch"
python -m uvicorn greek_nlp.api.main:app --host 127.0.0.1 --port 8000
```

**Keep this terminal open.** The first request to a new component can download
or load heavy model artifacts. A `/health` success does not prove readiness.

### Step B — Run explicit probes (Windows Anaconda Prompt #2)

Run in smaller groups to observe memory use and isolate failures:

```bat
python scripts\phase18_real_model_validation.py --probe classify --timeout 300 --report phase18-classify.json
python scripts\phase18_real_model_validation.py --probe search-dense --timeout 300 --report phase18-search-dense.json
python scripts\phase18_real_model_validation.py --probe search-hybrid --probe search-rerank --timeout 300 --report phase18-search-advanced.json
python scripts\phase18_real_model_validation.py --probe ask --timeout 300 --report phase18-ask.json
```

**Optional LLM** — only if enough RAM is available:

```bat
python scripts\phase18_real_model_validation.py --probe generate --timeout 600 --report phase18-generate-torch.json
python scripts\phase18_real_model_validation.py --probe rag --timeout 600 --report phase18-rag.json
```

Each command runs four lightweight preflight checks first, and *then* only the
requested expensive probes. A failed preflight stops model probes. `--probe`
is required: the script never loads models without explicit opt-in.

The `rag` probe uses the **original PyTorch generator** even if `LLM_BACKEND`
was switched to `llamacpp` for `/generate`. If memory is limited, restart the
FastAPI server between groups; lazy-loaded objects remain cached while it runs.

### Optional standalone llama.cpp `/generate` backend

Only if this was enabled in Phase 16: launch your local `llama.cpp` server bound
to `127.0.0.1` and restart FastAPI with
`LLM_BACKEND=llamacpp`, `LLAMACPP_BASE_URL=http://127.0.0.1:9931` and the
appropriate `LLAMACPP_MODEL_MATCH`. Repeat **only** the `generate` probe with
an alternate report filename. This does **not** test llama.cpp for `/rag/ask`.

## Interpreting results

- `technical_pass=true`: the selected endpoint responded with a valid shape and
  minimum sanity conditions (e.g., a nonempty retrieval list), **not** a quality grade.
- `manual_review_required=true`: judge actual article relevance, predicted labels,
  factual correctness, evidence spans and citation support on documented cases.
- `rag.quality_status=BLOCKED_BY_ABSTENTION`: structurally valid response, but
  no useful generated answer. This is **not** a quality success.
- `gate.release_approved=false`: intentional. A reviewer must complete the
  checks below and authorize any release; the script never approves one.
- A 503, timeout or missing weights fails the tested component; distinguish
  environment/configuration failure from a model-quality failure.

Reports contain timestamps, selected probe names, pass/fail status, durations
and small metadata only. They intentionally exclude raw article texts, user
prompts, retrieved snippets, token strings and generated responses. Keep
`phase18-*.json` out of version control regardless.

## Human assessment template

Use `docs/phase18_manual_review_template.csv` for a **new, independently
selected** sample of items. Record the observed result manually against a
known source article ID or a source excerpt; mark `uncertain` rather than
assuming the model is correct. Dataset queries used during training and tuning
are not a substitute for held-out evaluation.

For RAG, a fresh, validated answerable/unanswerable evaluation with source-span
annotations is necessary before claiming answer accuracy, calibrated
abstention or citation faithfulness. Phase 15 reported **80/80 abstentions**;
if unchanged, the generated-answer quality gate remains blocked.

## Final release gate (manual signoff)

- [ ] All 172+ existing and Phase 18 added unit tests pass, Ruff passes, new CI green.
- [ ] Core live probes (`classify`, dense, hybrid, rerank, extractive `/ask`) pass
      in the intended environment; exceptions and their root causes documented.
- [ ] Standalone `/generate` and `/rag/ask` separately tested **if** presented
      as supported release functionality; otherwise clearly marked experimental.
- [ ] Review completed on independently selected questions, with recorded
      relevance, factual correctness, source relevance and evidence support.
- [ ] Phase 15 RAG abstention explicitly disclosed; no unsupported claims about
      generated-answer quality; fix and re-evaluate if release promises that.
- [ ] Confirm model/data licenses, attribution and distribution permissions
      (repository MIT license does not cover every model/data artifact).
- [ ] No secrets, raw sensitive data, large model files or local reports committed.
- [ ] Dependency/security review and realistic deployment boundary: unauthenticated
      dev API or llama.cpp services must not be promoted to public production.
- [ ] Check wheel/sdist install in clean environment, Docker startup and real
      artifact-loading environment separately.
- [ ] Update README, model URLs, compatibility, known limitations, release notes.
- [ ] Choose and synchronize `pyproject.toml` version and FastAPI `app.version`
      deliberately; `2.0.0` is **not** automatic just because Phase 18 is complete.
- [ ] Only then create release tag and publish a GitHub Release.

**Completion rule:** Phase 18 is completed as a *validation phase* once all
applicable checks and manual decisions are documented. If a core functionality
fails, create targeted remediation work instead of marking the release ready.

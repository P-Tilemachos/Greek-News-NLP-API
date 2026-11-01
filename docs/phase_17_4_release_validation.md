# Phase 17.4 — Release validation (v2 candidate, NOT released)

This phase adds **repeatable release smoke checks** while retaining the existing API
implementation and package version. It does **not** publish a v2 release, solve the
Phase 15 RAG over-abstention issue or certify real-model answer quality.

## State at start of Phase 17.4

- Phase 17.1/17.2: 164 local pytest cases passed, with Ruff lint and formatting.
- Phase 17.3: the author confirmed the GitHub CI succeeded; reported local coverage
  was **75%** (informational, not a release threshold).
- The package in `pyproject.toml` and FastAPI in `src/greek_nlp/api/main.py` are
  **0.1.0**, *not* 2.0.0. A version/tag change requires a separate release decision.
- The Phase 15 exploratory RAG evaluation had 100% abstention (80/80); this must
  remain visible in public release notes. A successful contract check does not
  establish generated-answer accuracy, faithfulness or reliability.

## Automated (no model weights)

`scripts/phase17_release_validate.py` checks a **running** API using Python's
standard library only. Without `--probe`, the checks are:

1. `GET /health`: HTTP 200 and `{"status":"ok"}`.
2. `GET /openapi.json`: route/method contracts and API version matching
   `pyproject.toml`.
3. `POST /generate` with `{}`: HTTP 422 (invalid input must not initialize LLM).
4. `POST /rag/ask` with `{}`: HTTP 422 (invalid input must not initialize RAG).

New unit tests mock the HTTP transport and never load ML weights. The Docker CI
job now runs this smoke checker after the original `/health` readiness probe and
uploads `phase17-release-smoke.json` as an artifact.

From a Windows Anaconda Prompt, in the repository root:

```bat
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m pip check
```

Start the API in one terminal:

```bat
python -m uvicorn greek_nlp.api.main:app --host 127.0.0.1 --port 8000
```

From a **second** terminal in the project root:

```bat
python scripts\phase17_release_validate.py --base-url http://127.0.0.1:8000
```

The script writes `phase17-release-smoke.json`; keep it out of Git if it contains
local environment details. A zero exit code means the checked contracts passed.

### Docker release smoke

In one terminal, from the root with Docker running:

```bat
docker build -t greek-news-nlp-api:phase17 .
docker run --rm --name greek-nlp-phase17 -p 127.0.0.1:8000:8000 greek-news-nlp-api:phase17
```

In the second terminal run the same `phase17_release_validate.py` command.
Stop the container with Ctrl+C in the first terminal. **No real-model probe is
performed in the default Docker smoke.**

## Optional real-model smoke (explicitly opt in)

After configuring the public Hub repositories and sufficient RAM/disk/network,
start the API in the first terminal:

```bat
set "HF_MODEL_REPO=Tilemachos-P/greek-news-classifier"
set "HF_INDEX_REPO=Tilemachos-P/greek-news-index"
python -m uvicorn greek_nlp.api.main:app --host 127.0.0.1 --port 8000
```

In the second terminal (with the project's `ml` dependencies installed):

```bat
python scripts\phase17_release_validate.py --probe classify --probe search-dense --probe ask --timeout 300 --report phase17-real-model-smoke.json
```

Other opt-in probes: `search-hybrid`, `search-rerank`, `generate`, `rag`.
Each probe may download model weights or indexes and consume substantial memory;
run them only if that component and its dependencies are available. The
`generate` check uses the configured `LLM_BACKEND` (default PyTorch), and `rag`
continues to use PyTorch regardless of that setting.

**What a probe validates:** HTTP 200 and selected response fields.
**What it does not validate:** benchmark accuracy, semantic relevance,
answer correctness, citation support, abstention calibration, model licensing,
public deployment safety, or production readiness. Specifically, even an HTTP
200 with an abstained RAG answer is *not* a successful answer-quality result.

The checker defaults to loopback URLs. It refuses remote URLs unless explicitly
passed `--allow-remote`, which should be used only for a system you are
permitted to test. Do not expose the unauthenticated experimental llama.cpp
service to the public Internet.

## Release candidate gate — must be checked manually

- [ ] Fresh `main` CI is green: tests, Ruff, Docker, release smoke.
- [ ] CI artifacts exist: `phase17-test-reports` and `phase17-release-smoke`.
- [ ] Source build (wheel or sdist) can be installed in a clean environment.
- [ ] Real classifier and retrieval/index smoke checks pass with pinned artifacts.
- [ ] Optional LLMs/backend tested separately if shipped or advertised as ready.
- [ ] RAG Phase 15 over-abstention remains disclosed; corrective work and held-out
      evaluation required before any claim about answer faithfulness or quality.
- [ ] Confirm model/data licenses and attribution, third-party dependencies,
      and no secrets/private datasets/large artifacts in source or release assets.
- [ ] Confirm the API is not made public without auth, rate limiting and network
      policy; check error responses do not leak internal tokens or paths.
- [ ] Set and reconcile `pyproject.toml` version and FastAPI `app.version`
      **only when the version naming and release scope are approved**.
- [ ] Review README, API examples, documentation links and supported environment.
- [ ] Create a release tag and GitHub release only after those gates are met.

**Completion rule:** A green automated CI completes the *automated release
contract portion*. It does not alone authorize calling v2 production-ready.

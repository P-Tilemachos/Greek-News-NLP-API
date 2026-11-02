# Phase 17 — Integration and regression validation (in progress)

## Goal
Strengthen the automated API contract and regression checks, stabilize dependencies, and define gates before a future v2 release. This work does **not** resolve the Phase 15 over-abstention limitation.

## Current baseline
- Phase 16 CI run #46 on `main` passed `test` and `docker` on 2026-11-01.
- The repository's `pyproject.toml` currently identifies the package as `0.1.0`; do not describe it as v2 until release gates have been satisfied and a release has been cut.

## Proposed verification gates
- [ ] New Phase 17 smoke tests pass on Python 3.11 and are compatible with the current endpoint schema.
- [ ] All existing pytest regression tests pass.
- [ ] Ruff lint and formatting checks pass.
- [ ] Docker image builds and `/health` responds in CI.
- [ ] Regression tests cover dense, hybrid and reranked search using mocked dependencies.
- [ ] Regression tests cover both default PyTorch and opt-in llama.cpp, including unavailable backend and timeout behavior.
- [ ] Tests cover `/rag/ask` citation/abstention response contracts without claiming model quality.
- [ ] Dependency installation is reproducible; `requests` and `psutil` are declared in the appropriate optional-dependency group rather than added ad hoc in CI.
- [ ] Security review: local-only llama.cpp URL, public API exposure, artifact permissions and lack of authentication are documented.
- [ ] Reproducible real-model staging smoke tests are recorded separately from mocked CI tests.
- [ ] Version bump, release notes, tag and published release occur only after the required gates pass.

## Local commands
```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Notes
The included starter API tests are *proposed additions*, not validated against the whole source tree in this package. Run them locally and correct any endpoint/request-schema assumptions before committing. These smoke tests verify routing/validation only, not the correctness of downloaded ML models.

# Phase 17.3 — CI hardening

## Scope

This phase upgrades the existing GitHub Actions workflow; it does not change API
endpoints, model weights, benchmark results or public deployment behavior.

## Changes

- Development tools and Phase 16 test dependencies live in `pyproject.toml`
  under `dev`, so the CI installs them with `python -m pip install -e ".[dev]"`.
- `python -m pip check` verifies installed package metadata consistency.
- The test job runs Ruff lint and formatting, then pytest with `pytest-cov`.
- Every pytest run writes `test-results.xml` (JUnit) and `coverage.xml` (Cobertura)
  as long as it reaches the test command. GitHub Actions uploads the existing
  files as an artifact even if that step fails.
- GitHub Actions uses explicit read-only repository permissions, a Python 3.11
  package cache based on `pyproject.toml`, concurrency cancellation and timeouts.
- Docker still builds the application and probes `/health`; the container is
  cleaned up even if the smoke test fails.
- The Phase 17 API contract test mocks heavyweight FastAPI dependencies during
  malformed-request checks. This prevents test order from controlling whether
  ML weights are imported or downloaded.

## Local validation

From repository root (Windows Anaconda Prompt):

```bat
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --cov=greek_nlp --cov-report=term-missing:skip-covered --cov-report=xml:coverage.xml --junitxml=test-results.xml
```

The Docker smoke test can be run locally if Docker is installed. Its successful
execution on GitHub Actions is the primary gate in this phase.

## Coverage interpretation

Coverage is **informational** in this phase. No minimum coverage gate is enforced
until the team reviews the reported baseline. The suite uses mocks and cannot
validate the correctness, relevance, safety or faithfulness of real model outputs.
A successful `/health` smoke test checks liveness, not model/index readiness.

## Completion criteria

1. Local Ruff, pytest and `pip check` pass after the changes.
   A standalone `python -m pytest -q tests/test_phase17_api_contract.py` run also passes.
2. GitHub Actions on `main` reports green `test` and `docker` jobs.
3. The `phase17-test-reports` artifact includes `test-results.xml` and
   `coverage.xml`; record the actual reported coverage percentage.
4. Do not create a v2 tag/release yet — Phase 17.4 release validation remains.

## Repository hygiene

Keep locally generated ZIPs, test-result files and source-tree inventories out
of commits. Stage only files intended for GitHub. Review README/documentation
changes separately before adding them.

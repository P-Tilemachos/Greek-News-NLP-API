@echo off
setlocal
cd /d "%~dp0.."
echo ===== Phase 13 - install lightweight test dependencies =====
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
echo ===== Phase 13 - format only Phase 13-touched Python files =====
python -m ruff format src/greek_nlp/api/main.py src/greek_nlp/api/schemas.py src/greek_nlp/llm tests/test_generate.py
if errorlevel 1 exit /b 1
echo ===== Phase 13 - syntax and full unit/API tests (no model download) =====
python -m compileall -q src/greek_nlp tests
if errorlevel 1 exit /b 1
python -m pytest -q
if errorlevel 1 exit /b 1
echo ===== Phase 13 - Ruff lint and formatting =====
python -m ruff check .
if errorlevel 1 exit /b 1
python -m ruff format --check .
if errorlevel 1 exit /b 1
echo PHASE 13 OFFLINE CHECKS PASSED.
echo Real /generate inference is a separate opt-in smoke test.

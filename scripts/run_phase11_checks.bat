@echo off
setlocal
cd /d "%~dp0.."
echo ===== Phase 11 - install lightweight CI/test dependencies =====
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
echo ===== Phase 11 - format only modified Python files =====
python -m ruff format src/greek_nlp/api/main.py src/greek_nlp/api/schemas.py src/greek_nlp/rag/hybrid.py tests/test_hybrid.py
if errorlevel 1 exit /b 1
echo ===== Phase 11 - unit and API tests =====
python -m pytest -q
if errorlevel 1 exit /b 1
echo ===== Phase 11 - lint =====
python -m ruff check .
if errorlevel 1 exit /b 1
echo ===== Phase 11 - formatting =====
python -m ruff format --check .
if errorlevel 1 exit /b 1
echo PHASE 11 OFFLINE CHECKS PASSED.
echo A live HF-index smoke test is a separate optional step.

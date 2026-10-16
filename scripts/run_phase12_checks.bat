@echo off
setlocal
cd /d "%~dp0.."
echo ===== Phase 12 - install test dependencies =====
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
echo ===== Phase 12 - syntax and full unit/API suite =====
python -m compileall -q src/greek_nlp tests
if errorlevel 1 exit /b 1
python -m pytest -q
if errorlevel 1 exit /b 1
echo ===== Phase 12 - lint and formatting checks =====
python -m ruff check .
if errorlevel 1 exit /b 1
python -m ruff format --check .
if errorlevel 1 exit /b 1
echo PHASE 12 OFFLINE CHECKS PASSED.
echo A live rerank smoke test with HF weights remains a separate step.

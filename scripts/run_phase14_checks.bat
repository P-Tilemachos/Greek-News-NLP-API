@echo off
setlocal
cd /d "%~dp0.."
echo ===== Phase 14 - lightweight API/test dependencies =====
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
echo ===== Phase 14 - format changed Python files =====
python -m ruff format src/greek_nlp/api/main.py src/greek_nlp/api/schemas.py src/greek_nlp/rag/grounded.py src/greek_nlp/llm/generator.py tests/test_grounded_rag.py tests/test_generate.py
if errorlevel 1 exit /b 1
echo ===== Phase 14 - test suite (no model download) =====
python -m compileall -q src/greek_nlp tests
if errorlevel 1 exit /b 1
python -m pytest -q
if errorlevel 1 exit /b 1
echo ===== Phase 14 - lint and formatting checks =====
python -m ruff check .
if errorlevel 1 exit /b 1
python -m ruff format --check .
if errorlevel 1 exit /b 1
echo PHASE 14 OFFLINE CHECKS PASSED.
echo Live RAG inference with real indices/models is a separate opt-in test.

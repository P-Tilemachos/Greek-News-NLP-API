@echo off
setlocal
cd /d "%~dp0.."
echo ===== Phase 15 - install local package and test dependencies =====
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
echo ===== Phase 15 - format only new Phase 15 Python files =====
python -m ruff check src/greek_nlp/evaluation tests/test_phase15_rag_eval.py scripts/phase15_collect_predictions.py scripts/__init__.py --fix
if errorlevel 1 exit /b 1
python -m ruff format src/greek_nlp/evaluation tests/test_phase15_rag_eval.py scripts/phase15_collect_predictions.py scripts/__init__.py
if errorlevel 1 exit /b 1
echo ===== Phase 15 - offline tests (no model download) =====
python -m pytest -q
if errorlevel 1 exit /b 1
echo ===== Phase 15 - Ruff lint and format checks =====
python -m ruff check src/greek_nlp/evaluation tests/test_phase15_rag_eval.py scripts/phase15_collect_predictions.py scripts/__init__.py
if errorlevel 1 exit /b 1
python -m ruff format --check src/greek_nlp/evaluation tests/test_phase15_rag_eval.py scripts/phase15_collect_predictions.py scripts/__init__.py
if errorlevel 1 exit /b 1
echo PHASE 15 OFFLINE CHECKS PASSED.
echo Saved-prediction evaluation requires NO weights and performs NO LLM inference.
endlocal

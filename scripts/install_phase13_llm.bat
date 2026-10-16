@echo off
setlocal
cd /d "%~dp0.."
echo ===== Phase 13 - install OPTIONAL real LLM dependencies =====
echo This may download large packages. It does not download model weights until /generate.
python -m pip install -e ".[dev,llm]"
if errorlevel 1 exit /b 1
python -c "import torch, transformers; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'transformers:', transformers.__version__)"
if errorlevel 1 exit /b 1
echo PHASE 13 LLM DEPENDENCIES READY.

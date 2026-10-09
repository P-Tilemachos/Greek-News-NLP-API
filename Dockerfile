FROM public.ecr.aws/docker/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Πρώτα το PyTorch για CPU (πολύ μικρότερο από την έκδοση GPU)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[ml]"

# Μη-root χρήστης, και φάκελος για τα μοντέλα που κατεβαίνουν στην εκκίνηση
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/models_store \
    && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn greek_nlp.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
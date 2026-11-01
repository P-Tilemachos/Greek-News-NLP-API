# Phase 16 — Optional llama.cpp generation backend (experimental)

The existing `/generate` API keeps its request and response schemas. The default
continues to be the original lazy PyTorch `LocalGenerator`. A separate, opt-in
`llama.cpp` HTTP client can use the separately started **Krikri 8B Q4_K_M**
model. No server is launched by the Python application, and no model is
loaded during `/health` or import.

**Important:** `/rag/ask` deliberately continues to use the *original PyTorch*
`get_generator()` and remains subject to the Phase 15 over-abstention limitation.
This implementation **does not** enable GGUF for grounded RAG, resolve
faithfulness, or stream unverified RAG content.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BACKEND` | `torch` | `torch` or opt-in `llamacpp` for `/generate` only |
| `LLAMACPP_BASE_URL` | `http://127.0.0.1:9931` | HTTP loopback address, explicit port required |
| `LLAMACPP_MODEL_MATCH` | `Krikri` | Case-insensitive substring required in the served model ID |
| `LLAMACPP_TIMEOUT_SECONDS` | `300` | HTTP timeout (1–600 seconds) |

The existing `LLM_MODEL_ID` and `LLM_DEVICE` remain applicable to the original
PyTorch generator, including `/rag/ask`.

## Run on Windows (two distinct Anaconda Prompts)

**Server prompt** (start before API):

```bat
conda activate greek-nlp
llama serve -hf ilsp/Llama-Krikri-8B-Instruct-GGUF:Q4_K_M --host 127.0.0.1 --port 9931 -t 8 -c 2048 -ngl 0
```

**API prompt** (new terminal; from the repo root):

```bat
conda activate greek-nlp
set "LLM_BACKEND=llamacpp"
set "LLAMACPP_BASE_URL=http://127.0.0.1:9931"
set "LLAMACPP_MODEL_MATCH=Krikri"
python -m uvicorn greek_nlp.api.main:app
```

**Test prompt** (third terminal):

```bat
curl.exe http://127.0.0.1:8000/health
python -c "import requests; p={'prompt':'Τι είναι η μηχανική μάθηση;','max_new_tokens':64}; r=requests.post('http://127.0.0.1:8000/generate',json=p,timeout=360); print('HTTP:',r.status_code); print(r.json())"
```

Expected: HTTP 200 and a JSON object containing `mode: local_llm`, `model`,
`device: cpu`, `generated_text`, `input_tokens` and `output_tokens`. Do not
interpret this as a passing quality benchmark. Model quality still requires
independent evaluation; the 8-case Krikri development screen was only an
exploratory author-unverified pilot and had a conflict-evidence error.

## Failure handling and safety

- Loopback-only HTTP; remote/HTTPS URLs, userinfo, paths and redirects are
  rejected; proxies are disabled for this client.
- A served model lacking the required `LLAMACPP_MODEL_MATCH` substring is
  rejected on every generation request. This protects against switching back
  to the wrong local model on the same port.
- Non-200 responses, unavailable server, truncated/bad JSON, missing usage,
  empty text and invalid completion counts produce HTTP 503 with a generic
  error. Raw upstream responses are not returned to API callers.
- `/generate` is **ungrounded** text generation. The default `device: cpu`
  value denotes the instructed `-ngl 0` server configuration; the app does
  not interrogate the llama.cpp runtime to confirm compute placement.
- Only the existing `/generate` endpoint supports this experimental backend.
  Do not expose the unauthenticated llama.cpp or FastAPI servers publicly.
- The loopback restriction is not an authentication or OS-level sandbox.

## Local checks before Git commit

```bat
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

The included tests mock llama.cpp replies: no model files, server or internet
connection is required in CI. Before final Phase 16 release, inspect generated
Greek responses and expand independent quality evaluation. The overall Phase 16
quantization result combines *backend + weight precision* changes and must not
be attributed solely to quantization.

# Phase 14 — Opt-in Grounded Generative RAG

## Scope and relationship to earlier phases

This is an **experimental, opt-in** `POST /rag/ask` endpoint. It combines:

1. Phase 11 **Hybrid BM25 Full + Dense E5 (RRF)** on the deployed full-article corpus;
2. Phase 12 **Top-5 Cross-Encoder** reranking (`BAAI/bge-reranker-v2-m3`);
3. A small **corpus-backed excerpt selector**, operating on the *full original article texts* and IDs paired with the FAISS index;
4. Phase 13 **local open-source Qwen generator**, with a bounded context and output budget;
5. A structured response containing cited source excerpts and **conservative abstention**.

The existing extractive `POST /ask`, `POST /search` modes (`dense`, `hybrid`, `rerank`), and standalone `POST /generate` are unchanged. `POST /rag/ask` is not a new default and requires the existing optional ML/LLM dependencies, downloaded E5 index and Cross-Encoder weights.

This phase does **not** rerun or overwrite the frozen Notebooks 08–10 benchmarks, does not train or fine-tune an LLM, and does not prove answer faithfulness. A full independent human-assessed QA evaluation is planned for Phase 15.

## Example request

```json
{
  "question": "Ποιες αποζημιώσεις δόθηκαν στους αγρότες μετά τις πλημμύρες;",
  "k": 3,
  "max_new_tokens": 96
}
```

Use `http://127.0.0.1:8000/docs` → `POST /rag/ask` to execute. Do **not** expect a predetermined answer or particular article IDs: output depends on the indexed corpus, models and the question.

## Response contract

When a candidate excerpt passes the evidence gate:

```json
{
  "mode": "grounded_rag",
  "answer": "<model-produced Greek text; may be incorrect>",
  "abstained": false,
  "reason": null,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "sources": [
    {"reference":"[1]","doc_id":"<index document ID>","label":"<category>","excerpt":"<verbatim excerpt from the article>","retrieval_score":0.0}
  ],
  "faithfulness_verified": false
}
```

`retrieval_score` is an **uncalibrated Cross-Encoder score**, not a probability of evidence relevance or factual correctness. The `reference` marker points to an **actual excerpt** from the corresponding `doc_id`. Retrieved excerpts are not independent proof that *every generated claim* is supported; the model can ignore or misinterpret them. The `answer` may omit markers, or contain errors. Source markers not present in the returned `sources` are filtered.

When evidence is inadequate, a response will have `"abstained": true`, `"answer": null`, and `"reason": "insufficient_retrieved_evidence"` (the generator is **not loaded**). Other possible reasons are `"input_budget_exceeded"` and `"model_abstained"`. This is a heuristic abstention mechanism, **not a calibrated answerability classifier**; it may reject valid questions or accept weak passages.

The API returns HTTP **503** for model/index failures; internal file paths and tokens are not included in the response.

### Controlled extractive retry after model abstention

A diagnostic with the actual Qwen2.5-1.5B-Instruct showed that the original
strict prompt returned `ΑΝΕΠΑΡΚΗ ΣΤΟΙΧΕΙΑ` even when an article explicitly
contained the answer. A separate **single-source** probe using a shorter,
answer-oriented extraction prompt produced `το καλοκαίρι του 2017 [1]`
from a matching article excerpt. This probe is not a QA evaluation.

When the strict production prompt returns an empty response or this refusal
marker, the RAG implementation may now perform **at most one additional,
bounded extraction attempt** using only the top cited source. The fallback
response is accepted only if it ends in `[1]` and its answer phrase is a
**literal span** of that very excerpt. An unsupported phrase, missing/wrong
citation, or another refusal causes `model_abstained`, as before. The
response then includes only the article actually used for the retry.

This is a *conservative extractive fallback*, not evidence that the
open-source model can generate faithful Greek explanations. It can increase
CPU latency (one extra inference on abstention). A matching quotation also
does **not** prove that it answers the precise question; Phase 15 still needs
independent human assessment of answerability, citation relevance and factual
faithfulness. `faithfulness_verified` remains `false` in all cases.

## Implementation details

- The index and full-article text are paired by `document_id`, not selected by input supplied by the caller.
- The excerpt selector requires at least **two** matched significant Greek query terms when the query has two or more (one match when the query only has one term); matches are accent-insensitive word-prefix matches, not evidence entailment.
- At most **two** short source excerpts are included. Their source IDs and extracted text are derived from the returned full articles; arbitrary IDs and fabricated snippets are not cited.
- Untrusted article text is supplied to the LLM as data, never as a system instruction. Angle-bracket role-like markup in that data is escaped inside the prompt. This is a partial prompt-injection mitigation, not a guarantee.
- Input text length and the Qwen chat-template token count are bounded. The system trims excerpts when needed and abstains if the context cannot fit the generator's 1024-token input cap (with a conservative 896-token RAG prompt budget). Maximum generated tokens: 128.
- The LLM is lazy-loaded only after evidence is found. Existing `/health`, `/classify`, `/search`, `/ask` and `/generate` behavior remains intact.

## Offline checks (no model downloads)

On Windows, from the repository root and the `greek-nlp` Conda environment:

```bat
scripts\run_phase14_checks.bat
```

The command installs lightweight dev dependencies and runs the **full unit/API test suite** followed by Ruff lint/format checks. It never downloads model weights, invokes the Hugging Face Hub, or reruns historical benchmarks.

For an optional **live smoke test** (requires free disk/RAM; CPU is slow):

```bat
python -m pip install -e ".[ml,llm]"
set "LLM_DEVICE=cpu"
set "LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct"
python -m uvicorn greek_nlp.api.main:app
```

In Swagger, execute `POST /rag/ask` (or use the single-cell Notebook 14). A live smoke test verifies end-to-end wiring, **not** factual correctness.

If your system has insufficient RAM for Qwen or Cross-Encoder, stop rather than repeatedly downloading/running weights. You may complete the offline validation and defer the live test until adequate hardware is available.

## Limitations and Phase 15 evaluation requirements

- Small Qwen checkpoints produced **weak Greek-language responses** in Phase 13; integrating retrieval cannot, by itself, guarantee their quality. Report low-quality answers as failures, not successes.
- Retrieved excerpts are based on lexical matching; the system does **not** perform entailment validation, knowledge graph verification, or claim-by-claim attribution.
- Real faithfulness, hallucination rate, citation correctness, abstention precision/recall, answerability, and latency must be measured against an **independently authored human-assessed** held-out evaluation set.
- Comparing Phase 14 vs the Phase 10 extractive baseline requires careful controls: those phases have different scoring and answer formats, and the Phase 10 overall answer accuracy was **not** established.
- LLM inference and Cross-Encoder reranking on CPU are not expected to have production-grade latency. No model weights or restricted data should be committed to Git.

## Roadmap update after checks

Only after the full tests, live smoke test (if feasible), Git push and **green CI**, mark Phase 14 checked in the `README.md`. Keep Phases 15–17 unchecked until implemented and verified.

# Phase 18 — Conservative evidence refinement (experimental)

## Why

The live tests identified two separate behaviors:

1. The broad flood question returned `insufficient_retrieved_evidence` because no single allowed 155-character excerpt included two query stems. This can be appropriate when the answer is not present.
2. The narrower question about responsibility for the Mandra floods found a relevant source, but Qwen returned malformed text. The previously installed output guard correctly returned `invalid_generated_answer` rather than showing that text.

## What this patch changes

- The evidence extractor now optionally considers **two adjacent sentences from the same article** as one excerpt when they share a question-related keyword, the combination is an *exact substring* of the article and it fits the existing 155-character budget. The original per-sentence matching remains.
- On ties in question-term coverage, the extractor prefers this connected two-sentence excerpt because the following sentence may contain the relevant entity or action.
- For questions using Greek `ποιους` / `ποιες`, the existing **single bounded retry** now explicitly requests an *exact verbatim* answer span with citation `[1]`. This is **not** deterministic extraction and does **not** weaken the checks on generated output.
- **No thresholds were reduced.** The original abstention paths, citation checks, prompt-token budget and API response schema remain.

## Critical limitations

- The example Mandra source may *already* provide the key phrase in one excerpt; in that case the main potential improvement comes from the changed retry prompt, not sentence joining.
- A cited phrase, or an ungrounded paraphrase accompanied by `[1]`, is **not** proof of semantic correctness. `faithfulness_verified` intentionally remains `false`.
- A malformed or unsupported retry is still rejected; the system must continue to abstain when the evidence is insufficient.
- This patch has not been validated with actual model weights in this environment. Re-run the exact Mandra question against your running API before deciding whether it helps.
- Do not enable automatic extractive fallback based only on keyword overlap. Any such feature needs a human-reviewed benchmark, including hard negative examples.

## Run local checks

```bat
python -m pytest tests\test_phase18_contextual_evidence.py -v
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The previously reported total was 187 tests. This package adds 7 tests, so **194 total** is expected if the working tree otherwise remains unchanged.

## Live check

1. Stop the old Uvicorn server with Ctrl+C. Ensure you start only one server on port 8000.
2. Start it again from the repository root:

```bat
python -m uvicorn greek_nlp.api.main:app --host 127.0.0.1 --port 8000
```

3. In a second prompt, run the real-model request:

```bat
python -c "import requests; q='Σε ποιους επιρρίπτει πειθαρχικές ευθύνες το πόρισμα για τις πλημμύρες στη Μάνδρα;'; r=requests.post('http://127.0.0.1:8000/rag/ask',json={'question':q,'k':5,'max_new_tokens':128},timeout=600); d=r.json(); print('HTTP:',r.status_code); print('Abstained:',d.get('abstained')); print('Reason:',d.get('reason')); print('Answer:',d.get('answer')); print('Faithfulness:',d.get('faithfulness_verified')); print('Sources:',[(s.get('doc_id'),s.get('excerpt')) for s in d.get('sources',[])])"
```

Record whether the system returns a **verbatim and materially responsive** answer with a real citation, or safely abstains. Neither an isolated successful response nor these unit tests establish release-grade answer quality.

## Before publishing v2

Evaluate a fixed human-reviewed set of answerable and genuinely unanswerable questions, including the problematic Mandra example and broad flood questions. Report answer rate, supported-answer rate, citation correctness, false-answer rate and abstention rate independently. Review model/data licenses and deployment/security controls separately.

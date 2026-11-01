# Phase 15 — Grounded RAG evaluation (exploratory author-reviewed pilot)

## Status (1 November 2026)

**The Phase 15 evaluation run, saved-prediction collection, author review, and code quality checks have been completed.** This is an **exploratory pilot**, **not** an independently validated gold benchmark and **not** evidence of successful answer generation. The principal empirical finding is **complete over-abstention**: the evaluated `/rag/ask` system returned no answers in 80 attempts, including 40 author-reviewed answerable cases.

The evaluation tooling can be committed as a reproducible *workflow*. The current generative QA capability and the answerability labels for negative cases require further validation. This document does not claim that a research-quality benchmark or production-grade RAG system is complete.

## What was evaluated

- **Endpoint:** `POST /rag/ask` — generative, retrieval-grounded QA. This is distinct from the extractive `/ask` endpoint and its Notebook 10 benchmark.
- **Runtime used in the collection:** local `Qwen/Qwen2.5-1.5B-Instruct` with `LLM_DEVICE=cpu`, according to the execution configuration. Do not infer latency or quality on other hardware from this run.
- **Corpus:** local static index of 5,250 Greek news articles spanning 17 topic categories, not a live news source.
- **Cases:** 80 question records: 40 author-reviewed answerable positives and 40 *provisional* negative/unanswerable candidates. Each negative candidate is paired with a positive source article, so these are not 80 independent source-article samples.
- **Article/question selection:** positives were developed from an author-selected candidate pool covering 17 categories; candidate negatives were screened using BM25 top-5 and E5/FAISS top-8 retrieval. These screens do **not** establish corpus-wide absence of an answer.
- **Development/evaluation split:** `p15_a_001` and `p15_u_001` were development pilots. The other 78 cases were collected afterward and reported separately. The 80-case summary includes the two development pilots for illustration and must not be called a fully held-out result.
- **Reviewer:** dataset author, not a blinded independent adjudicator. All 80 abstentions have reviewer-attributed entries in the final locally imported review. No generative answers were available for human faithfulness, hallucination, or answer-correctness assessment.

Original model/prompt/index artifacts, checksums, and local run metadata should be preserved alongside the private evaluation outputs. The earlier Notebook 10 *question-level* overlap has not been fully verified; the sample is not a publication-grade independent test set.

## Observed results

| Metric | 78 cases excluding the development pilots | All 80, illustrative pilot view |
|---|---:|---:|
| Predictions collected | 78 / 78 | 80 / 80 |
| Answers returned | 0 / 78 | 0 / 80 |
| Abstentions | 78 / 78 (100%) | 80 / 80 (100%) |
| Answerable questions answered | 0 / 39 (0%) | 0 / 40 (0%) |
| Verified positive source ID present in returned sources | 37 / 39 (94.87%) | 38 / 40 (95.00%) |
| `model_abstained` reasons | 77 | 79 |
| `insufficient_retrieved_evidence` reasons | 1 | 1 |
| Authored review records imported | 78* | 80 / 80 |
| Answer correctness, response faithfulness, hallucination rate | N/A — no generated answers | N/A — no generated answers |

\* The 78 cases are contained within the 80 reviewed records, but the existing `phase15_complete_workflow.py` builds its separate 78-case aggregate from the unreviewed saved-predictions view. Consequently its **78-case `human_reviewed: 0` display is a reporting-path limitation**, not proof that the author omitted the 78 review entries. The 80-case reviewed aggregate counts the review records. This distinction must not be hidden or silently repaired by inventing judgments.

The 40 provisional negatives also all resulted in abstentions. An observed 40/40 abstention rate **on author-provisionally-labeled negatives** is not an independently validated true-negative or hallucination-avoidance rate. The negative labels require more rigorous full-corpus/independent review; some individual abstentions were recorded as uncertain.

### Interpretation

1. **Retrieval often succeeds at document level, but response generation does not.** A correct returned source document is not sufficient to prove that the *specific answer span* reached the generator. In multiple cases the returned excerpt is truncated or does not contain the answer.
2. **There are clear false-abstention examples.** For instance, development case `p15_a_001` retrieved an excerpt explicitly containing the album title `Moss`, yet the model abstained. This supports further investigation into prompt compliance, generation, post-generation safeguards, and evidence-span selection.
3. **No affirmative faithfulness or hallucination rates can be reported.** When every response abstains, the denominators for assessing the factual content of model-generated answers are zero. Do not report these N/A metrics as 0% hallucinations or 100% faithfulness.
4. **The problem remains open.** These baseline observations do not demonstrate an answer-capable grounded generative QA system, and the 78 evaluation cases should not be used for model/prompt tuning while still described as held out.

## Integrity and reporting constraints

- `answerable: true` reflects author review of positive source articles; `answerable: false` for negatives is **provisional**. Abstention must never be used as evidence to assign ground-truth answerability.
- Document-ID overlap (`gold_doc_ids`) is not an answer span match, entailment check, citation correctness, or proof of a faithful answer.
- The two development cases must remain separated from the 78-case exploratory result.
- Author review is not independent evaluation. No inter-rater agreement can be claimed, and no independent hallucination audit occurred.
- The 5,250-article static corpus and 256-token embedding limit constrain generalization and relevance of retrieved excerpts.
- Do not commit raw news text, local predictions/excerpts, author-level reviews, run outputs, model weights, or private metadata. Public documentation should contain aggregate statistics and methodology only.
- The generated `phase15_completion_summary_NOT_GOLD.md` may retain generic wording such as “not completed human validation” or show `human_reviewed: 0` for the separate 78-case section; see the reporting-path limitation above. Do not misrepresent this output as an independent audit.

## Repository layout

| File | Purpose |
|---|---|
| `src/greek_nlp/evaluation/rag_eval.py` | Offline scoring and review CSV import/export |
| `scripts/phase15_collect_predictions.py` | Resume-safe, explicitly authorized live API collection |
| `scripts/phase15_complete_workflow.py` | Assemble 78/80-case reporting views and HTML author-review workflow |
| `scripts/run_phase15_checks.bat` | Offline regression/quality checks |
| `notebooks/15_rag_evaluation.ipynb` | Offline reporting notebook — inspect/clear outputs before publication |
| `evaluation/phase15_cases_TEMPLATE.json` | Public empty case template, not evaluated data |
| `evaluation/phase15_run_metadata_TEMPLATE.json` | Public metadata template, not private run metadata |
| `tests/test_phase15_rag_eval.py` | Evaluation/scoring tests |
| `tests/test_phase15_complete_workflow.py` | Completion workflow tests |
| `results/phase15_local/` | **Git-ignored local-only** artifacts and reviewer data |

The root `.gitignore` should ignore `results/phase15_local/` in its entirety. Generated `.bak` and local debug logs should not be staged.

## Reproduce the *offline* report from saved local artifacts

The following commands **do not rerun Qwen**. They require existing local files and must be run from the repository root in the `greek-nlp` environment:

```bat
python -m greek_nlp.evaluation.rag_eval import-review --input results\phase15_local\phase15_pilot80_predictions_NOT_GOLD.json --csv results\phase15_local\phase15_output_author_review.csv --output results\phase15_local\phase15_pilot80_FINAL_AUTHOR_REVIEW.json
python scripts\phase15_complete_workflow.py reviewed-report --input results\phase15_local\phase15_pilot80_FINAL_AUTHOR_REVIEW.json
python -m pytest -q
python -m ruff check .
```

During final checks on 1 November 2026, the user reported **86 passed** and **`All checks passed!`** for Ruff. Those are code-quality checks, not evidence of successful generative QA answers. The reviewed summary lives in `results/phase15_local/phase15_final_summary_NOT_GOLD.json` and `results/phase15_local/phase15_completion_summary_NOT_GOLD.md`.

### Generic scoring/review interface

For a future, separately validated dataset, the evaluator also provides `score`, `export-review`, and `import-review`:

```bat
python -m greek_nlp.evaluation.rag_eval score --input results\phase15_local\predictions.json --output-dir results\phase15_local\report
python -m greek_nlp.evaluation.rag_eval export-review --input results\phase15_local\predictions.json --csv results\phase15_local\human_review.csv
python -m greek_nlp.evaluation.rag_eval import-review --input results\phase15_local\predictions.json --csv results\phase15_local\human_review.csv --output results\phase15_local\reviewed.json
```

The saved `/rag/ask` response must include its actual `answer`, `abstained`, `reason`, and `sources`; it must not be replaced with a fabricated answer. The collector requires `--confirm-live` for actual inference. Do **not** collect the 78 original cases again just to regenerate this report.

## Metric definitions

| Metric | Denominator | What it does and does not establish |
|---|---|---|
| Response coverage / abstention | Saved predictions | Operational response-selection behavior, not accuracy |
| Gold-source retrieval | Positive cases with curated `gold_doc_ids` | Source-ID overlap, **not** evidence-span support |
| Answerable answer rate | Author-reviewed positives | Whether any answer was produced, not whether it was correct |
| Provisional-negative abstention | Author-provisionally-labeled negatives | Descriptive pilot rate, **not** a validated true-negative rate |
| Citation syntactic validity | **Answered** cases only | Reference-format check, not semantic faithfulness |
| Correctness / faithfulness / hallucination | Answered cases with reviewer judgments | **N/A when there are no answered cases** |

The API output flag `faithfulness_verified: false` must not be overwritten by these offline reports.

## Outstanding work and release gate

- Preserve this 80-case evaluation as a **negative baseline**. Investigate over-abstention using development-only cases; keep generation, guardrail, and evidence-selection failures distinguishable. Validate modifications on a **new** held-out set rather than retuning on the 78 saved evaluation questions.
- Validate the negative answerability labels through more complete corpus-wide inspection and, preferably, independent blinded review. Document any disagreement and adjudication.
- Fix the reporting-path inconsistency for the 78-case human-review count in a subsequent, separately tested change if that breakdown is needed for publication.
- Audit Notebook 10 overlap and archive environment/model/index/metadata provenance for reproducibility.
- Rerun an independently constructed, post-fix benchmark that contains both answerable and unanswerable cases; only then assess nonzero-denominator faithfulness, correctness, and hallucination rates.

**Roadmap recommendation:** mark the *exploratory Phase 15 evaluation run and code checks* as completed in prose, but do **not** mark a broad “independently validated QA, faithfulness and hallucination evaluation” checkbox as fully complete yet.

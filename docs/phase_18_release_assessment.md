# Phase 18 — Final Validation & Release Assessment

**Project:** Greek-News-NLP-API
**Assessment date:** 2026-11-02
**Software version observed in Phase 18 report:** `0.1.0`
**Decision:** **Technical validation completed; full release approval BLOCKED pending quality, security, and licensing review.**

## Verified results

| Component | Observed result | Scope of evidence |
|---|---|---|
| Test suite | 204/204 passing | Local automated tests and mocks; not a real-model quality guarantee |
| Ruff | Lint and format passing | Local static checks |
| GreekBERT `/classify` | PASS | Real-model smoke probe, not independent accuracy evaluation |
| `/search` dense/hybrid/rerank | PASS | Real-model/API sanity checks, not human relevance assessment |
| Extractive `/ask` | PASS | Real endpoint validation, not exhaustive QA assessment |
| PyTorch `/generate` | PASS | Model generates output; factuality not validated |
| Grounded `/rag/ask` | Technical PASS | Does not imply answer-quality PASS |
| Final case: Mandra (known answerable) | **Abstained** (`Answer: None`) | Returned an apparently relevant source, but did not answer |
| Final case: broad flood query | **Abstained** | Exploratory question; not proven unanswerable |
| License/security review | Not yet verified | Separate release blockers |

## RAG evidence

For the known-answerable Mandra question, the final reviewer displayed source `macocu.el.9211668`, with an extract about disciplinary responsibilities and public-sector bodies, yet returned `Answer: None`. The final check labelled it `KNOWN_ANSWERABLE_ABSTENTION` (117.297 s in the provided run). The exploratory flood question also abstained (`ABSTENTION_NEEDS_CASE_REVIEW`, 32.61 s).

This is **over-abstention on a known-answerable case**. The newer output guard did prevent an earlier malformed generated answer from being displayed; that is a safety improvement, but not proof of useful answers or general faithfulness. The `faithfulness_verified` field should not be interpreted as an independent factuality assessment.

The Phase 15 exploratory review previously documented 80/80 abstentions. It remains a separate, limited historical evaluation with provisional negative labels; no answer-generation accuracy may be inferred from those abstentions.

## Release gates

- [x] Run local regression/contract tests and lint/format checks.
- [x] Run functional real-model probes for classification, retrieval, extractive QA and generation.
- [x] Run conservative RAG safety/abstention checks.
- [ ] Obtain an independently reviewed answerable QA set and assess correct answers, valid abstentions, and citation faithfulness.
- [ ] Decide whether RAG should be excluded/disabled in a deployment or clearly marked experimental; do not claim it is production-ready.
- [ ] Review license obligations for the corpus, all model weights, and redistributed assets separately from the code's MIT license.
- [ ] Review deployment security: authentication/authorization, rate limiting, resource exhaustion, access restrictions for external LLM services, and secrets handling.
- [ ] Confirm Docker and GitHub CI for the final committed code, then tag a version only after approving the intended release scope.

## Recommended decision

**Close Phase 18 as a completed *validation effort*, not a completed *quality certification*.** Keep Grounded RAG experimental or out of a public deployment until a human-reviewed benchmark supports release. The other endpoints have passed their specified functional sanity tests; deployment/readiness still requires separate security, licensing and documentation checks.

Do not change the current `0.1.0` package version to `2.0.0` solely because the automated tests are green.

## Next engineering work (if Grounded RAG is in release scope)

1. Freeze a small, independently reviewed set of answerable and genuinely unanswerable questions, with exact evidence spans and article IDs.
2. Log the RAG abstention reason and whether it happens during evidence selection, initial generation, or the strict extractive retry; avoid saving private article text in public CI artifacts.
3. Improve evidence selection and generation using the reviewed cases, without reducing evidence requirements merely to increase the answer rate.
4. Measure answer accuracy, valid abstention rate, citation support and latency on a held-out set. Compare with the original Phase 15 result, disclosing differences in prompts and models.

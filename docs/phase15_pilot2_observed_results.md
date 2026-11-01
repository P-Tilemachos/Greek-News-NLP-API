# Phase 15 — Observed real-model pilot (2 cases)

**Scope:** two CPU API requests, development/diagnostic, NOT a held-out faithfulness estimate.

| Case | Original status | Evidence / retrieval | Observed response | Assessment |
| --- | --- | --- | --- | --- |
| `p15_a_001` | Author-reviewed answerable | `macocu.el.16091235` retrieved, excerpt includes “Moss” | `answer: null`, `abstained: true`, `reason: model_abstained` | **False abstention** on an answerable question |
| `p15_u_001` | Author-screened negative candidate, **not gold** | Same article retrieved, excerpt gives no exact song duration | `answer: null`, `abstained: true`, `reason: model_abstained` | Reasonable *for this example*, but no proven corpus-wide negative label |

A separate guard-only test returned `VERBATIM_GUARD: Moss [1]`. This shows the guard can accept a directly cited string present in the source. It **does not** show what the model originally generated or that the retry logic was invoked as intended. The cause of the excessive abstention requires further targeted diagnosis; do not assert it is fixed.

The two previous HTTP responses were 200 OK. This indicates API success, not answer correctness. `faithfulness_verified:false` is the default Phase 14 status and is not proof of hallucination.

**Methodological rule:** preserve these two examples as development cases and run the remaining 78 without tuning on their answers or interleaving prompt/code changes. Keep positive-source hit and answer quality as separate measurements. Report preliminary negative-label metrics as provisional/author-assessed, not as independently verified ground truth.

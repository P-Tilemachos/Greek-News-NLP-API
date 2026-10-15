# Extractive QA Benchmark — Evaluation Summary

## Experiment

- API: `/ask`
- API Git revision: `f796f1570bb26efa34561937326a0e5c76e2640e`
- Retrieval setting: `k=10`
- Evaluation questions: **40**
- Unique source documents: **32**
- Question type: **Answerable only**
- Frozen dataset SHA256: `f60fb47301b3d5df2e2d60fff73588671e039717c83f22b234cce9cbd6ea772c`
- Predictions SHA256: `43cc5698bddf9807bd5a9bdb85006d14dcea39526a3500cc9deeb5ce51ccb778`

## Automated evaluation

| Metric | Result |
|---|---:|
| Source Hit@1 | 95.00% (38/40) |
| Source Hit@3 | 95.00% (38/40) |
| Source Hit@5 | 97.50% (39/40) |
| Source Hit@10 | 97.50% (39/40) |
| Evidence Document Hit | 97.50% (39/40) |
| Exact Evidence Quote Match | 65.00% (26/40) |
| Mean Evidence Token Recall | 0.7887 |
| Mean latency | 6.99 s |
| Median latency | 6.31 s |

Metrics were calculated from stored API responses.

## Flagged-case semantic review

15 cases were flagged for review.

| Judgment | Cases |
|---|---:|
| Correct | 4 |
| Partial | 3 |
| Incorrect | 8 |

These judgments were proposed through AI assistance and
approved by the project author. They are not an independent
blind human annotation study.

The remaining 25 questions were not individually
semantically adjudicated.

## Main findings

- The correct source was ranked first for 38/40 questions.
- The correct source was within the Top-10 for 39/40 questions.
- In `eval_008`, the gold document was not retrieved.
- In `eval_037`, the gold document was ranked fourth.
- Evidence selection sometimes omitted the facts required
  to support the reference answer, even when the correct
  article was retrieved.

## Interpretation and limitations

The existing API performs extractive evidence selection.
It does not generate standalone natural-language answers.

**Source retrieval accuracy must not be reported as
final answer accuracy.**

The benchmark contains only answerable questions.
It does not evaluate abstention or unanswerable queries.

The 40 questions come from 32 unique articles, with eight
articles contributing a second question.

The evaluation dataset should not be used for undocumented
model selection or parameter tuning.

GitHub publication of raw articles and evidence excerpts
requires separate data provenance and redistribution review.

## Reproducibility

- Frozen dataset and predictions were verified by SHA256.
- All five reviewed QA source batches were verified.
- Original saved predictions were reused for evaluation.
- No additional inference or model training was performed.
- The full file inventory is recorded in the audit JSON.

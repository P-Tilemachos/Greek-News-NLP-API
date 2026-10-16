"""Opt-in Phase 12 Cross-Encoder reranking of the Hybrid RRF Top-5.

Notebook 09 protocol: BAAI/bge-reranker-v2-m3 (pinned revision), CPU,
max_length=512, batch_size=2. Reranking is lazy and never alters /ask.
"""

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RERANK_TOP_K = 5
MAX_LENGTH = 512
BATCH_SIZE = 2


class HybridBackend(Protocol):
    docs_by_id: dict[str, dict]

    def search(self, query: str, k: int = 5) -> list[dict]: ...


class CrossEncoderBackend(Protocol):
    def predict(self, sentences: list[tuple[str, str]], **kwargs: object) -> object: ...


def rerank_top_five(
    query: str,
    hits: Sequence[dict],
    docs_by_id: dict[str, dict],
    model: CrossEncoderBackend,
    k: int,
) -> list[dict]:
    """Rerank first five *full-text* pairs; keep any remaining hits in place.

    With k < 5, still score the top five first, then truncate to k.
    Returns existing SearchHit-shaped dictionaries with the new rank score.
    """
    if k <= 0 or not hits:
        return []
    ids = [str(hit["id"]) for hit in hits]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document IDs in reranking candidates")
    top = list(hits[:RERANK_TOP_K])
    pairs = [(query, docs_by_id[str(hit["id"])]["text"]) for hit in top]
    raw = model.predict(
        pairs, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
    )
    scores = np.asarray(raw, dtype=float).reshape(-1)
    if scores.shape != (len(top),) or not np.isfinite(scores).all():
        raise ValueError("Cross-Encoder returned invalid scores")
    # Stable tie-break: preserve original Hybrid RRF rank, then document ID.
    ranked = sorted(
        (
            (float(score), rank, str(hit["id"]), hit)
            for rank, (score, hit) in enumerate(zip(scores, top, strict=True))
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    results = []
    for score, _, _, hit in ranked:
        result = dict(hit)
        # This score is an uncalibrated Cross-Encoder score, not an RRF or E5 score.
        result["score"] = score
        results.append(result)
    results.extend(dict(hit) for hit in hits[RERANK_TOP_K:])
    return results[:k]


class RerankingRetriever:
    """Load CrossEncoder only for explicit /search mode=rerank.

    Reuses the Phase 11 hybrid retriever, and never changes the FAISS index.
    """

    def __init__(self, hybrid: HybridBackend, model: CrossEncoderBackend | None = None):
        self.hybrid = hybrid
        self.docs_by_id = hybrid.docs_by_id
        if model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                MODEL_ID,
                revision=MODEL_REVISION,
                device=os.environ.get("RERANK_DEVICE", "cpu"),
                max_length=MAX_LENGTH,
                trust_remote_code=False,
            )
        else:
            self.model = model

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query.strip() or k <= 0:
            return []
        candidate_count = max(k, RERANK_TOP_K)
        hits = self.hybrid.search(query, candidate_count)
        return rerank_top_five(query, hits, self.docs_by_id, self.model, k)

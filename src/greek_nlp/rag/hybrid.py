"""Production Hybrid BM25 Full + Dense E5 retrieval (Notebook 08 protocol).

BM25 is built from the exact docs.json paired with the deployed FAISS index.
No benchmark or corpus files are modified. The dense backend is reused.
"""

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

import numpy as np

BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25
CANDIDATE_K = 50
RRF_K = 60


def tokenize_greek(text: str) -> list[str]:
    """Notebook 08 tokenization: casefold, strip accents, Unicode word tokens."""
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


class DenseBackend(Protocol):
    docs: list[dict]

    def search(self, query: str, k: int = 5) -> list[dict]: ...


def reciprocal_rank_fusion(
    bm25_results: Sequence[dict],
    dense_results: Sequence[dict],
    docs_by_id: dict[str, dict],
    k: int,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Fuse two ranked lists by doc ID with deterministic Notebook 08 tie breaks."""
    if k <= 0:
        return []
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    bm25_ranks = {str(hit["id"]): rank for rank, hit in enumerate(bm25_results, 1)}
    dense_ranks = {str(hit["id"]): rank for rank, hit in enumerate(dense_results, 1)}
    if len(bm25_ranks) != len(bm25_results) or len(dense_ranks) != len(dense_results):
        raise ValueError("duplicate document IDs in retrieval candidates")

    fused = []
    for doc_id in bm25_ranks.keys() | dense_ranks.keys():
        br = bm25_ranks.get(doc_id)
        dr = dense_ranks.get(doc_id)
        score = (1.0 / (rrf_k + br) if br is not None else 0.0) + (
            1.0 / (rrf_k + dr) if dr is not None else 0.0
        )
        doc = docs_by_id[doc_id]
        fused.append(
            {
                "id": doc_id,
                "score": score,
                "label": doc["label"],
                "snippet": doc["text"][:300],
                "bm25_rank": br,
                "dense_rank": dr,
            }
        )
    fused.sort(
        key=lambda hit: (
            -hit["score"],
            min(rank for rank in (hit["bm25_rank"], hit["dense_rank"]) if rank is not None),
            hit["id"],
        )
    )
    top_hits = fused[:k]
    for hit in top_hits:
        hit["score"] = round(hit["score"], 6)
    return top_hits


class HybridRetriever:
    """Construct full-text BM25 once, and share the already-loaded dense retriever."""

    def __init__(self, dense: DenseBackend):
        # rank-bm25 is an ML/search dependency, loaded only for hybrid mode.
        from rank_bm25 import BM25Okapi

        self.dense = dense
        self.docs = dense.docs
        self.docs_by_id = {str(doc["id"]): doc for doc in self.docs}
        if len(self.docs_by_id) != len(self.docs):
            raise ValueError("docs.json contains duplicate document IDs")
        if not self.docs:
            raise ValueError("docs.json is empty")
        tokens = [tokenize_greek(doc["text"]) for doc in self.docs]
        if any(not token_list for token_list in tokens):
            raise ValueError("docs.json contains articles without searchable BM25 tokens")
        self.bm25 = BM25Okapi(tokens, k1=BM25_K1, b=BM25_B, epsilon=BM25_EPSILON)

    def search_bm25(self, query: str, k: int = CANDIDATE_K) -> list[dict]:
        tokens = tokenize_greek(query)
        if not tokens or k <= 0:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")
        result = []
        for index in order:
            if scores[index] <= 0:
                continue
            doc = self.docs[int(index)]
            result.append({"id": str(doc["id"])})
            if len(result) >= k:
                break
        return result

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query.strip() or k <= 0:
            return []
        candidates = min(max(CANDIDATE_K, k), len(self.docs))
        bm25_hits = self.search_bm25(query, candidates)
        dense_hits = self.dense.search(query, candidates)
        return reciprocal_rank_fusion(bm25_hits, dense_hits, self.docs_by_id, k)

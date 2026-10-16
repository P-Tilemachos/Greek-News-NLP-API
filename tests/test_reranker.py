import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main
from greek_nlp.rag.reranker import RERANK_TOP_K, RerankingRetriever, rerank_top_five

DOCS = {
    str(i): {"id": str(i), "text": f"FULL ARTICLE {i} " * 80, "label": "news"} for i in range(8)
}
HITS = [
    {"id": str(i), "score": 0.04 - 0.002 * i, "label": "news", "snippet": f"SHORT {i}"}
    for i in range(8)
]


class FakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, sentences, **kwargs):
        self.calls.append((sentences, kwargs))
        return self.scores


class FakeHybrid:
    docs_by_id = DOCS

    def __init__(self):
        self.calls = []

    def search(self, query, k=5):
        self.calls.append((query, k))
        return HITS[:k]


def test_full_articles_order_and_preserved_tail():
    model = FakeModel([0.4, 0.2, 0.9, -0.2, 0.5])
    ranked = rerank_top_five("ελληνική είδηση", HITS, DOCS, model, 8)
    assert [h["id"] for h in ranked] == ["2", "4", "0", "1", "3", "5", "6", "7"]
    assert ranked[0]["score"] == pytest.approx(0.9)
    assert ranked[5]["score"] == HITS[5]["score"]
    assert ranked[0]["snippet"] == "SHORT 2"
    pairs, params = model.calls[0]
    assert len(pairs) == RERANK_TOP_K
    assert pairs[0] == ("ελληνική είδηση", DOCS["0"]["text"])
    assert params == {"batch_size": 2, "show_progress_bar": False, "convert_to_numpy": True}
    assert HITS[2]["score"] != 0.9  # input is not modified


def test_k_less_than_five_scores_five_then_truncates():
    hybrid = FakeHybrid()
    model = FakeModel([0.1, 0.0, 0.6, 0.2, 0.3])
    ranked = RerankingRetriever(hybrid, model).search("αποζημιώσεις", 2)
    assert hybrid.calls == [("αποζημιώσεις", 5)]
    assert [h["id"] for h in ranked] == ["2", "4"]


def test_tie_break_is_original_hybrid_order():
    model = FakeModel([1, 1, 1, 1, 1])
    assert [h["id"] for h in rerank_top_five("query", HITS, DOCS, model, 5)] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]


@pytest.mark.parametrize("scores", [[1, 2], [0, 1, np.nan, 0, 0], [0, 1, np.inf, 0, 0]])
def test_bad_scores_rejected(scores):
    with pytest.raises(ValueError, match="invalid scores"):
        rerank_top_five("query", HITS, DOCS, FakeModel(scores), 5)


def test_duplicate_candidate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        rerank_top_five("query", [HITS[0], HITS[0]], DOCS, FakeModel([0, 0]), 2)


def test_empty_queries_do_not_call_hybrid_or_model():
    hybrid = FakeHybrid()
    model = FakeModel([0.1] * 5)
    assert RerankingRetriever(hybrid, model).search("   ", 5) == []
    assert hybrid.calls == []
    assert model.calls == []


def test_loader_uses_pinned_revision_only_if_explicitly_initialized(monkeypatch):
    from greek_nlp.rag import reranker

    monkeypatch.delenv("RERANK_DEVICE", raising=False)
    settings = {}

    class FakeCrossEncoder:
        def __init__(self, model_id, **kwargs):
            settings["model_id"] = model_id
            settings.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    engine = RerankingRetriever(FakeHybrid())
    assert engine.model is not None
    assert settings == {
        "model_id": reranker.MODEL_ID,
        "revision": reranker.MODEL_REVISION,
        "device": "cpu",
        "max_length": 512,
        "trust_remote_code": False,
    }


def test_api_modes_and_validation(monkeypatch):
    class FakeRerank:
        def search(self, query, k=5):
            return [{"id": "rerank", "label": "news", "score": -0.22, "snippet": "reranked"}]

    class FakeHybridApi:
        def search(self, query, k=5):
            return [{"id": "hybrid", "label": "news", "score": 0.03, "snippet": "hybrid"}]

    class FakeDenseApi:
        def search(self, query, k=5):
            return [{"id": "dense", "label": "news", "score": 0.91, "snippet": "dense"}]

    monkeypatch.setattr(main, "get_reranking_retriever", lambda: FakeRerank())
    monkeypatch.setattr(main, "get_hybrid_retriever", lambda: FakeHybridApi())
    original = main.app.dependency_overrides.get(main.get_retriever)
    main.app.dependency_overrides[main.get_retriever] = FakeDenseApi
    try:
        client = TestClient(main.app)
        for mode, expected in [(None, "dense"), ("hybrid", "hybrid"), ("rerank", "rerank")]:
            payload = {"query": "αγροτικές αποζημιώσεις"}
            if mode:
                payload["mode"] = mode
            res = client.post("/search", json=payload)
            assert res.status_code == 200
            assert res.json()["mode"] == expected
            assert res.json()["results"][0]["id"] == expected
        bad = client.post("/search", json={"query": "αγρότες", "mode": "invalid"})
        over_limit = client.post("/search", json={"query": "αγρότες", "k": 21, "mode": "rerank"})
        assert bad.status_code == 422
        assert over_limit.status_code == 422
    finally:
        if original is None:
            main.app.dependency_overrides.pop(main.get_retriever, None)
        else:
            main.app.dependency_overrides[main.get_retriever] = original

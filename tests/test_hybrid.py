import pytest
from fastapi.testclient import TestClient

from greek_nlp.api import main
from greek_nlp.rag.hybrid import HybridRetriever, reciprocal_rank_fusion, tokenize_greek

DOCS = [
    {"id": "a", "label": "news", "text": "Η κυβέρνηση ανακοίνωσε μέτρα για αγρότες."},
    {"id": "b", "label": "news", "text": "Οι αγρότες ζητούν νέες αποζημιώσεις."},
    {"id": "c", "label": "sport", "text": "Η ομάδα κέρδισε τον αγώνα ποδοσφαίρου."},
]


class FakeDense:
    docs = DOCS

    def search(self, query: str, k: int = 5) -> list[dict]:
        return [{"id": doc_id} for doc_id in ("c", "b", "a")[:k]]


def test_tokenize_greek_is_accent_insensitive():
    assert tokenize_greek("ΑΓΡΌΤΕΣ, και λήξεις_λέξεις!") == [
        "αγροτεσ",
        "και",
        "ληξεισ",
        "λεξεισ",
    ]


def test_rrf_rank_order_and_id_tie_break():
    fused = reciprocal_rank_fusion(
        [{"id": "a"}, {"id": "b"}],
        [{"id": "b"}, {"id": "c"}],
        {d["id"]: d for d in DOCS},
        k=3,
    )
    assert [hit["id"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["bm25_rank"] == 2
    assert fused[0]["dense_rank"] == 1
    assert fused[0]["score"] == pytest.approx(round(1 / 62 + 1 / 61, 6))


def test_rrf_rejects_duplicate_candidate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([{"id": "a"}, {"id": "a"}], [], {d["id"]: d for d in DOCS}, 5)


def test_hybrid_bm25_full_and_dense_fused():
    engine = HybridRetriever(FakeDense())
    assert [hit["id"] for hit in engine.search_bm25("αγροτες", k=3)] == ["b", "a"]
    hits = engine.search("αγροτες", k=3)
    assert [hit["id"] for hit in hits] == ["b", "a", "c"]
    assert len({hit["id"] for hit in hits}) == 3
    assert hits[0]["bm25_rank"] == 1
    assert hits[0]["dense_rank"] == 2
    assert all(len(hit["snippet"]) <= 300 for hit in hits)


def test_hybrid_handles_out_of_vocabulary_and_empty_query():
    engine = HybridRetriever(FakeDense())
    assert engine.search("  ", k=3) == []
    hits = engine.search("xyzneverexists", k=3)
    assert [hit["id"] for hit in hits] == ["c", "b", "a"]
    assert all(hit["bm25_rank"] is None for hit in hits)


def test_hybrid_detects_duplicate_corpus_ids():
    class DuplicateDense(FakeDense):
        docs = [DOCS[0], DOCS[0]]

    with pytest.raises(ValueError, match="duplicate"):
        HybridRetriever(DuplicateDense())


def test_search_api_hybrid_dispatch_preserves_default_dense(monkeypatch):
    class FakeHybrid:
        def search(self, query: str, k: int = 5) -> list[dict]:
            return [{"id": "b", "label": "news", "score": 0.032, "snippet": "BM25 + E5"}]

    class FakeApiDense:
        def search(self, query: str, k: int = 5) -> list[dict]:
            return [{"id": "a", "label": "news", "score": 0.98, "snippet": "Dense E5"}]

    monkeypatch.setattr(main, "get_hybrid_retriever", lambda: FakeHybrid())
    original = main.app.dependency_overrides.get(main.get_retriever)
    main.app.dependency_overrides[main.get_retriever] = FakeApiDense
    try:
        client = TestClient(main.app)
        dense = client.post("/search", json={"query": "ποδόσφαιρο"})
        hybrid = client.post("/search", json={"query": "ποδόσφαιρο", "mode": "hybrid"})
        bad = client.post("/search", json={"query": "ποδόσφαιρο", "mode": "other"})
        assert dense.status_code == 200
        assert dense.json()["mode"] == "dense"
        assert dense.json()["results"][0]["id"] == "a"
        assert hybrid.status_code == 200
        assert hybrid.json()["mode"] == "hybrid"
        assert hybrid.json()["results"][0]["id"] == "b"
        assert bad.status_code == 422
    finally:
        if original is None:
            main.app.dependency_overrides.pop(main.get_retriever, None)
        else:
            main.app.dependency_overrides[main.get_retriever] = original

from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI

from greek_nlp.api.schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    SearchRequest,
    SearchResponse,
)

app = FastAPI(title="Greek News NLP API", version="0.1.0")


class Classifier(Protocol):
    def predict(self, text: str, top_k: int = 3) -> list[dict]: ...


class SearchEngine(Protocol):
    def search(self, query: str, k: int = 5) -> list[dict]: ...


class QAEngine(Protocol):
    def ask(self, question: str, k: int = 3) -> dict: ...


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    # Lazy import: το torch φορτώνεται μόνο όταν χρειαστεί
    from greek_nlp.models.classifier import NewsClassifier

    return NewsClassifier()


@lru_cache(maxsize=1)
def get_retriever() -> SearchEngine:
    from greek_nlp.rag.retriever import Retriever

    return Retriever.load()


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> SearchEngine:
    # Initialize BM25 lazily on the first opt-in hybrid /search request.
    # Reuse the existing E5/FAISS retriever; do not change extractive /ask.
    from greek_nlp.rag.hybrid import HybridRetriever

    return HybridRetriever(get_retriever())


@lru_cache(maxsize=1)
def get_qa() -> QAEngine:
    from greek_nlp.rag.qa import ExtractiveQA

    return ExtractiveQA(get_retriever())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(
    request: ClassifyRequest,
    clf: Annotated[Classifier, Depends(get_classifier)],
) -> ClassifyResponse:
    return ClassifyResponse(predictions=clf.predict(request.text, request.top_k))


@app.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    engine: Annotated[SearchEngine, Depends(get_retriever)],
) -> SearchResponse:
    if request.mode == "hybrid":
        results = get_hybrid_retriever().search(request.query, request.k)
    else:
        results = engine.search(request.query, request.k)
    return SearchResponse(results=results, mode=request.mode)


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    qa: Annotated[QAEngine, Depends(get_qa)],
) -> AskResponse:
    return AskResponse(**qa.ask(request.question, request.k))

from typing import Literal

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20000, description="Ελληνικό κείμενο είδησης")
    top_k: int = Field(3, ge=1, le=17, description="Πόσες κατηγορίες να επιστραφούν")


class Prediction(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    predictions: list[Prediction]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Ερώτηση ή θέμα")
    k: int = Field(5, ge=1, le=20, description="Πόσα άρθρα να επιστραφούν")
    mode: Literal["dense", "hybrid", "rerank"] = Field(
        "dense",
        description="Search mode: dense (default), hybrid (BM25 + E5 + RRF), or rerank.",
    )


class SearchHit(BaseModel):
    id: str
    score: float
    label: str
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchHit]
    mode: Literal["dense", "hybrid", "rerank"] = "dense"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="Ερώτηση στα ελληνικά")
    k: int = Field(3, ge=1, le=10, description="Πόσα άρθρα να χρησιμοποιηθούν")


class Evidence(BaseModel):
    doc_id: str
    sentence: str
    score: float


class AskResponse(BaseModel):
    mode: str
    evidence: list[Evidence]
    sources: list[SearchHit]


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=3, max_length=1500, description="Greek-language generation prompt"
    )
    max_new_tokens: int = Field(
        64, ge=1, le=128, description="Strict output token budget (greedy decoding)"
    )


class GenerateResponse(BaseModel):
    mode: Literal["local_llm"]
    model: str
    device: Literal["cpu", "cuda"]
    generated_text: str
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=0)


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=240, description="Greek factual question")
    k: int = Field(3, ge=1, le=5, description="Top reranked articles to inspect")
    max_new_tokens: int = Field(96, ge=1, le=128)


class RAGSource(BaseModel):
    reference: str = Field(description="Citation marker, e.g. [1]")
    doc_id: str
    label: str
    excerpt: str
    retrieval_score: float = Field(description="Raw, uncalibrated reranker score")


class RAGResponse(BaseModel):
    mode: Literal["grounded_rag"]
    answer: str | None
    abstained: bool
    reason: str | None
    model: str | None
    sources: list[RAGSource]
    faithfulness_verified: Literal[False] = False

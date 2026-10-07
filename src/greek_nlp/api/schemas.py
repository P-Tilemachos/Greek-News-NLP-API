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


class SearchHit(BaseModel):
    id: str
    score: float
    label: str
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchHit]

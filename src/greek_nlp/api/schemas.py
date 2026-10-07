from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20000, description="Ελληνικό κείμενο είδησης")
    top_k: int = Field(3, ge=1, le=17, description="Πόσες κατηγορίες να επιστραφούν")


class Prediction(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    predictions: list[Prediction]

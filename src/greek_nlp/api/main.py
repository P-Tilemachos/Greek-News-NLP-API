# In order to use it try the following domain
# http://127.0.0.1:8000/docs

from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI

from greek_nlp.api.schemas import ClassifyRequest, ClassifyResponse

app = FastAPI(title="Greek News NLP API", version="0.1.0")


class Classifier(Protocol):
    def predict(self, text: str, top_k: int = 3) -> list[dict]: ...


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    # Lazy import: το torch φορτώνεται μόνο όταν χρειαστεί
    from greek_nlp.models.classifier import NewsClassifier

    return NewsClassifier()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(
    request: ClassifyRequest,
    clf: Annotated[Classifier, Depends(get_classifier)],
) -> ClassifyResponse:
    return ClassifyResponse(predictions=clf.predict(request.text, request.top_k))

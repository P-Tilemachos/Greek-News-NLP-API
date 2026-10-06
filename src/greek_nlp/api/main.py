# In order to use it try the following domain
# http://127.0.0.1:8000/docs

from fastapi import FastAPI

app = FastAPI(title="Greek News NLP API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "models_store" / "rag_index"
EMBED_MODEL = "intfloat/multilingual-e5-small"
MAX_SEQ_LENGTH = 256


def load_corpus() -> list[dict]:
    """Διαβάζει όλα τα ελληνικά άρθρα (train + val + test) ως βάση γνώσης."""
    docs = []
    for split in ("train", "val", "test"):
        with open(DATA_DIR / f"el_{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(
                    {
                        "id": row["document_id"],
                        "text": row["text_clean"],
                        "label": row["GPT-IPTC-label"],
                    }
                )
    return docs


def load_embedder() -> SentenceTransformer:
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


class Retriever:
    def __init__(self, index: faiss.Index, docs: list[dict], model: SentenceTransformer):
        self.index = index
        self.docs = docs
        self.model = model

    @classmethod
    def build(cls, index_dir: Path = INDEX_DIR) -> "Retriever":
        docs = load_corpus()
        model = load_embedder()
        embeddings = model.encode(
            ["passage: " + d["text"] for d in docs],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        index = faiss.IndexFlatIP(embeddings.shape[1])  # εσωτερικό γινόμενο = cosine
        index.add(embeddings)

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "articles.faiss"))
        (index_dir / "docs.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        return cls(index, docs, model)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "Retriever":
        index = faiss.read_index(str(index_dir / "articles.faiss"))
        docs = json.loads((index_dir / "docs.json").read_text(encoding="utf-8"))
        return cls(index, docs, load_embedder())

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.model.encode(["query: " + query], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(q, k)
        return [
            {
                "id": self.docs[i]["id"],
                "score": round(float(s), 4),
                "label": self.docs[i]["label"],
                "snippet": self.docs[i]["text"][:300],
            }
            for s, i in zip(scores[0], ids[0])
        ]


if __name__ == "__main__":
    Retriever.build()
    print("Index built:", INDEX_DIR)

import re

import numpy as np

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
MIN_WORDS = 6


class ExtractiveQA:
    """Επιστρέφει τις πιο σχετικές προτάσεις των άρθρων, με παραπομπές."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.by_id = {d["id"]: d for d in retriever.docs}

    def ask(self, question: str, k: int = 3, n_sentences: int = 3) -> dict:
        hits = self.retriever.search(question, k)

        candidates = []
        for hit in hits:
            text = self.by_id[hit["id"]]["text"]
            for sentence in SENT_SPLIT.split(text):
                if len(sentence.split()) >= MIN_WORDS:
                    candidates.append((hit["id"], sentence.strip()))

        evidence = []
        if candidates:
            model = self.retriever.model
            q = model.encode(["query: " + question], normalize_embeddings=True)[0]
            s = model.encode(
                ["passage: " + sentence for _, sentence in candidates],
                normalize_embeddings=True,
                batch_size=32,
            )
            scores = s @ q
            for i in np.argsort(-scores)[:n_sentences]:
                doc_id, sentence = candidates[i]
                evidence.append(
                    {
                        "doc_id": doc_id,
                        "sentence": sentence,
                        "score": round(float(scores[i]), 4),
                    }
                )

        return {"mode": "extractive", "evidence": evidence, "sources": hits}

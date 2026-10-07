import html
import os
import re
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models_store" / "greek-bert-news-classifier"))
MAX_LENGTH = 256


def preprocess(text: str) -> str:
    """Ίδιος καθαρισμός με αυτόν της εκπαίδευσης (HTML entities, κενά, τόνοι, πεζά)."""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


class NewsClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text: str, top_k: int = 3) -> list[dict]:
        enc = self.tokenizer(
            preprocess(text), truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top = torch.topk(probs, k=min(top_k, len(probs)))
        return [
            {"label": self.id2label[i.item()], "score": round(p.item(), 4)}
            for p, i in zip(top.values, top.indices)
        ]

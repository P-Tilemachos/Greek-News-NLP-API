# Greek News NLP API

![CI](https://github.com/P-Tilemachos/Greek-News-NLP-API/actions/workflows/ci.yml/badge.svg)

Greek news classification served through a REST API: a fine-tuned **GreekBERT** classifier behind **FastAPI**, evaluated against a TF-IDF baseline. Retrieval-augmented question answering over the same articles is the next milestone.

## Results

Evaluated on a held-out test set of 788 Greek news articles (17 IPTC Media Topic categories), never used for training or model selection.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.753 | 0.667 |
| **GreekBERT (fine-tuned)** | **0.825** | **0.770** |

Macro-F1 is the main metric because the classes are imbalanced (sport has about 900 articles, weather about 30).

![Baseline confusion matrix](docs/baseline_confusion_matrix.png)

*Confusion matrix of the TF-IDF baseline on the validation set.*

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `POST /classify` | Returns the top-k predicted categories with scores for a Greek text |

Example request:

```json
{
  "text": "Η κυβέρνηση ανακοίνωσε νέα μέτρα στήριξης για τους αγρότες λόγω των πλημμυρών",
  "top_k": 3
}
```

Interactive documentation (Swagger) is available at `/docs` when the server is running.

## Quickstart

```bash
git clone https://github.com/P-Tilemachos/Greek-News-NLP-API.git
cd Greek-News-NLP-API
pip install -e ".[dev,ml]"
uvicorn greek_nlp.api.main:app --reload
```

The fine-tuned model weights are not stored in this repository. Place them in `models_store/greek-bert-news-classifier/`, or set the `MODEL_DIR` environment variable to their location.

## Project structure

```
src/greek_nlp/
  api/       FastAPI app and request/response schemas
  models/    GreekBERT inference
notebooks/   01 EDA, 02 TF-IDF baseline, 03 GreekBERT fine-tuning (Google Colab)
tests/       pytest suite (the classifier is mocked, so CI needs no model)
```

## Data

[EMMediaTopic 1.0](http://hdl.handle.net/11356/1991) (Kuzman and Ljubešić, Jožef Stefan Institute), Greek subset: 5,250 news articles, licensed under CC BY-SA 4.0. The data is not redistributed here. Re-split 70/15/15 (stratified, seed 42) into train, validation and test.

## Limitations

- The labels were assigned automatically by GPT-4o, not by human annotators. The dataset authors report a macro-F1 of 0.731 for these labels against a manually annotated sample, which caps what any model can reach.
- Rare categories (weather, lifestyle and leisure, religion) have very few test examples, so their per-class scores are noisy.
- The classifier struggles most with broad, overlapping categories such as *society* and *human interest*.
- Texts were truncated to 256 tokens during training and inference.
- A class-weighted loss improves recall on small classes at the cost of precision.

## Roadmap

- [x] Data exploration, cleaning and stratified split
- [x] TF-IDF baseline
- [x] GreekBERT fine-tuning and test-set evaluation
- [x] `/classify` endpoint with tests and CI
- [ ] RAG pipeline: embeddings, retrieval, reranking, `/ask` endpoint
- [ ] Docker image and cloud deployment with a live demo
- [ ] Model card
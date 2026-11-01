# Phase 16 — Πειραματική ενσωμάτωση llama.cpp στο `/generate`

## Τι είναι αυτό το πακέτο

Περιλαμβάνει **δύο νέα/αντικαταστάσιμα αρχεία κώδικα**, νέα offline tests και
τεκμηρίωση. Το `schemas.py` **δεν αλλάζει**: η υπάρχουσα απόκριση `GenerateResponse`
είναι συμβατή με τη νέα επιλογή.

## Αντιγραφή στο project

Κράτησε αντίγραφο του υπάρχοντος `src/greek_nlp/api/main.py`.
Αποσυμπίεσε το ZIP στον κεντρικό φάκελο του `Greek-News-NLP-API`, διατηρώντας
τους υποφακέλους, και επίτρεψε αντικατάσταση μόνο του `src/greek_nlp/api/main.py`.

Τα αρχεία είναι:

- `src/greek_nlp/api/main.py` (αντικατάσταση)
- `src/greek_nlp/llm/llamacpp_backend.py` (νέο)
- `tests/test_phase16_llamacpp_backend.py` (νέο)
- `docs/phase_16_optional_llamacpp_integration.md` (νέο)

**Μη διαγράψεις/αντικαταστήσεις** `src/greek_nlp/llm/generator.py`,
`src/greek_nlp/api/schemas.py` ή `src/greek_nlp/rag/grounded.py`.

## Βήμα 1 — Offline tests

```bat
python -m ruff format src/greek_nlp/api/main.py src/greek_nlp/llm/llamacpp_backend.py tests/test_phase16_llamacpp_backend.py
python -m ruff check .
python -m pytest -q
```

## Βήμα 2 — Krikri και FastAPI (δύο παράθυρα)

1. Πρώτο Anaconda Prompt: άφησε ανοιχτό τον Krikri GGUF server στη θύρα **9931**.
   (Αν χρειάζεται, εκτέλεσε την εντολή `llama serve` που υπάρχει στην τεκμηρίωση.)
2. Δεύτερο Anaconda Prompt, στο root του project:

```bat
set "LLM_BACKEND=llamacpp"
set "LLAMACPP_BASE_URL=http://127.0.0.1:9931"
set "LLAMACPP_MODEL_MATCH=Krikri"
python -m uvicorn greek_nlp.api.main:app
```

3. Τρίτο Anaconda Prompt, στο root του project:

```bat
python -c "import requests; r=requests.post('http://127.0.0.1:8000/generate',json={'prompt':'Τι είναι η μηχανική μάθηση;','max_new_tokens':64},timeout=360); print(r.status_code); print(r.json())"
```

Αν ο server δεν λειτουργεί ή το μοντέλο δεν είναι Krikri, η εφαρμογή επιστρέφει
**HTTP 503** αντί να κάνει fallback αθόρυβα στο PyTorch.

## Σημαντικό

- Το `/rag/ask` **δεν αλλάζει** και παραμένει στο PyTorch backend.
- Το `/generate` μένει PyTorch από προεπιλογή αν δεν ορίσεις `LLM_BACKEND`.
- Η δοκιμή των 8 ερωτήσεων **δεν είναι ανεξάρτητο gold benchmark**.
- **Μην κάνεις ακόμα `git add .` ή commit**. Πρώτα έλεγξε Ruff, tests και API smoke test.

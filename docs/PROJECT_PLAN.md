# GreekNews-LLM-Bench: Project Plan

## Context
This repo (Greek-News-NLP-API) already contains:
- FastAPI REST API: /health, /classify (fine-tuned GreekBERT), /search (multilingual-e5-small + FAISS over 5,250 Greek news articles), /ask (extractive, cited evidence sentences)
- Dataset: EMMediaTopic, Greek subset (lang == "el"), 17 IPTC topic labels, split 70/15/15 (3675/787/788)
- Test results: TF-IDF + LogReg accuracy 0.753 / macro-F1 0.667; GreekBERT accuracy 0.825 / macro-F1 0.77
- Retrieval evaluation: lead queries Recall@1 0.997; late queries Recall@10 0.53, MRR@10 0.386
- Docker and GitHub Actions CI

## Constraints
- Local GPU is a GeForce MX150 (2 GB), so LLM work runs on Google Colab (T4) with data in Google Drive
- No LLM API key and no budget: open models only
- Existing tests, ruff and CI must keep passing

## Goal
Show the full lifecycle of an LLM system with measurements: evaluation, fine-tuning, RAG, serving and safety. The value is in comparison and honest analysis, not in a chatbot demo.

## Phase 1: Classification benchmark (same test set for every model)
1.1 Zero-shot open LLM (Qwen2.5-1.5B-Instruct, then 3B in 4-bit)
1.2 Few-shot prompting and constrained output parsing
1.3 QLoRA fine-tuning of a small LLM on the train split
1.4 Comparison vs GreekBERT and TF-IDF: macro-F1, accuracy, latency per article, peak VRAM, cost per 1000 requests, per-class error analysis
Deliverable: results/benchmark_summary.csv, notebooks, README section

## Phase 2: RAG improvement with measurement
2.1 Hybrid retrieval (BM25 + dense) and a reranker, re-run retrieval eval (Recall@k, MRR)
2.2 Generative /ask with a quantized open LLM, answering only from retrieved articles with citations
2.3 Evaluation set of 40-60 hand-written questions: faithfulness, citation accuracy, hallucination rate, extractive vs generative
Deliverable: eval notebook, results tables, updated /ask endpoint with tests

## Phase 3: Serving and MLOps
3.1 Quantized inference and a streaming endpoint in FastAPI
3.2 Updated
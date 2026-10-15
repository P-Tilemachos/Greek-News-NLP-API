# Controlled Inference Benchmark — Final Four-Model Report

## Scope

Four measured model configurations on 64 fixed Greek validation articles, 3 warm-ups, batch size 1, Tesla T4 GPU. All results use the saved frozen protocol and same article IDs. Each model has one controlled benchmark run; training and checkpoint loading are excluded.

## Measured performance

| Model | Mean forward (ms) | Mean end-to-end (ms) | P95 E2E (ms) | Peak allocated VRAM (MiB) |
|---|---:|---:|---:|---:|
| GreekBERT | 33.393 | 42.679 | 59.021 | 448.70 |
| Qwen2.5-1.5B Zero-Shot | 273.456 | 277.076 | 352.203 | 3189.26 |
| Qwen2.5-3B Zero-Shot | 503.219 | 510.485 | 649.135 | 6307.31 |
| Qwen2.5-1.5B QLoRA NF4 | 299.936 | 303.439 | 422.657 | 1369.13 |

## Model inference protocols

- GreekBERT: FP32; max 256 tokens; supervised classification.
- Qwen2.5-1.5B and Qwen2.5-3B zero-shot: FP16; max 1200 article characters; first-token 17-label scoring.
- Qwen2.5-1.5B QLoRA epoch 2: 4-bit NF4 storage, FP16 compute; max 800 article characters; saved LoRA adapter; first-token 17-label scoring with an explicitly documented fixed prompt.

## Interpretation / limitations

- These are **configuration comparisons**, not matched-input model-size or tuning ablations.
- The QLoRA benchmark uses a fixed first-token 17-label inference prompt; its exact equivalence to the original training/evaluation prompt has not been established.
- Only one GPU session per model and 64 articles: do not generalize timings to other hardware or workloads.
- Results are **not production HTTP API latency** and do not measure autoregressive answer generation.
- Models have different input limits and precision. Peak PyTorch allocated memory excludes some runtime memory.
- GreekBERT checkpoint identity relative to the historical API remains unresolved.
- This benchmark measures inference speed/VRAM; it does not measure model accuracy or faithfulness.

## Reproducibility / artifacts

- Frozen validation SHA256: `d124de2fa8273aed325b80cb30f78adf677c876cee32797a4ca112929dfe22fb`
- QLoRA adapter weights SHA256: `b9591ad53c483209a387cecba8ea793516601660032e4c800b0de9ebfc6e56ab`
- Existing per-model CSV files and manifests remain unchanged.
- Four per-model manifest/timing pairs have been checksum-verified.
- `benchmark_comparison_partial.csv` and `benchmark_report_partial.md` remain unchanged.
- New outputs: `qwen_15b_qlora_epoch2_timings.csv`, `qwen_15b_qlora_epoch2_manifest.json`, `benchmark_comparison_final.csv`, `benchmark_report_final.md`.

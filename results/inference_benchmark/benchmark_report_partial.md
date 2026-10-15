# Controlled Inference Benchmark — Partial Report

## Status

**Partial benchmark:** 3 of 4 models completed.

The Qwen2.5-1.5B QLoRA Epoch 2 benchmark is pending
GPU availability. No results have been estimated for it.

## Experimental Setup

- GPU: NVIDIA Tesla T4
- Measured validation articles: 64
- Warm-up articles: 3
- Batch size: 1
- Random seed: 42
- Model loading excluded from latency measurements
- Explicit CUDA synchronization used
- Fixed article identifiers across all models
- Single measured run per model

Validation dataset SHA256:

`d124de2fa8273aed325b80cb30f78adf677c876cee32797a4ca112929dfe22fb`

## Model Configurations

| Model | Precision | Input budget | Inference |
|---|---|---|---|
| GreekBERT | FP32 | 256 tokens | 17-class classification |
| Qwen2.5-1.5B | FP16 | 1200 article characters | Zero-shot first-token scoring |
| Qwen2.5-3B | FP16 | 1200 article characters | Zero-shot first-token scoring |

For Qwen, the prompt includes all 17 class names. The
model performs a forward pass and selects the label
with the highest logit at the first token of its name.
No autoregressive text generation is performed.

## Latency and Memory Results

All latency values are in milliseconds (ms).
Peak memory is PyTorch allocated GPU memory in MiB.

| Model | Mean Forward (ms) | Mean End-to-End (ms) | P95 End-to-End (ms) | Peak Allocated VRAM (MiB) | Mean Input Tokens |
|---|---:|---:|---:|---:|---:|
| GreekBERT | 33.393 | 42.679 | 59.021 | 448.70 | 211.12 |
| Qwen2.5-1.5B Zero-Shot | 273.456 | 277.076 | 352.203 | 3189.26 | 989.56 |
| Qwen2.5-3B Zero-Shot | 503.219 | 510.485 | 649.135 | 6307.31 | 989.56 |

End-to-end measurements include preprocessing or
prompt construction, tokenization, input transfer,
model execution, synchronization and prediction
extraction. They are not production API latency.

## Paired Qwen Comparison

The Qwen2.5-1.5B and Qwen2.5-3B benchmarks use
identical article IDs, article order, article
character counts and prompt token counts.

- Mean end-to-end latency ratio (3B / 1.5B): 1.842x
- Mean paired latency difference: 233.409 ms
- Median paired latency difference: 250.202 ms
- Articles where 3B was slower: 64/64

Both zero-shot models were evaluated using FP16,
a 1200-character article limit and the same
first-token classification methodology.

These values describe the measured sample and
execution environment, not a universal performance
ratio or causal estimate.

## Reproducibility

Qwen2.5-1.5B model revision:

`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

Qwen2.5-3B model revision:

`aa8e72537993ba99e69dfaafa59ed015b17504d1`

GreekBERT checkpoint file hashes are stored in
`greekbert_manifest.json`.

The benchmark directory contains:

- `protocol.json`
- Per-model timing CSV files
- Per-model benchmark manifests
- `benchmark_comparison_partial.csv`

Individual CSV files are verified against the SHA256
checksums in their manifests.

## Limitations

1. The benchmark uses one Tesla T4 session and only
   64 selected validation articles per model.
2. There are no repeated timing sessions or
   confidence intervals for latency.
3. GreekBERT and Qwen use different input budgets,
   architectures and precision settings.
4. The Qwen results measure first-token classification,
   not full language-model response generation.
5. End-to-end latency includes explicit CUDA
   synchronization and excludes an actual HTTP API.
6. Peak allocated VRAM measures PyTorch allocations,
   not all GPU memory usage by the runtime.
7. The relationship between the saved GreekBERT
   checkpoint and the historical API checkpoint
   remains unresolved.
8. The QLoRA benchmark is incomplete; the current
   comparison must not be presented as final.

## Remaining Work

- Restore GPU access.
- Verify bitsandbytes NF4 compatibility.
- Load the saved QLoRA Epoch 2 adapter.
- Benchmark the QLoRA model using the frozen protocol.
- Update the comparison and prepare the final report.

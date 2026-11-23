# Qwen3.8-27B Inference Engine Comparison Report (Single AMD Radeon AI PRO R9700)

[日本語](engine-compare-qwen38-27b-r9700.md) | **English**

Measurement date: 2026-11-23 / Environment: Single AMD Radeon AI PRO R9700 (1 GPU, TP=1, ROCm 10.2, concurrency = 1)

This report compares four inference engines (`shelf`, `saber-engine`, `hipfire`, and `llama.cpp`) running `Qwen3.8-27B` on a single AMD Radeon AI PRO R9700 across **generation throughput, output quality, model footprint and memory efficiency, streaming stability, and reproducibility**.

---

## Executive Summary

- **Balance of throughput, quality, and footprint**: `shelf 3.05bpw` (with DFlash2) achieves **`104.66 tok/s`** on average across real-world tasks (and ranks 1st on `chat` at **`87.33 tok/s`**), matching the fastest engines (`saber` at `106.56 tok/s`, `hipfire` at `103.88 tok/s`) while using an effective text-inference footprint of **`3.82 bpw` (11.96 GiB)**—about 28% smaller than `saber W4A8` (`5.27 bpw` / `16.52 GiB`). For quality-first workloads, `shelf 4bpw` records the closest match to BF16 across all tested configurations (**`KL 0.0212` / `top-1 94.76%`**) while delivering **`93.98 tok/s`** on real-world tasks.
- **KV cache memory reduction via compression**: `shelf` supports `octave:c5v5`, an externally proposed KV cache compression scheme (not supported by the other three engines tested), reducing persistent KV memory for a 32k context from **`2.00 GiB` to `~0.63 GiB`** (~68% reduction). Quality impact is small (`KL 0.0212 → 0.0239` on `4bpw`), and real-world task throughput with DFlash2 remains `96.65–98.53 tok/s`.
- **Consistent streaming responsiveness**: Following the streaming stall fix, a 200-request continuous load test (stdio / HTTP × BF16 / `octave:c5v5`, 50 requests each, 37,783 update intervals total) recorded **0 stalls ≥ 300 ms (max `115.9 ms`)**. The 114 additional benchmark trials across three configurations (`4bpw BF16`, `4bpw octave:c5v5`, `3.05bpw octave:c5v5`) likewise recorded **0 stalls ≥ 300 ms (max `117.57 ms`)**.
- **In-repository reproducibility from quantization to evaluation**: `shelf` bundles its quantizer, format validator, benchmark suite, and BF16-referenced quality evaluation tools. Users do not need to rely on pre-quantized distributions—custom fine-tuned or LoRA-merged BF16 models can be quantized and verified using the same workflow.

---

## Engine Comparison Overview

Comparison of each engine's primary speculative-decoding configuration (`DFlash2` / `DFlash`).

| Engine / Configuration | Role | Effective Size<br>(bpw ↓ / GiB ↓) | Quality<br>(KL ↓ / top-1 ↑) | Real-Task Speed<br>(`real-v1` avg) | Dev / Chat Speed<br>(`code` / `chat`) | Long Prefill<br>(`pp2048`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **`shelf 4bpw`**<br>(EXL3-INT8 / BF16 KV + DFlash2) | Highest quality | 4.68 / 14.66 | **0.0212 / 94.76%** | 93.98 tok/s | 124.38 / 75.26 tok/s | 2,324.6 tok/s |
| **`shelf 3.05bpw`**<br>(EXL3-INT8 / BF16 KV + DFlash2) | Balanced default | **3.82 / 11.96** | 0.0597 / 91.11% | 104.66 tok/s | 145.37 / **87.33** tok/s | 2,350.8 tok/s |
| **`saber W4A8`**<br>(MACaQ W4A8 / FP8 KV + DFlash2) | High speed, larger weights | 5.27 / 16.52 | 0.0595 / 90.62% | **106.56 tok/s** | 137.72 / 74.83 tok/s | 4,236.5 tok/s\* |
| **`hipfire MQ4-XTS`**<br>(MQ4-XTS / FP8 KV + DFlash) | High speed, low VRAM | 4.46 / 13.96 | 0.0851 / 88.53% | 103.88 tok/s | **152.29** / 80.51 tok/s | **5,480.9 tok/s** |
| **`llama.cpp UD-Q4_K_S`**<br>(GGUF / FP16 KV + DFlash2) | High accuracy, general-purpose | 4.55 / 14.25 | 0.0407 / 93.39% | 34.46 tok/s | 53.41 / 42.19 tok/s | 1,438.5 tok/s |

- **Effective size (bpw / GiB)**: Average bits per parameter and total weight footprint used for text inference (excluding unused `vision` and `MTP` tensors), using a denominator of `26,895,998,464` parameters.
- **Quality (KL / top-1)**: Full KL divergence (lower is closer) and top-1 agreement rate (higher is closer) measured against the reference BF16 model.
- **`real-v1` / `code` / `chat`**: Average decode throughput across 6 English and Japanese code-editing tasks (`2k` / `8k` / `16k`), plus median `code` and `chat` throughput (`tok/s`) on the general benchmark suite.
- **`pp2048`**: Initial prompt processing throughput (prefill) for 2,048 tokens. \*`saber`'s `pp2048` is an approximation derived from non-speculative API TTFT (see "Trade-offs and Caveats").
- **Ready VRAM note**: By default, `shelf` and `saber` reserve most available VRAM upfront (`~31.5–31.7 GiB` on a 32 GiB GPU) to maximize KV / prefix cache capacity, so startup VRAM is omitted from the table above. In `shelf`, startup VRAM can be adjusted via `--arena-gib` with negligible impact on decode throughput (see "Trade-offs and Caveats").

---

## Key Characteristics of `shelf`

### 1. Quantization Quality and Model Footprint

- **`shelf 4bpw` matches BF16 most closely**: At `KL 0.0212` and `top-1 94.76%`, it cuts KL divergence roughly in half compared to the runner-up `llama.cpp UD-Q4_K_S` (`KL 0.0407` / `93.39%`) while sustaining `93.98 tok/s` on `real-v1`.
- **`shelf 3.05bpw` pairs compact size with competitive accuracy**: At `3.82 bpw` (`11.96 GiB`), it is ~28% smaller than `saber W4A8` (`5.27 bpw` / `16.52 GiB`) while maintaining comparable KL (`0.0597` vs `0.0595`) and higher top-1 agreement (`91.11%` vs `90.62%`). Compared to `hipfire MQ4-XTS` (`4.46 bpw` / `KL 0.0851` / `88.53%`), it is both smaller and more accurate.

### 2. Generation Throughput and Streaming Stability

- **On par with the fastest engines on real-world workloads**: Under identical prompt inputs, `shelf 3.05bpw` averages **`104.66 tok/s`** on `real-v1` (English `108.69` / Japanese `100.64`), on par with `saber` (`106.56 tok/s`) and `hipfire` (`103.88 tok/s`), and ~3× faster than `llama.cpp` (`34.46 tok/s`).
- **General benchmark throughput**: `shelf 3.05bpw` (DFlash2) records **`87.33 tok/s` on `chat` (1st overall)**, `145.37 tok/s` on `code`, `151.44 tok/s` on `json`, and `160.27 tok/s` on `math`.
- **No drafter degradation and smooth streaming**: The quantized DFlash2 drafter (EXL3-INT8 4bpw) shows no drop in expected acceptance length compared to the BF16 drafter (`4.253` vs `4.249`). After the stall fix, both the 200-request continuous streaming test (37,783 intervals total) and the 114 trials across the three additional configurations (`4bpw BF16`, `4bpw octave:c5v5`, `3.05bpw octave:c5v5`) recorded **0 stalls ≥ 300 ms** (max `115.9 ms` and `117.57 ms`, respectively).

### 3. KV Cache Compression (`octave:c5v5`)

`shelf` implements **`octave:c5v5`**, an externally proposed KV cache compression scheme (unsupported by the other three engines tested), shrinking persistent 32k-context KV memory from **`2.00 GiB` to `~0.63 GiB`** (~68% reduction). This is useful for long-context sessions and higher concurrency within fixed VRAM (see [kv-octave.md](kv-octave.md)).

| `shelf` Configuration (KV Mode) | 32k KV Footprint | Quality (KL ↓ / top-1 ↑) | Baseline `pp2048` / `tg ctx2048` | DFlash2 `real-v1` Avg | DFlash2 `code` / `chat` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `4bpw` (BF16) | 2.00 GiB | 0.0212 / 94.76% | 2,324.6 / 31.78 tok/s | 93.98 tok/s | 124.38 / 75.26 tok/s |
| `4bpw` (`octave:c5v5`) | **~0.63 GiB** | 0.0239 / 94.19% | 2,305.6 / 31.57 tok/s | 96.65 tok/s | 134.24 / 86.27 tok/s |
| `3.05bpw` (BF16) | 2.00 GiB | 0.0597 / 91.11% | 2,350.8 / 35.18 tok/s | 104.66 tok/s | 145.37 / 87.33 tok/s |
| `3.05bpw` (`octave:c5v5`) | **~0.63 GiB** | 0.0636 / 90.43% | 2,316.9 / 34.93 tok/s | 98.53 tok/s | 145.70 / 76.08 tok/s |

Quality shifts remain small (`4bpw + octave:c5v5` stays at `KL 0.0239` / `94.19%`, still higher accuracy than the other engines), and non-speculative baseline speed changes by around 1%. With DFlash2 enabled, `real-v1` average throughput relative to BF16 KV was +2.8% on `4bpw` and -5.9% on `3.05bpw`, varying with generated content and draft acceptance.

### 4. End-to-End Reproducibility (Quantization, Quality, and Benchmarks)

Alongside the inference runtime, the repository includes the full conversion, validation, and evaluation toolchain:

- **Quantization and validation**: [../user/quantize.md](../user/quantize.md) (`tools/quantize/shelf_quantize.py`, `validate_exl3.py`) lets users quantize and validate custom fine-tuned or LoRA-merged BF16 models without waiting for third-party quantized releases.
- **Speed benchmarks**: [../user/bench.md](../user/bench.md) and [methodology.md](methodology.md) (`shelf-bench`, `tools/bench/shelf_bench_suite.py`) cover reproducing both baseline and real-world task measurements.
- **Quality evaluation**: [quality-bf16/README.md](quality-bf16/README.md) and [../user/quantize.md](../user/quantize.md) document the BF16-referenced full KL and top-1 evaluation procedure.

---

## Configuration Guide by Use Case

| Priority | Recommended Configuration | Rationale |
| --- | --- | --- |
| **Balanced speed, quality, and size**<br>Chat, coding assistance, general tasks | **`shelf 3.05bpw`**<br>(DFlash2) | Smallest effective footprint (`3.82 bpw`) with practical quality (`KL 0.0597` / `91.11%`), delivering `104.66 tok/s` on `real-v1` and `87.33 tok/s` on `chat` (1st). |
| **Closest fidelity to BF16**<br>Accuracy-sensitive evaluation, technical writing, code quality | **`shelf 4bpw`**<br>(DFlash2) | Lowest KL (`0.0212`) and highest top-1 (`94.76%`) across all tested engines while sustaining `93.98 tok/s` (`KL 0.0239` and `96.65 tok/s` with `octave:c5v5`). |
| **Quantizing and verifying custom / LoRA models**<br>Self-contained reproducible workflow | **`shelf`**<br>(`4bpw` / `3.05bpw`) | Quantizer, validator, benchmark suite, and BF16-referenced quality evaluation are integrated in the same repository. |
| **Lowest time-to-first-token on long prompts**<br>Bulk document summarization, prefill-heavy workloads | **`hipfire MQ4-XTS`**<br>or **`saber W4A8`** | High `pp2048` prefill throughput (`4,236.5–5,480.9 tok/s`). Note the quality gap on `hipfire` (`KL 0.0851`) and larger weight size on `saber` (`5.27 bpw`). |
| **Sharing GPU VRAM with other workloads**<br>Coexistence with non-inference processes | **`hipfire MQ4-XTS`**<br>or **`shelf` (`--arena-gib`)** | `hipfire` allocates only required VRAM by default (`~16.88 GiB` at 32k); `shelf` supports capping allocation or leaving free VRAM via `--arena-gib` with negligible impact on decode speed. |

---

## Trade-offs and Caveats

- **Prefill throughput (`pp2048`)**: `shelf`'s `pp2048` (`2,305.6–2,350.8 tok/s`) is roughly half that of `saber` (`4,236.5 tok/s`) and `hipfire` (`5,480.9 tok/s`), and ~1.6× that of `llama.cpp` (`1,438.5 tok/s`). Workloads that repeatedly ingest uncached long prompts will see higher TTFT.
- **Origin of `octave:c5v5`**: `octave:c5v5` is an externally proposed KV cache compression scheme implemented in `shelf`.
- **Default upfront VRAM reservation and tuning**: By default, `shelf` and `saber` reserve most available VRAM upfront at startup (`~31.5–31.7 GiB` on a 32 GiB GPU) to allocate all remaining headroom to the KV / prefix cache (whereas `hipfire` uses `~16.88 GiB` and `llama.cpp` uses `~19.00 GiB` at 32k by allocating only what is required). When sharing the GPU with other processes, `shelf` lets you adjust startup VRAM via `--arena-gib N` (approximate total VRAM cap) or `--arena-gib -N` (approximate free VRAM left for other processes) with negligible impact on generation speed—reducing the budget only narrows KV / prefix cache headroom.
- **Quantization and KV cache differences across engines**: Each engine uses a different weight quantization format and default KV cache dtype (`shelf`: EXL3-INT8 + BF16 / `octave:c5v5` KV; `saber`: MACaQ W4A8 + FP8 KV; `hipfire`: MQ4-XTS + FP8 KV; `llama.cpp`: UD-Q4_K_S + FP16 KV). This report compares each engine's representative practical configuration.
- **`saber` `pp2048` measurement method**: Because step-capture was unavailable for `saber`'s W4A8 packed weights, `pp2048` is an approximation calculated from non-speculative API TTFT. (Synthetic `ctx2048` decode was also omitted because the run generated only 3 tokens.)
- **`shelf 3.05bpw BF16` measurement revision**: The `shelf 3.05bpw BF16` (DFlash2) throughput values in the comparison table were measured on `9c7be540` (prior to the streaming stall fix) and aggregate 38 normal trials after excluding 1 trial that experienced a ≥ 1 s stall. Following the fix (`b9002443`), 0 stalls ≥ 300 ms were observed across the 200-request continuous test and all 114 trials of the other three configurations.

---

## Measurement Environment and Target Configurations

### Common Environment

| Item | Setting |
| --- | --- |
| Measurement date | 2026-11-23 |
| GPU | Single AMD Radeon AI PRO R9700 (TP=1) |
| Toolchain | ROCm 10.2 (`gfx1201`, Release build) |
| Execution settings | Concurrency = 1, max context length 32,768 |
| Quality reference | Full KL divergence and top-1 agreement against `Qwen3.8-27B` (BF16) |

### Engines, Revisions, and Models

| Engine | Repository / Revision | Target Model | Speculative Drafter |
| --- | --- | --- | --- |
| `shelf` | This repository<br>`9c7be540` (`3.05bpw BF16` DFlash2 speed)<br>`b9002443` (`4bpw BF16`, `4bpw octave:c5v5`, `3.05bpw octave:c5v5` speed and streaming stability)<br>`dffb8851` (baseline `pp2048` / `tg`) | `EXL3-INT8 (mul24)` 4bpw / 3.05bpw | `DFlash2 EXL3-INT8 (mul24)` 4bpw |
| `saber-engine` | `GGZ14/saber-engine` @ `e43b8877` | `Launch80/Qwen3.8-27B-MACaQ-W4A8-SABER` | `Launch80/Qwen3.8-27B-DFlash2-Drafter-W4A8-SABER` |
| `hipfire` | `warpfront/hipfire` @ `d435a753` | `hipfire-models/qwen3.8-27b` (`qwen3.8-27b.mq4-xts`) | `hipfire-models/qwen3.8-27b` (`qwen38-27b-dflash-mq4.hfq`) |
| `llama.cpp` | `ggml-org/llama.cpp` @ `8d60873d` | `unsloth/Qwen3.8-27B-GGUF` (`UD-Q4_K_S`) | `Akicou/Qwen3.8-27B-DFlash2-GGUF` (`Q4_K_M`) |

### Reproduction Documentation and Tools

- Quantization and validation: [../user/quantize.md](../user/quantize.md) (`tools/quantize/shelf_quantize.py`, `validate_exl3.py`)
- Speed benchmarks and methodology: [../user/bench.md](../user/bench.md) / [methodology.md](methodology.md) (`shelf-bench`, `tools/bench/shelf_bench_suite.py`)
- Quality evaluation (BF16 reference full KL / top-1): [quality-bf16/README.md](quality-bf16/README.md) / [../user/quantize.md](../user/quantize.md)

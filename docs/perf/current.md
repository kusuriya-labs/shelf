# Current performance baseline

この文書は Shelf の qualified performance baseline を記録する。
ここに載る値は各節に記した measurement revision での実測結果であり、それ以後に追加された
変更に対する性能を保証しない。

測定原則は [methodology.md](methodology.md) を正本とする。

## 現在の状態

**現在の build に対する qualified baseline は、2026-11-03 の long-context prefill、
2026-11-05 の MTP decode（Qwen3.8-27B MXFP4）・MoE decode / prefill / quality・
MoE MTP decode（Qwen3.6-35B-A3B MXFP4）・MoE tensor parallel prefill（Qwen3.6-35B-A3B MXFP4）・
MoE pipeline parallel（Qwen3.6-35B-A3B MXFP4）、EXL3 W16A16 decode / prefill、
2026-11-04 の 9B BF16 linear allowlist である。**
2026-11-06 の model load（Flash Next MXFP4 TP4 / 27B FP8 TP2）と、同じ日の
BF16 GEMM / TP copy（Flash Next MXFP4 TP4）は reference run として下の節に記録する。

これまでに記録していた値は、計測の前提だった構成がソースから削除されたため、
再現できなくなっている。該当する計測を削除した。

| 削除した計測 | 失われた前提 |
| --- | --- |
| HF FP8 E2E prefill / decode（TP=1・TP=2） | resident model host（`shelf-model-host` / `--model-host`）を削除 |
| Target prefill pp2048・non-spec decode tg128 | 計測対象が `models/Qwen3.8-27B-Quark-AWQ-MXFP4`。MXFP4 / AWQ 量子化は非対応 |
| TP=2 decode（`test_qwen35_tp_e2e`） | 計測条件の `models/Qwen3.8-27B-PSQ` を削除 |
| DFlash2 decode・batched verify scaling | 融合カーネルが有効な構成と MXFP4 model での計測 |
| BetterBench によるカテゴリ別比較 | 参照実装が MXFP4 構成。現在の build では同じ条件を作れない |
| fp8 KV prefill の leg / query tile 比較、attention kernel 実効レート | kernel 構成が変更済み |
| GPU-MCU decode backend | 実装も `exp/gpu-mcu` branch もリポジトリに無い |

削除した数値そのものは Git history に残る。実験の経緯・採否判断は
[../research/](../research/) が正本である。

## 再計測が必要な断面

現在の performance target（`models/Qwen3.8-27B-FP8`、Qwen3.8-27B BF16、drafter DFlash2）で
次を測る。条件・コマンドは [methodology.md](methodology.md) に従い、measurement revision・
date・build・GPU 割り当て・生成結果の一致アンカーとともに同じ節へ書く。

| 断面 | 計測口 | 主指標 |
| --- | --- | --- |
| prefill（pp2048） | `shelf-bench pp --mode forward` | `gpu_tokens_per_sec` |
| non-spec decode（tg128 / ctx2048） | `shelf-bench tg --mode greedy` | `gpu_tokens_per_sec` |
| E2E decode（BF16 / HF FP8、TP=1・TP=2） | `shelf-compute` | `DECODE_TOKENS_PER_SEC` |
| DFlash2 decode | `shelf-compute --dflash2-model-dir <dir> --dflash2-drafts 7` | `DECODE_TOKENS_PER_SEC`、`DFLASH2_ROUNDS`、`DFLASH2_MEAN_ACCEPTED` |
| DFlash2 batched verify scaling | `test_dflash2_batch_perf` | round 内訳（`draft_gpu_ms` / `verify_gpu_ms`） |
| TP=2 scaling | `test_qwen35_tp_e2e` | decode tok/s p50 |
| カテゴリ別 serving | `vendor/BetterBench`（methodology §9） | decode t/s、TTFT p50 |
| fp8 KV の数値品質 | `tools/quality_gate.py` / `shelf-bench quality` | ppl 差、mean KL、top-1 一致率 |

## 2026-11-06 model load

起動時の weight load を短縮した。rank runtime を並列に生成し、Qwen4Exp MXFP4
checkpoint の PLE table（FP8 E4M3 per-tensor）を rank の weight load と同時に
prefetch する。変更内容と切り分けは
[../research/20261106/model_load_parallelism.md](../research/20261106/model_load_parallelism.md)。

### Qwen3.8-Flash-Next-Quark-MXFP4 / TP=4

| 状態 | `MODEL_LOAD_MS` | wall |
| --- | ---: | ---: |
| 変更前（warm） | 60.36 / 60.38 s | 63.35 / 64.98 s |
| 変更前（1 回目） | 69.18 s | 73.77 s |
| 変更後（warm） | **15.51 / 17.51 / 18.23 s** | 20.14 / 22.10 / 21.28 s |
| 変更後（page cache 追い出し後） | 30.42 s | 33.49 s |

- measurement revision: `4efa8de` の working tree（rank 並列化、PLE shard copy の
  16 worker 化、PLE prefetch）
- GPU: Radeon AI PRO R9700 × 4（device 0,1,2,3）
- command:
  `shelf-compute --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 --input-ids-file <3 token> --max-new-tokens 1 --max-seq-len 256 --device 0,1,2,3 --tensor-parallel 4 --prefix-cache 0 --kv-cache-dtype bf16 --temperature 0 --arena-gib 31`
- 生成一致アンカー: `GENERATED_IDS=1118`（3 token prompt）。変更前の同じコマンドと
  一致する。5 token prompt / 8 token 生成の 2 回は
  `9338,11,321,279,7526,3177,303,4357` で一致し、`--dump-logits` も bitwise 一致した。
- PLE table 単体（`test_qwen4exp_load --skip-gpu`）は 24.57 s → **7.23 s**
  （128 shard、47.684 GiB、16 worker）。

### 他の family（参考）

| model | 変更前 | 変更後 |
| --- | ---: | ---: |
| Qwen3.8-27B-FP8 / TP=2、1 回目 | 30.08 s | **19.15 s** |
| Qwen3.8-27B-FP8 / TP=2、2 回目 | 20.91 s | **10.47 s** |
| Qwen3.6-35B-A3B MXFP4 / TP=2 | 未測 | 18.96 s |

- 27B の command は
  `shelf-compute --model-dir models/Qwen3.8-27B-FP8 --input-ids-file <3 token> --max-new-tokens 1 --max-seq-len 128 --device 0,1 --tensor-parallel 2 --prefix-cache 0 --kv-cache-dtype bf16 --temperature 0`
- MoE の command は
  `shelf-compute --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 --input-ids-file <3 token> --max-new-tokens 1 --max-seq-len 512 --device 0,1 --tensor-parallel 2 --prefix-cache 0 --kv-cache-dtype bf16 --temperature 0`
- いずれも `GENERATED_IDS=1118` で変更前と一致する。

page cache の状態で値が変わるため、cold / warm を混ぜた比較はしない。

## 2026-11-06 Qwen3.8-Flash-Next MXFP4 TP4

`models/Qwen3.8-Flash-Next-Quark-MXFP4` を TP4 / bf16 KV / prefix cache なしで
計測した reference run である。qualified baseline ではない。

| 条件 | target-only | DFlash2 drafts=6 |
| --- | ---: | ---: |
| prompt 5 token、64 new token | 26.77 tok/s | 61.13 tok/s |
| prefill 5 token | 106.1 ms | 113.7 ms |
| mean accepted / round | - | 2.150 |
| full accept rate | - | 0.150 |
| DFlash2 round | - | 20 rounds / 1030.1 ms |

- measurement revision: `0a715da`
- GPU: Radeon AI PRO R9700 4 枚（device 0,1,2,3）
- command（target-only）:
  `shelf-compute --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 --input-ids-file <5 token> --max-new-tokens 64 --max-seq-len 256 --device 0,1,2,3 --tensor-parallel 4 --prefix-cache 0 --kv-cache-dtype bf16 --temperature 0 --arena-gib 31`
- command（DFlash2）: 上記に
  `--dflash2-model-dir models/Qwen3.8-Flash-Next-DFlash2 --dflash2-drafts 6` を加える
- 生成一致アンカー: `The capital of France is` →
  ` Paris. The capital of Germany is Berlin`
  （`11751,13,561,6511,314,9564,369,19241`）。drafts=6 の 64 token は target-only と
  完全一致する。drafts=7（8 行 verify）は 64 token で 46 token 目に分岐するため、
  性能値として採らない。
- 同一条件 2 回の `--dump-logits` は bitwise 一致する。

### context 128 / 512 の pp / tg

同じ revision・同じ 4 GPU で、prompt を 128 / 512 token にした断面。この表は当時
`shelf-bench` が Qwen4Exp の TP4 を扱わなかったため、`shelf-compute` の `PREFILL_MS` と
`DECODE_TOKENS_PER_SEC` で測ったものである。現在は `shelf-bench pp/tg/quality/batch` が
`--tensor-parallel 4` を受理する。DFlash2 の accept は corpus に依存する。

| 断面 | target-only | DFlash2 drafts=6 |
| --- | ---: | ---: |
| pp 128 token | 164.0 ms（780 tok/s） | 175.6 ms |
| pp 512 token | 293.1 ms（1747 tok/s） | 318.7 ms |
| tg ctx128、自然文 64 new | 25.74 tok/s | 35.95 tok/s（accepted 0.909） |
| tg ctx128、反復文 64 new | 25.79 tok/s | 129.15 tok/s（accepted 6.000） |
| tg ctx512、反復文 32 new | 20.21 tok/s | 95.33 tok/s（accepted 5.200） |

反復文は DFlash の accept が最大になる上限側の参考値であり、自然文の
`tg ctx128 = 35.95 tok/s` を通常の期待値として扱う。

### pp 2048（TP=4、shelf-bench）

`shelf-bench pp` の Qwen4Exp TP4 対応後に、同じ 4 GPU で 2048 token の prefill を
5 run ずつ 2 回測した。corpus は `vendor/BetterBench/corpus/v1` から
`tools/quality_gate.py corpus` で生成した 2203 token の先頭 2048 token である。

| 条件 | gpu_ms p50 | gpu_tokens_per_sec | run 間 |
| --- | ---: | ---: | --- |
| 1 回目 | 733.17 ms | 2793.35 | min 727.91 / p10 729.72 / p90 747.84 / stddev 10.67 ms |
| 2 回目 | 751.91 ms | 2723.74 | min 747.25 / p10 748.86 / p90 753.92 / stddev 2.78 ms |

- measurement revision: `4efa8de` の working tree（shelf-bench の TP4 対応、QSA top-k の
  radix select 化、mask 対応 prefill attention、BF16 / FP8 GEMM の出力 tile 再利用を含む）
- GPU: Radeon AI PRO R9700 × 4（device 0,1,2,3）
- command:
  `shelf-bench pp --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 --tokens-file <2203 token corpus> --tokens-offset 0 --prompt-tokens 2048 --prefill-chunk 2048 --runs 5 --warmup 1 --device 0,1,2,3 --tensor-parallel 4 --arena-gib 31`
- `shelf-bench pp` は top-level の `TpCoordinator` 経由で 4 rank を実行する。
  `PREFILL_MS` ではなく prefill ループの GPU event 区間を測るため、`shelf-compute` の
  `PREFILL_MS` とは区間が違う。混ぜて比較しない。
- 2 回の p50 差は 2.6 %。2 回目の run 間ばらつきは 1.5 % で、5 run の p50 として採用する。
- 最適化前（同じ条件、`4efa8de` の working tree で shelf-bench TP4 対応直後）は
  1202.36 / 1208.92 ms、1703.31 / 1694.07 tok/s だった。現在は約 38 % 短い。

#### kernel 内訳

rocprofv3 1.3.5 の `--kernel-trace` で `--runs 1 --warmup 0` の 1 run を取得した。
trace 下の wall は 809.02 ms で、通常の p50 733.17〜751.91 ms より約 8〜10 % 大きい。
GPU time は trace の kernel duration、launch は physical launch の 4 rank 合計である。

| kernel | launch（4 rank 合計） | GPU time（4 rank 合計、ms） | 1 call 平均（µs） | share |
| --- | ---: | ---: | ---: | ---: |
| `gemm_bf16_wmma` | 976 | 496.413 | 508.62 | 19.1% |
| `tp_copy_bf16` | 720 | 384.721 | 534.33 | 14.8% |
| `gemm_fp8_ptpc` | 960 | 376.251 | 391.93 | 14.5% |
| `gemm_mxfp4_w4a8_grouped` | 576 | 159.162 | 276.32 | 6.1% |
| `shelf_gdn_recurrence_wmma_lossy` | 144 | 148.730 | 1032.85 | 5.7% |
| `qsa_scores_batch` | 48 | 131.861 | 2747.10 | 5.1% |
| `v_prefill_bf16::attention_paged_prefill_bf16_impl` | 48 | 77.337 | 1611.19 | 3.0% |
| `hyper_mix` | 388 | 71.371 | 183.95 | 2.7% |
| `activation_quantize_mxfp4_rowref` | 768 | 68.333 | 88.97 | 2.6% |
| `qsa_topk_batch` | 48 | 62.962 | 1311.70 | 2.4% |
| `hyper_apply` | 384 | 61.229 | 159.45 | 2.4% |
| `qsa_key_cache_batch` | 48 | 59.213 | 1233.61 | 2.3% |
| `shelf_rmsnorm_bf16_pf_oneplus` | 400 | 51.343 | 128.36 | 2.0% |
| `shelf_mul_f32_f32_bf16` | 192 | 50.582 | 263.45 | 1.9% |
| `moe_combine` | 192 | 44.511 | 231.83 | 1.7% |
| `shelf_silu_f32_bf16` | 144 | 35.885 | 249.20 | 1.4% |
| `activation_quantize_fp8_pertoken` | 384 | 31.640 | 82.40 | 1.2% |
| `qsa_expand_mask` | 48 | 29.177 | 607.86 | 1.1% |
| `shelf_swiglu_bf16` | 384 | 26.903 | 70.06 | 1.0% |
| `gemm_mxfp4_w4a8_prefill_2d` | 576 | 26.803 | 46.53 | 1.0% |
| `gemm_bf16_splitk_wmma` | 576 | 25.769 | 44.74 | 1.0% |
| `shelf_gdn_conv1d_bf16_f32` | 144 | 25.352 | 176.06 | 1.0% |
| `tp_add_peers_bf16` | 240 | 22.442 | 93.51 | 0.9% |
| `shelf_rmsnorm_f32_pg_direct` | 144 | 22.248 | 154.50 | 0.9% |
| `shelf_sigmoid_bf16_f32` | 192 | 18.432 | 96.00 | 0.7% |
| `moe_slot_scan` | 192 | 17.542 | 91.36 | 0.7% |
| `moe_router` | 192 | 9.076 | 47.27 | 0.3% |
| `shelf_rope_f32_bf16_pair` | 96 | 8.276 | 86.21 | 0.3% |
| `shelf_l2_normalize_bf16_f32` | 288 | 8.242 | 28.62 | 0.3% |
| `shelf_gemm_bf16_exact_rows_1` | 4 | 8.064 | 2016.06 | 0.3% |
| `shelf_split_bf16` | 48 | 6.941 | 144.61 | 0.3% |
| `shelf_rmsnorm_bf16_f32_pg_oneplus` | 96 | 5.035 | 52.45 | 0.2% |
| `shelf_scale_f32` | 144 | 2.889 | 20.06 | 0.1% |
| `shelf_scale_bf16` | 148 | 2.826 | 19.09 | 0.1% |
| `ple_conv` | 4 | 2.777 | 694.26 | 0.1% |
| `moe_slot_scatter` | 192 | 2.717 | 14.15 | 0.1% |
| `silu_scale_bf16` | 388 | 2.610 | 6.73 | 0.1% |
| `model_dispatch` | 192 | 2.565 | 13.36 | 0.1% |
| `shelf_residual_add_bf16` | 8 | 1.663 | 207.90 | 0.1% |
| `prepare_batch_descriptor` | 4 | 1.063 | 265.78 | 0.0% |
| `amd_rocclr_copyBuffer` | 264 | 1.035 | 3.92 | 0.0% |
| `hyper_inject` | 384 | 0.943 | 2.46 | 0.0% |
| `ple_stream_dot` | 4 | 0.724 | 181.04 | 0.0% |
| `qsa_rope` | 48 | 0.689 | 14.34 | 0.0% |
| `shelf_kv_append_bf16` | 48 | 0.687 | 14.32 | 0.0% |
| `amd_rocclr_fillBufferUnAligned` | 36 | 0.628 | 17.46 | 0.0% |
| `amd_rocclr_copyBufferRectAligned` | 48 | 0.496 | 10.33 | 0.0% |
| `rmsnorm_impl` | 48 | 0.396 | 8.24 | 0.0% |
| `repeat_streams` | 4 | 0.367 | 91.67 | 0.0% |
| `ple_broadcast_mul` | 4 | 0.365 | 91.19 | 0.0% |
| `shelf_embedding_bf16` | 4 | 0.147 | 36.77 | 0.0% |
| `ple_gate` | 4 | 0.018 | 4.62 | 0.0% |
| `output_gather` | 4 | 0.010 | 2.50 | 0.0% |
| `argmax_single_impl` | 4 | 0.009 | 2.30 | 0.0% |
| `amd_rocclr_streamOpsWrite` | 4 | 0.009 | 2.19 | 0.0% |

- 4 rank 合計の GPU time は wall と加算できない。1 rank あたりは概ね 1/4 で、
  合計 2595.3 ms、1 rank 平均 648.8 ms である。
- `shelf_scale_bf16` だけ rank0 が 1 回、他 rank が 49 回。他は 4 rank で同数。
- 上位は `gemm_bf16_wmma` 19.1 %、`tp_copy_bf16` 14.8 %、`gemm_fp8_ptpc` 14.5 %、
  `gemm_mxfp4_w4a8_grouped` 6.1 %、`gdn_recurrence_wmma_lossy` 5.7 %。
- `__amd_rocclr_*` は prefill 区間の copy / fill を含む。
- 上位 4 カーネルの ISA と律速要因、採用した対策は
  [../research/20261106/pp2048_tp4_isa_analysis.md](../research/20261106/pp2048_tp4_isa_analysis.md)
  にある。ISA dump は `build/isa/pp2048_tp4/` に置いた。

## 2026-11-06 Qwen3.8-Flash-Next MXFP4 TP4 の BF16 GEMM と TP copy

`models/Qwen3.8-Flash-Next-Quark-MXFP4` を TP4 / bf16 KV / prefix cache なしで測った
reference run である。qualified baseline ではない。

- `tp_copy_bf16` / `tp_add_peers_bf16` / `tp_copy_f32` / `tp_add_peers_f32` を
  128-bit vector load/store 化した。
- BF16 linear の `rows >= 128` を hipBLASLt に切り替えた。`rows < 128` は従来カーネル。
- FP8 PTPC の hipBLASLt 化は採らない。判断と根拠は
  [../research/20261106/fp8_ptpc_hipblaslt.md](../research/20261106/fp8_ptpc_hipblaslt.md)
  にある。

| 断面 | 変更前 | 変更後 |
| --- | ---: | ---: |
| pp 2048 | 733.17 / 751.91 ms | **633.97 / 634.35 ms** |
| pp 2048 tok/s | 2793.35 / 2723.74 | **3230.42 / 3228.49** |
| pp 512 | 293.1 ms（`shelf-compute`） | 176.71 ms / 2897.41 tok/s |
| pp 128 | 164.0 ms（`shelf-compute`） | 81.81 ms / 1564.56 tok/s |
| tg ctx128、8 new | 25.91 tok/s | 25.94 tok/s |
| trace の `tp_copy_bf16` | 384.72 ms | 313.75 ms |

- measurement revision: `981d492` の working tree（TP copy の vector 化、BF16 GEMM の
  hipBLASLt 化）
- GPU: Radeon AI PRO R9700 × 4（device 0,1,2,3）
- command（pp）:
  `shelf-bench pp --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 --tokens-file <2203 token corpus> --tokens-offset 0 --prompt-tokens 2048 --prefill-chunk 2048 --runs 5 --warmup 1 --device 0,1,2,3 --tensor-parallel 4 --arena-gib 31`
- command（tg）:
  `shelf-bench tg --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4 --mode greedy --context 128 --tokens 8 --warmup 1 --device 0,1,2,3 --tensor-parallel 4 --arena-gib 31`
- 生成一致アンカー: `GREEDY_TOKEN_SUM=505644`。BF16 linear を全行 hipBLASLt にした場合と
  全行従来カーネルにした場合はどちらも `819968` で、prefill と decode で経路を分けた
  ときだけ分岐する。これは丸め差による greedy の分岐であり、同一性の判定には使えない。
  `shelf-compute --max-new-tokens 1` の 3 token prompt は `GENERATED_IDS=1118` で
  変更前と一致する。
- pp 2048 の改善は 733.17 → 634.35 ms（13.5 %）。

## 2026-11-06 Qwen3.8-Flash-Next EXL3 decode reference

`models/Qwen3.8-Flash-Next-exl3` を TP=2 / bf16 KV / prefix cache なしで計測した
単発の reference run である。qualified baseline ではない。

| 条件 | 値 |
| --- | ---: |
| prefill 5 token | 98.9 ms |
| decode | 22.28 tok/s（44.9 ms/token、15 token） |

- command:
  `shelf-compute --model-dir models/Qwen3.8-Flash-Next-exl3 --input-ids-file <5 token> --max-new-tokens 16 --max-seq-len 128 --device 0,1 --tensor-parallel 2 --prefix-cache 0 --kv-cache-dtype bf16 --temperature 0`
- GPU: Radeon AI PRO R9700 2 枚（GPU 0,1）
- 生成結果: `The capital of France is Paris. The capital of the United Kingdom is London.`
- decode は rank あたり 26 GiB の weight traffic による DRAM bound にほぼ一致する。

## 2026-11-05 EXL3 W16A16 decode / prefill

`models/Qwen3.8-27B-exl3`（mul1、4 bpw、lm_head 6 bpw）を TP1 / bf16 KV / GPU 1 枚で
計測した。EXL3 は production target（FP8 / BF16）とは別の経路であり、以下の値は
EXL3 経路の現在値である。

| 条件 | 値 |
| --- | ---: |
| decode | 20.09 / 20.16 / 20.47 tok/s（48.9〜49.8 µs/token） |
| prefill 128 | 632.98 / 633.17 / 636.39 tok/s（1.571〜1.580 ms/token） |
| prefill 256 | 800.07 / 800.13 / 801.87 tok/s（1.247〜1.250 ms/token） |
| prefill 512 | 1096.12 / 1097.37 / 1104.84 tok/s（0.905〜0.912 ms/token） |
| prefill 1024 | 1293.05 / 1295.17 / 1298.60 tok/s（0.770〜0.773 ms/token） |
| prefill 2048 | 1389.53 / 1390.41 / 1398.79 tok/s（0.715〜0.719 ms/token） |

- measurement revision: `ca5af8f`（dense 経路の rowsum 事前計算、hipBLASLt algo の
  16 候補選択、recon の row-major 書き込み。prefill `rows >= 256` は dense recon +
  hipBLASLt、`rows < 256` / decode / verify / LM-Head は融合 W16A16）
- `feature-moe-mtp` の木（EXL3 の実行 workspace を条件つき確保へ変えた後）で同じ条件を
  `--device 0` / `--arena-gib 28` で再測した: prefill 2048 **1439.4** tok/s、decode 128
  **20.70** tok/s。上表の帯と同じかややくらい。dense recon 経路が落ちていると prefill 2048 は
  **715 tok/s** になる（落ち方の記録は
  [../research/20261105/exl3_workspace_conditional.md](../research/20261105/exl3_workspace_conditional.md) §6）
- GPU: Radeon AI PRO R9700 1 枚（`--device 1`、他プロセスなし）
- decode command:
  `shelf-bench tg --model-dir models/Qwen3.8-27B-exl3 --mode forward --context 128 --tokens 16 --warmup 1 --device 1 --arena-gib 28`
- prefill command:
  `shelf-bench pp --model-dir models/Qwen3.8-27B-exl3 --mode forward --prompt-tokens <N> --runs 3 --warmup 1 --device 1 --arena-gib 28`
- split は `grid_x <= 16 → 8`、`grid_x <= 160 → 4`、それ以上は 1 を基本に、
  K chunk 数と 12 MiB の scratch 容量で制限する。`SHELF_EXL3_SPLIT` で上書きできる。
  rows=2048 では scratch 制限により split=1 になる。
- prefill は `rows >= 256` かつ dense / accum の容量に収まるとき、recon kernel が
  trellis を row-major FP16 へ展開し、hipBLASLt の GEMM と epilogue で計算する。
  `SHELF_EXL3_DENSE` でしきい値を変更できる（0 で無効）。LM-Head、decode、verify、
  `rows < 256` は融合 W16A16 のまま。
- 融合経路の N tile は `rows >= 4` かつ `out_features % 256 == 0` かつ
  `out_features >= 1024` のとき 2。row group は `rows >= 256` かつ `grid_x >= 8` の
  とき 4、`rows >= 32` かつ `grid_x * ceil(rows/32) * splits >= 128` のとき 2。
  `SHELF_EXL3_GROUPS` で 1 / 2 / 4 を強制できる。
- 入力回転は `rows >= 4` かつ `rows * k * 2 <= 80 MiB` のとき linear ごとに 1 回だけ
  計算する。pre 経路は xrot を LDS にコピーせず global から B fragment を読む。
  `SHELF_EXL3_PRE=0` で fused 経路（毎 CTA 再計算、LDS pitch 136）に切り替えられる。
- decode（rows=1）は fused のまま。reduce kernel は平均 3.5 µs。
- 生成一致アンカー: `The capital of France is` → ` Paris.`（`11751,13,198`）。
- DFlash2（drafts=7）は EXL3 target でも動作する（DFlash2 の target lm_head に
  EXL3 経路を追加）。5 token prompt、32 new token、bf16 KV、`--arena-gib 28`:

  | 条件 | 値 |
  | --- | ---: |
  | non-spec decode | 20.82 tok/s |
  | DFlash2 drafts=3 | 40.53 tok/s（accepted 1.58/round） |
  | DFlash2 drafts=5 | 47.73 tok/s（accepted 2.10/round） |
  | DFlash2 drafts=7 | 50.59 / 51.65 / 51.67 tok/s（accepted 2.44/round） |
  | DFlash2 drafts=7、64 new token | 60.74 tok/s（accepted 3.20/round） |

  生成 32 token は non-spec と一致する。303 token の合成 random prompt では
  accepted 0.48/round まで落ち、9.81 tok/s と non-spec を下回る（自然文では有効）。

  E2E decode は 22.40 tok/s、MTP-2 は 18.3 tok/s（`--arena-gib 27`）。
- 8 warp 化前（`2504163`）の同一 decode 条件は 1.32〜1.34 tok/s。split-K 前の
  9.04〜9.06 tok/s からの内訳と棄却した実験は
  [../research/20261105/exl3_isa_analysis.md](../research/20261105/exl3_isa_analysis.md)
  を参照。

## 2026-11-07 MTP TP=2（Qwen3.8-27B-exl3）

`models/Qwen3.8-27B-exl3`（5 bpw）の MTP を TP=2 で動かした。MTP block は rank に
分割せず、rank0 の drafter と全 rank の target verify / GDN restore で実行する。
契約は [../developer/mtp.md](../developer/mtp.md) の Tensor parallel 節。

| 条件 | 値 |
| --- | ---: |
| target-only TP2、128 token | 22.89 tok/s |
| MTP TP2 drafts=4、32 token | 54.18 tok/s |
| MTP TP2 drafts=4、128 token | **59.28 tok/s** |
| MTP TP2 drafts=4、128 token の mean accepted | 2.750 |
| DFlash2 TP2 drafts=6、128 token | **72.87 tok/s** |

- measurement revision: MTP の TP=2 対応を入れた working tree（TP partition plan に
  MTP entry なし、`emit_normed_token_hidden`、per-rank GDN snapshot / compact、
  MTP+TP の working set reserve 4 GiB）
- GPU: Radeon AI PRO R9700 2 枚（device 0,1）
- command:
  `shelf-compute --model-dir models/Qwen3.8-27B-exl3 --input-ids-file <5 token> --max-new-tokens 128 --max-seq-len 256 --device 0,1 --tensor-parallel 2 --prefix-cache 0 --kv-cache-dtype bf16 --temperature 0 --arena-gib 30 --mtp --mtp-drafts 4`
- 生成一致アンカー: target-only TP2 と MTP TP2 の 128 token `GENERATED_IDS` は一致する。
  2048 token prompt / 64 token の `test_mtp_parity_tp2` も PASS（rounds 12、
  GDN restore 5）。
- MTP は TP1 でも動く（32 token、drafts=4 で 44.51 tok/s）。TP2 で +33 %。
- DFlash2 TP2 の方が速い。MTP は drafter model の追加 load が不要な点が利点。
- この計測は EXL3 の ka=5 fast path を含まない build で取った。fast path を入れた
  build では target / MTP とも再測が必要。

## 2026-11-03 long-context prefill

Qwen3.8-27B-FP8 / TP2 / FP8 KV / `--rope-factor 2`、selector の範囲内
（`max_visible <= 262144`）で計測した。warmup 1 + 5 run の p50。

| 条件 | 値 |
| --- | ---: |
| prompt tokens | 262,128（要求 seq len 262,144） |
| prefill chunk | 2048 |
| gpu_ms p50 | 272468.22 |
| gpu_tokens_per_sec p50 | 962.05 |
| run 間 p10 / p90 | 272200.69 / 272688.71 ms |

- measurement revision: `bb2e56e` + 計測用 harness 変更
- GPU: Radeon AI PRO R9700 × 2（device 0,1）
- command: `shelf-bench pp --model-dir models/Qwen3.8-27B-FP8 --tensor-parallel 2 --device 0,1 --kv-cache-dtype fp8 --prompt-tokens 262128 --prefill-chunk 2048 --runs 5 --warmup 1`
- 524,288 は production の値が未取得（`kMeasuredMaxVisible24 = 524288` を採用済みだが、
  warmup + 5 run の production 計測は未実施。1 run の参考値のみ research 記録にあり）。詳細は
  [../research/20261103/gate6_long_context.md](../research/20261103/gate6_long_context.md)。

## 2026-11-04 Qwen3.6-35B-A3B MoE decode

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8）を TP=1 / bf16 KV で
計測した。decode の correctness path は `EMBEDDING_LOOKUP`（1 本 / token）だけである。

**生成一致アンカー（`GREEDY_TOKEN_SUM`）は combine の shared expert gate stride 修正で変わった**
（context 64 で 2199 → 976467）。decode 速度そのものは修正前後で変わらず、修正後の再測は
[§2026-11-05 MoE combine 修正後の再測](#2026-11-05-moe-combine-修正後の再測)。

| 条件 | 値 |
| --- | ---: |
| context tokens | 64 |
| decode tokens | 32 |
| warmup | 1 |
| gpu_ms p50（clean 6 run） | 497.27 |
| gpu_tokens_per_sec p50 | 64.35 |
| run 間 | 496.28 / 496.70 / 497.21 / 497.34 / 497.42 / 498.87 ms |
| 生成一致 anchor | `GREEDY_TOKEN_SUM=2199`（11 run すべて一致） |

- measurement revision: `4b8d413` + `RMS_NORM` @2048（per-feature）と `PAGED_ATTENTION`
  `(q_heads 16, kv_heads 2, head_dim 256)` の registered 化、split-KV splits 決定の一本化、
  `EMBEDDING_LOOKUP` @2048 の registered 化、`MOE_COMBINE` の feature 方向分割と
  `MOE_PLAN_SCAN` の expert 合計 LDS 前段（2026-11-05、
  [../research/20261105/moe_decode_launch_roofline.md](../research/20261105/moe_decode_launch_roofline.md)）
- GPU: Radeon AI PRO R9700（`--device 0`、測定時は device 0 / 1 idle）
- command: `shelf-bench tg --mode greedy --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 --context 64 --tokens 32 --warmup 1`
- kernel 単位 counter の内訳（`SHELF_QWEN35_PERF_STATS=1`、32 token 区間）:
  logical_dispatches 43360 = optimized_operations 43360、**correctness_fallbacks 0**
  （`physical_kernel_launches` 43392 は grouped 系の multi-launch ぶん。`EMBEDDING_LOOKUP`
  登録前は fallback 32、`RMS_NORM` / `PAGED_ATTENTION` 登録前は 2944）
- split の on / off は `physical_kernel_launches - logical_dispatches` で見てよい
  （context 4096 / 8 token で 88 = grouped の 8 + split attention の 80 = 10 層 × 8 decode）。
  context 64 では 32（grouped のみ）で split は入らない
- 前の値（同一条件、combine / plan scan 変更前）は `gpu_ms p50` 542.29 / **59.01 tok/s**。
  その前は `gpu_ms p50` 3128.75 / **10.22 tok/s**（anchor `GREEDY_TOKEN_SUM=2987`）。その前は
  5512.97 / 5.81 tok/s（anchor 15232）。登録と計測の経緯は
  [../research/20261104/moe_backbone_linear_allowlist.md](../research/20261104/moe_backbone_linear_allowlist.md) と
  [../research/20261104/moe_rmsnorm_paged_attention_allowlist.md](../research/20261104/moe_rmsnorm_paged_attention_allowlist.md)
- `shelf-compute` E2E（prompt 14 token、`--max-num-batched-tokens 64 --max-concurrent-requests 1`、
  8 decode）は DECODE_TOKENS_PER_SEC 56.52（1 run）。tg とは条件が違うので直接比較しない
- 長い context（device 1、各 2 run、`--prefill-chunk 2048`）。split-KV 有効後:
  `--context 4096` は 56.19 / 56.25 tok/s、`--context 2048` は 56.60 / 55.01 tok/s。
  capability を外した同一 tree の A/B は 51.67 / 51.73 と 55.02 / 55.11 なので、
  **context 4096 で +8.7 %、context 2048 は差が無い**（visible 2048 は attention が短く、
  split の効く余地が無い）。`q_per_kv = 8` の split-KV は bf16 で有効（`max_visible_tokens >= 2048`）、
  fp8 は非 split のまま
- 同じ条件を MoE combine / plan scan の変更後に測ると `--context 4096` は **58.79 / 61.64**
  （anchor `GREEDY_TOKEN_SUM=2192` 不変。combine だけの中間測定で 60.40 / 60.43）。
  context 4096 の run 揺れは 2.9 ms/token（5.3 %）あるので、2 run の差を性能主張に使わない
- context 2048 は split の on / off で greedy の合計が 745578 → 827368 に変わる
  （reduction 順序の差。split は softmax を可視域ごとに online で走らせ最後に合成する）。
  数値品質は context 2048 / 64 positions の A/B で mean KL 0.00623・top-1 一致 0.984・
  ppl 差 −2.70 %（TP1 と TP2 の差と同じ帯。根拠は research 記録）
- context 64 は split の閾値（2048）未満なので無変更。3 run で 58.74 / 58.86 / 59.17 tok/s、
  anchor `GREEDY_TOKEN_SUM=2199`
- MoE prefill chunk（decode、tg、device 1、各 2 run）: `--prefill-chunk 1024 / 2048 / 4096` の
  decode は 51.36 / 51.28 / 49.31 tok/s。context 8192 では chunk 4096 が 55.09 / 54.86 tok/s
  （`GREEDY_TOKEN_SUM=710`）
- MoE prefill throughput（`shelf-bench pp --prompt-tokens 8192 --runs 3 --warmup 1`、device 1、
  `LoadedModel` 配線後、group 行数 capacity 厳密化後の tree）:
  `--prefill-chunk 1024 / 2048 / 4096` で **3662 / 3807 / 3631 tok/s**（chunk 1024 は 2 rep で
  3662.29 / 3645.01、run 揺れ 0.5 %）。**chunk 2048 が最良**。grouped expert の row bucket は
  4096 token（group 36608 行）が上限で、
  chunk 8192 は allowlist 外なので `unsupported`（上限を 8192 に上げたときの値は
  3569 tok/s で最良より 5.3 % 低く、raise の根拠にはならなかった。
  経緯は `docs/research/20261104/moe_group_row_elementwise_envelope.md` と
  `docs/research/20261105/moe_group_row_capacity_exactness.md`）
- 数値の連続性は teacher-forced A/B（base の生成列をプロンプトへ埋めた 14 token 入力、
  `--dump-logits` 9 行）で top-1 一致 7/9、入力列が同一だった row 0〜6 はすべて一致。
  HF との絶対照合は未実施（§8 の 40 層 E2E gate）

## 2026-11-05 MTP decode（Qwen3.8-27B MXFP4）

`models/Qwen3.8-27B-Quark-AWQ-MXFP4` を TP=1 / BF16 KV で計測した。target-only と
MTP drafter（`--mtp --mtp-drafts 7`）を交互に実行し、各 3 run の p50 を取った。
全 run で `GENERATED_IDS` が target-only と一致した。

| prompt | target t/s | MTP K=7 t/s | speedup | accepted/7 | full accept | prefill target/MTP ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 26.07 | **91.89** | **3.52x** | 6.88 | 1.00 | 407.1 / 468.3 |
| 2048 | 26.33 | 63.55 | 2.41x | 4.25 | 0.58 | 821.8 / 938.0 |
| 4096 | 26.08 | 80.27 | 3.08x | 6.00 | 0.89 | 1657.1 / 1938.9 |

- measurement revision: executor の keepalive thread を削除した `feature-moe-mtp` build
  （combine gate 修正・MTP requant 採否なしの状態と同じ forward）
- GPU: Radeon AI PRO R9700 × 1（`--device 0`、他の GPU は idle）
- command（MTP）: `shelf-compute --model-dir models/Qwen3.8-27B-Quark-AWQ-MXFP4 --input-ids-file <prose N tokens> --max-new-tokens 64 --max-seq-len 4096（4096 は 8192） --device 0 --prefix-cache 0 --mtp --mtp-drafts 7`
- prompt: `tests/data/mtp_perf/prose.txt` を tokenizer で 1024 / 2048 / 4096 token に伸ばしたもの
- 生成一致 anchor（`GENERATED_IDS` の sha1、keepalive thread 削除前後で同一）:
  1024 `6c07c6fe36929d9ad2f750304422b1a6f6035edc`、
  2048 `08e9e2c949de068a2d18621e676067db5170e070`、4096 `4aba2b4cee835fa60e6696f5528fb4ece21d6e42`
- run 間の散布は小さい（1024 MTP 91.89 / 91.89 / 93.16、2048 MTP 63.39–63.59、
  4096 MTP 75.98–81.00）。4096 target-only の 3 run は 26.03 / 26.08 / 26.08
- verify 時間 / round は 1024 で 43.6 ms（349.0 / 8 round）、2048 で 44.0 ms（527.6 / 12 round）、
  4096 で 45.6–46.3 ms（410.1–416.8 / 9 round）。draft 時間 / round は 41.1–42.1 / 38.6–38.8 /
  40.8–45.8 ms
- 旧値（`2c3cdb1`、executor keepalive thread が既定 on だった build、5 run p50）は
  1024 target 26.03 / MTP 56.55（2.17x）、2048 26.25 / 37.91（1.44x）、
  4096 25.93 / 49.46（1.91x）で、**MTP 側は無効**。keepalive thread が verify round に
  約 1.6 倍の費用を載せていた（切り分けは
  [../research/20261105/executor_keepalive_thread.md](../research/20261105/executor_keepalive_thread.md)）。
  target-only 側は削除前後で同じ帯（26.0–26.3 t/s）
- 参考: K=4 の 5-run p50 は 1024 41.00、2048 34.95、4096 37.73。計測した範囲では K=7 が上回る
  （K=4 は keepalive on build の実測で、削除後は K の比較を改めてしていない）
- target の prefill chunk は target-only と同じ `max_batch_rows` を使う。MTP 側の first pass だけ
  512 行以下に分割するため、MTP の prefill は target より長い（上表の prefill 差）

## 2026-11-04 MoE teacher-forced quality（初回）

`shelf-bench quality` を `LoadedModel` 経由にしたことで、grouped expert checkpoint も
teacher forcing できる。

**この節の数値は combine の shared expert gate stride 欠落のある forward で測ったもので、
絶対値は無効**（修正後は [§2026-11-05 MoE combine 修正後の再測](#2026-11-05-moe-combine-修正後の再測)）。
on / off A/B という比較の形自体に関する結論は同じ節の notes を参照。

| 条件 | 値 |
| --- | ---: |
| model | Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 / TP=1 / bf16 KV |
| corpus | `vendor/BetterBench/corpus/v1` を `tools/quality_gate.py corpus`（2191 token、sha256 `9136d38b…`） |
| context / scored positions | 128 / 64 |
| `QUALITY_PPL` | 75.146 |
| 再現性 | 同一条件の self-compare: mean KL 0、top-1 一致 1.000、ppl 差 0（決定論的） |

- **これは HF との絶対照合ではない**。生成列の一致 anchor は `docs/perf/current.md` の MoE
  decode 節と research 記録にあり、数値の正しさの gate は未実施のまま（§8）
- split-KV が走る条件（context 2048 / 64 positions / score start 2049）での on / off A/B:
  ppl 31.754（split）/ 30.896（非 split、−2.70 %）、mean KL 0.00623、top-1 一致 0.984、
  top-5 overlap 0.966、max abs logit delta 1.216。reduction 順序の差として
  TP1 と TP2（mean KL 0.00494・top-1 0.9375）と同じ帯
- 同じ `quality` 経路の dense 側確認（Qwen3.8-27B-FP8、corpus 2203 token、context 128 /
  positions 32）: TP1 `QUALITY_PPL` 4.3905、TP2 4.5421（+3.45 %）、mean KL 0.00494、
  top-1 一致 0.9375、top-5 overlap 0.9563。TP1 と TP2 の差は reduction 順序によるもので、
  `quality` の `LoadedModel` 化で生じた変更ではない（`--tensor-parallel 2` は dense のみ受理）

## 2026-11-04 Qwen3.5-9B BF16 linear allowlist

`models/Qwen3.5-9B`（BF16）の hidden 4096 系 linear shape を
`kLinearShapeRules` に追加した。201 token prompt / 8 decode token / BF16 KV で、
変更前と変更後をそれぞれ 5 run 計測した。

| 項目 | 変更前 p50 | 変更後 p50 |
| --- | ---: | ---: |
| prefill ms | 12392.6 | 294.3 |
| decode tok/s | 3.77 | 13.69 |

- measurement revision: 変更後 `500367c`、変更前 `5b4885f`（target-only 経路は同一。
  変更前は MTP 無効の target-only build で計測）
- GPU: Radeon AI PRO R9700 × 1（`--device 0`）
- command: `shelf-compute --model-dir models/Qwen3.5-9B --input-ids-file <prose 201 tokens> --max-new-tokens 8 --max-seq-len 512 --device 0 --prefix-cache 0`
- prompt: `tests/data/mtp_perf/prose.txt` を Qwen3.5-9B の tokenizer で 201 token に切ったもの
- 生成一致 anchor（`GENERATED_IDS` の sha1）: `db823356300f09e2258f9f9615ffd0df7abebb74`
- 変更前の decode は 5 run 中 1 run が 2.48 tok/s、他は 3.77–3.78。変更後は
  13.62–13.70 で安定。
- 変更前は `SHELF_LINEAR_DEBUG=1` で 7 shape の `LINEAR_SELECT_MISS` が出ていた。
  変更後は通常 decode で miss が出ない。生成列は correctness mode と一致する。

## 2026-11-05 MoE fp8 KV cache

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8）を TP=1 / device 0 で、
KV cache を bf16 と fp8 E4M3 で比べた。測定向けに `shelf-bench tg` へ
`--kv-cache-dtype bf16|fp8` を追加している。fp8 は登録前は `PAGED_ATTENTION` `(16, 2, 256)` と
`KV_APPEND` `(2, 256)` の rule が無く correctness fallback だった。

| 条件 | fp8 | bf16 |
| --- | ---: | ---: |
| `tg --context 64 --tokens 32 --warmup 1`（gpu_ms / tok/s） | 495.81 / **64.54** | 497.27 / 64.35 |
| `tg --context 4096 --prefill-chunk 2048 --tokens 8` run1 / run2（tok/s） | **62.23 / 62.23** | 62.18 / 62.07 |
| census（context 4096、fp8） | logical 10840 = optimized 10840、**fallback 0**、physical 10928（split 80 + grouped 8） | 同一内訳 |
| `quality --context 2048 --positions 64` | ppl 4.4925（bf16 4.5247、**-0.71 %**）、KL 0.00695、top-1 0.9531、top-5 0.9625 | 4.5247 |
| `quality --context 512 --positions 64` | ppl 27.4009（bf16 29.6400、-7.55 %）、KL 0.07542、top-1 0.7969、top-5 0.8563 | 29.6400 |
| `quality --context 128 --positions 64` | ppl 94.3507（bf16 75.1460、**+25.56 %**）、KL 0.16576、top-1 0.8125、top-5 0.7781 | 75.1460 |
| KV 容量（`shelf-compute --max-seq-len 8192`、auto arena） | KV budget 10.07 GiB = **1,039,728 tokens** | 10.07 GiB = **527,968 tokens** |

- **上の quality 3 行と fp8 / bf16 の生成列差は、combine の shared expert gate stride 欠落が
  入った forward での計測で無効**。修正後は fp8 KV は bf16 と同じ帯に入り、context 128 の
  ppl 差 +25.56 % は観測されなくなった。修正後の値は
  [§2026-11-05 MoE combine 修正後の再測](#2026-11-05-moe-combine-修正後の再測)

- measurement revision: `feature/moe-fp8-kv` の fp8 rule 追加後（bf16 側は同 tree の再測）
- GPU: Radeon AI PRO R9700（`--device 0`、測定中は他の GPU に workload なし）
- command: `shelf-bench tg --mode greedy --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 --context 64 --tokens 32 --warmup 1 --kv-cache-dtype fp8`
- 生成一致 anchor: bf16 KV の `GREEDY_TOKEN_SUM` は context 64 で 2199（既存記録）、fp8 KV では
  **14441**（生成列が変わる）。context 4096 / 8 token は fp8 / bf16 とも `GREEDY_TOKEN_SUM=8`
  （id 1 の退化列で検出力が無い）
- dense `Qwen3.8-27B-FP8` の同条件（corpus 2203 token、`--positions 64`）: context 128 で
  bf16 12.3659 → fp8 11.3385（-8.31 %、KL 0.00504）、context 2048 で bf16 5.8388 → fp8 6.1002
  （+4.48 %、KL 0.00724）。**fp8 KV の短 context 劣化は dense 側には無い**
- fp8 KV の decode 速度は bf16 と同等。kernel 単体では可視 16384 で bf16 split 178.74 µs に対し
  fp8 split 173.85 µs、非 split は bf16 701.17 µs に対し fp8 785.36 µs（fp8 非 split kernel は
  自分の roofline の 27 %）。経緯と内訳は
  [../research/20261105/moe_fp8_kv.md](../research/20261105/moe_fp8_kv.md)

## 2026-11-05 MoE MTP speculative decode

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4` を TP=1 / bf16 KV / device 0 で、target-only と
`--mtp` を比べた。prompt は corpus から切った 1600 token、`--max-new-tokens 32`、
`--max-seq-len 4096`。combine の shared gate stride 修正後、かつ executor の keepalive
thread を削除した build の実測。

| 条件 | t/s | vs target | rounds | accepted drafts | mean accepted | verify rows | draft ms | verify ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target-only | 61.59 / 60.87 | 1.00x | - | - | - | - | - | - |
| `--mtp --mtp-drafts 1` | **83.83** | 1.37x | 16 | 14 | 0.875 | 32 | 43.42 | 309.89 |
| `--mtp --mtp-drafts 4` | **113.18** | **1.85x** | 9 | 22 | 2.444 | 43 | 86.89 | 186.93 |
| `--mtp --mtp-drafts 7` | 90.93 | 1.49x | 9 | 22 | 2.444 | 63 | 137.08 | 203.71 |

- 生成一致アンカー: 三条件の `GENERATED_IDS` は target-only の greedy 列と**完全一致**
  （`264,13759,506,3603,440,799,75727,13,561,25786,2942,513,220,16,11,220,17,11,220,20,11,321,220,16,15,4276,13,8771,628,5144,506,2957`）
- measurement revision: combine の shared gate stride 修正 + executor keepalive thread 削除
  （`feature-moe-mtp`）
- 直前 revision（keepalive thread を opt-in にして無効で動かした build）では
  61.56 / 61.44、84.80、115.83、93.37 t/s だった。削除後 build との差は 1.3〜2.6 % で
  GPU 共有下の run 間振れの範囲内
- verify 時間 / round は 2 行 19.4 ms、平均 4.8 行 20.8 ms、7 行 22.6 ms で行数に対して緩やかに増える。
  同じ model の 2 行 forward 単体（`shelf-bench pp --prefill-chunk 2` = 11.24 ms/token）と同じ帯で、
  **round に forward 外の費用は载っていない**
- draft 受理率は dense 対照（accepted/7 = 6.88 = 98 %、§2026-11-05 MTP decode）に対し 2.444 / 7 = 35 % のまま。
  それでも break-even を超えるため速度上の利点がある。drafts=4 の break-even:
  round 273.89 / 9 = 30.4 ms ÷ 16.2 ms/token = 1.88 token / round、実測 emitted/round = 3.444
- weight load 実測（MTP block 込み、`test_qwen35moe_model_load_real`）: load 28.6 s、
  arena 21.000 GiB（うち MTP expert 平面 0.399 GiB）、peak RSS 22.98 GiB
- 経緯（受理率の評価、verify round 費用の切り分け）は
  [../research/20261105/moe_mtp.md](../research/20261105/moe_mtp.md)、
  round 費用の原因は [../research/20261105/executor_keepalive_thread.md](../research/20261105/executor_keepalive_thread.md)
- 上記以外の条件（keepalive thread が既定 on だった build、および combine 修正前の forward
  で測った t/s 28.09 / 45.92 / 41.23 と 22.09 / 17.20 / 15.78）は無効
## 2026-11-05 MoE combine 修正後の再測

`moe_combine_kernel` が shared expert gate の平面を packed 前提で読み、**2 行目以降の token で
shared expert の寄与が落ちていた**（原因と修正は
[../research/20261105/moe_combine_gate_stride.md](../research/20261105/moe_combine_gate_stride.md)）。
この bug は multi-row forward（= prefill）に入るので、MoE の数値品質の実測と greedy 生成
アンカーはすべて修正前のものだった。修正後 build（`feature-moe-mtp`）・device 0・TP=1 で
再測した。corpus は `vendor/BetterBench/corpus/v1` を `tools/quality_gate.py corpus` した
2191 token（sha256 prefix `9136d38b`）。

`shelf-bench quality --positions 64` の `QUALITY_PPL`:

| context | bf16 KV | fp8 KV | fp8 差 | mean KL | top-1 一致 | top-5 overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 9.7973 | 9.7530 | **-0.45 %** | 0.008783 | 0.96875 | 0.934375 |
| 512 | 14.1440 | 14.4508 | +2.17 % | 0.017500 | 0.921875 | 0.921875 |
| 2048 | 4.1820 | 4.0059 | -4.21 % | 0.006101 | 1.0000 | 0.953125 |

- 修正前の同条件（**無効**）: context 128 で bf16 75.1460 / fp8 94.3507（+25.56 %）、
  context 512 で 29.6400 / 27.4009、context 2048 で 4.5247 / 4.4925。**fp8 KV の短 context
  劣化（+25.6 %）は bug の側で、修正後は fp8 は bf16 と同じ帯**に入っている
- `shelf-bench tg --mode greedy --tokens 32 --warmup 1 --prefill-chunk 2048`（2 run の
  gpu_ms と `GREEDY_TOKEN_SUM`）:

| context | bf16 KV | fp8 KV |
| ---: | ---: | ---: |
| 64 | 494.80 / 495.34 ms、64.67 / 64.60 tok/s、**976467** | 494.01 / 495.10 ms、64.78 / 64.63 tok/s、**976467** |
| 1024 | 512.99 / 541.72 ms、62.38 / 59.07 tok/s、**975006** | 515.70 / 543.79 ms、62.05 / 58.85 tok/s、**975269** |
| 2048 | 507.78 / 507.77 ms、63.02 / 63.02 tok/s、**975006** | 507.56 / 507.42 ms、63.05 / 63.06 tok/s、**904565** |

- decode throughput は kernel 作業が変わらないため修正前と同程度（context 64 の clean anchor
  で 497.27 → 494.80 / 495.34 ms）。**生成アンカーは変わった**（context 64 の
  `GREEDY_TOKEN_SUM` 2199 → 976467）。fp8 KV は context 64 で bf16 と同一の生成列になった
  （修正前は 14441 で不一致）
- 上表は executor の keepalive thread が既定 on だった build の実測。thread を削除した build で
  同じ条件を再測すると context 64 で 499.63 / 500.15 ms（64.05 / 63.98 tok/s）、context 2048 bf16 で
  512.11 / 512.04 ms（62.49 / 62.49 tok/s）で、`GREEDY_TOKEN_SUM`（976467 / 975006）を含む
  生成列は同一だった。差異は 1.0 % で GPU 共有下の run 間振れの範囲内。
  `shelf-bench pp --prompt-tokens 8192 --prefill-chunk 2048 --runs 3 --warmup 1` も
  3809.14 / 3762.39 tok/s で、上記の 3807 と同じ
- context 2048 の fp8 greedy は `GREEDY_FIRST_TOKENS=1,1,1,...`（id 1 の退化列）で検出力が
  無い。fp8 の比較としては上の quality 表が有効
- 行数をまたいだ teacher-forced logits の一致（walk 突合）の実測は
  [../research/20261105/moe_combine_gate_stride.md](../research/20261105/moe_combine_gate_stride.md) §3

## 2026-11-05 MoE tensor parallel（TP=2）

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8・moe_intermediate 512・
shared 512）を TP=2 / device 0,1 で測った。分割方針と契約は
[../developer/tensor_parallel_execution.md](../developer/tensor_parallel_execution.md) §11-1、
expert 軸を選んだ理由・数値破綻の経過・allowlist 登録の内訳は
[../research/20261105/moe_tp_expert_axis.md](../research/20261105/moe_tp_expert_axis.md) §6 / §7。

### prefill throughput（`shelf-bench pp --prompt-tokens 8192 --runs 3 --warmup 1`、median）

| prefill chunk | TP=1 bf16 KV | TP=2 bf16 KV | TP=2 / TP=1 |
| ---: | ---: | ---: | ---: |
| 1024 | 3755.3 | 6204.3 | **1.65 倍** |
| 2048 | 3782.8 | 6510.8 | **1.72 倍** |
| 4096 | 3618.7 | 6439.1 | **1.78 倍** |

- fp8 KV（chunk 2048）: TP=1 **3732.6** / TP=2 **6508.7**（1.74 倍）
- 上の値は `main`（EXL3 W16A16 系）を取り込み、EXL3 の実行 workspace を条件つき確保へ変えた
  tree の再測（measurement revision: `ef80238` 以降、required 168/168・gpu2 6/6）。同一条件の
  合体前の値は TP=1 3777.7 / 3785.0 / 3616.7、TP=2 6209.5 / 6543.6 / 6466.5 で、差は run 揺れの
  範囲（0.6 % 以下）。executor footprint は 1.31 → **1.33 GiB**（scratch 8192 → 3145728 float
  分）、KV pool は 18.86 → **18.64 GiB**（auto slack 0.20 GiB を確保した分）
- 上記は TP=2 の rank local shape を registered 側へ載せた後の値。載せる前は chunk 1024 / 2048 /
  4096 で **123.4 / 152.4 / 151.6 tok/s**（TP=1 の 1/25）で、backbone が correctness path へ
  落ちていた。kernel 別の倍率と登録した shape は research 記録 §7
- fallback 数（`SHELF_OP_FALLBACK_DEBUG=1`、`pp --prompt-tokens 512 --prefill-chunk 512 --runs 1`）:
  bf16 KV で TP=1 **0** / TP=2 **0**、fp8 KV でも TP=1 **0** / TP=2 **0**
- 上の TP=1 値は `pp` が mode にかかわらず最終 chunk の logits を計算するようになった tree の
  ものである。同じ tree で logits を計算しなかった測定は 3763.0 / 3740.6 / 3574.1（chunk 1024 /
  2048 / 4096 の median）で、差は run 揺れの範囲（1 % 未満）。logits 変更前の別 revision の
  記録値は 3662 / 3807 / 3631（§2026-11-04 Qwen3.6-35B-A3B MoE decode）
- barrier の費用は kernel 別倍率では読めない。collect 直後の op（`RESIDUAL_ADD`）に peer send /
  sum の待りが載り、per-rank 6.4 倍に見える（research 記録 §7）

### 数値一致（TP=1 vs TP=2、teacher-forced 4 位置、bf16 KV、同一 tree・同一 corpus）

corpus は `vendor/BetterBench/corpus/v1` を `tools/quality_gate.py corpus` した 2191 token
（sha256 prefix `9136d38b`）。

| context | 行ごとの最大 KL | top-1 一致 | 非有限 |
| ---: | ---: | --- | --- |
| 32 | 1.6e-3 | 4/4 | 0 |
| 64 | 3.6e-3 | 3/4 | 0 |
| 96 | 7.7e-3 | 4/4 | 0 |
| 128 | 1.4e-2 | 4/4 | 0 |
| 192 | 2.6e-3 | 4/4 | 0 |
| 256 | 4.9e-3 | 4/4 | 0 |

- 同じ tree で backbone を registered 側に載せる前は、同じ比較で最大 KL が ctx 96 で 8.2e-2、
  ctx 256 で 3.5e-2、top-1 不一致が 3 あった。**optimized kernel の方が TP=1 と近い**（correctness
  path は reduction 順序・精度が違い、registered 側は dense TP と同じ丸め経路を通る）
- 数値破綻（`NUMERIC_ERROR`）を直した直後の測定値と、破綻の切り分け・原因は
  [../research/20261105/moe_tp_expert_axis.md](../research/20261105/moe_tp_expert_axis.md) §6
- TP=1 側が prefill chunk を変えたときの KL 幅（1e-4 〜 1.4e-2）と同じ帯に入っている

### decode

- `shelf-bench tg` には `--tensor-parallel` が無いので tg では測れない。TP=1 の tg
  （`--mode greedy --context 64 --tokens 32 --warmup 1`、2 run）は **63.88 / 64.65 tok/s**、
  `GREEDY_TOKEN_SUM=976467`（combine gate stride 修正後の現行 anchor と一致）
- 合体後の tree で teacher-forced の ctx 192 を TP=1 vs TP=2 で測り直すと最大 KL **2.6e-3**・
  top-1 全一致・非有限 0 で、下表と同一の帯
- TP=2 の decode は `shelf-compute` E2E で測る（`--tensor-parallel 2` は `--prefix-cache 0` が
  必須）。prompt は上の corpus の 1200 番地以降 16 token（special token を除いたもの、
  `/tmp` 上の作業ファイルなので再現には同様のスライスを使う）、
  `--max-new-tokens 48 --max-num-batched-tokens 1024 --max-concurrent-requests 1 --max-seq-len 512`

| | TP=1 | TP=2 | DECODE_TOKENS |
| --- | ---: | ---: | ---: |
| run 1 | 64.76 | 71.39 | 47 |
| run 2 | 64.79 | 50.46 | 47 |
| `PREFILL_MS` | 41.8 / 41.8 | 49.0 / 46.5 |  |

  → **DECODE_TOKENS_PER_SEC +10 %**（run 2 の 50.46 は GPU 0 共有時の外れ値。合体前の同条件は
  TP=1 64.24 / 64.24、TP=2 71.84 / 70.53）。`GENERATED_IDS` は 48 token すべて TP=1 と TP=2 で
  一致した
- TP=2 の E2E は auto reserve の slack 修正前（row bucket 16）は
  `arena capacity exceeded` で起動できなかった。修正の経緯は
  [../research/20261105/moe_tp_expert_axis.md](../research/20261105/moe_tp_expert_axis.md) §5

## 2026-11-05 MoE pipeline parallel（PP=2）

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8）を device 0,1 の
PP=2（stage `[0,20)` / `[20,40)`）で測った。分割契約と stage model の interface は
[../developer/pipeline_parallel.md](../developer/pipeline_parallel.md) §2、
実装の経過と採否は
[../research/20261105/moe_pipeline_parallel.md](../research/20261105/moe_pipeline_parallel.md)。

command（`shelf-compute`。PP は `shelf-bench` に無いので E2E で測る。prompt は corpus
`9136d38b` の 100 番地以降で special token を除いた 512 / 1024 / 1943 token）:

```text
shelf-compute --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 \\
  --input-ids-file <prompt N ids> --max-new-tokens 32 --max-seq-len <N+64> \\
  --max-concurrent-requests 1 [--pipeline-parallel 2 --device 0,1]
```

| prompt token | prefill ms（PP=1） | prefill ms（PP=2） | decode tok/s（PP=1） | decode tok/s（PP=2） |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 142.8 / 142.6 | 148.4 / 145.4 | 63.01 / 62.97 | 57.70 / 57.82 |
| 1024 | 260.2 / 261.7 | 257.4 / 255.0 | 42.28 / 48.77 | 31.44 / 56.93 |
| 1943 | 600.2 / 579.0 | **487.3 / 487.3** | 58.63 / 58.66 | 55.13 / 55.11 |

- 生成列は PP=1 と PP=2 で**全 case 一致**（`GENERATED_IDS` 30〜31 token。`DECODE_TOKENS` は
  31 / 31 / 30）
- PP=2 の decode は 5〜8 % 遅い。stage 間 hand-off は pinned host 経由の直列で overlap が
  無い（1 batch を stage 順に通す）ため、1 step の壁時刻は stage 時間の合計になる
- prefill は 512 / 1024 で等価、1943 で +19 %（487.3 vs 579.0〜600.2 ms）。stage ごとに
  weight traffic は半分へ寄るが、直列なので上限は出ない。1024 の decode 側 run 値
  （31.44 / 56.93）は GPU 0 共有の揺れ
- 容量（1943 token case）: 1 stage あたり weight_bytes **11,003,576,320 B**（checkpoint
  24,600,713,248 B の layer 分割分）、`KV_POOL_RESERVED_BYTES` **20,746,240,000 B**。
  PP=1 では同一 GPU に全 weight が入り `KV_POOL_RESERVED_BYTES` **9,750,446,080 B** で、
  PP=2 は KV pool が **2.13 倍**。`ARENA_USED_GB` 30.157（PP=2、stage 每）/ 30.746（PP=1）、
  auto reserve 0.78 GiB（row bucket 1944、auto slack 0.20 GiB を含む）
- measurement revision: `a14a7ce`（stage model interface 導入後）で、required 168/168・
  gpu2 6/6。dense checkpoint の PP=2 は起動と生成を確認したが warmup + 複数 run の計測は
  していない
- 3 stage 以上は未検証（この環境は GPU 2 枚まで）。TP と PP の同時利用は非対応

## 記録規則

- 新しい計測をしていない数値は更新しない。
- 速度の数値は、生成結果が一致している rep からのみ採用する。一致アンカー
  （`GENERATED_IDS` の sha1、`GREEDY_TOKEN_SUM` など）を同じ節に併記する。
  `GENERATED_IDS` の sha1 は次で取る。

  ```sh
  grep '^GENERATED_IDS=' <output> | sed 's/^GENERATED_IDS=//' | sha1sum
  ```

- 速度に関する主張には計測条件（revision / commit / GPU / コマンド）を添える。
- 絶対値は GPU の温度・クロック状態でも振れる。同一条件の交互測定と run 間中央値で
  判定し、日付の違う値を直接比較しない。
- ここに無い性能値が必要な場合は推測せず、methodology の手順で新規計測し、
  日付・revision・commit とともに追記する。

# MXFP4 活性の producer 融合（non-spec prefill / decode）

実験 ID: `MXFP4-PRODUCER-1`

`models/Qwen3.8-27B-Quark-AWQ-MXFP4` の非投機 prefill / decode で、
活性の MXFP4 量子化を producer kernel に畳み、quantize の launch と
中間テンソルの往復を削った記録である。

## 1. 背景

profile（device 1、bf16 KV、`SHELF_QWEN35_PERF_STATS=1` と rocprofv3）で、
1 token あたりの GPU 時間は 30.24 ms、その内訳は次だった。

| kernel | us/token | 備考 |
| --- | ---: | --- |
| `gemv_mxfp4_w4a8_kernel` | 22,445 | 12.9 GB の重み、576 GB/s |
| `activation_quantize_mxfp4_rowref_kernel` | 2,402 | 256 launch、平均 9.4 us |
| `lm_head_int8_w32_kernel` | 2,130 | 1.27 GB、596 GB/s |
| `gdn_recurrence_wmma_lossy` | 769 | 48 launch |
| `rmsnorm_bf16_pf_oneplus` | 627 | 128 launch |
| `attention_paged_bf16` | 530 | 16 launch |
| その他 elementwise / conv / rope | 1,340 | |

prefill 2048 token は 923.6 ms で、`LINEAR_MXFP4_W4A8` 674 ms（148 TFLOPS、
FP8 WMMA の実測上限に近い）、`GDN_RECURRENCE` 69.5 ms、
`ACTIVATION_QUANTIZE_MXFP4_ROWREF` 36.2 ms、`PAGED_ATTENTION` 34.2 ms、
残りが elementwise だった。

量子化 kernel は 1 row あたり 1 block で、128 K group ごとの E8M0 を求め、
row の最大 exponent を参照値として code を折り畳む。decode では 1 row しか
入力がないため block 数が 1 になり、6〜18 us が launch と直列依存に費やされる。
prefill でも実効 316 GB/s で、DRAM 640 GB/s の半分だった。

## 2. 実装

`src/shelf/kernels/fused/mxfp4_activation_producer.hip` に 4 つの fused kernel を足し、
`optimized_dispatch.hip` で隣接する dispatch を見て置き換える。

| 置き換え | 融合する dispatch | 削減 launch |
| --- | --- | ---: |
| `mxfp4_residual_norm_quantize` | RESIDUAL_ADD + RMS_NORM + QUANT | 3 → 1 |
| `mxfp4_norm_quantize` | RMS_NORM + QUANT | 2 → 1 |
| `mxfp4_swiglu_quantize` | SWIGLU + QUANT | 2 → 1 |
| `mxfp4_gdn_gated_quantize` | RMS_NORM + SILU + MUL + QUANT | 4 → 1 |

すべて `ACTIVATION_QUANTIZE_MXFP4_ROWREF` と同じ row-reference folded 形式
（E4M3 code を GEMM の fragment 配置に書き、`scales[row]` に 2^ref、
残りの group slot は 0）を書く。数値は個別 kernel の連結と bit 一致する。
`tests/kernels/optimized/test_mxfp4_activation_producer.hip` が
rows 1 / 5 / 16 / 128 で codes と scales を memcmp する。

rows <= 32（decode / verify）は 1024 thread、それ以上（prefill）は 256 thread で
起動する。quantize 単体でも rows <= 32 を 1024 thread にすると decode が
30.7 → 31.7 tok/s になったため、同じしきい値を fused kernel にも使う。

dispatch 側は、量子化の入力が producer の出力だけに読まれ、quantize の
workspace slot が後続の `LINEAR_MXFP4_W4A8` と一致することを確認してから置き換える。
`SHELF_DISABLE_MXFP4_PRODUCER=1` で分離経路に戻せる。

## 3. 測定

`models/Qwen3.8-27B-Quark-AWQ-MXFP4`、R9700 1 枚（`--device 1`）、bf16 KV。
`SHELF_DISABLE_MXFP4_PRODUCER` を交互に切り替え、各 3 回の p50。

```sh
./build/shelf-bench pp --model-dir models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --device 1 --prompt-tokens 2048 --prefill-chunk 2048 \
  --kv-cache-dtype bf16 --warmup 1 --runs 3
./build/shelf-bench tg --mode greedy --model-dir models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --device 1 --context 512 --tokens 128 --prefill-chunk 512 \
  --kv-cache-dtype bf16 --warmup 1
```

| 断面 | 分離（off） | 融合（on） | 差 |
| --- | ---: | ---: | ---: |
| prefill 2048 p50 | 937.8 ms | 906.2 ms | -3.4 % |
| decode ctx512 p50 | 31.72 tok/s | 32.87 tok/s | +3.6 % |
| decode ctx512（quantize 1024 thread の前後） | 30.64 tok/s | 31.7 tok/s | +3.5 % |

E2E の単一リクエスト（prose 512 token、16 token 生成、`shelf-compute`）でも
30.86 → 32.02 tok/s、`GENERATED_IDS` は一致した。

fused 後の decode 内訳は 28.5 ms/token、launch 数は 1653 → 1165 だった。

## 4. 棄却した候補

- **GEMV の 64 output/block 化**（lane あたり 2 output、32 byte の連続 load）:
  decode 30.64 → 30.27 tok/s（-1.2 %）。並列度が下がる損失が load 幅の利得を上回った。
- **prefill の quantize を 1024 thread で起動**: prefill 928 → 951 ms（-2.4 %）。
  prefill は 1 row あたりの要素数に対して thread が過剰になり、occupancy が落ちる。
- **producer 融合を swiglu だけ 256 thread で起動**: 24.5 us/launch かかり、
  分離（swiglu 1.8 us + quantize 18 us）より遅い。1024 thread にすると 10.2 us になる。

## 5. 残っている律速

fused 後も prefill 906 ms のうち `LINEAR_MXFP4_W4A8` が約 696 ms を占める。
2048 row の実測は 17408:5120 で 160 TFLOPS、6144:5120 で 186 TFLOPS で、
これ以上の tiling 変更は register 圧迫で成立しない。
`GDN_RECURRENCE` は 71.9 ms（48 layer、1.5 ms/layer）で、chunk が 32 token の
直列 scan のため並列度が低い。q/k を v_head ごとに 6 回読む構成もあり、
chunk-parallel scan への置き換えは未実施である。
`PAGED_ATTENTION` は 34.8 ms（16 layer）で BF16 WMMA の約 25 % にとどまる。

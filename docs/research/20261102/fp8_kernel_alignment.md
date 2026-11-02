# FP8 kernel を radiance (libr4d) と突き合わせて修正した記録

日付: 2026-11-02
対象: `models/Qwen3.8-27B-FP8` / gfx1201 (R9700) / ROCm 7.1
比較先: `../references/radiance`（`libr4d/` の device kernel 群。gfx12 専用ビルド）
前提: 変更はすべて FP8 path（weight codes / activation codes / W8A8 GEMM / activation quantize）。
方針は「radiance と違う点を直す。Prefill の tok/s が下がっても spill は 0 を保つ」。

## 変更をいれたもの

### F8A-1 WMMA fragment order を 16 row x 64 K block へ（ACCEPT）

radiance `libr4d/r4d_fp8_frag.h` は fragment を 4 本（16 row x 64 K = 1024 B）単位で
置き、lane が 32 B 連続、wave が 1 KiB 連続要求になると書いてある。
「1 fragment（16 row x 16 k = 256 B）だと wave の要求が 256 B になり、LDS staging
にも負ける。4 fragment が全 shape で勝つ」と実測根拠付きで記録されている。

Shelf の codes 配置は 16 row x 32 K chunk（lane 8 B、wave 要求 256 B x 2）だった。
これを 16 row x 64 K block、lane slot 32 B へ変更した。

```text
block = 16 rows x 64 K = 1024 bytes
byte(row, k_local) = (row + ((k_local & 15) >> 3) * 16) * 32 + (k_local >> 4) * 8 + (k_local & 7)
codes_row_stride_bytes = (padded_k / 64) * 1024 = padded_k * 16   (総バイト数は不変)
```

定義は `include/shelf/quantization/fp8/fp8_frag.h` に置き、host preshuffle
（`preshuffle_fp8_native`）、activation quantize kernel、correctness path、W8A8 GEMM
kernel が同じ式を使う。on-disk の canonical layout
（`fp8_e4m3_row_major_block128_v1`）は変えていない。

`padded_k` は 64 の倍数が必要になった。`Int8ActivationWorkspaceLayout::k_padded_for`
は 32 align から 64 align へ変更。production の weight `k_padded` は 128 倍数なので影響なし。

結果（`shelf-bench gemm --dtype fp8 --impl auto`、device 1、old/new 交互 2 rep、p50）:

| shape (n x k) | rows | 変更前 us | 変更後 us | 差分 |
| --- | ---: | ---: | ---: | ---: |
| 5120 x 5120 | 1 | 26.67 | 19.37 | −27.4% |
| 5120 x 5120 | 8 | 26.82 | 22.14 | −17.4% |
| 5120 x 5120 | 64 | 51.70 | 42.89 | −17.0% |
| 17408 x 5120 (gate/up) | 1 | 150.4 | 144.2 | −4.1% |
| 17408 x 5120 (gate/up) | 512 | 485.1 | 465.8 | −4.0% |
| 17408 x 5120 (gate/up) | 2048 | 2661 | 2555 | −4.0% |
| 5120 x 17408 (down) | 1 | 165.1 | 149.1 | −9.7% |
| 5120 x 17408 (down) | 64 | 184.8 | 174.2 | −5.7% |
| 6144 x 5120 (q_proj) | 1 | 28.57 | 21.25 | −25.6% |
| 6144 x 5120 (q_proj) | 512 | 173.6 | 181.0 | +4.3% |

L3 に乗る小 shape（5120 x 5120 / 6144 x 5120）で大きく改善。DRAM 帯域まで張っている
gate/up・down は −4〜10%。q_proj の rows=512（Prefill2D wide）だけ +4.3% 悪化した。
 Prefill の悪化は許容する方針なので採用。

### F8A-2 block scale の fold 周期（ACCEPT）

radiance は 128 K 分を unscaled で accumulate し、block 末尾で 1 回だけ scale を
畳む（inner loop に scale 演算を入れない）。
Shelf の decode 系 kernel（`gemm_fp8_finegrained_w8a8_wmma_kernel`）は 32 K ごとに
fold していた（16 VALU / 2 WMMA）。64 K block へ変更したため fold は 64 K ごと
（16 VALU / 4 WMMA）になった。prefill wide kernel は元から 128 K ごと（`kDeferScale`）。

128 K への延長（`raw` accumulator を追加）も試したが、F8A-5 の local memory 問題と
同時に検証できたため、64 K で留めた（MB4 kernel の VGPR 181 / spill 0）。
gfx1201 では WMMA と VALU が co-issue しない（radiance `r4d_fp8_wmma.h` の hazard 記録と
同じ結論）ので、fold VALU 本数はそのまま rate に効く。

### F8A-3 RowBlock8 の削除（ACCEPT）

`gemm_fp8_finegrained_w8a8_wmma_kernel<8>` は VGPR 256 / **vgpr_spill 75**
（private segment 168 B）だった。production selector
（`fp8_finegrained_row_block_config`）は最大で RowBlock4 を選ぶため、
この config は shelf-compute / shelf-server から到達不能だった。
bench と test から選んでいたため、enum ごと削除した。

### F8A-4 spill / private segment の contract test（ACCEPT）

`tools/check_kernel_spills.py` を追加。build の `compile_commands.json` から実フラグを
使い、FP8 kernel を assembly へコンパイルして `.vgpr_spill_count` /
`.sgpr_spill_count` / `.private_segment_fixed_size` を読む。
CTest `test_fp8_kernel_spill_contract`（label `cpu;required`）として登録。

計測後（すべて spill 0 / private 0）:

| kernel | VGPR | LDS |
| --- | ---: | ---: |
| `gemm_fp8_finegrained_w8a8_wmma_prefill_2d_wide_kernel<true>` | 144 | 1024 |
| `gemm_fp8_finegrained_w8a8_wmma_prefill_2d_wide_kernel<false>` | 159 | 1024 |
| `gemm_fp8_finegrained_w8a8_wmma_prefill_2d_kernel` | 164 | 0 |
| `gemm_fp8_finegrained_w8a8_wmma_kernel<4>` | 181 | 0 |
| `gemm_fp8_finegrained_w8a8_wmma_kernel<2>` | 98 | 0 |
| `gemm_fp8_finegrained_w8a8_wmma_kernel<1>` | 80 | 0 |
| `activation_quantize_fp8_finegrained_*` | 26 | 0 |
| `attention_paged_fp8_impl` | 67 / 63 | 8800 |
| `attention_paged_split_fp8_impl` | 145 / 200 | 21144 / 31448 |

変更前の同一 kernel 群は MB4 = 192 / MB2 = 135 / MB1 = 76 VGPR、
prefill wide = 210 / 182 VGPR だった。断片配置の変更で address 計算が減り、
全 kernel で VGPR が下がった。

## 棄却したもの

### F8A-5 Frag4 構造体 / device helper で断片を束ねる書き方（REJECT）

`struct Frag4 { int32x2_t f0..f3; }` を `load_frag4()` で返して配列に受けると、
LLVM が SROA できず **private segment 96 B (MB1) / 160 B (MB2) / 288 B (MB4)** が
発生した。`.vgpr_spill_count` は 0 だが ISA 内に `scratch_load_b128` / 
`scratch_store_b128` がループあたり数百本入り、実質 local memory トラフィックになる。

`int32x4_t` 2 本の値とマクロ展開（`FP8_LOAD4` / `FP8_WMM4`）に置き換えたところ
private segment は 0、命令数も減った。集約型を cross-inlining 境界で受け渡さない、
という radiance の kernel 群と同じ運用にしている。

### F8A-6 paged attention fp8 の E4M3 LUT を hardware convert へ（REJECT）

radiance は `v_cvt_f32_fp8` / `v_cvt_pk_f32_fp8`（2 code / 1 命令）で E4M3 を復号し、
LUT を使わない。Shelf の fp8 decode attention は LDS の 256 entry bf16 LUT を
使っている。LUT を `__builtin_amdgcn_cvt_pk_f32_fp8` 1 code 版へ置き換えて測った
（`shelf-bench paged-attention --variant optimized --kv-dtype fp8`、qh 32 / kvh 4 /
hd 128 / pt 16 / rows 1、device 1、3 rep 中央値）:

| visible tokens | LUT (us) | hw convert (us) | 差分 |
| ---: | ---: | ---: | ---: |
| 128 | 15.0 | 17.6 | +17.2% |
| 1024 | 74.6 | 91.9 | +23.2% |
| 4096 | 250.5 | 322.6 | +28.8% |

LDS 512 B と LUT 初期化（256 変換）は消えるが、現状の load が 1 lane 1 byte
（`d = lane + 32 * j`）なので 1 命令 / 1 code になり、`v_cvt_pk_f32_fp8_e64` の
op_sel 版も挟まる。LUT 側の `ds_read_b16` + bf16 → f32 は VALU と別系統で
masked できるため勝つ。radiance が得をしているのは **2 code を 1 命令で復号できる
packed load**（V8 code tile、`byte_gather`）を先に作っているからで、
load 側を 2 byte / lane に変えずに convert だけ差し替えると負ける。
差し替えは撤回した。やるなら load / tile 再構成と同時に行うこと。

### F8A-7 `__hip_cvt_fp8_to_halfraw` を production で使わない（確認）

`shelf::quantization::codec::device::e4m3_decode_f32` は gfx1201 では software 変換で、
分岐込み 20 命令前後だった。hardware 版（`v_cvt_pk_f32_fp8_e32`）は 1 命令。
LUT 初期化と reference path にしか使われていないので、production path には影響ない。
reference path の速度改善候補として残す。

## 生成結果が変わっていない確認

- required acceptance: 129/129 PASS（`test_gemm_fp8_optimized` 283 case、
  `test_activation_quantize_fp8_finegrained` 312 case、`test_activation_quantize_tp_split`、
  `test_fp8_frag_layout`、`test_fp8_kernel_spill_contract` を含む）
- correctness path（`SHELF_QWEN35_KERNEL_MODE=correctness`）の logits は
  変更前後で bit 一致（top5 まで同一、max abs diff 0）。
  linear.inc の reader / writer 変更は配置のみで数値は不変であることの確認。
- `shelf-compute`（Qwen3.8-27B-FP8、TP=1、prompt 128 ids、24 生成、bf16 KV）:
  `GENERATED_IDS` は変更前後で完全一致。
  `PREFILL_MS` 124.9 → 117.4（−6.0%）、`DECODE_TOKENS_PER_SEC` 17.71 → 18.41（+3.9%）。
  prompt は意味を持たない id 列なので、この 2 値は performance baseline として
  `docs/perf/current.md` には記載しない（regression check のみ）。
- optimized path と correctness path の logit 差（この OOD prompt で top1 が別 token）は
  変更前から存在する。変更前の build でも opt = 220 / 2.815、correctness = 3966 / 7.1174
  で同一の差が出た。fp8 GEMM 自体は `test_gemm_fp8_optimized` の exact host 比較で
  worst rel 5.7e-3 に収まっている。

## radiance にあって Shelf に残っている差分（今回の対象外）

- split-K と last-arriver reduce（独立 reduce launch を消す。radiance は
  dependent dispatch を 1 本 3.5-6.4 us と見積もっている）
- prefill の activation code を LDS へ double buffer staging + pitch +8 の
  bank conflict 回避（radiance は 128 B read / 32 B store per WMMA を目標にしている）
- `__attribute__((amdgpu_waves_per_eu(n, n)))` による occupancy pinning
  （compiler release で scheduler の選択が動くproblemへの対策）
- `__syncthreads()` の代わりに LDS scope fence（`global_inv` で L0/L1 が
  flush される問題）、`__builtin_amdgcn_sched_group_barrier`
- attention prefill の log2 domain softmax + octave 超過時のみ rescale
- weight preload kernel、chained launch（AQL barrier を落として device 側 counter で待つ）
- fp8 lm_head の行内 scale 配置（`[K codes][K/128 scale][pad]`）

# QSA 長文 decode の top-k タイル選択と decode 経路

測定日: 2026-11-11、R9700 (gfx1201) × 2、TP=2。
対象: `src/shelf/kernels/optimized/qsa.hip`、`src/shelf/kernels/optimized/attention/paged_decode.hip`、
`src/shelf/inference/execution/paged_attention_dispatch.hip`、
`models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`。
比較: `1c5376f`（変更前）と `2407d24`（変更後）。

## 結論

長文 decode のコンテキスト依存コストを 65K で切り分けたところ、増分 17.8 ms/token
（両 rank 合計）のうち 15.4 ms（87%）が `qsa_topk_kernel` だった。dot 積を計算する
`qsa_score_wmma_kernel` は 0.54 ms（3%）、decode attention は 1.79 ms（10%）である。

原因は 2 つあった。

- 8192 block を超えると、top-k は 512 候補ごとに bitonic sort と merge を繰り返す経路に入る。
  1 block tile（8192 block）当たり約 258 µs かかり、radix select の約 3 倍だった。
- `qsa_score_wmma_kernel` が tile 2 以降で `args.block_begin` を使わず、常に pooled
  block 0 から読んでいた。32K を超える context では選択する block がずれていた。

tile ごとの radix select と 1 回の merge に変更し、score の block offset を修正した。
さらに decode の top-k を 512 thread で走らせ、BF16 KV の decode で選択 page を直接
列挙するようにした。FP8 KV の decode は 65K で 20.48 → 23.85 tok/s（+16.4%）、
128K で 21.60 → 22.99 tok/s、BF16 KV は 65K で 21.47 → 24.03 tok/s（+11.9%）に
なった。8K 以下は変化しない。

## 計測

`shelf-bench tg --mode forward --context <N> --tokens 32 --prefill-chunk 2048
--kv-cache-dtype <fp8|bf16> --device 2,3 --tensor-parallel 2 --warmup 1`。
32 token の p50、tok/s（括弧内は ms/token）。

FP8 KV:

| context | `1c5376f` | `2407d24` |
|---|---:|---:|
| 128 | 26.45（37.81） | 26.46（37.79） |
| 8K | 25.02（39.96） | 25.09（39.86） |
| 32K | 22.67（44.10） | **24.47（40.87）** |
| 65K | 20.48（48.84） | **23.85（41.92）** |
| 128K | - | 22.99（43.49） |

BF16 KV:

| context | `1c5376f` | `2407d24` |
|---|---:|---:|
| 8K | 25.30（39.52） | 25.24（39.59） |
| 65K | 21.47（46.55） | **24.03（41.61）** |
| 128K | - | 23.16（43.17） |

8K からの増分は、FP8 KV の 65K で 8.87 → 2.06 ms/token、BF16 KV の 65K で
7.03 → 2.02 ms/token になった。

32K の改善は、32 token の decode が position 32768 以降で走るため、最初の token から
複数 tile 経路に入ることで説明できる。32768 と 32772 の A/B では差が出ない。
どちらも decode 区間が境界を越えており、比較したい経路の切り替えを分離できない。

## カーネル内訳

65K、8 token の decode 区間、両 rank 合計、ms/token。

| kernel | `1c5376f` | `2407d24` |
|---|---:|---:|
| `qsa_topk_kernel` | 16.28 | **2.42** |
| decode attention | 7.78 | 6.90 |
| `qsa_score_wmma_kernel` | 1.09 | 1.23 |
| `shelf_gdn_recurrence_wmma_lossy` | 0.99 | 0.99 |
| その他（MoE / EXL3 / elementwise） | 50.69 | 50.69 |
| 合計 | 76.84 | **62.23** |

`qsa_topk_kernel` の 1 起動（8192 block）は約 258 µs から約 38 µs になった。
131K でも 1 起動 40.6 µs、4.50 ms/token である。decode attention は 8K 6.02、
65K 6.90、131K 6.71 で、差は run 間のばらつきの範囲に収まる。

## 変更

- `qsa_topk_kernel`
  - 各 tile の score を radix select で上位 k に絞り、直前までの上位 k と 1 回だけ merge する。
    512 候補ごとの sort と merge の繰り返しを除去した。
  - batch 行数が `kQsaTopkSmallRows` 以下なら 512 thread で起動する。decode は
    1 行 1 CTA なので、128 thread では GPU の大部分が遊んでいた。
- `qsa_radix_select`
  - 選択した score と block index を同時に書き出せるようにした。
- `qsa_score_wmma_kernel`
  - pooled key の読み出しに `args.block_begin` を加算した。scores は tile 先頭を 0 とした
    相対位置のままなので、workspace の幅は変わらない。
- `attention_paged_bf16_impl`
  - 全 page の走査と bitmap による skip をやめ、FP8 経路と同じく選択 page を
    wave へ分配して直接列挙する。
- `resolve_paged_attention_physical_impl`
  - mixed batch でも `block_mask` 付きの prefill 要求が prefill kernel を使うようにした。
    条件は単一 request と同じ `kMaskedPrefillMinRows`（256 行）である。
- `test_qsa_select`
  - `blocks = 12000` を 2 分割・3 分割するケースを追加した。CPU 参照の top-k、
    同点時の小さい index 優先、bitmap の tail block を比較する。

## 検証

- `test_qsa_select`: PASS（追加した複数 tile ケースを含む）
- `test_qsa_attention_mask`: PASS
- `test_qsa_key_path`: PASS
- `test_paged_attention`: 56 passed
- `test_qwen4exp_qsa_site`: 11 passed
- `test_qwen4exp_qsa_layer`: 7 passed
- mixed batch: `shelf-bench batch --requests 2 --prefix-tokens 512` の kernel trace で
  `v_prefill_bf16::attention_paged_prefill_bf16_impl<0u>` が使われ、decode kernel が
  起動しないことを確認した。

score offset の修正により、32K 超の選択結果は変更前と異なる。変更前の 32K 超は
pooled block 0 側の score で別の block を選んでいたため、出力の一致は比較できない。

## prefill

FP8 KV、`shelf-bench pp --prefill-chunk 2048 --warmup 1 --runs 1`。

| prompt | 修正後 |
|---|---:|
| 32K | 2,811 tok/s |
| 64K | 2,622 tok/s |
| 128K | 2,419 tok/s |

128K の kernel trace では prefill attention が GPU 時間の 18.4%、`qsa_score` が 11.0%、
`qsa_topk` が 3.4% だった。page 単位の和集合は最大 4 倍の token を WMMA に渡すが、
32K から 128K まで share はほぼ一定で、崖は出ていない。

## 残る課題

- decode の top-k は 65K で 2.42 ms/token、131K で 4.50 ms/token。radix select は
  全 block を数回走査するため、なお block 数に比例する。境界 bucket の候補を圧縮して
  残りの pass を小さくする案が残っている。
- prefill attention は選択 page の 16 token をまとめて WMMA に渡すため、4 token block
  の選択が page 内で疎でも 16 token 分を処理する。token 粒度で K/V を詰め直すには
  内側ループの再設計が要る。
- mixed batch の masked prefill は kernel 選択のみ変更した。multi-request の
  end-to-end 品質は server 側の受入で確認する必要がある。
- BF16 の split-KV decode は担当範囲の全 page を走査する。Flash-Next の
  `q_heads / kv_heads = 12` では split の instantiation が無いため、現状この経路は
  使われない。

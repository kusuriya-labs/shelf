# QSA 長文 decode の top-k タイル選択

測定日: 2026-11-11、R9700 (gfx1201) × 2、TP=2。
対象: `src/shelf/kernels/optimized/qsa.hip`、`models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`、
`--kv-cache-dtype fp8`。
比較: `1c5376f`（変更前）と `3e81dd6`（変更後）。

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
decode は 65K で 20.48 → 23.04 tok/s（+12.5%）、1 token 当たりの追加コストは
8.87 → 3.45 ms（-61%）になった。8K 以下は変化しない。

## 計測

`shelf-bench tg --mode forward --context <N> --tokens 32 --prefill-chunk 2048
--kv-cache-dtype fp8 --device 2,3 --tensor-parallel 2 --warmup 1`。
32 token の p50、tok/s（括弧内は ms/token）。

| context | `1c5376f` | `3e81dd6` |
|---|---:|---:|
| 128 | 26.45（37.81） | 26.43（37.83） |
| 8K | 25.02（39.96） | 25.03（39.95） |
| 32K | 22.67（44.10） | **23.99（41.68）** |
| 65K | 20.48（48.84） | **23.04（43.40）** |
| 128K | - | 21.60（46.30） |

8K からの増分は、変更前が 65K で 8.87 ms/token、変更後が 3.45 ms/token である。

32K の改善は、32 token の decode が position 32768 以降で走るため、最初の token から
複数 tile 経路に入ることで説明できる。32768 と 32772 の A/B では差が出ない。
どちらも decode 区間が境界を越えており、比較したい経路の切り替えを分離できない。

## カーネル内訳

65K、8 token の decode 区間、両 rank 合計、ms/token。

| kernel | `1c5376f` | `3e81dd6` |
|---|---:|---:|
| `qsa_topk_kernel` | 16.28 | **5.51** |
| decode attention | 7.78 | 6.95 |
| `qsa_score_wmma_kernel` | 1.09 | 1.24 |
| `shelf_gdn_recurrence_wmma_lossy` | 0.99 | 1.00 |
| その他（MoE / EXL3 / elementwise） | 50.69 | 50.95 |
| 合計 | 76.84 | **65.65** |

`qsa_topk_kernel` の 1 起動（8192 block）は約 258 µs から約 90 µs になった。
8K（2048 block、radix select）の 35 µs と比べても、block 当たりのコストは同等以下である。

## 変更

- `qsa_topk_kernel`
  - 各 tile の score を radix select で上位 k に絞り、直前までの上位 k と 1 回だけ merge する。
    512 候補ごとの sort と merge の繰り返しを除去した。
  - 512 要素の sort と 1024 要素の merge は tile 当たり 1 回になる。
- `qsa_radix_select`
  - 選択した score と block index を同時に書き出せるようにした。
- `qsa_score_wmma_kernel`
  - pooled key の読み出しに `args.block_begin` を加算した。scores は tile 先頭を 0 とした
    相対位置のままなので、workspace の幅は変わらない。
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

score offset の修正により、32K 超の選択結果は変更前と異なる。変更前の 32K 超は
pooled block 0 側の score で別の block を選んでいたため、出力の一致は比較できない。

## 残る課題

- top-k は依然として最大のコンテキスト依存項である。65K で 5.51 ms/token、
  131K で 10.25 ms/token。radix select は全 block を数回走査するため、
  block 数に比例する。threads の増加、threshold に同点がない場合の index pass の省略、
  score と選択の融合が候補になる。
- decode attention は 65K で 6.95 ms/token。page が 16 token、圧縮 block が 4 token の
  ため、選択 block の page を処理すると最大 4 倍の token を WMMA に渡す。
  また `q_heads / kv_heads = 12` では split-KV の instantiation が無く、
  q head ごとに同じ K/V を読み直す。
- mixed batch では `block_mask != nullptr` の prefill 要求が decode 経路に落ちる
  （`paged_attention_dispatch.hip`）。単一 request では prefill 経路を使う。
- prefill は最大 32 行の選択 page の和集合を WMMA に渡す。選択が行間で分散すると
  密な prefill に近づく。

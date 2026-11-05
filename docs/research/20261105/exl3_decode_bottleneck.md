# EXL3 W16A16 decode のボトルネック

## 条件

- model: `models/Qwen3.8-27B-exl3`（mul1、4 bpw、lm_head 6 bpw）
- GPU: Radeon AI PRO R9700 1 枚（`--device 1`、他プロセスなし）
- build: `2504163` の tree（W16A16 fused kernel、multi-warp 化前）
- decode 測定: `shelf-bench tg --mode forward --context 128 --tokens 16 --warmup 1`
- per-op 測定: `SHELF_OP_DUMP=1 SHELF_OP_SYNC=1`

## 全体

3 run の decode は **1.32 / 1.33 / 1.34 tok/s**（747〜756 µs/token）だった。

per-op の内訳（1 decode step、`SHELF_OP_SYNC=1`）:

| kernel | 回数 | 合計 | 割合 |
| --- | ---: | ---: | ---: |
| `LINEAR_EXL3_W16A16` | 401 | 766.5 ms | 95.8 % |
| `RMS_NORM` | 209 | 6.7 ms | 0.8 % |
| `LINEAR_BF16` | 96 | 4.8 ms | 0.6 % |
| `RESIDUAL_ADD` | 128 | 4.0 ms | 0.5 % |
| その他 | — | 18.0 ms | 2.3 % |

decode は EXL3 linear だけで決まっている。

## shape 別

401 launch の内訳（代表 run、合計 781.7 ms）:

| n | k | 回数 | 合計 | 1 launch | 実効重み帯域 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5120 | 17408 | 64 | 283.0 ms | 4.42 ms | 10.1 GB/s |
| 17408 | 5120 | 128 | 180.4 ms | 1.41 ms | 31.6 GB/s |
| 5120 | 6144 | 64 | 105.5 ms | 1.65 ms | 9.5 GB/s |
| 6144 | 5120 | 48 | 68.5 ms | 1.43 ms | 11.0 GB/s |
| 10240 | 5120 | 48 | 66.6 ms | 1.39 ms | 18.9 GB/s |
| 1024 | 5120 | 32 | 44.7 ms | 1.40 ms | 1.9 GB/s |
| 12288 | 5120 | 16 | 21.9 ms | 1.37 ms | 22.9 GB/s |
| 248320 | 5120 | 1 | 11.1 ms | 11.12 ms | 85.7 GB/s |

全体の実効重み帯域は **16.8 GB/s**。DRAM の実測 636〜640 GB/s の約 2.6 %。
もっとも条件の良い lm_head でも 85.7 GB/s で、DRAM の 13 %。

## 読み取れること

1. **k に比例し、n にはほぼ依存しない。**
   k=5120 の n=1024 / 6144 / 10240 / 17408 はすべて 1.37〜1.43 ms/launch。
   n が 17 倍違っても launch 時間が同じ。
   k=6144 は 1.65 ms、k=17408 は 4.42 ms で、ほぼ k に比例する。
   これは DRAM 帯域ではなく、**1 block の k ループの latency** が支配していることを示す。

2. **1 block = 1 warp で、8 個の n subtile を直列に処理している。**
   128 K ごとに 8 ks × 8 subtile = 64 tile を順に復号する。
   各 tile は ring の LDS staging、tile の LDS 書き込み、A fragment の読み出しの間に
   3 回の `lds_barrier()` を挟む。n が小さい形状では block 数も少なく、
   n=1024 では 8 block しかないため 56 CU が遊ぶ。

3. **lm_head だけは block 数が多いため帯域寄りになる。**
   n=248320 は 1940 block で、85.7 GB/s。それでも decode 全体の 1.4 % にすぎない。

4. **prefill は decode よりはマシだが同じ構造。**
   `pp --prompt-tokens 128` は 70.0 tok/s（14.3 ms/token）。
   16 行 block を使うため、row tile 数に比例して重みを復号し直す。

## 試して棄却したもの

- **ring の `%` を条件減算へ変更**（`lo`/`hi` が 2×word_count 未満であることを利用）:
  781.7 → 779.2 ms。差は 0.3 % で、測定揺れの範囲。**REJECT**。
- **ring を LDS ではなく warp shuffle で読む**（LDS ring と 2 barrier を削除、
  tile を double buffer 化、`__syncwarp()` へ置換）:
  692 → 1209 ms。shuffle が 1 window あたり最大 4 回必要で、LDS より遅い。**REJECT**。
- **rows=1 のとき入力回転を有効行だけに制限**:
  単独の A/B では有意差を確認できず。回転は全体の数 % 程度。**保留**。

## 次の候補

1. **multi-warp block**。8 warp を 1 warp = 1 n subtile（16 出力）に割り当て、
   ring / tile は warp ごとの LDS、同期は `__syncwarp()`、xrot と出力だけ block 同期にする。
   1 block の k ループ latency を約 1/8 にし、同時に warp 数で LDS/barrier の stall を隠す。
2. **rows == 1 専用 GEMV**。mul1 は `state * 0x83DCD12D` の byte 和に対してアフィンなので、
   exllamav3 の `exl3_gemv_int8` と同様に `dp4a` / `v_sad_u8` で 4 重みずつ復号・積和できる。
   現行の 1 重みずつの `__hfma` 復号より桁で速い可能性がある。
3. **小 n 形状の split-K**。n=1024 / 6144 / 10240 は block 数が少なく CU が遊ぶ。
   K を分割して部分和を reduction する。workspace と 2 段 launch が必要。
4. barrier の削減。LDS 自体は shuffle より速いので、ring staging と tile 書き込みの
   間の同期を warp 単位に落とせるかは、multi-warp 化と合わせて再評価する。

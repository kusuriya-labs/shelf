# QSA score WMMA の warp 重複除去

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
対象: `src/shelf/kernels/optimized/qsa.hip` の `qsa_score_wmma_kernel`。

## 結論

`qsa_score_wmma_kernel` は 4 warp が同じ 16x16 WMMA tile を重複して計算していた。
1 warp に 1 つの row tile（4 query row）を割り当て、WMMA fragment から直接
4 head の ReLU 和を取る形に変更した。32K prefill は修正前の 13.15–13.23 s に対し
12.74–12.78 s（約 3%）となった。8K の同時刻 A/B は 2,903 ms → 2,887 ms（0.6%）で、
score の対象ブロックが 4 分の 1 になるぶん差も小さい。

## 重複の内容

修正前は `mt` が全スレッド共通のループ変数で、warp ごとの割り当てがなかった。

```cpp
for (uint32_t mt = 0u; mt < 4u; ++mt) {   // 4 warp すべてが同じ mt を実行
    ...
    wmma_out[8u * hi + e][lo] = c[e];     // 同じ値を 4 warp が書き込む
}
```

`lane = threadIdx.x & 31u` で A/B fragment を作るため、4 warp は同一の
WMMA を 4 回実行していた。`wmma_out` を読む合成段も `t < 64` の 2 warp しか
使っておらず、残り 2 warp は待つだけだった。

WMMA の仕事量は CTA あたり 16 tile から 4 tile へ減る。同時に `wmma_out`
(1 KiB) への書き込みと読み直し、および `nt` ごとの `__syncthreads()` が不要になる。

## 変更

- `mt = threadIdx.x >> 5u` とし、warp ごとに 1 row tile を担当する。
- lane は 1 列 `lo` と 8 行 `8 * hi + e` を持つため、同じ lane 内で
  head 0..3 の和が取れる。`hi == 0` が r=0,1、`hi == 1` が r=2,3 を受け持つ。
- `score_tile` へは各 lane が直接 2 行分を書く。`wmma_out` を削除した。
- `__syncthreads()` は chunk あたり 18 回から 3 回へ減った。

合成順序は修正前と同じで、`fmaxf` の和と `args.scale` の積も同一である。
`tests/unit/test_qsa_select.cpp` の split 経路が scalar 経路と一致することを確認した。

## リソース

| 項目 | 修正前 | 修正後 |
|---|---:|---:|
| VGPR | 92 | 84 |
| SGPR | 28 | 27 |
| LDS | 30,720 B | 29,696 B |
| spill | 0 | 0 |
| 命令数 | 545 | 389 |

以前の計測記録にある VGPR 248 は古いビルドの trace 値であり、現行カーネルには
当てはまらない。

## 性能

FlashNext 3.05 bpw、`--prefill-chunk 2048 --kv-cache-dtype fp8 --device 2,3
--tensor-parallel 2 --warmup 1 --runs 3`。

| prompt | 修正前 | 修正後 |
|---|---:|---:|
| 8K（同時刻 A/B） | 2,903 ms / 2,822 tok/s | 2,887 ms / 2,838 tok/s |
| 32K（前セッションの baseline と比較） | 13,152 ms / 2,491 tok/s | 12,778 ms / 2,564 tok/s |

32K の修正前値は同じコード状態で別セッションに取得した median で、
修正後は run 1–2 が 12.74–12.78 s、run 0（load 直後）が 13.41 s だった。
32K の修正前後を同一セッションで取り直す作業は、並行して変更されている
作業ツリー（KV reserve が 3.44 GB に増加）の影響で中断した。

32K では QSA score が 738 ms（両 rank 合計、rocprof trace）から約 4 分の 1 に
なる計算で、実測差 0.37 s と整合する。8K は score の対象ブロックが 4 分の 1 で
差が小さい。

## 次の候補

- `qsa_key_cache_batch_kernel` は 1 request を 1 CTA で逐次処理しており、
  32K で 530 ms（両 rank 合計）。prefill の並列化余地が大きい。
- `qsa_topk_kernel` は 32K で 484 ms。bitonic sort + merge を
  radix select に置き換える余地がある。
- grouped EXL3 は gate/up（`n_tiles=1`）が 4,666 ms、down（`n_tiles=2`）が
  1,750 ms で、同じ FLOPs に対して gate/up が 2.7 倍遅い。

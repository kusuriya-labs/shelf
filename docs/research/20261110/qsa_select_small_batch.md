# QSA select の small batch 経路

測定日: 2026-11-10、R9700 (gfx1201) × 2、TP=2。
対象: `qsa_select_batch_kernel` の適用途絶と `qsa_score_wmma_kernel` の block tile。

## 結論

`qsa_select_batch_kernel`（fused 経路）は `args.rows < 256` で使われており、
decode と 128 token の prefill chunk でまだ遅かった。
FlashNext 3.05 bpw、8K context の `shelf-bench tg --tokens 32` で
qsa_select は 755.8 ms（2,304 launch、両 rank 合計）を占めていた。

split 経路（score WMMA + radix top-k）を small batch にも使い、
score kernel の block tile を batch 行数で変えることで、
QSA の合計を 893.8 ms から 375.2 ms に減らした。
decode の throughput は 16.44 → 17.88 tok/s（同一作業ツリーでの比較）。

## 適用条件

`try_launch_qsa_select` の split 条件から `args.rows >= 256` を外した。
引き続き `qsa_scores` workspace、`staged_requests`、4 head、head_dim 128 が
そろう場合だけ split を使う。fused カーネルは 8 head などの fallback として残す。

## block tile

`QsaScoreArgs` に `block_tile` を追加し、score kernel の 1 CTA が担当する
block 数を起動時に決める。

| rows | block_tile | 8K の score |
|---|---:|---:|
| 1024（従来） | 1024 | 635.4 ms |
| 128 | 128 | 425.1 ms |
| 1–63（decode） | 64 | 411.3 ms |
| 64–255（128 token chunk） | 256 | **154.7 ms** |

8K context の `tg` は 128 token の prefill chunk を使うため、
rows=128 の tile を 256 にすると CTA 数が 16 → 64 になり大きく改善する。
decode（rows=1）は tile 64 で 16 CTA になる。

## 測定

8K context、32 token、`shelf-bench tg --mode forward`、両 rank 合計 GPU 時間:

| 経路 | qsa_select / score | top-k | 合計 |
|---|---:|---:|---:|
| fused（従来） | 755.8 ms | — | 893.8 ms |
| split + tile 1024 | 635.4 ms | 81.6 ms | 846.2 ms |
| split + tile 128 | 425.1 ms | 81.7 ms | 637.0 ms |
| split + tile 64 / 256 | **154.7 ms** | 81.7 ms | **375.2 ms** |

この作業ツリーは MTP 周りの変更を含むため、tok/s の絶対値は参考値。
同じツリーでの fused 16.44 tok/s に対し split は 17.88 tok/s。

## 注意

- key cache の並列化は `actual_rows >= 256` のままなので、decode と
  128 token chunk では従来の逐次カーネルが使われる。
- top-k は単発選択なので radix select が効く。decode でも 81.7 ms。

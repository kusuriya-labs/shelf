# BF16 small-rows GEMM の kernel 選択（ACCEPT）

## 背景

マルチリクエスト decode と投機 verify は `rows = リクエスト数`（verify は
`リクエスト数 × block_size`）で GEMM を回す。既定の並列数は 4 なので、
`rows = 4`（decode）と `rows = 32`（verify）が主戦場になる。

## 見つけた問題

bf16 GEMM の selector は multi-row `ExactRows` を `verify_exact`（= `role == Verify`）
のときだけ選んでいた。通常 decode は `rows >= 2` で WMMA 系に落ちるため、
`rows=2` が `rows=4` より遅いという逆転が起きていた。

`shelf-bench gemm --dtype bf16 --shapes 2560:2560`（1 GPU、p50）:

| rows | 変更前 | `ExactRows` 単体 |
| ---: | ---: | ---: |
| 1 | 10.33 µs | 10.3 µs |
| 2 | 40.26 µs | 11.15 µs |
| 4 | 36.41 µs | 12.97 µs |
| 8 | 37.10 µs | 25.30 µs |
| 16 | 36.88 µs | 33.72 µs |
| 17 | - | 44.77 µs |
| 32 | 37.21 µs | 68.10 µs |
| 64 | 41.16 µs | 136.50 µs |

`ExactRows` は kRows 行をまとめて計算するため、weight の読み出しが 1 回で済む。
launcher は 1 回の launch で扱えるのが `rows <= kBf16GemmExactRowsKernelMax = 16` までで、
それを超えると 16 行ずつに分割して weight を毎回読み直すため遅くなる。

## 変更

- selector の条件から `verify_exact` を外し、`2 <= rows <= 16` を role に依らず
  `ExactRows` にする。`rows >= 17` は従来どおり WMMA 系。
- `Bf16GemmSelectorInput::verify_exact` を削除した（role 依存の分岐が無くなったため）。
- `shelf-bench gemm` の `--impl auto` は bench ローカルの古い policy ではなく
  production の `select_bf16_gemm_config` を使うようにした（計測対象のずれの修正）。

結果（同じ bench、変更後）:

| rows | p50 | 変更前比 |
| ---: | ---: | ---: |
| 2 | **10.97 µs** | 0.27x |
| 4 | **13.04 µs** | 0.36x |
| 8 | **24.57 µs** | 0.67x |
| 16 | **33.22 µs** | 0.90x |
| 17 | 37.58 µs | 1.00x |

`rows=2` が `rows=4` より遅い逆転も解消した。

## 副作用と確認

- decode と verify の kernel 選択は行数だけで決まるようになり、batch 構成が変わっても
  kernel が切り替わらない。role 依存で一致していた従来の契約より厳しい。
- `rows=1` は従来どおり `ExactRows` の 1 行版なので、単一 request decode の数値は変わらない。
  `shelf-bench tg` の `GREEDY_TOKEN_SUM=505644`、`gpu_tokens_per_sec=25.96` は変更前と一致。
- `test_gemm_bf16_*`、`test_gemm_lt`、`test_gemm_fp8_optimized`、`test_fp8_ptpc`、
  `test_tp_transport` は PASS。

## 残課題

マルチリクエストの end-to-end スループットを測る計測口がまだ無い
（`shelf-bench batch` は合成トークンで即 EOS し、token/s も出さない）。
並列数を上げたときの効果測定は [../../homework.md](../../homework.md) の
「マルチリクエスト + 投機推論」に残した。

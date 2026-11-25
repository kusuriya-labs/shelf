# EXL3 prefill の kernel 境界に固定 262 µs の空白

## 観測

`d4fae4a2`、R9700（gfx1201）1 GPU、TP=1、`shelf-bench pp --mode forward`、
2048 token / chunk 2048、BF16 KV。`rocprofv3 --kernel-trace` の
`Start_Timestamp` / `End_Timestamp` から、連続する 2 dispatch の間隔
（前の End から次の Start まで）を求めた。

1 run あたり wall 1012.6 ms、カーネル実行の合計 912.8 ms、差の 97.8 ms（9.7%）が
カーネル外時間である。内訳（1 run 換算）:

| 遷移 | 回数 | gap p50 | 合計 |
| --- | ---: | ---: | ---: |
| `exl3_mul24_prefill_super64_kernel` → `exl3_mul24_gemm_kernel`（flagged） | 400 | 261.5 µs | 67.8 ms |
| `shelf_gdn_recurrence_wmma_lossy_v32` → 直後 | 48 | 262.7 µs | 12.3 ms |
| `attention_paged_prefill_bf16_impl` → `shelf_sigmoid_bf16_f32` | 16 | 263.2 µs | 4.2 ms |
| EXL3 chain の残り 4 遷移 | 400 ずつ | 4.2〜4.3 µs | 6.8 ms |

262 µs が乗る 3 遷移の前にあるカーネルは、いずれも 1 回の起動が 1 ms を超える
（super64 1.25 ms 平均、gdn_recurrence 3.1 ms、attention 2.0 ms）。
他のカーネルはすべて 0.5 ms 未満で、遷移は 4.2 µs である。

## 切り分け（棄却した説）

- **ホスト律速ではない。** `--runtime-trace` を併用すると `hipLaunchKernel` の
  p50 は 3.0 µs、カーネル開始は対応する launch 完了より中央値 245 ms 先行する。
- **stream / queue の跨ぎではない。** 全 dispatch が `Stream_Id=1`、`Queue_Id=1`。
  `Dispatch_Id` も連続しており、間に挿入された packet はない。
- **LDS サイズ説は棄却。** 合成テスト（LDS 0 / 4 / 16 / 32 / 34 KB のカーネルの直後に
  小さなカーネルを置く）で 1 組あたり 0.75〜0.78 ms、差は出なかった。
- **HW queue / MES 説は棄却。** `GPU_MAX_HW_QUEUES=1` で 3 ペア測り、
  +0.09% / −0.02% / +0.00% と差なし（分割版と併用しても分割の利得は変わらない）。
- **単純な duration 説も合成では再現しない。** 4.8〜92 ms の合成カーネル
  （2048 blocks、メモリ無し）の直後にも空白は出ない。実 trace の相関は完全なので、
  起動時間そのものに加えて実カーネルの何か（メモリを大量に読み書きする構造）が
  関与している可能性があるが、機序は未特定のままである。

## 修正: 1 回の起動を 0.84 ms 未満に分割する

`super64` GEMM の起動を、所要時間の推定
（`row blocks × n tiles × k × 3.0e-7 ms` を実測から係数化）が 0.84 ms を超えない
範囲で分割した。

- **行バンド分割（行方向）**: 空白は消える（262 µs → 13 µs p50）。ただし重み平面を
  バンドごとに読み直すため、GEMM 自体が 498.6 → 527.4 ms（+28.8 ms）遅くなる。
  E2E は −2.7 / −2.5 / −2.3%。
- **列バンド分割（採用）**: `n_base` を追加して n 方向に分割する。重み平面の読みは
  (row block, n tile) あたり 1 回のまま変わらず、増えるのは活性 codes の再読だけ。
  E2E は −3.40 / −3.36 / −3.42%（base 1007.9 / 1012.0 / 1012.6 ms →
  973.7 / 977.9 / 978.0 ms）。
- しきい値の比較（列バンド、同じ 3 ペア）: 0.84 ms が最良、0.60 ms と 1.10 ms は
  それぞれ +0.4〜0.7% 遅い。分割しすぎると起動回数の増加が勝つ。

分割後も `gdn_recurrence`（12.3 ms）と `attention_paged_prefill`（4.2 ms）の
空白は残る。前者は系列方向の依存、後者はページ単位の構造のため、単純な分割は
できない。合計 16.5 ms（1.6%）が残課題である。

## 結論

- 1 回の起動が約 1 ms を超えると、次の起動に約 262 µs の空白が乗る。実 trace の
  相関は完全で、分割すると消える。
- 機序は未特定（合成テストでは再現せず、HW queue / LDS / ホスト要因はいずれも棄却）。
  ただし「1 ms 未満に分割すれば消える」という実務的な規則は 3 ペアの E2E で再現する。
- 分割は **列方向** に取る。行方向は重み平面の再読で相殺される。

## 関連

- 内訳の全体: [../../perf/exl3-prefill-selection-table.md](../../perf/exl3-prefill-selection-table.md)
- 計測原則: [../../perf/methodology.md](../../perf/methodology.md)

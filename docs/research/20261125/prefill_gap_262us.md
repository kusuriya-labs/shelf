# EXL3 prefill の kernel 境界に固定 262 µs の空白

## 計測

`d4fae4a2`、R9700（gfx1201）1 GPU、TP=1、`shelf-bench pp --mode forward`、
2048 token / chunk 2048、BF16 KV。`rocprofv3 --kernel-trace` の
`Start_Timestamp` / `End_Timestamp` から、連続する 2 dispatch の間隔
（前の End から次の Start まで）を求めた。

1 run あたり wall 1012.6 ms、カーネル実行の合計 912.8 ms、差の 97.8 ms（9.7%）が
カーネル外時間である。内訳（1 run 換算）:

| 遷移 | 回数 | gap p50 | 合計 |
| --- | ---: | ---: | ---: |
| `exl3_mul24_prefill_super64_kernel<true,true>` → `exl3_mul24_gemm_kernel<4,4,true,true>` | 400 | 261.5 µs | 67.8 ms |
| `shelf_gdn_recurrence_wmma_lossy_v32` → 直後の kernel | 48 | 262.7 µs | 12.3 ms |
| `attention_paged_prefill_bf16_impl<0>` → `shelf_sigmoid_bf16_f32` | 16 | 263.2 µs | 4.2 ms |
| EXL3 chain の残り 4 遷移（rotq→rowprep→recon→super64、flag→fixup） | 400 ずつ | 4.2〜4.3 µs | 6.8 ms |

262 µs は p50 261.5 / p95 266.1 で、linears の形状（n 1024〜17408）によらず一定である。
3 遷移の合計は 84.3 ms/run で、464 回のうち約 6 割に 262 µs が乗り、残りは 4.2 µs である
（p50 だけを見ると全回発生に見える）。差し引き 97.8 ms のうち約 6.7 ms が
EXL3 chain 外（layer 間）の遷移である。

## 切り分け

- **ホスト律速ではない。** `--runtime-trace` を併用すると `hipLaunchKernel` の
  p50 は 3.0 µs（p95 4.9）、カーネル開始時刻は対応する launch 完了より
  中央値 245 ms 先行している。queue は常に埋まっており、GPU が次 packet を
  持っていない状態ではない。
- **stream / queue の跨ぎではない。** 6 launch はすべて `Stream_Id=1`,
  `Queue_Id=1`。
- **LDS / scratch の値と完全に相関する。** 遷移を (前 kernel の Scratch_Size,
  次 kernel の Scratch_Size) で分類すると、262 µs の組は `544 → 12` の
  800 回だけで、`0 → 0`、`0 → 544`、`12 → 0` はいずれも 4.2 µs である。
  LDS で見ると 262 µs が乗るのは「前が LDS 14336 B 以上」の 3 種類のみで、
  `0 → 34816`（rotq → super64）と `512 → 0`（rowprepare → recon）は 4.2 µs。

## 仮説（未検証）

LDS を大量に確保した kernel の終了直後、次の kernel の LDS / scratch 構成へ
切り替えるための drain が固定費として入る。増加方向（0 → 34816）では
発生せず、減少方向でも確保量が小さい場合（512 → 0）は発生しないため、
「大 → 小」の切り替えにだけ課金されているように見える。

gfx1201 の PMC は 0 を返すため、この drain を直接観測できない。
確定には「LDS / scratch を減らした版で gap が縮むか」の A/B が必要である。

## 次の実験

1. `prefill_super64` の scratch（544 B/thread）を消す。VGPR 232 / LDS 34816 の
   削減は occupancy のレバー（3 → 4 blocks/CU）と同一なので、同時に検証できる。
2. 同 kernel の LDS を 32768 B 以下へ落とす（A staging の kStride 72 → 64 か、
   epilogue の scratch を 16 行 → 8 行）。
3. gap が消えれば 67.8 ms/run（6.7%）が回収できる。消えなければ本仮説を棄却し、
   「大 kernel の終了固定費」として別の説明（EOP fence 等）を調べる。

## 関連

- 内訳の全体: [../../perf/exl3-prefill-selection-table.md](../../perf/exl3-prefill-selection-table.md)
- 計測原則: [../../perf/methodology.md](../../perf/methodology.md)

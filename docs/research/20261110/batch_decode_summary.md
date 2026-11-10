# バッチ decode の改善（1 GPU、Qwen3.8 27B）

2026-11-10〜11 の作業の記録である。バッチ decode の合計速度を上げるために main へ入れた変更、測定、
残っている課題をまとめる。測定は R9700 1 枚、27B EXL3 4bpw（mul24）、短い prompt（約 60〜250 token）、
出力 160 token、全員が生成中の区間の合計 tok/s、各 1〜2 回である（±5% 程度のばらつきがある）。
スクリプトは `batch_decode_scripts/`（実行時のまま、machine 固有のパスを含む。`README.md` 参照）。

## 1. 入れた変更

### 1.1 DFlash2 の session 領域（VRAM が同時数にほぼ比例しなくなった）
- `--dflash2-max-sessions N`（既定 `min(--max-concurrent-requests, 8)`、`shelf-server` も転送）。
  session pool、batch verify の行数、drafter の scratch がこの値で決まる。空きがない request は投機せず
  通常 decode になる。
- session workspace から decoder 専用の history / compact log を外した（`step_buffers=false`）。
- batched verify は live の recurrent state を書き換えない（`GdnCompactLogDeviceView::defer_state_store`）。
  受理 prefix を compact log の replay（全受理でも K+1 行）で確定するので、72 MiB の recurrent snapshot が
  要らない。prefix cache の shadow prefill 用に全 session で 1 本を共有し、conv snapshot（2.8 MiB）だけ
  session ごとに持つ。
- GDN verify kernel の launch 上限を 4 request / 32 行から 64 request / 512 行へ広げた（MTP の chunk 定数
  とは分離）。
- serve の行数の見積もり（scratch、batch の history / compact log、target の出力行数、起動時の検査）を
  `sessions × block_size` ではなく `sessions × (K+1)` から作る（`dflash2_session_rows`）。K=7 では従来と同じ。
- 効果（drafts=7、drafter runtime）: 同時 8/16/32/64 で 1.86/3.51/6.80/16.99 GiB → 1.11 GiB（上限 8 の既定）。
  上限＝N にすると 1.90/3.48/6.66 GiB（同時 16/32/64）。K=2 では上限 8 で 0.80 GiB、同時 64・上限＝N で
  6.63（起動不可）→ 4.34 GiB。

### 1.2 EXL3-INT8 の 17〜64 行の GEMM
詳細は [exl3_int8_mid_rows.md](exl3_int8_mid_rows.md)。
- 17 行以上でも K 方向を split する（約 3000 wave を目標、`groups/8`・16・scratch で制限）。分割数は
  1..16 行、17..64 行のそれぞれで一定。
- row tile 幅を MT=1/2/3/4（1..16 / 17..32 / 33..48 / 49..64 行）に細分し、重みの展開は 1 launch に 1 回のまま。
- bits=4 の MT>=3 だけ `amdgpu_waves_per_eu(8)`。3/5/6bit では 20〜35% 遅くなるので適用しない
  （6bit の lm_head の 64 行が 2523 → 3299 µs になるのを統合測定で見つけて限定した）。
- `shelf-bench exl3-int8` が 64 行・`--random`・`--dump` に対応。`test_exl3_mul24_linear` に 17〜64 行の
  split-K ケースを追加。

### 1.3 attention の region 上限（16 → 64）
mixed batch（verify / prefill の region が複数ある batch）の region が 16 を超えると、最適化された attention
が使われず遅い経路に落ちていた。DFlash2 の batched verify が 17 request 以上のとき、PAGED_ATTENTION が
3 ms から 327〜570 ms になった。`kPhysicalPagedAttentionMaxPrefillRegions` を 64 にした（plan 構造体は
約 11 KB → 約 40 KB でスタック上）。効果: K=2・上限＝N・同時 32 が 134 → 313 tok/s、K=7 では 141 → 215。

### 1.4 DFlash2 の prefix cache（1024 token を超える prefix）
DFlash2 の prefill は、prompt の最後の page 境界でしか prefix を保存しなかった。prompt が 1 chunk
（2048 token）を超えると chunk 終端がそこに一致せず、1024 token 超の共有 prefix が cache に入らなかった。
chunk 終端ごとに保存する（通常 prefill と同じ）ようにした。共有 16k prefix・8 request で TTFT 最大 98 → 15 秒、
64k・4 request で 252 → 20 秒（`restored` は 0 → 16384 / 65536）。復元した出力が、cache 無効の DFlash2 と
非投機 greedy の両方に一致することを `test_compute_dflash2_prefix_long.py` で確認する。

## 2. 測定

### 2.1 投機なしのバッチ decode（GEMM 変更の前後）
| 同時数 | 変更前 | 変更後 |
| ---: | ---: | ---: |
| 4 / 8 / 16 | 98.6 / 177 / 285 | 99.5 / 177 / 286 |
| 24 | 303 | 361 |
| 32 | 360 | 420 |
| 64 | 418 | 507（再測定で 497〜510） |

N=64 の 1 step は 152.9 → 127.0 ms（INT8 GEMM 103.9 → 75.1、GDN 36.8 → 37.0）。1..16 行は変えていない。

### 2.2 DFlash2（drafts=7、上限 8 が既定）
| 同時数 | 投機なし | DFlash2 変更前 | DFlash2 変更後 |
| ---: | ---: | ---: | ---: |
| 4 | 99 | 176 | 193 |
| 8 | 177 | 225 | 233 |
| 16 | 286 | 208 | 214〜225（上限 16 なら 281〜283） |
| 32 | 420 | 245 | 265〜278（上限 16 なら 283〜287） |
| 64 | 500 | 未測定 | 348〜362 |

この表は benchmark prompt（`rows_cases.json`、受理 約 1.85 draft/request）の値である。有利なのは同時 8 まで、
16 で並び、32 以上は投機なしが速い。GEMM の改善は投機なしのほうにより効き、差は開いた。

### 2.3 K=2（K に合わせた行数の見積もりの後）
同じ run の投機なしは同時 16/32/64 で 288 / 370 / 498（投機なしは run 間で 370〜421 になることがある）。

| 同時数 | K=2 上限 8 | K=2 上限 16 | K=2 上限＝N |
| ---: | ---: | ---: | ---: |
| 16 | 191 | - | 279 |
| 32 | 265 | 298 | 313 |
| 64 | 371 | - | 315 |

K=2 は K=7（上限 8）とほぼ同じ。検証の行は減るが、draft 段（draft_gpu 17〜35 ms）は縮まなかった。

### 2.4 MTP（`--mtp --mtp-drafts 4`、同時 1〜8 のみ起動可能）
| 同時数 | 投機なし | MTP | DFlash2 |
| ---: | ---: | ---: | ---: |
| 1 | 27.7 | 54.3 | 45.5 |
| 4 | 98.7 | 125.8 | 176 |
| 8 | 177 | 134.7 | 225 |

MTP は同時 16 以上でメモリ不足で起動しない（drafter runtime が 1 slot あたり約 0.86 GiB 増える）。

### 2.5 追加検証
- **prefill が混ざったとき（投機なし、8 本が decode 中に 8192 token の prompt が到着）**

  | `--max-num-batched-tokens` | 到着 request の TTFT | 他の 8 本の ITL 最大 / p99 | prefill 中の他 8 本の合計 |
  | ---: | ---: | ---: | ---: |
  | 2048（既定） | 5.4 s | 2.05 s / 1.17 s | 約 25 tok/s |
  | 512 | 6.8 s（+26%） | 0.95 s / 0.38 s | 約 34 tok/s |

  到着前の ITL 中央値は 45 ms（8 本で約 177 tok/s 相当）。chunk を小さくすると最大 ITL は半分以下になるが、
  到着側の TTFT が伸びる。
- **長文脈のバッチ decode（共有 prefix + prefix cache、1 request あたりの decode tok/s、ログの中央値）**

  | 条件 | 投機なし | DFlash2（prefix 修正後） | 短文脈の投機なし |
  | --- | ---: | ---: | ---: |
  | 8 本、16k | 約 17 | 約 47 | 約 22 |
  | 4 本、64k | 約 16〜20 | 約 34 | 約 25 |

  request の重なりが完全でなく、全員生成中の窓が取れなかった（出力 160 token では足りない）ため、定常の
  合計ではなく参考値である。16 本・16k（投機なし）の約 17 は、重なりで甘く出ている可能性がある。
- **サーバー経由**: 投機なしの overhead は 1〜5%（同時 16 で 284 / compute 287、同時 8 で 168 / 177）。
- **temperature=1 と c4v4（同時 8、DFlash2、短い prompt）**: FP8・temp=0 が 235、temp=1 が 204、c4v4 が 216、
  c4v4・temp=1 が 198 tok/s。c4v4・temp=1・投機なし・同時 16 は 279（FP8・temp=0 は 287）。エラーなく完了し、
  出力は退化していない（distinct token 比 0.40〜0.43）。品質の定量評価（KLD）はしていない。
- **DFlash2 の効果は内容に依存する**: 日本語の説明文（thinking 無効）をサーバー経由で流すと、受理は約 1.1
  draft/request（benchmark prompt は 1.85）で、同時 8 で投機なし 167 に対し DFlash2 110 tok/s、同時 4 で 98
  に対し 91 tok/s だった。受理が約 1.5 draft を下回る内容では投機なしが速い。想定する主な用途（コマンドと
  英語の reasoning）の受理は未測定である。

## 3. 残っている課題
- **DFlash2 の draft 段**: draft_gpu が K=2 でも 17〜18 ms（上限 8）、25〜35 ms（上限＝N の 32〜64 同時）で、
  行数を減らしても縮まない。内訳は未計測（drafter の重みの読み出しなどの固定費の疑い）。
- **GDN recurrence**: N=64 で 37 ms（step の 29%）。1 行 約 0.57 ms で、fp16 state の帯域下限（約 0.23 ms/行）
  の約 2 倍。
- **GEMM**: 49〜64 行（MT=4）は重みを読む下限（75 µs）の 2 倍以上（17408×5120 で 173〜178 µs）。65〜255 行は
  split なしの MT=4、256 行以上は dense（hipBLASLt）経路のまま。
- 上限を超えた request の通常 decode は、投機する request とは別の forward で走る（重みを 2 回読む）。
  統合すれば同時 16 で約 1 割（推定、未実装）。
- **prefill の混在**: 既定の `--max-num-batched-tokens 2048` では、長い prompt の到着中に他の request の ITL が
  最大 2 秒になる（§2.5）。
- **未検証**: TP（`--tensor-parallel`）経路の session 変更（複数 GPU が必要）。MoE（Flash-Next）。長文脈の定常合計
  （§2.5 は参考値）。バッチを組んだ状態の品質評価（KLD）。
- 計測は各条件 1〜2 回、時間窓 2〜5 秒。±5% 程度の差は区別できない。

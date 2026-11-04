# group 行数 capacity の厳密化と、tail 行の実測コスト（MoE grouped MXFP4）

手順書「SHELF Grouped MXFP4 MoE 最適化調査」の実施記録。目的は
`logical_slots` / `actual_group_rows` / `capacity_group_rows` の差を定量化し、
それを消す方式を correctness と実測性能で判定すること。**実行経緯と採否判断の記録**で、
現在の contract は `docs/developer/qwen35moe.md`、性能値は `docs/perf/current.md` を正本とする。

## Phase 0 固定条件

| 項目 | 値 |
| --- | --- |
| branch / base | `feature/moe-group-rows`（`main` の `3a435bd` から） |
| GPU | AMD Radeon AI PRO R9700（gfx1201）、測定は device 0 / 1、他 process 非動作を確認 |
| driver 属性 | `MaxGridDimX=2147483647`、`MaxGridDimY=MaxGridDimZ=65535`、`MaxThreadsPerBlock=1024`（`hipDeviceGetAttribute` で実測） |
| compiler | ROCm clang 23.0.0git（`_rocm_sdk_devel`）、`-O3 -DNDEBUG --offload-arch=gfx1201`、CMake Release |
| model | `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層、experts 256、top_k 8、moe_intermediate 512、hidden 2048、layer_types `LLL A` × 10） |
| 変更禁止条件 | `kGroupedRowTile=16`、routing semantics、MXFP4 の数値表現、expert をまたぐ tile 共有 |

## Phase 1 データフロー（実装から確定した事実）

| 段階 | 実行対象の行数 | 出所 |
| --- | --- | --- |
| workspace / plan scratch の確保 | **bucket の token 行数**から決まる capacity | `program.cpp` `rows_for(GROUPED_ROWS)` = `grouped_row_capacity(token_rows(), shape)`。`token_rows() = min(bucket limit, max_token_rows)` |
| dispatch の行数（elementwise / activation 量子化 / grouped GEMM） | **その batch の実 token 行数**（`ctx.actual_rows`）から決まる capacity | `moe_dispatch.hip` の `moe_group_rows()` と `resolve_moe_plan()`。`view.row_capacity` / `view.group_capacity` |
| activation 量子化（`ACTIVATION_QUANTIZE_MXFP4_ROWREF`） | `b.activation_max_rows`（= capacity 行数ぶん launch。未使用行は `row_index < 0` で zero operand） | `program.cpp` の grouped linear 展開、`moe_slot_scan_kernel` が `row_token` を容量ぶん `-1` で埋める |
| SWIGLU（`SwigluBf16 @512`） | capacity 行数ぶん launch | `elementwise_dispatch.hip` → `moe_group_rows()` |
| grouped MXFP4 GEMM | grid `(out_features/16, group_capacity)` だが、kernel は **`if (g >= group_count[0]) return;`** で GPU 側の実 group 数で早期終了する | `linear/mxfp4.hip`（`group_count` は scan が書く） |
| MOE combine | token 行数（group 行数ではない） | `combine.hip` |

重要な帰結を 2 つ先に書いておく。

1. **GEMM は既に tail を処理していない**（`group_count[0]` で早期終了）。tail 分だけ実行されるのは
   activation 量子化と SWIGLU などの elementwise で、それらは host が決めた capacity 行数ぶん走る
2. capacity は**その batch の実 token 数**から決まる（bucket 固定値ではない）。だから
   decode（1 token）なら capacity は 128 行で、`actual`（8 expert × 1 tile）と一致する

## Phase 2 capacity の厳密値と census

旧式は `slots / 16 + experts`（group 数）。これは sound だが slack だった。厳密な最大値は

```
tiles <= slots                                   （group は空にならない）
tiles <= (slots + experts * (16 - 1)) / 16        （expert ごとに最大 15 slot 無駄）
tiles <= experts * ceil(rows / 16)               （1 token が同じ expert を選ばないので c_e <= rows）
```

の min。**DP で全 routing を列挙した最大値と一致する**（`tests/unit/test_grouped_row_shape.cpp`、
experts 4..256 / top_k 1..8 / rows 4..64 の 7 形状で `bound == exact` を確認）。

MoE geometry（experts 256、top_k 8、tile 16）での capacity 行数:

| tokens | slots | 旧 group 行数 | 新 group 行数 | 差 | 完全平衡時の実 group 行数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 128 | 128 | 0 | 128 |
| 8 | 64 | 1024 | 1024 | 0 | 1024 |
| 16 | 128 | 2048 | 2048 | 0 | 2048 |
| 64 | 512 | 4608 | 4352 | 256 | 4096 |
| 128 | 1024 | 5120 | 4864 | 256 | 4096 |
| 512 | 4096 | 8192 | 7936 | 256 | 4096 |
| 1024 | 8192 | 12288 | 12032 | 256 | 8192 |
| 2048 | 16384 | 20480 | 20224 | 256 | 16384 |
| 4096 | 32768 | 36864 | 36608 | 256 | 32768 |

- tokens ≤ 32 では `slots` 項が効くので capacity = 16 × slots であり、これは**厳密値でもある**
  （1 slot しか持たない expert が 16 行タイルを 1 個 each 使う）。つまり decode 付近の
  「無駄」は capacity の過大評価ではなく **tile 粒度そのもの**で、`kGroupedRowTile` を変えない
  限り消えない
- 旧式の slack は常に 256 行（16 group）で、4096 token bucket では 0.7 %
- 合成形状（experts 4 / top_k 2 / rows 16）では旧式 96 行 → 厳密 64 行と**1.5 倍**差が出る。
  旧式は小さい bucket で最大 2 倍過大（`test_program_moe_plan` / `test_qwen35moe_lowering` の
  期待値がそれを写していた）

実 router 分布の census は**未実施**（plan の `group_count` を読む診断口が無い）。ただし後述のとおり
判定は分布に依存しない: tail の上限は `capacity − slots` で、distribution によらず 4096 token では
3840 行（10.5 %）以下に固定される。分布を実測したい場合は scan が書く `group_count[0]` を
読み出す診断経路を増やす（`moe_dispatch.hip` の plan view に stats 欄を足し、run 終了時に
readback する。perf 計測中の同期は入れない）。

## Phase 3 tail 行の実測コスト（改善案 A / B の上位限界）

`shelf-bench elementwise --variant optimized --profile swiglu --features 512 --rows N`
（device 0、samples 20、launches 5）。capacity 行数と、完全平衡 routing の実行数を並べる:

| tokens | capacity rows → p50 | balanced rows → p50 | Δ / layer |
| ---: | ---: | ---: | ---: |
| 512 | 7936 → 25.49 µs | 4096 → 18.19 µs | 7.30 µs |
| 1024 | 12032 → 34.22 µs | 8192 → 25.56 µs | 8.66 µs |
| 2048 | 20224 → 54.66 µs | 16384 → 44.99 µs | 9.67 µs |
| 4096 | 36608 → 187.70 µs | 32768 → 168.36 µs | 19.34 µs |

`pp` の MoE prefill（device 1、`--prompt-tokens 8192`、3 run）は chunk 2048 / 4096 で
**3807.01 / 3630.98 tok/s**（厳密 capacity 適用後）。旧 capacity での同一条件の測定は
3759.43 / 3683.91 tok/s で、**差は run 揺れの範囲**（chunk 2048 の run 揺れは 3759 / 3769 / 3807）。
4096 token chunk あたり約 1110 ms。**tail 行を全部消しても**
（完全平衡という最良仮定。実 router はこれより不平衡なので実際の减少は以下より小さい）

- SWIGLU: 19.34 µs × 40 層 = 0.77 ms / chunk = **0.07 %**
- activation 量子化（bench が MXFP4 rowref 経路を持たないため交通量から評価）:
  量子化 2 本ぶん 3840 行 × (1088 + 272) B × 40 層 = 166 MB = 0.26 ms = **0.02 %**
- grouped GEMM: tail を既に処理していないので **0**

decode 側は capacity == actual（1 token で 128 行）なので対象がなく、per-op 実測でも
SWIGLU 8.34 µs / ACTIVATION_QUANTIZE 9.85 µs は launch 下限（3.6 µs / dispatch、event ペア込みで
約 8 µs）に張り付いている。**行数を減らしても CTA 数は減らない**（手順書の注記どおり）ので、
decode では収益が 0 に近い。

## Phase 5 vLLM との比較（2026-11-05 の main を確認）

`vllm/model_executor/layers/fused_moe/moe_align_block_size.py`:

```python
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
if topk_ids.numel() < num_experts:
    max_num_tokens_padded = min(topk_ids.numel() * block_size, max_num_tokens_padded)
```

- これは今回入れた厳密式と**同じ形**（slot 単位で `S + E(B−1)`、`S < E` のとき `S·B` へ clamp）。
  SHELF の新 capacity は vLLM の host 側容量と一致する（旧式はこれより slack だった）
- 中间 buffer（`intermediate_cache` 系）は padded 容量で確保され、GEMM 側の仕事は
  `num_tokens_post_pad`（device 側実行数）と invalid index skip で縛られる
  → **SHELF の `group_count[0]` 早期終了と同じ構造**。vLLM も「compact logical slot 中間 buffer」
  （改善案 B）はしていない。`scatter_idx` は追加 option で、元 routing 行への写像を返すだけで
  中間を compact 化する経路ではない
- 注意点（手順書の指示どおり）: vLLM には Triton / AITER など複数の fused MoE backend が
  あり、上の式は `moe_align_block_size` の契約であって backend 共通の仕様として扱わない

## 採否

| 案 | 判定 | 根拠 |
| --- | --- | --- |
| **D: capacity の厳密化** | **採用（実装済み）** | DP 列挙と一致する厳密値。workspace と launch 行数が 256 行減り、小さい bucket では最大 2 倍の過大確保が消える。ただし**性能主張はしない**（上の実測で ≤ 0.1 %）。正しさの強化（capacity が実最大と一致）としての採用 |
| A: GPU 実 group 数での早期終了（elementwise / 量子化） | **不採用** | 上位限界が prefill の 0.09 %（SWIGLU + 量子化）。decode は対象 0。CTA 数は減らない。`ElementwiseArgs` の ABI 拡張 + plan scratch の pointer を elementwise binding へ通す変更に見合わない |
| B: compact logical slot 中間平面 | **不採用** | 削除できる traffic は A と同じ tail ぶんだけ。追加される scatter store / indirect gather は coalescing を悪化させ、vLLM も同じ形状（padded 容量）を維持している |
| C: Gate/Up 入力の token 単位量子化 | **保留（未実装）** | quantum 対象が 36608 行 → 4096 行（書き込み 35 MB/layer、読み 150 MB/layer が L3→少数 row へ）で、上限 estimate は chunk あたり 0.2 %（書き込み）+ 0.4 %（L3 読み）程度。実装には grouped GEMM の activation 参照の間接化が必要で、fragment order の 16 行連続読み（`rowref` の契約）を壊すリスクを測っていない。A/B より收益はあるが同じ桁なので、他の 1 % 級 lever（launch 数削減、weight bytes 削減）の後 |
| grid 制約 | **済（別コミット）** | elementwise は `blockIdx.y + blockIdx.z * gridDim.y` で address するので 65535 は kernel の限界ではない。MoE の 4096 token 上限（group 36608 行）は policy。grid.z 経路は `test_elementwise` の rows 69632 / 69633 で押さえている |

## 受け入れ

- `tests/unit/test_grouped_row_shape.cpp`: 契約値を厳密値へ更新（experts 4 / top_k 2 / rows 16 →
  4 group・64 行、4096 token → 2288 group・36608 行）+ **DP で全 routing の最大値を列挙して
  capacity と一致させる**検証を追加。bound が小さすぎれば（unsound）ここで落ち、
  大きすぎれば（slack）これも落ちる
- `tests/unit/test_moe_combine.hip`: router → scan → scatter の後に
  `group_count[0] <= moe_group_count_upper_bound(rows, experts, top_k)` を確認
  （実 router が生成する分布で capacity が壊れないこと）
- `tests/unit/test_program_moe_plan.cpp` / `test_qwen35moe_lowering.hip`: plan scratch の期待値を
  厳密 capacity へ更新
- `tests/unit/test_qwen35_elementwise_selector.cpp`: group 行数 bound を 36608 に戻し、
  36609 と 69632（8192 token 相当）が Correctness 側へ落ちることを確認。grid helper の
  行数カバレッジ検査（`elementwise_grid_y` / `elementwise_grid_z`）は残す
- `tests/kernels/optimized/test_elementwise.hip`: MoE ケースを 12032 / 36608 行へ更新。
  69632 / 69633 行ケースは「production は到達しないが kernel の address 契約を固定する」
  ものとしてラベルを明示
- required acceptance: **162 PASS / 0 FAIL / 0 ERROR / 0 SKIP**（`feature/moe-group-rows` の最終 tree）
- **fail-first は tamper で実演**: capacity から experts 項を落とす（`tiles = slots / 16`）と
  `test_grouped_row_shape` の DP 一致検査が 4 形状で失敗（例 `experts 4 top_k 1 rows 4: 0 vs 4`）、
  `test_moe_combine` は rows 64 / top 8 で group 数が capacity を超えて scan が group を落とし、
  combine の sync 失敗として出る（過大すぎる bound も過小も、この 2 本で捕まる）

## 次の実装が必要な人向けの interface メモ

A や C を distribution 実測後にやり直す場合:

- scan は既に `group_count[0]`（実 group 数）と `expert_row_begin[experts]`（総 slot 数）を持つ。
  診断で実分布が知りたいだけなら、plan scratch に 2 語の stats を足して run 終了時に readback
  するだけで、hot path は変わらない
- elementwise / activation 量子化が早期終了するには、その binding が plan scratch の
  `group_count` 参照を持てばよい（`DispatchBinding::scratch_slot` は既に MoE 系で使われている）。
  ただし `ElementwiseArgs` に pointer 欄を追加することになり、`sizeof` / offset の
  static_assert を更新する（ABI 変更）
- 行数の上限は `kMeasuredMaxGroupRows`（policy）で、kernel 側の address は grid.z で無制限
  （grid.y ≤ 65535、grid.z ≤ 65535 なので 42 億行まで）

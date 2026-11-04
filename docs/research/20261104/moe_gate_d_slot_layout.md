# Gate D 実施記録 第二段 Step 1a / 1b: rank-3 weight slot と group row 行空間

MoE FFN を graph / kernel に配する前の下ごしらえ。`WeightSlot`（program の weight 記述）を
rank-3 expert 平面に対応させ、expert 間 stride の式を 1 箇所に集約した。
現在形の contract は `docs/developer/qwen35moe.md` §3。

## 1. 実験 D-2a: stride 式の重複を layout contract 側へ寄せる

配線前に、同じ式を 4 箇所が別々に書いていた。

```text
moe_weight_plan.cpp      (rows/16) * blocks * 256        ... arena 割当
moe_expert_load.cpp      tiles * native.codes_row_stride  ... 実平面の stride
mxfp4.hip launcher       (weight_rows>>4) * nb * kWeightBlockBytes ... kernel の受理条件
Mxfp4NativeView::validate batch * tiles * blocks * 256    ... host payload 検証
```

host 側の weight table 検証（`validate_host_weight_slot`）は rank-3 を一切見ておらず、
kernel の受理条件と検証条件が食い違っても launch まで気づかない。

採用: 式を `include/shelf/quantization/mxfp4/mxfp4_native.h` の
`mxfp4_native_codes_batch_stride_bytes` / `mxfp4_native_scale_batch_stride_bytes` に置き、
上記 4 箇所と `validate_host_weight_slot` の 5 箇所すべてがそれを呼ぶ。
常量も `mxfp4_native_block_bytes` / `mxfp4_native_output_tile` を使い、
`kernels/optimized/linear/mxfp4.hip` に
`static_assert(kMxfp4GemmRowTile == mxfp4_native_output_tile)` を入れた。

却下: 式を `kernels/optimized/linear/mxfp4.h` に置く案。grouped GEMM の受理条件である前に
native SoA の layout contract なので、配置は quantization 側が正しい
（`models/qwen35moe` が `kernels/optimized` を include する辺も消える）。

## 2. 実験 D-2b: plan は k_padded が 128 の倍数でないと拒否する

`moe_weight_plan.cpp` の padding は 64 単位（checkpoint の declared padding と一致させる値）の
まま、`k_padded % 128 != 0` の plan を `Status::unsupported` にした。grouped GEMM は K を
128 単位で読む（`ng = k_padded >> 7`）ため、64 単位の平面は kernel が読めない。
従来は `hidden_size = 2064` のような geometry で plan が通り、load まで進んで
launch 時に `hipErrorInvalidValue` になっていた。

この判断で `test_moe_weight_plan` の既存 assert
（`hidden 2064 → k_padded 2112` を受理）は契約変更になった。代わりに
`moe_intermediate_size = 200 → w2_k_padded 256`（切り上げは残る）を受理し、
`2064` を拒否する形にしている。

## 3. 実験 D-2c: weight table は構造体そのものを写す

`upload_host_weight_table` は字段を 1 つずつ写しており、`bytes` と `compute_spec` が
device 側の table に落ちていた（`WeightSlot::bytes` は host 側 program build
（`program.cpp` の bf16 検証）でしか効いていないため、実害は出ていなかった）。
字段追加と同じ穴があくので `host[i] = s;` の構造体代入に変えた。
device 側 kernel が `bytes` / `compute_spec` を読む箇所は無い（grep 0 件）ため、
dense の挙動は変わらない。

## 4. fail-first 実演

`tests/unit/test_weight_slot_validation.cpp`（CPU、required）を先に置き、検証本体を
変えない状態で実行した。

```text
FAIL experts 0 rejected
FAIL rank-3 with zero codes batch stride rejected
FAIL rank-3 with zero scales batch stride rejected
FAIL rank-3 with tampered codes batch stride rejected
FAIL rank-3 with tampered scales batch stride rejected
FAIL rank-3 with output rows not a multiple of 16 rejected
FAIL rank-3 with k_padded not a multiple of 128 rejected
FAIL rank-3 bf16 encoding rejected
（+ expert count 不一致の 1 件、後述の理由で削除）
FAILED 9 checks
```

rank-3 検査を入れた後に 0 FAIL。`experts` 数と arena 平面長の突合は、slot 自体が
plane byte 数を持たないため表現できない（grouped launcher も expert 数を読まない）。
これは loader 側の `moe_ffn_weights_match` が plan と突合して固定するものとして、
slot 検証の case からは外した。

## 5. 実測

`build/`（`SHELF_BUILD_OPTIONAL_TESTS=ON`、gfx1201）で実施。

required acceptance（`python3 tests/run_required_acceptance.py --build-dir build`）:

```text
Passed: 149  Failed: 0  Errors: 0  Skipped: 0
```

Step 1a 前の baseline が 148 PASS で、増えた 1 件が `test_weight_slot_validation`。
`test_weight_load` / `test_qwen35_mtp_weight_load` / `test_qwen35_tp_weight_load` は
そのまま PASS（dense の slot 記述は変化していない）。

実 checkpoint の MoE load（`test_qwen35moe_model_load_real`、optional / external files）:

```text
load 25.4 s, arena used 20.527 GiB, expert planes 15.967 GiB (plan sum), peak RSS 21.40 GiB
fold-exact planes: 80 / 80 (both planes in 40 layers)
```

arena・expert 平面・fold-exact 数は第一段の実測（20.527 GiB / 15.967 GiB / 80 本）と同一。
load 時間の差（32.7 s → 25.4 s）は同一 build 系の再現実行のばらつきで、
今回の変更による短縮としては計測していない。

## 6. Step 1b: group row の行空間（`ValueRowDomain::GROUPED_ROWS`）

grouped GEMM は group row（`group * 16 + 内offset`）でアドレスするので、activation 平面・
grouped 出力・combine の読む行は token 行ではない。既存の program は全 value が
`rows_for(TOKEN_ROWS)` または `rows_for(OUTPUT_ROWS)` の行数を持つ前提で、
`moe_group_row_capacity(rows, experts, top_k)` 相当の行数を扱えなかった。

### 配置の判断: 容量の式は runtime が持つ

候補と採否:

| 案 | 判断 |
| --- | --- |
| 容量の式を `kernels/optimized/moe/router.h` に残し、program から include する | 却下。link 辺は要らなくても（constexpr）、`shelf_runtime` は `shelf_core` / `shelf_gpu` のみを link する契約（`docs/developer/architecture.md`）に依存方向が逆 |
| lowering（models 側）が容量を計算して node に定数で持たせる | 却下。program は 1 つの graph を row bucket 変換で 10 本作る（`build_program_set`）ため、容量は bucket の token rows から決まり、graph の時点で確定できない |
| `runtime/graph/grouped_row_shape.h`（header-only）に容量を置き、kernel 側はそれを呼ぶ alias にする | 採用。`shelf_kernels_optimized` は `shelf_runtime` に link する既存方向で、`static_assert(kMxfp4GemmRowTile == kGroupedRowTile)` が kernel の行 tile と plan の group tile を張り合わせる |

`Int8ActivationWorkspaceLayout`（runtime が kernel の 16 / 64 の定数で workspace 容量を決める）
が同じ形の前例である。

語彙は `GROUPED_ROWS` / `grouped_row_*` にした。`MOE` という単語を runtime に置く必要は
なく、行空間の本質は「grouped GEMM の plan が決まる行」である
（`include/shelf/runtime` に `MOE` は増えない）。

### plan geometry は graph が 1 度宣言する

`PrimitiveGraph::grouped_row_shape{experts, top_k}`。value ごとに容量を持たせないのは、
容量が bucket の token rows に依存する一方で `experts` / `top_k` は model の定数だからで、
定数だけを graph に置き、行数は program build が計算する。検証は両側に入れた:

- `validate_primitive_graph`: `GROUPED_ROWS` の value があるのに shape 未宣言なら fail
- `build_program`: 同じ条件で fail（graph 検証を通さずに呼ぶ経路があるため）

### fail-first 実演

`tests/unit/test_grouped_row_shape.cpp`（CPU、required）を先に置いた。

```text
tests/unit/test_grouped_row_shape.cpp:1:10: fatal error:
  'shelf/runtime/graph/grouped_row_shape.h' file not found
```

実装後は 16 checks PASS。固定したのは

- 容量の数値（rows=1 / experts=256 / top_k=8 → group 8・group row 128、
  rows=512 → group 512・group row 8192）
- 不変条件（group 数は slot 数を超えない、容量は slot 数を覆う、容量は 16 の倍数）
- program build で `GROUPED_ROWS` value の binding 行数が容量基準になること
  （bucket R16 / experts 4 / top_k 2 → 96 行）
- shape 未宣言の graph が graph 検証と program build の両方で拒否されること
- dense graph（token rows のみ）には grouped 行空間が現れないこと

### 実測

required acceptance:

```text
Passed: 150  Failed: 0  Errors: 0  Skipped: 0
```

149 → 150 の 1 件は新規 `test_grouped_row_shape`。kernel 側の 3 関数を runtime alias に
置き換えた後、`test_moe_router` / `test_moe_grouped_linear` / `test_moe_combine` /
`test_moe_layer_e2e` はそのまま PASS（Gate C の数値 gate は不変）。

### Step 1b で意図的に置かなかったもの

plan 配列（`sel_expert` / `tile_counts` / `group_expert` / `row_token` / `slot_row_of` など）の
workspace 確保は Step 3（MoE node と dispatch）へ移した。消費者が無いまま program に
確保項目だけ置くのは production 到達不能な placeholder になるため、node 追加と同じ commit に
する。`tile_counts` の層毎 zero 初期化もそこでの決定になる
（scan kernel が消費後に zero に戻す案と、launcher が memset する案を追加 launch 実測で選ぶ）。

## 7. Step 2a: 层 block の emit を qwen_hybrid へ抽出（FFN は各 family が持つ）

MoE lowering を `shelf::qwen35moe` 側に置く前に、dense lowering の层ループから
**FFN を含まない block**（input norm → GDN / full attention → attention residual →
post norm）を `qwen_hybrid::emit_hybrid_block`（`models/common/qwen_hybrid/block_lowering.h`）へ
抽出した。dense 側はこれを見て FFN を emit し、MoE 側は同じ block を見て自分の FFN を
emit する。抽出しないと backbone の 60 行が両家に複製され、norm 順序や state index 採番の
逸脱が 2 所に生まれる。

同じ理由で lowered 保持型も `qwen_hybrid::HybridLoweredPrimitives` に置いた。
`Qwen35LoweredPrimitives` はその alias で、呼び出し側は `auto` しか使っていないため
code path の変化は無い。

### 等価性の実測（A/B）

合成 checkpoint `tests/fixtures/qwen35/qwen35_gdn_tiny`（hidden 512 / 2 层 / vocab 512）で
`shelf-compute --max-new-tokens 4 --hidden-taps 0` を `SHELF_WS_DEBUG=1` を付けて
抽出前後で実行した。

| | program workspace ranges | 生成 id |
| --- | --- | --- |
| 抽出前（`4e147c3`） | 10 プログラムすべて `ranges=43` | `186,507,39,454` |
| 抽出後 | 10 プログラムすべて `ranges=43` | `186,507,39,454` |

graph 検証・value binding・lifetime から決まる range 数と、greedy 生成列が一致した。
required acceptance は 150 PASS / 0 FAIL / 0 ERROR / 0 SKIP
（`test_qwen35_lowering_contract` / `test_qwen35_mtp_lowering` /
`test_qwen35_gdn_gate_reference` を含む）。

### 境界 test が実際の契約として機能した例

抽出直後のコメントに「a dense model and an MoE model …」と書き、
`test_architecture_boundaries` が

```text
FAIL: include/shelf/models/qwen35/model/lower_to_primitives.h contains forbidden MoE
```

で落ちた。dense 側のヘッダに MoE の語彙を書かない、というルールが文言レベルでも
機械的に適用されている。表現を「whatever FFN a model emits」へ変えて解消した。

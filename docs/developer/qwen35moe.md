# Qwen3.5 MoE（`qwen3_5_moe`）contract

MoE 固有の contract をここに集約する。dense 側の配線・kernel・runtime は
[qwen35.md](qwen35.md) を参照。層構造と checkpoint 内レイアウトの実測は
[../research/20261103/qwen36_35b_a3b_moe_structure.md](../research/20261103/qwen36_35b_a3b_moe_structure.md) が正本。

## 1. module 境界

| 項目 | 値 |
| --- | --- |
| namespace | `shelf::qwen35moe` |
| headers | `include/shelf/models/qwen35moe/` |
| sources | `src/shelf/models/qwen35moe/` |
| CMake target | `shelf_qwen35moe`（`shelf_qwen35` に依存） |

checkpoint の `model_type` は `qwen3_5_moe` だが、runtime family は `qwen35` のまま
（rename しない）。MoE 固有の型・loader はこの module に置き、`shelf::qwen35` 側へ
MoE の分岐を増やさない。

実装済みは config 契約、routed experts の weight 集約ロード、そして
routing / grouped linear / combine の GPU kernel（standalone launcher）である。
**graph 配線（`PrimitiveKind` / `KernelId` / `WeightSlot` の experts 化 / selector /
lowering）は未実装**なので、`shelf-compute` は MoE checkpoint を production path では
実行できない。kernel 契約は §5、配線未実装の範囲は §7 に書く。

## 2. config 契約

`shelf::qwen35moe::Qwen35MoeTextConfig` は `shelf::qwen35::Qwen35TextConfig` を基底とし、
expert 幾何を追加する。読み込みは
`Result<Qwen35MoeTextConfig> read_qwen35_moe_text_config(const std::string& model_dir)`。

MoE 判定の正本は `text_config.num_experts > 0`。`model_type` 文字列での分岐はしない。

| key | 意味 |
| --- | --- |
| `num_experts` | routed expert 数 |
| `num_experts_per_tok` | 1 token に選択される expert 数 |
| `moe_intermediate_size` | routed expert の intermediate size |
| `shared_expert_intermediate_size` | shared expert の intermediate size |

受理条件（違反は `Status::invalid_argument`）:

- `num_experts > 0`（dense config を MoE reader が拒否する）
- `num_experts_per_tok` は `[1, num_experts]`
- `moe_intermediate_size` と `shared_expert_intermediate_size` は必須
- `intermediate_size` を宣言してはならない（MoE には dense FFN が無い）

`text_config.output_gate_type` が欠落している場合は `output_gate_silu = true`
（Qwen3.5 MoE checkpoint はこの key を持たず、参照実装は常に silu を適用する）。
`sigmoid` の宣言はそのまま尊重される。dense 側の既定（key 省略時 sigmoid）は
[qwen35.md](qwen35.md) §1.2 のまま変更しない。

## 3. routed experts の weight 契約（rank-3 native SoA）

expert 毎の `MatrixWeight` 配列は作らない。**1 layer あたり rank-3 の平面 2 本**に集約する。

| 平面 | 中身 | 1 expert の幾何 |
| --- | --- | --- |
| `w13` | `gate_proj` + `up_proj` 融合。**gate rows 先行** | `rows = 2 * moe_intermediate_size`、`cols = hidden_size` |
| `w2` | `down_proj` | `rows = hidden_size`、`cols = moe_intermediate_size` |

`weights::MatrixWeight` の batch 字段:

| 字段 | 意味 |
| --- | --- |
| `experts` | expert 数。1 は rank-2 path（従来の 2D weight） |
| `codes_batch_stride_bytes` / `scales_batch_stride_bytes` | 1 expert 分の平面内 stride |

`rows` / `cols` / `k_padded` / `codes_row_stride_bytes` / `scale_row_stride_bytes` は
**1 expert あたり**の意味のまま。`row_ref` は `[experts][rows]` の byte plane（row stride 1）。
native layout 自体は rank-2 と同一で、expert をまたぐ tile は作らない
（[canonical_soa_payload.md](canonical_soa_payload.md) §6bis）。

### plan

```cpp
// models/qwen35moe/weights/moe_weight_plan.h
Result<MoeExpertGroupPlan> build_moe_expert_group_plan(const Qwen35MoeTextConfig&, std::size_t layer);
std::size_t moe_expert_tensor_count(const Qwen35MoeTextConfig&);   // 物理 tensor 数（layers x experts x 3 x 2）
std::string moe_expert_tensor_name(std::size_t layer, uint32_t expert, const char* projection, bool scale);
```

plan の source は expert 単位・checkpoint ファイル順（`down_proj` → `gate_proj` →
`up_proj`）で、1 source は 1 logical tensor を指す。その payload は codes 平面と
E8M0 scale 平面の両方を含む。scale tensor は物理名 `<name>_scale` であり、
`QuantizedModelReader::resolve` が受け付ける logical name は `<name>` だけである。

### load 順序

```text
disk canonical SoA（1 layer の expert payload は shard 内で連続）
  → host で w13 / w2 平面へ集約（expert 昇順、gate rows 先行）
  → mandatory native preshuffle（batch = experts）
  → codes / scales / row_ref を平面毎に 1 転送
  → GPU 上は native SoA のみ常駐
```

expert 毎の個別 H2D は使わない。ファイル順を native layout に採用もしない
（codes / scale の interleave は canonical の AoS 禁止に反する）。

```cpp
// models/qwen35moe/weights/moe_expert_load.h
Result<MoeExpertGroupWeights> load_moe_expert_group(
    const QuantizedModelReader&, const MoeExpertGroupPlan&,
    gpu::GpuArena&, hipStream_t);
```

### 受理条件（いずれも `Status::invalid_argument`）

- encoding が `Mxfp4` でない
- `logical_shape` / `k_padded` が plan の幾何と一致しない
- codes / scale payload size が padded 幾何から求められる byte 数と一致しない
- source の row range が行き先平面を超える
- plan の `experts` と `sources` 数が一致しない
- E8M0 byte 255（NaN）を含む平面

`row_ref_exact` は平面単位で計算される。1 expert を単独で preshuffle した結果と
byte 一致する（rank-3 でも fold band の判定は expert 毎）。

MXFP4 の rank-3 (N-D) canonical partition / TP materialization は unsupported のまま
（[tensor_partition.md](tensor_partition.md) §5）。

## 4. 対象 checkpoint の幾何アンカー

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層、MXFP4 E4M3 codes + E8M0 group 32）:

| 項目 | 値 |
| --- | --- |
| `hidden_size` / `moe_intermediate_size` / `shared_expert_intermediate_size` | 2048 / 512 / 512 |
| `num_experts` / `num_experts_per_tok` | 256 / 8 |
| routed expert tensor 数 | 61,440（1 layer 1,536 本） |
| 1 layer の expert payload | 427,819,008 B（shard 内で連続） |
| 全 expert payload | 17,112,760,320 B |

shared expert は BF16 rank-2、router は `mlp.gate.weight` `[num_experts, hidden_size]`、
`mlp.shared_expert_gate.weight` `[1, hidden_size]`。

## 5. GPU kernel 契約（routing / grouped linear / combine）

3 つの kernel family は standalone launcher で、`KernelId` も selector も持たない
（§7）。source は `src/shelf/kernels/optimized/moe/` と
`src/shelf/kernels/optimized/linear/mxfp4.hip` の grouped 形。

### routing / group plan

```cpp
// kernels/optimized/moe/router.h
launch_moe_router       // 1 workgroup = 1 token: softmax -> top-k -> renormalize
launch_moe_slot_scan    // 1 workgroup: expert 別 slot 範囲・group table・tile 別 base
launch_moe_slot_scatter // 1 workgroup = 64 token: slot を group row へ写す
moe_slot_tile_count / moe_group_count_upper_bound / moe_group_row_capacity
```

決定順は §2 の contract と同一で、**すべて device 側で決定的**である（expert 内の slot は
token 昇順、atomic は expert 別件数と tile 内 cursor のみに使い、cursor のblock跨ぎ衝突は
tile 別 prefix で起こらない）。

slot は (token, 選択 expert) の組で、数は正確に `rows * top_k`。slot は expert 昇順、
同一 expert 内は token 昇順で番号付けする。さらにその slot 空間を
`kMxfp4GemmRowTile`(=16) slot 以下の **group** に切り、group は expert をまたがない。

**group row 番号でアドレスする**ことが plan の出力形式の核心である。activation 平面・grouped
GEMM 出力・w2 入力・combine の読む行はすべて group row 番号
（`group * 16 + 内offset`）で、slot 番号ではない。これで grouped GEMM は fragment order の
16 行 tile を常に連続して読める。plan が渡すのは

| 配列 | 意味 |
| --- | --- |
| `expert_row_begin[experts + 1]` | expert 別 slot 範囲（最後は slot 総数） |
| `expert_group_begin[experts + 1]` | expert 別 group 範囲 |
| `group_expert[g]` / `group_rows[g]` / `group_count[0]` | group table |
| `row_token[r]` | group row r の token。group の padding 行と `group_count` 以降の全行は -1 |
| `slot_row_of[t * top_k + j]` | token t の j 番目の選択の group row |

group row 数が expert 昇順なので、**token の選択を group row 昇順に並べることは expert
昇順に並べることと同じ**である。combine はこの順序を使う。

host 側の容量は `moe_group_count_upper_bound(rows, experts, top_k)`（=
`min(rows*top_k, rows*top_k/16 + experts)`）と `moe_group_row_capacity` の单一源から取る。
plan 配列はすべてこの容量で確保する。

### activation 平面（group row 順）

grouped linear の activation 平面は `Int8ActivationWorkspaceLayout`（行基準 fold 形、
FP8 W8A8 と同一の fragment order）で、行は group row である。生成は
`launch_activation_quantize_mxfp4_rowref` の `row_index` 間接参照を使う
（`row_index[m] < 0` の行は codes 0・scale 1.0 の zero 行になる）。
token あたり `top_k` 回（+ padding）量子化するコストは現状そのまま許容しており、
1 回量子化してから group row へ複製する形へ替えるかは Gate D の実測で決める。

### grouped linear

```cpp
// kernels/optimized/linear/mxfp4.h
launch_gemm_mxfp4_w4a8_grouped(...)
```

- 1 workgroup = 16 output column × 1 group。expert index・activation 行 tile・出力行 tile
  はすべて group index から決まる。
- weight は rank-3 native SoA。`codes_batch_stride_bytes` /
  `scales_batch_stride_bytes` / `row_ref`（`[experts][rows]`）を launcher が検証し、
  native layout と一致しない stride は `hipErrorInvalidValue` で拒否する。
- `out_features % 16 == 0` と `k_padded % 128 == 0` を要求する（MoE geometry では成立）。
  列 guard は無く、`group_rows[g]` を超える行は **計算するが store しない**。
- weight E8M0 の fold は dense と同じ規則（`weight_ref` を渡すと fold、null なら 32 K
  block ごとに rescale）。呼び出し側は平面の `row_ref_exact` から選ぶ。合成 fixture では
  w13 = true / w2 = false。実 checkpoint の E8M0 幅は未実測（Gate D）。

### combine

```cpp
// kernels/optimized/moe/combine.h
launch_moe_combine(output, expert_output, sel_weight, slot_row_of,
                   shared_output /*nullable*/, shared_gate /*nullable*/, ...)
```

1 workgroup = 1 token。token の `top_k` 個の group row を **昇順（= expert 昇順）** に
並べ替え、`out[f] = Σ_j weight_j * expert_output[row_j][f]` を **f32 で蓄積**して最後に
1 回 bf16 へ丸める。`shared_output` を渡した場合は
`shared_gate_weight[t] * shared[t]` を routed 合計の**後**に足す。
`shared_gate_weight` は**活性化済みの gate 値**で、生成は既存の
`launch_sigmoid_bf16_f32`（`shared_expert_gate` の [1,H] linear 出力が bf16 なのでその
形）が担う。combine 自身は超越関数を使わない。
`shared_output` と `shared_gate_weight` は対で渡す（片方だけなら拒否）。

### 数値の受け入れ

reference（correctness path）ではなく **double 参照に対する dtype 誤差**で受け入れる
（`docs/developer/testing.md` の方針と同じ）。

| kernel | gate |
| --- | --- |
| grouped linear | bf16 exact ≥ 99.9%（GPU が実際に使った operand の double 参照）、rel_l2 < 4e-3 |
| combine | expert 昇順 f32 参照と **bit 一致** |
| router / plan | 選択 expert index は double 参照と完全一致、renormalize 重みは相対 1e-5、実行間 bit 一致 |
| 1 layer E2E | 段ごとは上の gate（GPU operand の double 参照）。**HF f64 に対する相対誤差はクラス誤差として別 gate**: routed rel_l2 < 8e-2、layer 合計 rel_l2 < 8e-2 |

grouped linear の rel_l2 を 4e-3 にしたのは、bf16 の出力丸め自体が相対 2^-9 ≈ 2e-3 で、
rows=1 / top_k=2 のように書かれる行が 2 行しかない断面では正しい丸めの RMS が 2e-3 を
超えるためである（primary gate は bf16 exact 比）。

1 layer E2E で HF f64 への gate を 8e-2 にしたのは、activation を E4M3 group 128 にする
数値 class の差が支配的だからである（design 案 §5 の「A4 にしない」と同じ扱い）。段ごとの
gate（GPU の使った operand の double 参照）はクラス誤差を含まないので 4e-3 で固定できるが、
HF への gate は含む。層合計の relative 値は shared expert 項の大きさで動くため回帰検知としては
弱く、**段ごと gate が正本**である。クラス誤差の実測値と内訳は
[../research/20261104/moe_gate_c_kernels.md](../research/20261104/moe_gate_c_kernels.md) §6。

## 6. test contract

| test | label | 固定する内容 |
| --- | --- | --- |
| `test_moe_config` | `cpu;required` | MoE field の読込、`num_experts > 0` が正本、dense config の拒否、`intermediate_size` 併記拒否、gate 既定（key 省略時 silu） |
| `test_moe_weight_plan` | `cpu;required` | plan の幾何・stride・平面サイズ、実測 payload との一致、source の file order と gate 先行 row range |
| `test_moe_expert_slice_fixture` | `cpu;required` | 合成 checkpoint の shard 内連続配置、manifest の logical / 物理名契約、payload bit 一致 |
| `test_moe_expert_load` | `gpu1;required` | GPU 常駐平面が native block layout、expert 単位で dense path と bit 一致、`row_ref_exact` の平面別判定、arena 使用量、不受理条件 |
| `test_moe_router` | `gpu1;required` | softmax/top-k/renorm の決定順（同値は小 index）、group table と slot→group row 写像の不変条件、padding 行 = -1、実行間 bit 一致、容量 bound |
| `test_moe_grouped_linear` | `gpu1;required` | grouped w13 / w2 が GPU の使った operand の double 参照へ bf16 exact 1.0000、`group_rows[g]` 超の行を書かない、fold / unfold が bit 一致（厳密帯域内）、plan 由来の expert 選択（空 expert・複数 group・skew routing） |
| `test_moe_combine` | `gpu1;required` | expert 昇順 f32 蓄積との bit 一致、shared expert の gate 値を最後に加算、順序を判別する工作例（ascending 0 / descending 2）、入力の不受理条件 |
| `test_moe_layer_e2e` | `gpu1;required` | 1 MoE layer 全体（router → plan → 量子化 → grouped w13 → swiglu → 量子化 → grouped w2 → combine）が HF `Qwen3_5MoeSparseMoeBlock` の f64 参照と、段ごと / routed / 層合計の各 gate で一致。device router の選択は HF と完全一致、Shelf の codec が fixture payload を再現 |

`test_moe_layer_e2e` が読む合成 fixture は `tests/fixtures/qwen35moe/`（commit 済み、
torch 不要で読むだけ）で、再生成は

```sh
python3 tools/moe_layer_oracle_fixture.py --out-dir tests/fixtures/qwen35moe
```

referee は HF transformers の `Qwen3_5MoeSparseMoeBlock` を float64 で実行したものである
（生成時に op-by-op 参照と module forward の bit 一致を検証している）。expert 重みは
Shelf の MXFP4 codec と同一規則で量子化した**あと**の値を HF に入れているため、
量子化誤差は参照側に含まれない。

GPU 不足による skip は required では失敗として扱う（テスト受入ルール）。

## 7. 配線が未実装な範囲

kernel と loader は存在するが、production の model load / lowering はまだ MoE を通さない。
`shelf-compute --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4` は
`quantized_model_reader.cpp` の logical tensor 解決で失敗する。残っているのは

- `model_weights.cpp` の logical tensor 名列挙に expert 名を載せる（`hf_quantized_logical_names`
  は dense FFN 名のみ。`architecture != "qwen35_dense"` の拒否もそのまま）
- `Qwen35LayerWeights` への expert weight 保持と `load_moe_expert_group` の呼び出し元
- `PrimitiveKind` / `PrimitiveNode` への MoE node、`emit_node` の case、
  `KernelId::MOE_ROUTER` / `LINEAR_MXFP4_GROUPED` / `MOE_COMBINE` と
  `try_launch_optimized` / `host_execute_binding` の分岐
- `WeightSlot` の `experts` / batch stride 字段と `make_weight_slot` /
  `upload_host_weight_table` / `validate_host_weight_slot` の追従
- linear selector の allowlist に MoE geometry（`1024:2048` = w13、`2048:512` = w2、
  `256:2048` = router）と grouped 形の選択
- program workspace の group row 容量（`rows * top_k` ではなく
  `moe_group_row_capacity` から決まる行数）
- correctness path には grouped linear も router も無い（fallback は成立しない）
- `shelf-bench` に MoE の計測口が無い（性能主張は配線と計測口の両方が揃うまでしない）

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

実装済みは config 契約と routed experts の weight 集約ロードのみ。
router / grouped linear / combine / lowering / KernelId は未実装で、
`shelf-compute` は MoE checkpoint を production path では実行できない。

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

## 5. test contract

| test | label | 固定する内容 |
| --- | --- | --- |
| `test_moe_config` | `cpu;required` | MoE field の読込、`num_experts > 0` が正本、dense config の拒否、`intermediate_size` 併記拒否、gate 既定（key 省略時 silu） |
| `test_moe_weight_plan` | `cpu;required` | plan の幾何・stride・平面サイズ、実測 payload との一致、source の file order と gate 先行 row range |
| `test_moe_expert_slice_fixture` | `cpu;required` | 合成 checkpoint の shard 内連続配置、manifest の logical / 物理名契約、payload bit 一致 |
| `test_moe_expert_load` | `gpu1;required` | GPU 常駐平面が native block layout、expert 単位で dense path と bit 一致、`row_ref_exact` の平面別判定、arena 使用量、不受理条件 |

GPU 不足による skip は required では失敗として扱う（テスト受入ルール）。

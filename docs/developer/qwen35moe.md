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
| CMake target | `shelf_qwen35moe`（`shelf_qwen_hybrid` に依存。`shelf_qwen35` には依存しない） |

checkpoint の `model_type` は `qwen3_5_moe` だが、runtime family は `qwen35` のまま
（rename しない）。MoE 固有の型・loader はこの module に置き、`shelf::qwen35` 側へ
MoE の分岐を増やさない。逆に `shelf::qwen35moe` は `shelf::qwen35` を include しない。
両家が共有するのは `shelf::qwen_hybrid`（`include/shelf/models/common/qwen_hybrid/`）だけ
であり、dense / MoE の直接依存は `test_architecture_boundaries` が source と
`cmake/targets.cmake` の両方で拒否する。

実装済みは config 契約、routed experts の weight 集約ロード、そして
routing / grouped linear / combine の GPU kernel（standalone launcher）である。
**graph 配線（`PrimitiveKind` / `KernelId` / `WeightSlot` の experts 化 / selector /
lowering）は未実装**なので、`shelf-compute` は MoE checkpoint を production path では
実行できない。model load 配線は §5、kernel 契約は §6、テスト契約は §7、未実装の範囲は
§8 に書く。

## 2. config 契約

`shelf::qwen35moe::Qwen35MoeTextConfig` は FFN に依存しない geometry を持つ
`shelf::qwen_hybrid::QwenHybridConfig` を基底にし、expert 幾何は `MoeFfnConfig ffn` に
まとめる。dense の `intermediate_size` はこの型に存在しない（`Qwen35TextConfig` は
兄弟であり、基底でも MoE の特殊形でもない）。読み込みは
`Result<Qwen35MoeTextConfig> read_qwen35_moe_text_config(const std::string& model_dir)`。
config.json の parse は `qwen_hybrid::read_qwen_hybrid_config` と
`qwen_hybrid::read_text_config_sizes` / `has_text_config_key` の 1 箇所にあり、
FFN の key は各自の family reader が読む。

モデルの識別（identity）と FFN の kind は別のキーで決める。

- **identity**: checkpoint が宣言する `model_type`（`text_config.model_type` も見る）。
  受理集合は `qwen_hybrid::supported_model_identities()` が持つ 4 語
  （`qwen3_5` / `qwen3_5_text` / `qwen3_5_moe` / `qwen3_5_moe_text`）で、ビルド時に固定した
  allowlist である。宣言が無い、または集合外なら `Status::unsupported` で拒否する。
  検査は `qwen_hybrid::check_config_identity` にあり、config を読む入口
  （`read_qwen_hybrid_config` / `read_declared_num_experts` / `read_qwen35_rope_parameters`）
  が全部最初に呼ぶ。したがって未対応アーキテクチャは tensor 名を検索する前に名指しで
  拒否される。identity は**拒否にだけ**使い、どのコードで読むかの選択には使わない。
- **FFN kind**: 正本は `text_config.num_experts > 0`。実 weight 集合（expert tensor 本数）と
  突き合わせて検証できる方が正本である。identity の宣言は検証できないので、
  family を決める根拠にはしない。

identity が routed expert を宣言する config も dense reader は拒否する。ただし構造の検査
（`intermediate_size` と `num_experts`）が先に通り、identity は構造では見えない場合の
backstop として働く。expert 数キーがこの build の語彙と違う綴り（例 `n_routed_experts`）で
書かれていた場合がそれで、dense として受理すると expert tensor を取りこぼす。
identity 名と「routed expert を宣言するか」の対応は `qwen_hybrid` の 1 箇所だけに置く。
dense family の source に MoE の語彙は現れないことを `test_architecture_boundaries` が
静的に調べる（§1）。
この判定設計の調査記録（未対応モデルの失敗形態の実測を含む）は
[../research/20261104/model_family_detection_boundaries.md](../research/20261104/model_family_detection_boundaries.md)。

| key（`config.ffn.*`） | 意味 |
| --- | --- |
| `num_experts` | routed expert 数 |
| `num_experts_per_tok` | 1 token に選択される expert 数 |
| `moe_intermediate_size` | routed expert の intermediate size |
| `shared_expert_intermediate_size` | shared expert の intermediate size |

受理条件（違反は `Status::invalid_argument`）:

- `num_experts > 0`（dense config を MoE reader が拒否する）
- `num_experts_per_tok` は `[1, num_experts]`
- `moe_intermediate_size` と `shared_expert_intermediate_size` は必須
- `intermediate_size` を宣言してはならない（MoE には dense FFN が無い）。
  逆方向も対称で、dense reader は `num_experts` を宣言する config を拒否する（§5）

`text_config.output_gate_type` は読まない。ゲート活性は dense と共通の固定契約
（GDN gate = silu、attention output gate = sigmoid）で、
[qwen35.md](qwen35.md) §1.2 を参照。

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

batch stride の式は `include/shelf/quantization/mxfp4/mxfp4_native.h` の
`mxfp4_native_codes_batch_stride_bytes` / `mxfp4_native_scale_batch_stride_bytes` が単一源で、
plan・expert plane loader・weight table validation・grouped GEMM launcher の 4 箇所はいずれも
この式を読む（式を書き直す箇所は作らない）。`Mxfp4NativeView::validate()` も同じ式で
codes 平面長を検証する。

plan は `k_padded` が **128 の倍数**であることを要求する。grouped GEMM は K を 128 単位で
読むため、64 単位で padding した平面は kernel が読めない。padding 境界で不一致が起きても
launch 時ではなく plan 時点で `Status::unsupported` にする。

`runtime::WeightSlot` も同じ `experts` / batch stride 字段を持ち、両 family の lowering は
共通の `qwen_hybrid::make_weight_slot`（`models/common/qwen_hybrid/weight_slot.h`）で
`MatrixWeight` を写す。weight table を device へ送るのは slot の構造体そのものの写しであり、
字段を個別に列挙しない（列挙形式は字段を追加すると落ちる）。
`validate_host_weight_slot` は `enc == Mxfp4` の rank-3 slot に対して
`rows % 16 == 0` / `k_padded % 128 == 0` / batch stride が上記の式と一致を要求し、
`experts > 1` の slot が MXFP4 以外であることを拒否する。grouped launcher の受理条件と
同じ条件であり、host 側で落ちていれば launch まで待たずに失敗する。

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
- source の expert が plan の expert 範囲外（分割 plan の名前がその rank の範囲を指さない）
- plan の `experts` と `sources` 数が一致しない
- E8M0 byte 255（NaN）を含む平面

`row_ref_exact` は平面単位で計算される。1 expert を単独で preshuffle した結果と
byte 一致する（rank-3 でも fold band の判定は expert 毎）。

### expert 軸の分割（rank あたりの平面）

routed expert 平面は matrix axis を持たないため、`TensorPartitionPlan` では分割できない。
分割は expert 軸そのもので、`MoeExpertGroupPlan` が持つ。

```cpp
// models/qwen35moe/weights/moe_weight_plan.h
Result<MoeExpertGroupPlan> build_moe_expert_group_plan(
    const Qwen35MoeTextConfig&, std::size_t layer, uint32_t tp_size, uint32_t tp_rank);
```

- `experts` はその plan が平面に持つ expert 数、`expert_begin` はその最初の global expert id、
  `global_experts` は checkpoint の平面 depth であり router が top-k を選ぶ幅。
  `experts * tp_size == global_experts` と `expert_begin == tp_rank * experts` が contract
- `num_experts` が `tp_size` で割り切れない場合は `Status::unsupported`。分割は contiguous な
  expert 範囲のみ（非均等な expert 配分は無い）
- source の名前は global expert id を保つ（checkpoint の名前は global 単位）。集約時の
  行き先 slot は `expert - expert_begin`
- fused rank-3（MTP block）も同じ範囲で切る。checkpoint の tensor は `[global_experts][rows][cols]`
  のまま読まれ、再量子化するのは所有範囲の expert だけ
- 1 expert の layout（rows / cols / k_padded / batch stride / native preshuffle）は分割前後で
  同一。したがって rank 平面を global 順に並べた byte 列は単一 GPU の平面と一致する
  （`test_qwen35moe_model_load` が byte 一致で固定）

routed MXFP4 expert は `materialize_rank_local_canonical` を通らない（MXFP4 の N-D canonical
partition は今も unsupported。[tensor_partition.md](tensor_partition.md) §5）。分割は上の
expert 軸で起き、matrix axis の分割は backbone と shared expert だけが使う
（`build_qwen35moe_tensor_partition_plan`）。

## 4. 対象 checkpoint の幾何アンカー

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層、MXFP4 E4M3 codes + E8M0 group 32）:

| 項目 | 値 |
| --- | --- |
| `hidden_size` / `moe_intermediate_size` / `shared_expert_intermediate_size` | 2048 / 512 / 512 |
| `num_experts` / `num_experts_per_tok` | 256 / 8 |
| routed expert tensor 数 | 61,440（1 layer 1,536 本） |
| 1 layer の expert payload | 427,819,008 B（shard 内で連続） |
| 全 expert payload | 17,112,760,320 B |
| GPU 常駐 expert 平面（codes + scales + row_ref） | 17,144,217,600 B = `moe_expert_group_bytes` の 40 層合計 |

shared expert は BF16 rank-2、router は `mlp.gate.weight` `[num_experts, hidden_size]`、
`mlp.shared_expert_gate.weight` `[1, hidden_size]`。

## 5. model load 配線（dense / MoE の入口）

MoE checkpoint の weight load は `shelf::models` の factory が選ぶ。family を決めるのは
config.json の `num_experts` だけで、`model_type` は選択に使わない（§2 と同じ正本）。
ただし §2 の identity gate が先に通り、この build が実装していないアーキテクチャは
factory に到達する前に拒否される。

```cpp
// shelf/models/model_weights_factory.h
enum class ModelFamily { Dense, Moe };
using LoadedModelWeights = std::variant<
    shelf::qwen35::Qwen35ModelWeights,          // family == Dense
    shelf::qwen35moe::Qwen35MoeModelWeights>;   // family == Moe
ModelFamily model_family(const LoadedModelWeights&) noexcept;
Result<ModelFamily> probe_model_dir(const std::string& model_dir);
Result<LoadedModelWeights> load_model_weights(model_dir, arena, stream, options);
```

- factory が保持するのは **有効な 1 family の weight だけ**。両 family を同時に持つ
  フィールドと、MoE から dense を取り出す `body()` は存在しない。具体型への access は
  `std::get_if<shelf::qwen35moe::Qwen35MoeModelWeights>(...)` で明示する
  （null は「その family では無い」ことを意味する）。
- `shelf_models` は `shelf_qwen35` と `shelf_qwen35moe` を link する唯一のターゲット。
  両 family が互いを参照する必要はない（§1）。
- factory はその family の validation を呼ぶ。GPU 実行方式は決めない。

### MoE 側の weight 保持

```cpp
// shelf/models/qwen35moe/weights/moe_ffn_weights.h
struct MoeFfnWeights {          // 1 layer の MoE FFN（GPU 常駐）
    MatrixWeight router;          // [experts, hidden]
    MatrixWeight expert_w13;      // rank-3 native SoA、gate 行が先
    MatrixWeight expert_w2;       // rank-3 native SoA
    MatrixWeight shared_gate;     // [1, hidden]
    MatrixWeight shared_gate_proj, shared_up_proj, shared_down_proj;
};

// shelf/models/qwen35moe/weights/qwen35moe_model_load.h
struct Qwen35MoeLayerWeights : shelf::qwen_hybrid::HybridLayerWeights {
    MoeFfnWeights ffn;
};
struct Qwen35MoeModelWeights : shelf::qwen_hybrid::HybridModelLevelWeights {
    std::vector<Qwen35MoeLayerWeights> layers;
};
std::vector<std::string> moe_aux_tensor_names(layer);   // router / shared 5 本
std::vector<std::string> moe_layer_logical_names(plan); // aux + experts（plan 順）
uint64_t moe_expert_group_bytes(plan);                  // 常駐平面 byte の単一源
Status moe_ffn_weights_match(ffn, plan, ffn_config);
Status validate_qwen35moe_weights(weights, config);
Result<Qwen35MoeModelWeights> load_qwen35moe_weights(model_dir, arena, stream, options);
```

- **layer 配列は 1 つ**。dense FFN の空フィールドも、dense model の内包も無い。
  backbone（layernorm 2 本 + GDN / full attention）は `HybridLayerWeights` の字段で、
  層の `is_gdn` と同じ構造体に入るので index の暗黙的同期は発生しない。
- MoE layer 型は dense FFN 字段を表現できず、逆もまた同様
  （`test_qwen35moe_model_load` の concept + `static_assert`）。
- `moe_layer_logical_names` は logical 名だけを返す。物理 `<name>_scale` は要求しない（§3）。
- `moe_ffn_weights_match` は expert 平面を plan で、router と shared expert を
  `MoeFfnConfig` で検証する。shared expert の幅は `shared_expert_intermediate_size` であり、
  routed expert の `moe_intermediate_size`（plan の `w13_gate_rows`）とは混同しない。
  両者が異なる checkpoint も受理し、routed 幅の shared expert は拒否する。

### 共通 backbone loader

FFN の形に依存しない部品は `shelf/models/common/qwen_hybrid/weight_loader.h` にある。
dense 側が公開する load 部品は存在せず、両家が同じ部品の上位に自分の FFN を置く。

```cpp
qhy::layer_tensor_prefix(layer)            // "model.language_model.layers.<l>."
qhy::hf_quantized_body_names(config)       // embed / lm_head / norm + 層 layernorm / attn
qhy::open_quantized_model_reader(model_dir, logical_names, verify_crc)
qhy::load_hybrid_model_level_quantized(reader, config, arena, stream, options, weights)
qhy::load_hybrid_layer_body_quantized(arena, stream, reader, config, layer, options, w)
```

`load_hybrid_layer_body_quantized` は `qhy::layer_is_gdn(config, layer)` で topology を
決める（`layer_types` が層数分あればそれを優先し、無ければ `full_attention_interval` から
導出する）。呼び出し側が `is_gdn` を渡さないため、load 結果と config が食い違えない。
options は `Qwen35MoeLoadOptions`（`qhy::HybridLoadOptions` を base に `load_mtp_layers` を足した型）。
vocab partition の dense 固有オプションは MoE 側は要求できない。

### 共通 invariant

`shelf/models/common/qwen_hybrid/weight_validation.h`:

```cpp
qhy::validate_hybrid_model_level(model, config, loaded_layer_lo, loaded_layer_hi)
qhy::validate_hybrid_layer_body(layer, config, layer)
qhy::validate_hybrid_layer_body(layer, hidden, is_gdn, tag)
qhy::check_matrix_shape(w, rows, cols, tag)
```

- model level: 読み込んだ layer 範囲が layer 0 から始まるなら embed（tied head を持つ最終範囲も
  含む）、最終層で終わるなら lm_head と final norm を要求し、中間範囲はどちらも要求しない。
  vocab は partition を模型側の次元へ戻して照合し、hidden 幅と `lm_head_tied` と実体の一致も見る
- model level は layer 数を検証しない。vector の長さと config / stage 範囲の突き合わせは
  呼び出し側（`validate_qwen35_weights` と `validate_qwen35moe_weights`）が持つ
- layer body: 2 つの layernorm と、その topology が要求する attention tensor。
  逆 topology の tensor が残っていれば拒否する
- partition は axis と count で模型側の次元へ戻して比較するため、TP rank でも成立する

FFN の検査は各 family の validation が持つ。dense は `validate_qwen35_weights`
（`Qwen35TextConfig` を取る）で gate / up = `[intermediate, hidden]`、
down = `[hidden, intermediate]`、MTP block を見る。MoE は `validate_qwen35moe_weights`
で plan の幾何・stride・row_ref と router / shared 形状を見、`mtp.present` のときは
`validate_qwen35moe_mtp_weights` を同じ validation の一部として通す。

### MTP block

checkpoint が `mtp.*` として持つ 1 層ぶんの transformer を読む。transformer layer 配列とは
別個に、`Qwen35MoeModelWeights::mtp` へ入れる。

- body（layernorm と full attention）は `qhy::load_hybrid_layer_body_quantized_at` を
  prefix `mtp.layers.0.` と `is_gdn = false` で読む。名前列は
  `qhy::hf_layer_body_names(prefix, is_gdn)` が同じ列表を使う
- routed expert は transformer layer と同じ plan 幾何だが、checkpoint では fused rank-3 の BF16
  （`mtp.layers.0.mlp.experts.gate_up_proj` = `[experts][2 * moe_intermediate][hidden]`、
  `experts.down_proj` = `[experts][hidden][moe_intermediate]`）で保存される。
  `build_moe_mtp_expert_group_plan` は `fused_rank3 = true` と 2 つの平面名を持ち、
  `load_moe_expert_group` は `gather_fused_plane` で MXFP4 へ再量子化して native SoA 平面に落とす。
  gate rows が前半、up rows が後半（HF の `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` と同じ順）
- block 固有 tensor は `mtp.fc.weight` = `[hidden, 2 * hidden]`、
  `mtp.pre_fc_norm_embedding.weight` / `mtp.pre_fc_norm_hidden.weight` / `mtp.norm.weight` = `[hidden]`
- router と shared expert は transformer layer と同じ名前の並びを prefix `mtp.layers.0.mlp.` で読む
  （`moe_aux_tensor_names(mlp_prefix)`）

### 受理条件

| 条件 | 結果 |
| --- | --- |
| `options.weights.partition_plan != nullptr` | その rank の load。plan の `tp_size` / `tp_rank` が 2 未満なら `Status::invalid_argument`。backbone と shared expert は plan の matrix axis で分割、routed expert は expert 軸で分割（§3-1） |
| MTP 層の要求で checkpoint に `mtp.*` が無い | reader の要求名前列が tensor 欠落で失敗する（block を無いものとして通さない） |
| `mtp.present` のくに `mtp.fc` が `[hidden, 2 * hidden]` で無い / block norm が `[hidden]` で無い / body が full attention tensor を持たない / FFN が MTP plan と不一致 | `validate_qwen35moe_mtp_weights` の `Status::invalid_argument` |
| Shelf quantized bundle（`shelf_quantization.json` がある dir） | `Status::unsupported`（MoE は HF real-quantized dir のみを読む） |
| dense config（`num_experts == 0`） | `read_qwen35_moe_text_config` の `Status::invalid_argument` |
| loaded 平面が plan の幾何 / stride / row_ref と不一致、router・shared expert が config の幅と不一致 | `moe_ffn_weights_match` の `Status::invalid_argument` |
| 層 topology 違反、backbone tensor 欠落、layer 数不一致 | `validate_qwen35moe_weights` の `Status::invalid_argument` |

dense loader は MoE checkpoint を config 段で拒否する。`read_qwen35_text_config` は
`intermediate_size` を必須とし、`num_experts` を宣言する config を
`dense config must not declare num_experts` で弾く（`num_experts: 0` は dense のまま）。
この拒否と上の表は `test_qwen35moe_model_load` が固定する。

load 時間 / host RSS / 実 checkpoint の `row_ref_exact` 平面数など、この段階の実測は
[../research/20261104/moe_gate_d_load_wiring.md](../research/20261104/moe_gate_d_load_wiring.md)
と [../research/20261104/moe_ffn_architecture_split.md](../research/20261104/moe_ffn_architecture_split.md)。

## 6. GPU kernel 契約（routing / grouped linear / combine）

3 つの kernel family は standalone launcher で、`KernelId` も selector も持たない
（§8）。source は `src/shelf/kernels/optimized/moe/` と
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

**router の non-finite 契約**。logits に 1 つでも non-finite（NaN / +INF / -INF）を含む行がある
routing round は数値異常とする。Qwen3.5 MoE の router は logits へ正当な `-INF` マスクを置か
ないので、`-INF` も異常に含める。

- kernel は plan をメモリ安全に保つ。選択 expert は常に `[0, experts)` に収まり、`sel_weight` は
  finite である。全 entry が non-finite の行は index 昇順で `top_k` 個を選び、weight は 0 とする
- 同じ行で program の error word へ `ProgramStatus::NUMERIC_ERROR` を立てる
- 上位はそれを見て推論を中断し、エラーとして報告する。executor は commit 時点で word を読み、
  pending を poison して `Status::invalid_state` を返す

この fallback は illegal memory access で GPU context を壊さないための処置であり、異常な推論
結果の継続を許容するものではない。範囲外の expert index は `tile_counts` への累算と scatter の
アクセスを plan 配列の容量外へ導くので、producer 側であり得なくする（`test_moe_router` の
non-finite case と `test_moe_program_execute` の 1 entry が NaN の case が張る）。

scan は `tile_counts` を**消費してゼロに戻す**。層を跨いだ routing round は同じ scratch を
使うが、層毎の memset launch は足さない（消費を張った契約は `test_moe_router` の chained
round case）。

**expert 軸を分割した rank の routing**。`launch_moe_router` は logits 幅（model 全体の
expert 数）で softmax / top-k / 再正規化を行い、`expert_begin` から `owned_experts` 個の
範囲だけで plan を作る。範囲内の選択は `sel_expert` へ**範囲内の index** を書き、範囲外の
選択は `kMoeSlotNotOwned`（-1）を書く。`sel_weight` は全 selection に対する再正規化値のまま
どの rank でも同じなので、rank の combine 出力は部分和となり、その和が分割しない run と
同じ値になる（barrier の f32 累積 → 1 回だけ bf16 へ丸める契約は
[tensor_parallel_execution.md](tensor_parallel_execution.md) §6）。
`moe_slot_scatter_kernel` はマーカーの slot を plan に載せず `slot_row_of` へ -1 を書く。
`tile_counts`・`expert_row_begin`・`group_expert`・grouped 行容量は `owned_experts` 基準で、
plan scratch と activation 平面もその分だけ小さくなる（footprint は
`moe_executor_grouped_bytes(config, rows, tp_size)`）。
`moe_combine_kernel` は `slot_row_of < 0` の行を加算しないので、combine 側の契約は変わらない。
選択の順序・決定順・non-finite 契約は分割前後で同一である。

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

host 側の容量は `include/shelf/runtime/graph/grouped_row_shape.h` が単一源である
（`grouped_row_tile_count` = `min(rows*top_k, rows*top_k/16 + experts)`、
`grouped_row_capacity` = 容量行数、`kGroupedRowTile` = 16）。
`kernels/optimized/moe/router.h` の `moe_group_count_upper_bound` /
`moe_group_row_capacity` / `moe_slot_tile_count` はこれを呼ぶ alias で、
`static_assert(kMxfp4GemmRowTile == kGroupedRowTile)` が kernel の行 tile と張り合わせる。
plan 配列はすべてこの容量で確保する。

graph 側では group row 空間は `ValueRowDomain::GROUPED_ROWS` の value であり、plan geometry
（`experts` / `top_k`）は `PrimitiveGraph::grouped_row_shape` に 1 度宣言する（value 毎に
繰り返さない）。`GROUPED_ROWS` の value があるのに shape が未宣言なら
`validate_primitive_graph` と `build_program` の両方が拒否する。program の
`rows_for(GROUPED_ROWS)` は bucket の token rows（`max_token_rows` を適用した後）から容量を
計算するので、activation 平面・grouped 出力の binding byte 数は plan 配列と同じ容量基準に
なる。token rows と group row を混同した確保はできない。

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
                   shared_output /*nullable*/, shared_gate /*nullable*/,
                   rows, top_k, features, expert_row_stride, shared_row_stride,
                   shared_gate_row_stride, stream)
```

1 workgroup = 1 token。token の `top_k` 個の group row を **昇順（= expert 昇順）** に
並べ替え、`out[f] = Σ_j weight_j * expert_output[row_j][f]` を **f32 で蓄積**して最後に
1 回 bf16 へ丸める。`shared_output` を渡した場合は
`shared_gate_weight[t * shared_gate_row_stride] * shared[t]` を routed 合計の**後**に足す。
`shared_gate_weight` は**活性化済みの gate 値**で、生成は既存の
`launch_sigmoid_bf16_f32`（`shared_expert_gate` の [1,H] linear 出力が bf16 なのでその
形）が担う。combine 自身は超越関数を使わない。
`shared_output` と `shared_gate_weight` は対で渡す（片方だけなら拒否、gate の row stride が
0 なら拒否）。**gate は 1 列でも row-major value の平面なので stride を渡す**——packed
（添字 1 刻み）で読むと token 0 以外は padding か別データの bytes を読み、shared expert の
寄与が消えるまたは汚染される。この規則は `test_moe_combine` の padded gate case と
`test_qwen35moe_rows_parity`（§7）で固定する。

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

段ごと・1 層 E2E の gate はいずれも**固定 token 数**の比較である。同じ token を
**違う batch 行数**で forward したときの一致は `test_qwen35moe_rows_parity`（required）で
受け入れる。合成 MoE model（GDN 層を含む fixture と全 layer full attention の fixture）を
executor で走らせ、1 行ずつの walk と multi-row walk（4+4 / 2x4 / 8 / 3+5 / 1+7 / 2+6 /
5+3 / 3+3+2 / 1+1+6 / 4+1x4 / 1x4+4）の token 行ごと hidden を突き合わせ、
rel_l2 < 5e-2 を要求する（実測は全 case・全 row で 0、bit 一致）。
実 checkpoint では chunked prefill と stepwise decode の GDN 丸め差（layer 0 の router logits
で rel 3e-5 級）が残る。grouped expert を使う model では router の top-k 選択が離散的なので
この差が expert 集合の反転へ増幅し、walk をまたいだ token 列の bit 一致は要求しない。
経緯と実測は [../research/20261105/moe_combine_gate_stride.md](../research/20261105/moe_combine_gate_stride.md)。

## 6-1. graph / program の MoE 契約（配線済み）

primitive は 3 種で、tensor arity は固定である（`primitive_graph.cpp` の `arity_of`）。

| PrimitiveKind | in / out | KernelId |
| --- | --- | --- |
| `MOE_ROUTER` | 1 / 0 | `MOE_ROUTER` + `MOE_PLAN_SCAN` + `MOE_PLAN_SCATTER`（1 node から 3 dispatch） |
| `MOE_GROUPED_LINEAR` | 1 / 1 | `ACTIVATION_QUANTIZE_MXFP4_ROWREF` + `LINEAR_MXFP4_W4A8_GROUPED`（1 node から 2 dispatch。dense の LINEAR と同じ形） |
| `MOE_COMBINE` | 3 / 1 | `MOE_COMBINE` |

- `MOE_ROUTER` の入力は router linear の **f32** logits（features = `experts`、TOKEN_ROWS）。
  選択と plan は value にしない（§6）ので、この node は tensor を産まない。
- `MOE_GROUPED_LINEAR` の `shape` は **1 expert の {output, input} features**。expert 数と
  plane stride は weight slot が持つ（§3）。`k_padded` が activation 平面の padding
  （`activation_k_padded_for(64, cols)`）と一致しない expert 平面は program build が拒否する。
  平面が rank-3（`experts >= 2`）でない weight も拒否する。
- gate / up 融合平面の 2 半分は `PrimitiveValueView`（`feature_offset = 0` と `= inter`）で
  `SWIGLU` に渡す。複製・split はしない。
- `MOE_COMBINE` の入力は **expert 出力（GROUPED_ROWS）・shared expert 出力（TOKEN_ROWS）・
  活性済み shared gate（TOKEN_ROWS, f32, 1 列）**、出力は TOKEN_ROWS の hidden。
  shared expert は Qwen3.5 MoE の必須部品なので arity は固定で、gate 活性は既存の
  `SIGMOID` primitive が担う（§6）。

### plan scratch

routing round の scratch（`sel_expert` / `sel_weight` / `tile_counts` / `expert_row_begin` /
`tile_base` / `expert_group_begin` / `group_expert` / `group_rows` / `group_count` /
`row_token` / `slot_row_of`）は **program workspace の 1 本の range** であり、配置は
`runtime/program/moe_plan_scratch.h` の `MoePlanScratch::make(tokens, experts, top_k)` が
単一源である（unit は uint32、region は end to end）。program は 1 本だけ確保し、
routing round を使う全 dispatch が同じ range を指す。

`DispatchBinding` は `scratch_slot` と `scratch_tokens` / `scratch_experts` /
`scratch_top_k` を持つ。offset そのものを binding に写さず、consumer は同じ
`MoePlanScratch::make` を呼んで offset を導く（二重に持たせない）。
`tokens` はその program の bucket の token rows なので、確保容量と読む容量がズレない。
`MOE_ROUTER` の binding だけが `scratch_logit_experts` と `scratch_expert_begin` を持つ。
expert 軸を分割した rank では logits 幅（model 全体の expert 数）と plan の深さ
（`scratch_experts`）が異なるためで、0 は「logits 幅は plan の深さと同じ」を意味する
（§6 の分割契約。scan / scatter の binding は local 幅のみを見る）。

`Program::validate()` は `scratch_slot` が range を指すことと geometry が非ゼロであることを
検証し、program は `plan_scratch_slot` を持つ。

**plan scratch は dispatch span ではなく program 全体で offset を確保する**。workspace packer
は生涯の重ならない block を使い回すが、router より前に死ぬ block を同じ offset に置くと、run
開始時の zeroing がその byte で上書きされ、`tile_counts` が zero から累算されない（壊れた
plan は scatter の不正アクセスで context を止める）。`validate()` は plan range が他の
workspace range や workspace storage の value と重ならないことを検証する。

### 実行側の契約

- router は token tile 別の件数を **累算**する。したがって routing round は zero から始まら
  なければならず、`prepare_program_run` が program run の開始時に plan scratch range を 1 回
  memset する（層毎ではない。layer L の scatter までが layer L の round を使い、layer L+1 の
  router はその後に走り、scan が件数を消費してゼロに戻す）。memset は **実行経路を選ぶ直前の
  1 箇所**で置く。single rank でも multi rank でも同じ前処理を通すため、前処理は経路側
  （`execute_program_range` / `execute_program_tp_ranges`）では行わない。production の入口は
  `launch_program_run`、test 入口は `execute_program`。multi rank では rank が自分の stream 上で
  自分の scratch を準備する。
- fallback した dispatch は correctness interpreter へ渡るが、grouped 系 kernel に
  correctness fallback は無い。どれが fallback したかは kernel 単位の計数が正本で、
  `SHELF_QWEN35_PERF_STATS=1` の dump JSON が kernel 毎に `logical` / `optimized` /
  `fallback`（と `fallback_rows` の min・max）を出す。contract は
  `docs/developer/qwen35.md` §5。optimized dispatch が失敗したときの status には
  op index・kernel 名・rows が載る。
- kernel 別の計数は thread 単位で集まるので、multi rank の rank thread 分は main thread の
  dump に乗らない（dump が 0 になる）。multi rank で fallback を見るには
  `SHELF_OP_FALLBACK_DEBUG=1` を使い、op index・kernel・io の feature count / dtype /
  row domain を読む。kernel 別の所要時間を見るなら `SHELF_OP_DUMP=1`（per-op 同期が入るので
  絶対値は使わない）。
- device 側から実行側への報告は program の error word 1 語だけで行う。kernel は
  `ProgramStatus::COMPLETE` から `atomicCAS` で最初の異常を記録し、executor は commit 時点で
  staged された word を読んで、非 COMPLETE なら pending を poison して `Status::invalid_state`
  を返す。`NUMERIC_ERROR`（値が 6）は kernel が non-finite を見つけた場合で、router が
  `ctx.error_word` へ立てる。optimized dispatch は kernel に `ctx.error_word` を渡す責務を持つ。
- group-row value に触れる dispatch は、plan 配列を読まない場合でも `scratch_tokens` /
  `scratch_experts` / `scratch_top_k` を持つ（`build_program` が emit 後に付与する）。
  kernel 側はこれを `moe_group_rows(b, ctx)` で実際の token 数から引き直した group 行数に
  使うので、`rows * top_k` でも bucket の token 行数でもない、**その batch の group 行数**
  で work が決まる。
- grouped 系の kernel には correctness fallback が無い。加えて **correctness interpreter は
  group 行数を数えられない**（`value_rows()` は token 行数を返す）。したがって
  `GROUPED_ROWS` の value に触れる dispatch は optimized が NotApplicable でも interpreter へ
  落とさず、executor が `Status::unsupported` を返す（op index・kernel 名・group 行数を載せる）。
  geometry は allowlist に載っている必要がある: elementwise selector の `SwigluBf16 @512` と
  `@128` は group 行数専用の bound `kMeasuredMaxGroupRows`(36608) まで、
  `SigmoidBf16ToF32 @1` と `ResidualBf16 @2048` は token 行数で `kOptimizedMaxRows`(8192) まで。
  group 行数 capacity は `grouped_row_capacity(rows, {experts, top_k})` で、group 数は routing
  に対する**厳密な最大値**である（`tiles <= slots`、
  `tiles <= (slots + experts * (tile - 1)) / tile`、
  `tiles <= experts * ceil(rows / tile)`。最後の項は 1 token が同じ expert を選ばないこと
  由来で、これが無い旧式は token が少ない bucket で最大 2 倍過大だった）。MoE geometry
  （experts 256・top_k 8）なら token 1 / 16 / 64 / 512 / 1024 / 2048 / 4096 が
  128 / 2048 / 4352 / 7936 / 12032 / 20224 / 36608 行。
  **grouped expert の token row bucket は 4096 まで**で、それを enforce するのが
  `kMeasuredMaxGroupRows` である（8192 token 相当の 69376 行は allowlist 外なので
  `Status::unsupported`）。elementwise kernel は行を
  `blockIdx.y + blockIdx.z * gridDim.y` で address する（grid.y 単独の上限は 65535 で、
  `grid.y = 69632` の launch は `invalid configuration argument` になる）ので、
  4096 という上限は kernel の限界ではなく policy である。
  厳密性の検証は `tests/unit/test_grouped_row_shape.cpp`（小さな形状で DP により全 routing
  の最大値を列挙して一致させる）と `tests/unit/test_moe_combine.hip`（実 router 出力の
  group 数が host capacity を超えないこと）。
- fused gate / up 平面は 2 つの view として読まれる。elementwise dispatch の alias 検査は
  **読み専用 operand 同士は同じ buffer を共有してよい**（行 stride で span が重なるため）、
  出力だけが各 operand と同一または非重複であることを要求する
  （`elementwise_dispatch.hip` の `output_alias_ok`）。

## 6-2. lowering の契約（`shelf::qwen35moe`）

`lower_qwen35_moe_to_primitives(config, weights, options)` は `Qwen35MoeTextConfig` と
`Qwen35MoeModelWeights` を受け取り、family 中立な `qwen_hybrid::HybridLoweredPrimitives` を
返す。保持型が同一なので executor は family に関係なく 1 型を消費する。

- 入口で `validate_qwen35moe_weights` を通す。load 済みの weight に対しても lowering は
  自分の graph の幾何を保つため、検証を省略しない。
- `Qwen35MoeLowerOptions` に tensor partition 字段は無い。MoE の lowering は 1 device を
  前提とする（rank-3 expert 平面は partition contract の外。§5 と同じ理由）。
  `emit_normed_token_hidden` は target graph の external output を 1 本増やす
  （[mtp.md](mtp.md) の target graph 節）。rope は config の `rope_theta` /
  `partial_rotary_factor` と options の `rope_attention_factor` から決める。
- MTP block は `lower_qwen35_moe_mtp_to_primitives` が別 graph として lower する。
  外部入力は `[token_ids, hidden]`、state は KV cache 1 本（`state_index 0`）だけで GDN state は
  持たない。body は `emit_hybrid_block`（full attention 固定）、FFN は target と同じ
  `emit_moe_ffn` を MTP plan で呼び、head は `mtp.norm.weight` を final norm として
  `tail_output_is_normed_hidden = true` で `emit_hybrid_head` を使う。
  plan は §5 の MTP block（fused rank-3 を load 時再量子化したもの）を読む。
- `pg.grouped_row_shape = {num_experts, num_experts_per_tok}` を 1 度だけ宣言する。
  value にも node にも容量は書かない（§6-1）。expert 軸を分割した rank では
  `{local_experts, top_k}` になり、`MoeRouterNode` が logits 幅としての `experts` と
  その rank の `local_experts` / `expert_begin` を併せ持つ（§3「expert 軸の分割」）。
  分割時は `MOE_COMBINE` node に `tp_combine = 1` を立て、combine 出力を collective point
  にする（`o_proj` / `gdn_out` と同じ機構。単一 device では execution schedule を作らない
  ため、この annotation は inert）。
- weight table の並びは embedding → 層ごと（backbone 線形 → router → expert w13 →
  expert w2 → shared expert gate / up / down → shared expert gate）→ lm_head
  （tied の場合は embedding の index を使い回す）。expert 平面は rank-3 のまま
  （`experts` と native batch stride を持つ）で、平面を 2D へ作り替えない。
- FFN の node 列は §6-1 の primitive 表のとおり。shared expert の gate 活性は `SIGMOID`
  （bf16 logit → f32）で、`MOE_COMBINE` の 3 番目の入力へ渡す。
- backbone（input norm → GDN / full attention → attention residual → post norm）と
  model head（`OUTPUT_GATHER` → final norm → lm_head → `SAMPLING`）は `qwen_hybrid` の
  `emit_hybrid_block` / `emit_hybrid_head` を使う。head の外部出力 4 本
  （selected_hidden → logits → sampled → 末尾出力）の順は executor の staging 順なので
  呼び出し側で並べ替えない。
- hidden tap は層の出力 value を外部出力へ足す。最終層の tap は head の末尾出力と同一 value
  になるため graph 検証が duplicate として拒否する（dense と同じ挙動）。

## 7. test contract

| test | label | 固定する内容 |
| --- | --- | --- |
| `test_moe_config` | `cpu;required` | MoE field の読込、`num_experts > 0` が正本、dense config の拒否、`intermediate_size` 併記拒否、`output_gate_type` を宣言した config も受理すること。`Qwen35MoeTextConfig` に dense 幅字段が無く `Qwen35TextConfig` に expert 字段が無いことを concept + `static_assert` で固定 |
| `test_moe_weight_plan` | `cpu;required` | plan の幾何・stride・平面サイズ、実測 payload との一致、source の file order と gate 先行 row range 、expert 軸分割の plan（rank の contiguous 範囲・global 名を持つ source・平面 depth が半分・per-expert layout 不変）、`num_experts` が tp_size で割れない場合と tp_size / tp_rank の範囲外の拒否 |
| `test_moe_expert_slice_fixture` | `cpu;required` | 合成 checkpoint の shard 内連続配置、manifest の logical / 物理名契約、payload bit 一致 |
| `test_moe_expert_load` | `gpu1;required` | GPU 常駐平面が native block layout、expert 単位で dense path と bit 一致、`row_ref_exact` の平面別判定、arena 使用量、不受理条件。fused rank-3 plan では checkpoint の BF16 payload を再量子化して preshuffle した結果と平面が bit 一致すること、fused plan を per-expert checkpoint へ / unfused plan を fused checkpoint へ適用したときの拒否も固定 |
| `test_moe_router` | `gpu1;required` | softmax/top-k/renorm の決定順（同値は小 index）、group table と slot→group row 写像の不変条件、padding 行 = -1、実行間 bit 一致、容量 bound、外部 zeroing なしの 2 つ目の round が同じ plan になること（scan が `tile_counts` を消費する契約）、non-finite logits で expert index が `[0, experts)` に収まり weight が finite であること、その round が error word へ `NUMERIC_ERROR` を立てること（有限 logits では word が立たないことも一緒に固定）。expert 軸分割では、同じ logits で unsplit と 2 rank を比べ、選択と再正規化 weight が全 slot で完全一致、`sel_expert` が local index か `kMoeSlotNotOwned`、slot がちょうど一方の rank でのみ group row を持つ、両 rank の slot 数の和が全 slot と一致、`group_expert` が local 平面を指すことを固定。範囲外の owned 範囲を要求する launch の拒否も固定 |
| `test_moe_grouped_linear` | `gpu1;required` | grouped w13 / w2 が GPU の使った operand の double 参照へ bf16 exact 1.0000、`group_rows[g]` 超の行を書かない、fold / unfold が bit 一致（厳密帯域内）、plan 由来の expert 選択（空 expert・複数 group・skew routing） |
| `test_moe_combine` | `gpu1;required` | expert 昇順 f32 蓄積との bit 一致、shared expert の gate 値を最後に加算（gate 平面は padded row stride で置く case を含む。packed 前提だと 2 行目以降が壊れる）、順序を判別する工作例（ascending 0 / descending 2）、入力の不受理条件 |
| `test_qwen35moe_model_load` | `gpu1;required` | model-level MoE load: 単一 layer 配列で backbone（layernorm / attention）と MoE FFN（router・rank-3 expert 平面・shared expert）が plan の幾何・stride・row_ref と一致、shared expert が routed expert と異なる `shared_expert_intermediate_size` を持つ checkpoint も受理し routed 幅へ伪造は拒否、layer topology 違反 / backbone tensor 欠落 / layer 数不一致の拒否、factory が dense weight を持たないこと、MTP block を持つ checkpoint を `load_mtp_layers` で load したときに block が present になり fc / block norm / body / FFN plan と一致すること、MTP block を持たない checkpoint の MTP load 失敗、MTP tamper（fc 幅・block norm 欠落・expert 平面欠落・attention tensor 欠落）の拒否、dense loader による MoE config 拒否と MoE loader による dense config 拒否、`moe_ffn_weights_match` の tamper 検知。expert 軸分割の rank load では、rank 平面の byte 列が単一 GPU load の対応する expert と一致し、shared expert が半幅で partition を持ち、router は全 expert 数のまま、rank の weights が whole-model invariant を通らないことを固定。dense / MoE FFN 字段の非共存と `HybridLoadOptions` の MTP 字段不在、`Qwen35MoeLoadOptions` の MTP 字段存在を `static_assert` で固定 |
| `test_qwen35moe_lowering` | `gpu1;required` | 合成 MoE checkpoint（GDN 層と full attention 層を含む）を load して lower し、graph 構造を固定: `grouped_row_shape`、層ごとの FFN node 列、fused 平面を 2 view で読むこと、group-row value の数と幅、expert 平面が rank-3 native stride で weight table へ入ること、router が f32 / experts 列の logits を読むこと、combine の行空間 crossing、head の外部出力 4 本の shape と順。bucket R16 の `build_program` で dispatch 数・plan scratch 1 本・group row 容量 96 行・view の slot / offset を固定。さらに hidden tap（inner 層は受理、最終層と range 外は拒否）、expert 数 1 化・shared expert 幅伪造・top_k > experts・batch stride 伪造の各拒否 |
| `test_qwen35moe_tensor_parallel_plan` | `cpu;required` | rank の matrix axis partition plan: shared expert の gate / up が axis 0、down が axis 1 で分割されること、router と `shared_expert_gate` が plan に入らないこと（分割すると rank が別の expert 集合を選ぶため）、routed expert が plan に入らないこと、backbone entries が dense plan と byte 単位で一致すること、lm_head の vocab 分割と tied 時の不在、`num_experts` / `shared_expert_intermediate_size` / head 数が tp_size で割れない場合の拒否 |
| `test_grouped_row_shape` | `cpu;required` | group row 容量の数値（rows=1 / experts=256 / top_k=8 → group 8・group row 128 など）と不変条件（group 数は slot 数を超えない・容量は slot 数を覆う・16 の倍数）、`GROUPED_ROWS` value の binding 行数が容量基準になること、`grouped_row_shape` 未宣言の graph を graph 検証と `build_program` の両方が拒否すること、dense graph に grouped 行空間が現れないこと |
| `test_program_moe_plan` | `cpu;required` | 1 MoE FFN の graph / program 配線: `MOE_ROUTER` 1 node から 3 dispatch・grouped linear 1 node から量子化+grouped の 2 dispatch・combine まで dispatch 列で固定、plan scratch が workspace に 1 本だけ確保され scratch を読む 7 dispatch すべてが同じ range を指すこと、fused 出力と activation 平面が group row 容量で確保されること、router が shape と不一致・combine が TOKEN_ROWS を読む・weight が rank-2・`k_padded` が平面 padding と不一致、いずれも拒否 |
| `test_moe_program_execute` | `gpu1;required` | 実測 geometry（experts 256・top_k 8・hidden 2048・moe_inter 512）の 1 MoE FFN を program 経由で実行し、同じ kernel を直接呼んだ参照路径と combine 出力が bit 一致。9 dispatch すべて optimized path を通り、同一 workspace での 2 回目の実行が 1 回目と一致（plan scratch の zeroing が効いていることの固定）。logits の 1 entry が NaN の run が fault せず error word が `NUMERIC_ERROR` になり、clean な run では word が COMPLETE のままなこと |
| `test_qwen35moe_executor` | `gpu1;required` | 合成 MoE checkpoint を `LoadedModel` 経由で load し、`create_executor` の実行経路で prefill + decode を回す。row bucket 16 / 64 / 128（group 行数容量と plan scratch 容量が変わる）で、program が `plan_scratch_slot` を持つこと、hidden 行が finite かつゼロででないこと、同一 rounds を別 sequence で実行すると同一 token になること（run 跨ぎ・層跨ぎで plan scratch が漏れないことの固定） |
| `test_qwen35moe_mtp_executor` | `gpu1;required` | MTP graph を executor（`external_hidden_input`、1 attention layer の KV / GDN pool）で実行し、routed expert 平面と shared expert 平面をそれぞれ zero したときに block output の hidden 行が変わること（MoE FFN の両分岐が実際に出力へ届いていることの固定）。create_executor 直後の最初の batch が失敗しないこともこの経路が通ることで固定する |
| `test_moe_layer_e2e` | `gpu1;required` | 1 MoE layer 全体（router → plan → 量子化 → grouped w13 → swiglu → 量子化 → grouped w2 → combine）が HF `Qwen3_5MoeSparseMoeBlock` の f64 参照と、段ごと / routed / 層合計の各 gate で一致。device router の選択は HF と完全一致、Shelf の codec が fixture payload を再現 |
| `test_qwen35moe_tp_execution` | `gpu2;optional` | 合成 MoE checkpoint を 2 device で TP=2 実行し、TP=1 と同じ greedy token になること（rank 双方）、expert 平面が expert 数の半分・shared expert が半幅・KV pool が半分の kv heads・router は全 expert 数を持つこと、hidden / logits 行が TP=1 と 5 % 以内、barrier 後の logits が rank 間で bit 一致すること |
| `test_qwen35moe_rows_parity` | `gpu1;required` | 同じ prompt を executor で違う batch 行数の walk（1 行ずつ / 4+4 / 2x4 / 8 / 3+5 / 1+7 / 2+6 / 5+3 / 3+3+2 / 1+1+6 / 4+1x4 / 1x4+4）で forward し、token 行ごとの hidden が 1 行 walk と一致すること（forward の batch 行数非依存）。GDN 層を含む fixture と全 layer full attention の fixture の両方で行う。combine が shared gate 平面の row stride を仮定すると 2 行目以降の shared expert が落ちてこの一致が壊れる |

`test_moe_layer_e2e` が読む合成 fixture は `tests/fixtures/qwen35moe/`（commit 済み、
torch 不要で読むだけ）で、再生成は

```sh
python3 tools/moe_layer_oracle_fixture.py --out-dir tests/fixtures/qwen35moe
```

referee は HF transformers の `Qwen3_5MoeSparseMoeBlock` を float64 で実行したものである
（生成時に op-by-op 参照と module forward の bit 一致を検証している）。expert 重みは
Shelf の MXFP4 codec と同一規則で量子化した**あと**の値を HF に入れているため、
量子化誤差は参照側に含まれない。

`test_qwen35moe_model_load_real`（`gpu1;optional;external_files`）は実 checkpoint を
factory 経由で load し、`moe_ffn_weights_match` と `validate_qwen35moe_weights` の一致と常駐 byte・load 時間・
peak RSS・`row_ref_exact` の平面数を表示する（`SHELF_MODEL_DIR_MOE`）。

GPU 不足による skip は required では失敗として扱う（テスト受入ルール）。

## 8. 配線が未実装な範囲

model load（§5）、graph / program の MoE 語彙（§6-1）、device dispatch、model への lowering
（§6-2）、executor 入口と footprint の group 項、`shelf-compute` と `shelf-bench tg` の load
入口（`shelf::models::LoadedModel` 経由）は配線済みである。実 checkpoint（40 層）の prefill /
decode は `shelf-compute` で通る。実行の経緯と計測値は `docs/research/`、現在の性能値は
`docs/perf/current.md` を正本とし、計測していない性能主張はここに書かない。

- `shelf-bench` の `pp` / `batch` は `LoadedModel` 経由で load するようになり、grouped expert
  checkpoint の prefill / continuous batching が測れる。`shelf-server` は `shelf-compute
  --serve-stdio` を起動する側なので `LoadedModel` を持たず、compute 側は既に配線済みなので
  server 側の配線は無い。`quality` も `LoadedModel` 経由で、`--tensor-parallel 1` で grouped
  expert を teacher forcing できる
- 40 層 E2E の数値 gate（§6 の受け入れ）と MoE decode の性能実測。`shelf-bench tg` の
  fallback 内訳（`SHELF_QWEN35_PERF_STATS=1`）から未登録 geometry を洗い出し、**計測した
  ものだけ** linear / elementwise allowlist へ載せる。MoE FFN の bf16 線形
  （`256:2048`・`512:2048`・`2048:512`・`1:2048`）と backbone 線形
  （`8192:2048`・`4096:2048`・`32:2048`・`2048:4096`・`248320:2048`）、`RMS_NORM` @2048、
  `PAGED_ATTENTION` `(16, 2, 256)`・`EMBEDDING_LOOKUP` @2048 は載っている。decode の
  correctness fallback はゼロで、census が `logical_dispatches == optimized_operations`
  （context 64 / 32 token で 43360 / 43360）を出す。これは bf16 KV・fp8 KV のどちらでも
  成立する（`KV_APPEND` `(2, 256)` と `PAGED_ATTENTION` `(16, 2, 256)` は bf16 / fp8 双方に
  載っている）
- `q_per_kv = 8`（MoE の 16/2）の split-KV は bf16・fp8 の双方で有効で、
  `max_visible_tokens >= 2048` の decode で planner が splits=16 を選ぶ。capability は
  dtype 別に `paged_attention_bf16_split_supports` / `paged_attention_fp8_split_supports`
  （いずれも 4 / 6 / 8）が単一源。split と reduce は `q_per_kv` ごとの別 instantiation で、
  pairing のずれは launch 成功のまま静かに誤るため
  `test_paged_attention`（split16 比較を bf16 / fp8 双方で）と
  `test_paged_attention_split_reduce_exact`（q_per_kv 6 / 8）が対にする（経緯と計測は
  `docs/research/20261104/moe_rmsnorm_paged_attention_allowlist.md` と
  `docs/research/20261105/moe_fp8_kv.md`）
- MoE の expert 軸分割は weight load・lowering・program・kernel・executor・multi-rank runtime
  まで配線済みで、`test_qwen35moe_tp_execution` が 2 GPU で TP=1 と TP=2 の greedy 一致を
  検証する（契約は §3「expert 軸の分割」と §6 の routing、
  [tensor_parallel_execution.md](tensor_parallel_execution.md) §11-1）。multi-rank runtime は
  `shelf::qwen35` にあるので family をまたぐ判断は持てず、rank の model は
  `shelf::models::TpRankModel` 経由で注入される（builder の選択は `shelf::models`）
- ただし実 model（Qwen3.6-35B-A3B）では TP=2 と TP=1 の生成列が**一致していない**。
  合成 fixture の一致は expert 分割・collective・rank geometry の契約が成立していることの
  証明であって、実 model の kernel 選択（TP=2 の局所 shape が optimized allowlist の外側で
  correctness path へ落ちる）を含んでいない。経緯は
  [../research/20261105/moe_tp_expert_axis.md](../research/20261105/moe_tp_expert_axis.md)。
  MTP block は分割しない（§5）
- pipeline parallel も未対応。stage 分割は dense の weight load と executor footprint を前提に
  しており、group 行数容量と plan scratch の stage 別配分が無い。`--pipeline-parallel` を
  grouped expert checkpoint へ向けると `shelf::models::probe_model_dir` により起動時に
  `unsupported` を返す（拒否の実装と条件は `pipeline_parallel.md` §4 が正本）

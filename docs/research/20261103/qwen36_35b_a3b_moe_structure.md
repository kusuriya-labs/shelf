# Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 レイヤー構造調査

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`
目的: MoE 関連コードの契約を決める前に、この checkpoint の実際のレイヤー構造・
tensor 配置・量子化フォーマットを実測で確定させる。
本ドキュメントは R&D 記録であり、contract は未確定のまま（contract が決まり次第
`docs/developer/` へ現在形で書く）。

## 1. 測定方法

- `config.json` / `generation_config.json` を直接読んだ。
- `model.safetensors`（単一 shard 24.6 GB）のヘッダ（JSON, 8,816,952 B）を parse して
  全 62,405 tensor の dtype / shape / data_offsets を列挙した。
- 選択した tensor の payload を実読みして、FP4 codes と e8m0 scale の bit 列を
  histogram・group 単位で検証した（推測ではなく byte 実測）。
- MXFP4 の pack 規則は同梱 checkpoint の生成元 Quark 実装
  （`references/phaseshift-fp8/vendor/quark/quark/torch/kernel/mx/triton.py` の
  `downcast_to_mxfp` / `quark/torch/quantization/utils.py` の `even_round`）と
  突き合わせて確認した。
- MoE の forward semantics の参照は `references/qwen35-f64-oracle`
  （`src/qwen35.hpp` の `moe_forward`、`tiny_moe` fixture）。

## 2. checkpoint サマリ

| 項目 | 実測値 |
| --- | --- |
| architectures | `Qwen3_5MoeForConditionalGeneration` |
| model_type | `qwen3_5_moe` / text `qwen3_5_moe_text` |
| safetensors | 単一ファイル 24.601 GB、tensor 62,405 個、ヘッダ 8.82 MB |
| dtype 内訳 | U8 61,440 個、BF16 965 個（F8_E4M3 / F32 は存在しない） |
| quantization_config | `quant_method = "quark"`, version `0.13+93232b592e8` |
| weight | fp4, per_group, group_size 32, e8m0 scale, static |
| input_tensors | fp4, per_group, group_size 32, e8m0 scale, dynamic |
| export | `weight_format = "real_quantized"`, `pack_method = "reorder"` |
| AWQ algo_config | `scaling_layers`: `mlp.experts.*.down_proj`（inp）、`prev_op = up_proj` |
| exclude | 462 module。`lm_head` / `model.visual.*` / `mtp.*` / `*mlp.gate` / `*shared_expert_gate*` / `*.linear_attn.*` / `*.self_attn.*` / `*.shared_expert.*` |

`exclude` のとおり、**量子化されているのは routed experts だけ**である。
attention・shared expert・router・vision・MTP はすべて BF16 のまま。

### size 内訳（実測）

| 分類 | tensor 数 | GiB |
| --- | ---: | ---: |
| routed experts（MXFP4 codes+scale） | 61,440 | 15.938 |
| 言語モデル attention（BF16） | 330 | 2.392 |
| MTP experts（BF16, rank-3 fused） | 2 | 1.500 |
| embed_tokens + final norm 他 | 2 | 0.947 |
| lm_head（BF16） | 1 | 0.947 |
| vision tower（BF16） | 333 | 0.832 |
| shared expert（BF16） | 120 | 0.234 |
| MTP 非 expert（BF16） | 17 | 0.073 |
| router `mlp.gate`（BF16） | 40 | 0.039 |
| `mlp.shared_expert_gate`（BF16） | 40 | 0.000 |
| layernorm 類（BF16） | 80 | 0.000 |
| 合計 | 62,405 | 22.903 |

言語モデル本体（embed + attn + MoE + router + norms）は約 19.6 GiB、
lm_head を含めて約 20.5 GiB。32 GiB ボードで text-only なら載る。

## 3. MXFP4 on-disk 契約（実測で確定）

routed experts の tensor 名と形状は次の形。

```text
model.language_model.layers.{L}.mlp.experts.{E}.gate_proj.weight         U8 [512, 1024]
model.language_model.layers.{L}.mlp.experts.{E}.gate_proj.weight_scale   U8 [512, 64]
model.language_model.layers.{L}.mlp.experts.{E}.up_proj.weight           U8 [512, 1024]
model.language_model.layers.{L}.mlp.experts.{E}.up_proj.weight_scale     U8 [512, 64]
model.language_model.layers.{L}.mlp.experts.{E}.down_proj.weight         U8 [2048, 256]
model.language_model.layers.{L}.mlp.experts.{E}.down_proj.weight_scale   U8 [2048, 16]
```

- codes: `[N, K/2]`。**1 byte に 2 element、下位 nibble が偶数 index、上位 nibble が
  奇数 index**（Quark packer の `evens | (odds << 4)` と一致）。
- scale: `[N, K/32]` row-major。1 byte = e8m0、`value = 2^(byte - 127)`。
  各行（output channel）ごとに K 方向 32 element に 1 つの scale。
- E2M1 code 値: `0, 0.5, 1, 1.5, 2, 3, 4, 6`（bit7 = sign）。
- K は 2048 / 512 でいずれも 32 の倍数。**padding は発生しない**。
  逆に K が 32 の倍数でない tensor はこの checkpoint に存在しない。
- scale の作り方は OCP MX §6.3 に従う `even_round`
  （`floor(log2(amax)) - 2`、指数を [-127, 127] に clamp、amax==0 は 2^-126）。

### 実測 scale byte 分布

400 tensor（scale 13.1 MB）をサンプルした結果、byte の実測範囲は **1 〜 129**。
0（2^-127）と 255（e8m0 spec 上 NaN）は出現しない。

| 用途 | 主な scale byte | 復元最大値 |
| --- | --- | --- |
| gate/up_proj | 112〜121（最頻 119） | 2^(119-127) × 6 ≈ 0.023 |
| down_proj | 122〜127（最頻 125） | 0.375 〜 1.5 |

byte 1（2^-126）と byte 8（2^-119）は実質ゼロの行に紐づく特殊値（次節）。

### dead row（実質ゼロの中間チャネル）

gate_proj / up_proj に、行全体の scale が 2^-126 前後の「死んだ行」が存在する。
全 256 expert × 40 layer の gate_proj scale を全数スキャンした実測:

| layer | 死んだ行を持つ expert 数 | 死んだ行数 | 最大（/512 行） |
| --- | ---: | ---: | ---: |
| 0 | 256（全 expert） | 54,257 | 419 |
| 1 | 129 | 2,153 | 371 |
| 2 | 40 | 952 | 158 |
| 3〜39 | 1〜7 | 1〜370 | 183 |

合計 59,093 行 = gate/up 全行 5,242,880 行の **1.13%**。
**完全に死ぬ expert は 0**（256 × 40 のうち 1 も無い）。
dead row は gate_proj と up_proj で**同じ行 index に一致**し、down_proj には存在しない。
codes は 0 ではなく ±6 に飽和している（scale が下限に clamp された結果）。

契約上の含意:

- dequant は `code × 2^(byte-127)` を f32 で行い、byte 1 でも inf / NaN を出してはならない。
  （`2^-126 × 6 ≈ 7e-38` で f32 正規域。）
- byte 255（NaN）は受理しない = 検証で弾く、という契約を採れる（実データに無い）。
- 「expert がまるごと死んでいる」前提の最適化（expert pruning / 読み飛ばし）は
  この checkpoint では利益 0。行単位のゼロでもあり、それは gate/up にだけ。

### AWQ の自己完結性

checkpoint には AWQ の `input_scale` 等の追加 tensor は**存在しない**
（`weight` と `weight_scale` のみ）。scaling_layers は `down_proj` の入力チャネル
（中間 512 次元）に対する per-channel alpha で、

- up_proj の行に alpha、down_proj の列に 1/alpha が folded されている。
- 中間チャネル軸の elementwise 変換なので `silu(gate) * up` の乗算と交換可能で、
  変形は厳密。したがって **runtime は通常の MoE 計算でよく、alpha は不要**。

ただし folded 後の重みは元の BF16 モデルの重みそのものではない点に注意
（down_proj の scale が gate/up より 2 指数大きいのはこの影響と考える。仮説）。

## 4. ファイル内レイアウト（ローディング契約に直結する実測）

名前は辞書順（`layers.10` が `layers.2` より前に来る）。payload はギャップ無く連続、
領域は次の 3 ブロック。

```text
[0.000 GB .. 2.034 GB)  lm_head, embed_tokens
[2.034 GB .. 4.897 GB)  各 layer の non-expert tensor（辞書順、1 layer あたり 16 or 13 本）
                        + model.language_model.norm.weight
[4.897 GB .. 7.479 GB)  model.visual.* と mtp.*
[7.479 GB .. 24.592 GB) 各 layer の experts 全 1536 本（辞書順）
```

- non-expert 領域の 1 layer 分は 74.79 MB（linear_attention 層、16 tensor）
  または 61.88 MB（full_attention 層、13 tensor）で**連続**。
- experts 領域の 1 layer 分は **427,819,008 B（427.82 MB）で完全連続**。
  expert 昇順、expert 内は名前辞書順で
  `down_proj.weight, down_proj.weight_scale, gate_proj.weight, gate_proj.weight_scale,
  up_proj.weight, up_proj.weight_scale`。
- 1 expert あたり 1,671,168 B、6 tensor の codes/scale はすべて同一サイズ
  （codes 524,288 B / scale 32,768 B）。

つまり **「1 layer の全 expert を 1 回の連続読み込みで GPU へ運べる」**。
61,440 回の個別 copy を要求するわけではない。ファイル順は
expert-major（interleaved codes/scale）なので、plane 集約レイアウトへ直すか、
ファイル順をそのまま native layout にするかは契約判断になる。

## 5. 言語モデルのレイヤー構造

### text_config（実測）

| key | 値 |
| --- | --- |
| num_hidden_layers | 40 |
| hidden_size | 2048 |
| vocab_size | 248320 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |
| shared_expert_intermediate_size | 512 |
| num_attention_heads / num_key_value_heads | 16 / 2 |
| head_dim | 256 |
| partial_rotary_factor | 0.25（→ rotary_dim 64） |
| attn_output_gate | true（`output_gate_type` は**無い**） |
| full_attention_interval | 4 |
| linear_num_key_heads / linear_key_head_dim | 16 / 128 |
| linear_num_value_heads / linear_value_head_dim | 32 / 128 |
| linear_conv_kernel_dim | 4 |
| mamba_ssm_dtype | float32 |
| rms_norm_eps | 1e-6 |
| max_position_embeddings | 262144 |
| mtp_num_hidden_layers | 1 |
| `intermediate_size` | **存在しない**（dense MLP を示す key は無い） |

`layer_types` は 40 個で `linear_attention × 30` + `full_attention × 10`。
full attention は index 3, 7, 11, …, 39（4 毎）。safetensors 側の tensor 構成と一致する。

### 層ごとの tensor（1 layer あたり）

linear_attention 層（16 tensor、全部 BF16）:

```text
input_layernorm.weight               [2048]
post_attention_layernorm.weight      [2048]
linear_attn.in_proj_qkv.weight       [8192, 2048]   # q(16x128) + k(16x128) + v(32x128)
linear_attn.in_proj_z.weight         [4096, 2048]   # 32 x 128
linear_attn.in_proj_b.weight         [32, 2048]
linear_attn.in_proj_a.weight         [32, 2048]
linear_attn.conv1d.weight            [8192, 1, 4]
linear_attn.norm.weight              [128]
linear_attn.out_proj.weight          [2048, 4096]
linear_attn.A_log                    [32]
linear_attn.dt_bias                  [32]
mlp.gate.weight                      [256, 2048]
mlp.shared_expert.gate_proj.weight   [512, 2048]
mlp.shared_expert.up_proj.weight     [512, 2048]
mlp.shared_expert.down_proj.weight   [2048, 512]
mlp.shared_expert_gate.weight        [1, 2048]
```

full_attention 層（13 tensor、全部 BF16）:

```text
input_layernorm.weight / post_attention_layernorm.weight   [2048]
self_attn.q_proj.weight   [8192, 2048]   # 16 heads x 256 x 2（query|gate が head 単位 interleave）
self_attn.k_proj.weight   [512, 2048]    # 2 kv heads x 256
self_attn.v_proj.weight   [512, 2048]
self_attn.o_proj.weight   [2048, 4096]
self_attn.q_norm.weight / k_norm.weight  [256]
mlp.gate.weight           [256, 2048]
mlp.shared_expert.*       gate/up [512,2048], down [2048,512]
mlp.shared_expert_gate.weight [1, 2048]
```

routed experts（MXFP4、論理 shape）:

```text
gate_proj  [512, 2048]  x 256 expert
up_proj    [512, 2048]  x 256 expert
down_proj  [2048, 512]  x 256 expert
```

GDN / gated full attention の geometry は現在の Qwen3.8-27B dense と
同一の式で書ける（counts だけが違う）。
`q_proj` が 2 倍幅 = attn output gate、GDN の z gate も既存の
`emit_gated_attention` / `emit_gdn` の形そのまま対応。

## 6. MoE の参照セマンティクス（oracle 実測）

`references/qwen35-f64-oracle/src/qwen35.hpp::moe_forward` は HF transformers の
`Qwen3_5MoeTopKRouter` / `Qwen3_5MoeExperts` を op-by-op で再現しており、
`tiny_moe`（8 experts / top-2 / shared 24）の fixture を持つ。順序込みの契約は次のとおり。

1. shared expert MLP（gate/up → silu → down）と `shared_expert_gate` [1,H] の
   sigmoid を先に計算する。
2. router: `scores = xn @ mlp.gate^T`（gate `[E, H]`、bias 無し）。
3. probs = softmax（HF は f32 でmaterialize）。
4. top-K を**確率の大きい順**、同値なら**小さい expert index**優先。
5. 選択重みを top-K 合計で除算して renormalize（`norm_topk_prob` のような
   フラグはこの config には無く、常に renormalize）。
6. expert 側は **expert index 昇順**にループし、token は昇順に append された順に
   `out[t] += w * expert_out`（HF の `index_add_` と同一の加算順）。
7. 最後に `out += sigmoid(shared_expert_gate) * shared_expert_out`。

expert 重みの serialization について oracle は 2 通り両対応で、
「transformers' checkpoint conversion makes them equivalent（gate rows first）」と
コメントされている。

```text
fused: mlp.experts.gate_up_proj [E, 2*Im, H]（先頭 Im 行が gate）
split: mlp.experts.{e}.gate_proj / up_proj / down_proj
```

本 checkpoint の routed experts は split 側、MTP は fused 側（後述）。

## 7. MTP と vision

- MTP は 1 層（`mtp_num_hidden_layers = 1`、tensor は `mtp.layers.0.*`）。
  層そのものは full_attention + MoE。
- MTP の expert だけ **rank-3 fused BF16**、かつ tensor 名に `.weight` が無い。

```text
mtp.layers.0.mlp.experts.gate_up_proj   BF16 [256, 1024, 2048]
mtp.layers.0.mlp.experts.down_proj      BF16 [256, 2048, 512]
```

  この形は既存 `load_bf16_matrix_axis_slice`（rank-3 `[E,N,K]` の unit 範囲ロード）と
  ぴったり一致する。MTP 自体は未実装（`docs/homework.md` TODO）で、
  現行 drafter は DFlash2。MTP は今回の実装範囲に含めない。
- vision tower（`model.visual.*`, 333 tensor, 0.832 GiB, BF16）は既存 model
  （Qwen3.8-27B 系）と同じで、Shelf は読まない。読む tensor を固定列挙する
  現行 loader 方針のままなら自然に無視される。

## 8. 現行 Shelf との差分

この checkpoint を現在のコードに通した場合の現状の壁は次のとおり
（以下はコード読みで確認した挙動であり、実行して再現したものは明記する）。

1. `weights/weight_loader.cpp::is_quantized_model_dir` は
   `shelf_quantization.json` 不在 → config `shelf_quantization` 不在 →
   `quantization/bundle/hf_manifest.cpp::is_hf_quantized_model_dir` へ落ちるが、
   `supported_hf_quant_method_name` は `quant_method == "fp8"`
   （+ `fmt=e4m3` + `weight_block_size=[128,128]`）のみ受理するので false。
   `quant_method = "quark"` では **量子化 model dir として認識されない**。
   その結果 plain safetensors 経路（`load_qwen35_weights_from_safetensors`）に入り、
   dense FFN の名前を要求して
   `tensor not found: model.language_model.layers.0.mlp.gate_proj.weight`
   （`weights/weight_loader.cpp` の名前解決失敗）で止まる。
2. `quantization::QuantFormatId` は `Bf16` / `Fp8Finegrained` の 2 個。
   `QuantSpecId` / `ComputeSpecId` にも fp4 / e8m0 は無い。
   `QuantScalarType::E8M0` は manifest の型名としてだけ存在し、format 定義は無い。
3. `qwen35_config.cpp` は `intermediate_size` を読むが、この config には**無い** → 0。
   `output_gate_type` も**無い** → `output_gate_silu = false`。
   27B dense は同じ key 群を持つが、この MoE config では GDN z gate が
   sigmoid に落ちる。HF 既定は `swish`、oracle も silu を使うので
   **key 欠落時の既定を silu にする**必要がある。
   `attn_output_gate` は全く読まない（Qwen3_5Moe では常に true。oracle も
   false を unsupported として弾く）。
4. `Qwen35LayerWeights` は dense FFN 3 本（`mlp_gate_proj/up_proj/down_proj`）固定。
   router・shared expert・experts 配列の表現が無い。
5. `runtime/graph/primitive_node.h` の `PrimitiveKind` に MoE 系が無い
   （`LINEAR` / `SWIGLU` / `SIGMOID` / `MUL` / `SPLIT` はあるが、
   router の top-k・grouped expert GEMM・expert 集約を表現できない）。
6. `load_bf16_matrix_axis_slice` / `load_fp8_matrix_axis_slice` は rank-3 tensor 前提。
   routed experts は rank-2 × 61,440 本なので**この axis-slice 経路は使えない**。

## 9. decode 1 token あたり weight traffic（算術見積、未実測）

top-8 / 256 experts を実測 shape から積算した**上限の目安**（帯域で割った値は
ベンチマークではなく算術。`docs/perf/current.md` には載せない）。

| 分類 | 1 token あたり byte |
| --- | ---: |
| routed experts（MXFP4 + e8m0） | 535 MB |
| linear_attn（BF16, 30 層） | 2,021 MB |
| self_attn（BF16, 10 層） | 545 MB |
| shared expert（BF16, 40 層） | 252 MB |
| router（BF16, 40 層） | 42 MB |
| lm_head（BF16） | 1,017 MB |
| 合計 | **約 4.41 GB / token** |

L3 64 MB に収まらないので DRAM 供給が支配的。
この checkpoint は attention・shared expert・lm_head が BF16 のままのため、
**MXFP4 expert は decode traffic の約 12% で、支配的なのは BF16 attention と lm_head**。
「A3B = 3 GB/token」ではない。expert を 4 bit 化した効果は期待より小さく、
速度目標を置くなら attention / lm_head 側の量子化が主戦場になる
（これは本 checkpoint の契約ではなく次の課題）。

## 10. 次に決める契約項目（仮案つき）

構造が確定したので、契約として次に固定すべき点は次の 6 群。

1. **on-disk MXFP4 format 受理**
   `quant_method = "quark"` + `weight.dtype = U8 [N, K/2]` + `weight_scale.dtype = U8 [N, K/32]`
   + `group_size = 32` + `scale_format = "e8m0"` + `weight_format = "real_quantized"`
   を受理条件とし、それ以外（group 32 以外、swizzled scale、NVFP4）は明示拒否。
   `QuantFormatId::Mxfp4` / `QuantSpecId::MXFP4_E8M0_G32_V1` / scale dtype E8M0 の
   canonical stream 定義を `docs/developer/canonical_soa_payload.md` 側に追加。
   受理条件の検証として「K % 32 == 0」「e8m0 byte 255 拒否」を入れる。
2. **config 契約**
   `Qwen35TextConfig` に `num_experts` / `num_experts_per_tok` /
   `moe_intermediate_size` / `shared_expert_intermediate_size` を追加し、
   `num_experts == 0` が dense（現行 Qwen3.8-27B）、`> 0` が MoE 分岐の正本判定にする。
   `output_gate_type` 欠落時の既定を silu に直す。
3. **expert 重みコンテナ**
   per-expert `MatrixWeight` 配列（40×256×3 = 30,720 個）を採るか、
   layer あたり rank-3 集約（`[256, 1024, 2048]` codes + `[256, 1024, 64]` scale の
   gate+up 融合と `[256, 2048, 512]`+`[256,2048,16]`）を採るか。
   現行 `MatrixWeight` の rows/cols が uint32 で 1 tensor あたり 1 arena 確保なので、
   後者（1 layer 1 確保、427.82 MB 連続読み）がファイル順と Phase 1 の
   kernel 形状に合う。gate+up を 1 GEMM に融合するか（oracle の fused 表現と一致）も同一決定。
4. **primitive 契約**
   `MOE_ROUTER`（softmax + top-k + renorm、決定論的 tie-break）、
   `MOE_GROUPED_LINEAR`（experts × 3 本、activation 量子化つき）、
   `MOE_COMBINE`（expert 昇順 index_add + shared expert × sigmoid gate）の 3 つか、
   shared expert を既存 `LINEAR`/`SWIGLU` で下ろす組み合わせか。
   scheduler 側の `ScheduledBatch` には漏らさない（Topology 哲学）。
5. **compute 契約**
   RDNA4（gfx1201）に native MXFP4 MMA は無い。MXFP4 weight を何に落として計算するか
   （native layout の preshuffle 定義、FP8 WMMA で exp を fold する経路等）を
   `linear_quant_contract.h` の表に追加する。activation 側は dynamic MXFP4 か FP8 か。
6. **単体テスト契約**
   - MXFP4 codec の CPU reference（`downcast_to_mxfp` / `even_round` と bit 一致）
   - e8m0 境界値（byte 0 / 1 / 254 / 255、amax==0、dead row）
   - router の tie-break と renormalize（`tiny_moe` の 8 expert / top-2 と同一条件）
   - expert 集約順（expert index 昇順が一致すること）
   - unfused split 名寄せと rank-3 集約の同値性
   - fixture は `expert_slice_fixture.h` 形式の合成 safetensors（`shelf_io::SafetensorsWriter`）

## 11. 未確認事項・仮説

- down_proj の scale が gate/up より大きい理由が AWQ fold の影響かどうかは未確認（仮説）。
- dead row が base BF16 モデル由来か、AWQ の alpha 変換由来かは未確認。
  元 BF16 checkpoint が手元に無いため比較できていない。
  MoE の正しさテストで「dead row を含めて一致」を要求するかは契約判断に残す。
- `pack_method = "reorder"` がファイル内並び（expert-major・名前辞書順）を指すのか、
  nibble 並べ替えを指すのかは未確認。実測した codes の nibble 順序は Quark packer の
  `evens | (odds << 4)` と一致しているので、reorder は tensor 配置側の用語と考える。
- 実際の GPU 実装（MXFP4 native layout）は本調査の範囲外。

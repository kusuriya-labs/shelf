# Qwen3.6-35B-A3B MoE / MXFP4 contract 設計案

構造実測は [qwen36_35b_a3b_moe_structure.md](qwen36_35b_a3b_moe_structure.md) を正本。
ここに置くのは **MoE 側の contract 設計案と決定記録** であり、実装の現在形は
`docs/developer/qwen35moe.md`。Gate B（config と weights 集約）は実装済みで、
設計案からの差分（`shelf::qwen35moe` module へ分離、plan の source は projection 単位）は
[20261104/moe_gate_b_weights.md](../20261104/moe_gate_b_weights.md) を参照。
Gate C 以降（router / grouped linear / combine / 40 層 E2E）は未実装。

Gate A（Quark MXFP4 format の受理）は実装済みで、現在形の contract は
`docs/developer/canonical_soa_payload.md` §6bis と `docs/developer/quantization.md` へ移した。
実測と採否の経緯は [mxfp4_gate_a.md](mxfp4_gate_a.md) を参照。
設計案との差分は 1 点だけ: correctness path は canonical 直接読みではなく
**native layout 読み**で、MXFP4 の native layout は当面 canonical と同一
（row-major 4-bit codes + row-major E8M0）と決定した。

## 0. 確定した方針（確認済み）

| # | 決定 | 理由 |
| --- | --- | --- |
| D1 | routed experts は **layer あたり rank-3 集約 + gate/up 融合** | 実測で 1 layer の expert 全 1536 tensor が 427,819,008 B 連続。1 確保 / 1 読み込みで運べて、grouped GEMM の stride 契約と一致する。per-expert `MatrixWeight` 配列（30,720 個）は確保・preshuffle・dispatch が tensor 数に比例して重くなるため棄却 |
| D2 | correctness path は canonical を直接読む block-scaled reference、production は **FP8 WMMA に 4-bit codes のまま** | gfx1201 に native MXFP4 MMA は無い。**MXFP4 を FP8 に展開して常駐させてはならない**（下記 §6 の VRAM 条項） |
| D3 | Quark MXFP4 の受理契約は **dense / MoE 共通**（format 層は model 非依存） | 同じ checkpoint 書式で `models/Qwen3.8-27B-Quark-AWQ-MXFP4` も解ける。model 側は tensor name binding のみを増やす |
| D4 | 正しさの ground truth は **f64 oracle `tiny_moe` + Quark packer bit 一致** | router / 集約順は oracle が HF を op-by-op 検証済み。MXFP4 codec は `downcast_to_mxfp` / `even_round` と bit 一致で固定する |

範囲: 言語モデルのみ。`model.visual.*` は読まない（現行 model と同じ）。
`mtp.*` は今回の contract に含めない（`docs/homework.md` の MTP 未実装と同一）。

## 1. MXFP4 canonical format 契約

`docs/developer/canonical_soa_payload.md` の既存 stream 機構のまま追加できる
（新しい metadata mode は不要）。

```text
Mxfp4（OCP MX, group 32, e8m0）
  block_elements        = 32
  code_bytes_per_block  = 16          # 2 weight / byte → 32 weight あたり 16 byte
  codes stream          : row-major [rows][padded_k / 2] U8
                          byte[i] = low nibble = element 2i, high nibble = element 2i+1
  metadata1 stream      : MetaCountMode::PerBlock, element_bytes = 1, MetaStorage::U8
                          [rows][padded_k / 32]、1 byte = e8m0
  metadata2..4          : size-0（持たない）
  padded_k              = align_up(logical_k, 32)
  code 値               : E2M1 = 0, .5, 1, 1.5, 2, 3, 4, 6（bit3 = sign）
  復元                  : w[row][k] = e2m1(codes[row][k/2] nibble) * 2^(scale[row][k/32] - 127)
```

`QuantFormatId::Mxfp4 = 3`（`Bf16 = 1` / `Fp8Finegrained = 2` の次）、
`QuantizedEncoding::Mxfp4`（on-disk 名 `mxfp4_e2m0_g32`）、
`QuantSpecId::MXFP4_E8M0_G32_V1`、scale dtype は既存 `QuantScalarType::E8M0` を使う。

### 受理条件（HF Quark 由来 safetensors）

`quantization_config` すべてに一致した場合のみ受理する。

```text
quant_method          == "quark"
weight.dtype          == "fp4"
weight.qscheme        == "per_group"
weight.group_size     == 32
weight.scale_format   == "e8m0"
weight.is_dynamic     == false
export.weight_format  == "real_quantized"
```

それ以外（group ≠ 32、`scale_format` が e4m3/float、NVFP4、swizzled scale、
`weight_format` が real_quantized 以外）は `Status::unsupported` で明示拒否する。
`input_tensors`（dynamic MXFP4）は activation 側の宣言であり、
本 contract では activation を FP8 に固定するので **読んでも反映しない**（§6）。

tensor naming は 2 経路を受理する。

```text
dense : <base>.weight U8 [N, K/2]        + <base>.weight_scale U8 [N, K/32]
expert: <base>.experts.{e}.{proj}.weight + .../weight_scale   （rank-2 × E）
```

`scales` を 1 本の opaque stream に融合する旧方針に戻らない（canonical §8）。
gate/up 融合は **host 側の集約操作**であり、disk 上の 2 tensor をつなげて
`[2*Im, K]` を作る（gate 行が先、oracle / HF の fused 表現と一致）。

### validation 追加

`validate_weight_scale_spec` は現在 `granularity == MatrixBlock` しか許さないが、
MXFP4 は `PerBlock` なので分岐を追加する。追加検証:

```text
padded_k % 32 == 0
codes bytes == rows * padded_k / 2
metadata1 bytes == rows * padded_k / 32
e8m0 byte == 255 → reject（MX spec 上 NaN。実 checkpoint には出現しない）
```

`pack_method = "reorder"` については、codes の nibble 順が Quark packer の
`evens | (odds << 4)` と実測一致しているため、**contract は nibble 順のみ固定**し、
reorder の意味は検証対象にしない。

## 2. weights 層契約（generic layer に置く）

`shelf::weights` が format と集約を担当し、model 側は名前 binding のみに留める
（canonical §14 を維持）。

### rank-3 集約の表現

`MatrixWeight` に batch 次元を追加する（別 struct を作らない）。

```text
MatrixWeight
  + uint32_t experts = 1            # 1 = 現行の 2D path
  + uint64_t codes_batch_stride_bytes
  + uint64_t scales_batch_stride_bytes
```

- `rows` / `cols` / `k_padded` / `weight_scale_group` / `codes_row_stride_bytes` /
  `scale_row_stride_bytes` は **1 expert あたり**の意味のまま。
- `experts > 1` のとき `codes` / `scales` は batch 連続の単一 arena 確保。
  preshuffle は expert 毎に同一変換を stride 分ずらして適用する
  （native fragment tile は 16 出力 row × 64 K で expert をまたがない）。
- `partition`（TP）は rank-3 で未対応のまま据え置く
  （`docs/developer/tensor_partition.md` §5 の rank-3 unsupported を維持）。
  現在の production top-level load path は TP partition を受け付けないので影響なし。

### load 順序 contract

```text
Disk canonical SoA（1 layer = 427,819,008 B 連続領域を 1 読み）
   ↓ host
canonical host buffers（w13 codes/scale、w2 codes/scale に分離・結合）
   ↓ mandatory native preshuffle（host）
H2D（w13 / w2 それぞれ 1 転送）
   ↓
GPU native SoA のみ常駐
```

- 1 expert ずつの個別 H2D（61,440 回）は使わない。
- ファイル内順は expert-major の interleaved
  （`down_w, down_s, gate_w, gate_s, up_w, up_s`）なので、host 側で
  plane 集約へ組み替える。**ファイル順を native layout に採用するのは棄却**
  （codes / scale の interleave は canonical §4 の AoS 禁止に反する）。
- preshuffle 済み native を disk に保存しない方針は維持する。

### 既存 axis-slice API との関係

`load_bf16_matrix_axis_slice` / `load_fp8_matrix_axis_slice` は rank-3 tensor 専用。
routed experts は rank-2 × 61,440 本なので、この 2 API は使わない
（MTP の fused rank-3 BF16 にのみ残る。今回範囲外）。

## 3. config 契約

`Qwen35TextConfig` に追加する。

```text
std::size_t num_experts = 0;
std::size_t num_experts_per_tok = 0;
std::size_t moe_intermediate_size = 0;
std::size_t shared_expert_intermediate_size = 0;
```

- **MoE 判定の正本は `num_experts > 0`**。`model_type` 文字列
  （`qwen3_5_text` / `qwen3_5_moe_text`）で分岐しない。
- `intermediate_size` は MoE config に **存在しない**。MoE では dense FFN を
  表す field を 0 のままにし、`moe_intermediate_size` と
  `shared_expert_intermediate_size` を要求する。
- `output_gate_type` も MoE config には存在しない。key 欠落時は
  **`output_gate_silu = true`（silu）を既定**にする。
  実測根拠: 27B dense は `output_gate_type = "swish"` を持ち、
  f64 oracle は `output_gate_type` を読まず常に silu を使う
  （`rmsnorm_gated_rows`）。key 欠落を false 扱いにした現状は MoE で誤り。
- `attn_output_gate` は読まない。Qwen3.5 / 3.6 系は gate 無し path が無い
  （oracle も `attn_output_gate == false` を unsupported で弾く）。
  検証は `q_proj` の rows が `2 * num_attention_heads * head_dim` であることで行う
  （`gated_attention_feature_sizes` が既に検証している）。
- `layer_types` と safetensors の tensor 構成の一致（linear 層は
  `linear_attn.*`、full 層は `self_attn.*` を持つ）を load 時検証に加える。
  不一致は `Status::invalid_argument`。

## 4. primitive 契約

`PrimitiveKind` に 3 つ追加する。scheduler / `ScheduledBatch` の contract には
MoE の語彙を漏らさない（AGENTS.md Topology 哲学）。expert ごとの token 数は
program 内部の device 側データに閉じる。

```text
MoeRouterNode
  RowwiseShapeKey input_shape          # hidden rows [T, H]
  MatrixwiseShapeKey router_shape      # [E, H]
  uint32_t weight_index
  uint32_t top_k
  ValueDType weight_dtype = BF16       # router は BF16 のまま（実測）

MoeGroupedLinearNode
  uint32_t experts
  uint32_t top_k
  MatrixwiseShapeKey weight_shape      # w13: [2*Im, H] / w2: [H, Im]（1 expert あたり）
  uint32_t weight_index                # rank-3 MatrixWeight
  uint32_t activation_k_alignment      # FP8 W4A8 では 64
  ValueDType out_dtype = BF16

MoeCombineNode
  RowwiseShapeKey shape                # [T, H]
  uint32_t experts
  uint32_t top_k
  bool apply_shared_gate               # shared_expert_gate の sigmoid を適用
```

lowering（1 MoE layer）:

```text
LINEAR(router)  →  MOE_ROUTER                    # softmax + top-k + renorm
MOE_GROUPED_LINEAR(w13)  →  [T*K, 2*Im]
SWIGLU（grouped 領域内）  →  [T*K, Im]           # 融合候補（§7 O2）
MOE_GROUPED_LINEAR(w2)   →  [T*K, H]
MOE_COMBINE（expert 昇順 index_add、shared expert 加算）
```

shared expert は既存 `LINEAR` × 3 + `SWIGLU` + `SIGMOID` で下ろす
（BF16・1 本なので format 分岐不要）。

### 決定順の contract（数値）

oracle と同一の順序を contract として固定する。

1. router probs = softmax（**f32** で materialize。HuggingFace も f32）。
2. top-k は確率降順、同値は **小さい expert index 優先**（strict `>` 比較）。
3. 選択重みは top-k 合計で除算して renormalize（フラグ無く常に）。
4. 集約は **expert index 昇順**、同一 expert 内は token 昇順の `out[t] += w * y`。
5. 最後に `out += sigmoid(shared_expert_gate) * shared_expert`。

unfused split serialization と fused rank-3 serialization は同一結果とする
（oracle が「transformers' conversion makes them equivalent（gate rows first）」）。

## 5. activation / compute 契約

`resolve_linear_quant_contract` に 1 行追加し、activation 方針の分岐を増やさない。

| weight encoding | compute_spec | linear kernel | activation | activation K padding |
| --- | --- | --- | --- | --- |
| `Mxfp4` | `MXFP4_W4A8_F32_BF16` | `LINEAR_MXFP4_GROUPED` / `LINEAR_MXFP4` | **E4M3, group 128**（既存 spec を再利用） | 64 |

- activation を MXFP4（group 32, e8m0）にはしない。checkpoint の
  `input_tensors` は dynamic MXFP4 を宣言しているが、RDNA4 で FP4 MMA は
  無いため activation を 4-bit にする利点が計測できない。
  **A4 にしないことは数値 class の差**であり、`docs/perf/` と
  品質テストの tolerance に明示する（`docs/developer/testing_f64_oracle.md`
  の FP8 と同じ扱い）。
- weight の e8m0 scale は group 32 なので、accumulation は 32 K ごとに
  partial sum を rescaleする（`gemm_block_scaled_correctness.hip` の
  `BlockScaledKind::Mxfp4` として correctness 側に同一契約を置く）。
- correctness path は canonical codes / metadata1 を直接読む
  （production kernel の canonical 直接参照禁止は production のみ。
  既存 FP8 correctness と同じ位置）。

### VRAM 条項（D2 の核心）

`docs/developer/canonical_soa_payload.md` §12 の resident VRAM 不増原則より、

```text
MXFP4 native layout の codes は 4-bit のまま保つ
  codes 0.5 B/weight + metadata1 1/32 B/weight = 4.5 B / 8 weights
MXFP4 を E4M3 に展開して常駐させる経路は禁止
  （1.0 B/weight で 2 倍。routed experts 15.938 GiB → 31.9 GiB となり
    32 GiB board で KV を載せられない）
```

E2M1 code と e8m0 exponent の積（例 6 × 2^-8 ≈ 0.0234）は E4M3 の
正規域では正確に表現できるが、scale byte が 112 前後の group では
E4M3 subnormal 下限 2^-9 を割って 0 に潰れる。したがって
**exponent を code へ fold する経路でも metadata1（e8m0）は必ず残す**。

### 常駐 size（実測 shape から）

| 単位 | codes | scale | 合計 |
| --- | ---: | ---: | ---: |
| w13 `[256, 1024, 1024]` B | 256 MiB | 16 MiB | 272 MiB |
| w2 `[256, 2048, 256]` B | 128 MiB | 8 MiB | 136 MiB |
| 1 layer | | | 408 MiB |
| 40 layers | | | **15.938 GiB**（checkpoint 実測と一致） |

expert 以外の常駐: attention BF16 2.392 GiB、shared expert 0.234 GiB、
router 0.039 GiB、embed 0.947 GiB、lm_head 0.947 GiB。

## 6. 単体テスト contract

`cmake/tests.cmake` の `shelf_add_test` で登録する。ラベルは既存語彙
（`cpu;required` / `gpu1;required` / `gpu1;optional;external_files`）。
GPU 不足の skip は required では失敗として扱う（テスト受入ルール）。

| test | label | 固定する内容 |
| --- | --- | --- |
| `test_mxfp4_codec` | `cpu;required` | E2M1 encode / decode と Quark `downcast_to_mxfp`（EVEN rounding）の bit 一致。all zero / mixed sign / 小値 / 大値 / amax==0（scale byte 1）/ 指数 clamp / K = 1, 15, 31, 32, 33, 63, 64, 65 の padding / nibble 順（偶 = 下位） |
| `test_mxfp4_canonical_payload` | `cpu;required` | stream size formula、`padded_k % 32`、e8m0 byte 255 拒否、codes / scale の size 不一致拒否、canonical → CPU dequantize round trip |
| `test_quark_hf_manifest` | `cpu;required` | `quant_method = "quark"` の受理条件と拒否条件（group ≠ 32、scale_format 違い、weight_format 違い）、dense 名寄せ `<base>.weight` + `.weight_scale`、BF16 tensor を bf16 entry として同一経路に載せる、合成 manifest の architecture 名 |
| `test_moe_config` | `cpu;required` | 新 field の読込、`num_experts > 0` が MoE 判定、`intermediate_size` 欠落 + `output_gate_type` 欠落で `output_gate_silu == true`、`num_experts_per_tok > num_experts` 等の不正 config 拒否 |
| `test_moe_weight_plan` | `cpu;required` | 61,440 名の実測列挙（40 × 256 × 3 × 2）、expert-major interleaved のファイル順から w13（gate 先行）/ w2 へ集約する計画、1 layer の連続 span 検証、rank-3 `MatrixWeight` の stride 契約、gate/up 融合が unfused と同値 |
| `test_moe_router` | `cpu;required` | softmax(f32)、top-k tie は小 index、renormalize、E = 8 / top-2（oracle `tiny_moe` と同一条件）と E = 256 / top-8、決定論（実行順不変） |
| `test_moe_expert_slice_fixture` | `cpu;required` | 合成 safetensors writer で unfused MXFP4 experts + BF16 router / shared expert を生成（`tests/unit/expert_slice_fixture.h` の後継。dead row と e8m0 境界値を含む） |
| `test_moe_expert_load` | `gpu1;required` | 合成 fixture → 1 layer 集約 load → native preshuffle → CPU dequant と bit 一致。dead row（byte 1）で inf / NaN が出ない。preshuffle inverse transform の bit 一致（canonical §17.3） |
| `test_moe_grouped_linear` | `gpu1;required` | grouped w13 / w2 が block-scaled reference（group 32 rescale）と一致。expert 別 token 数 0 / 1 / 非整列、decode T = 1..64、prefill T = 512 |
| `test_moe_combine` | `gpu1;required` | expert 昇順 index_add と shared expert + sigmoid gate の加算順が CPU 参照と bf16 rounding で一致 |
| `test_moe_layer_e2e` | `gpu1;optional;external_files` | 合成 MXFP4 MoE model を f64 oracle `tiny_moe` と照合（相対 L2 / 平均絶対差で tolerance 固定） |

fixture は commit するが、実 checkpoint を要求する test は
`external_files` に置く（`SHELF_MODEL_DIR` 指定）。
実 checkpoint を使う初回の受け入れは §7 Gate 4 の計測 test に含める。

## 7. Gate 分割（実装順の合意）

契約は 1 度に固定せず、次の順で受理条件を確定させる。

1. **Gate A — MXFP4 format（dense で通す）**
   `test_mxfp4_codec` / `test_mxfp4_canonical_payload` / `test_quark_hf_manifest`
   + `Qwen3.8-27B-Quark-AWQ-MXFP4` を load して correctness path で greedy 1 token。
   MoE が不要なうちに format 契約だけを確定させる。
2. **Gate B — MoE weights 集約**
   `test_moe_config` / `test_moe_weight_plan` / `test_moe_expert_slice_fixture` /
   `test_moe_expert_load`。GPU 上へ rank-3 native SoA として載るまで。
3. **Gate C — MoE semantics**
   `test_moe_router` / `test_moe_grouped_linear` / `test_moe_combine` /
   `test_moe_layer_e2e`。1 layer を oracle と照合。
4. **Gate D — 40 層 E2E と計測**
   `shelf-compute` / `shelf-bench` で実 checkpoint。ここから `docs/perf/` に数値を書く。

## 8. routing / small-M linear の実測アンカー（2026-11-03）

routing kernel は未実装なので実測値は無い。ただし router が要求する形は
既存の `shelf-bench gemm` で測れるため、R9700 (device 0) で測った。
コマンドは `shelf-bench gemm --dtype bf16 --shapes 256:2048 --rows ... --warmup 50 --samples 30`、
値は sample 20-30 の p50（`--launches 20` の per-launch 換算）。

| 条件 | 実測 p50 |
| --- | ---: |
| router 形 256:2048 BF16、correctness path（rows=1） | 373 µs |
| 同（rows=64） | 372 µs |
| 同（rows=2048、prefill chunk） | 8,316 µs |
| correctness path の下限（32:32、rows=1） | 163 µs |
| 同形を WMMA へ強制（`--impl wmma`、rows=1） | 32 µs |
| 同（rows=8 / 32 / 64 / 128） | 28.8 / 29.1 / 29.1 / 29.2 µs |
| 同（rows=2048） | 224 µs |
| greedy argmax 248,320（sampling single variant） | 180 µs |

測定から確定した事実:

1. **`--variant auto` は 256:2048 を correctness へ落とす**。
   `selectors/linear_selector.cpp` の allowlist に `(256, 2048)` が無い
   （`(256, 5120)` は bf16 で登録済み）。35B-A3B の router（hidden 2048）は
   もちろん、expert FFN の `(512, 2048)` / `(2048, 512)` も allowlist 外なので、
   MoE を載せると全部 correctness path に落ちる。
2. correctness path の per-dispatch 下限が 163 µs あり、router 形は work が
   1 MB / 0.5 MFLOP しか無いのに 373 µs かかる。**overhead 支配**である。
   `--launches` を 20 → 1000 にすると per-launch が 373 → 566 µs へ悪化するため、
   この値は kernel 単体コストではなく launch / host 経路込みの数値である。
   kernel 単位の比較には使えないが、「dispatch 本数 × 数百 µs」が
   correctness path の支配項であること自体は実測で裏取れている。
3. WMMA は tiny-N で効率が悪く、1 MB 読込に 29 µs（実効 36 GB/s）。
   `docs/developer/roadmap.md` §4 の small-M split-K が未実装のため、
   router 形の optimized 経路は使えても遅い。

contract への反映（Gate C で固定する）:

- router は **`linear` + 別 top-k kernel に分割しない**。softmax / top-k /
  renormalize を 1 kernel に融合し、dispatch を 1 本に寄せる。
  selection 自体の論理幅は 256 で、vocab-wide argmax（実測 180 µs）より
  常に小さいため、cost は dispatch 本数側が支配する。
- router GEMM を separate linear として残す場合は、`(256, 2048)` を
  allowlist へ**実測値つきで**追加する（28.8 µs 以上であることを示してから追加する）。
- 40 層での router 相当の追加コスト（実測値の単純乗算。未検証の外挿）:
  correctness path なら 373 µs × 40 ≒ 15 ms/token、
  WMMA 経路なら 29 µs × 40 ≒ 1.2 ms/token。
  現行 MXFP4 correctness decode が 1.44 s/token なので、correctness 段階では
  router は支配項にならない。optimized path へ移した後は
  weight traffic 算術上限 4.41 GB/token ÷ 636 GB/s ≒ 6.9 ms/token に対し
  1.2 ms/token が 17% 上乗せになるため、fusion の判断は optimized kernel 実装時に行う。

## 9. 未決事項

- shared expert を grouped expert 表の 1 行として w13 / w2 に融合するか
  （launch 数 −3、ただし COMBINE での加算順が最後に一致することが条件）。
  Gate C までに採否を決める。
- `SWIGLU` を grouped 領域で個別 launch にするか grouped w13 の epilogue に
  融合するか。launch 数と workspace の trade-off を計測して決める。
- decode の grouped GEMM 方針（small-M split-K 等、`docs/developer/roadmap.md` §4 の
  優先順に従う）。contract には影響しない。
- down_proj の scale が gate/up より大きい原因（AWQ fold の影響か模型由来か）は未確認。
  品質 test の tolerance は実測で決める。
- dead row（gate/up の 1.13%）を kernel 側で短絡する最適化は、
  正しさ test の条件から除外する（Gate D 以降の計測対象）。

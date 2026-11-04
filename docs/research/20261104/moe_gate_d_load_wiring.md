# Gate D 実施記録 第一段: MoE weight load の production 配線

Gate B の expert loader（[moe_gate_b_weights.md](moe_gate_b_weights.md)）と Gate C の
kernel（[moe_gate_c_kernels.md](moe_gate_c_kernels.md)）を、`shelf-compute` が使う
model load 経路に載せた第一段。現在形の contract は `docs/developer/qwen35moe.md` §5。

## 1. 判断: dense 側は「FFN を含まない body」だけ公開する（`include_dense_ffn` は不採用）

当初は dense 側の名前列挙と layer load に `include_dense_ffn` フラグを足した。
これは **MoE のために dense API に knob を置く**形で、`qwen35moe` を別 module に分けた
理由（MoE の語彙を dense 側に漏らさない）と矛盾する。採用した形:

```text
qwen35 が公開する部品 (weights/quantized_model_body.h)
  qwen35_hf_quantized_body_names(config)       ... embed / lm_head / norm + 層 layernorm / attn
  qwen35_load_layer_body_quantized(...)        ... layernorm + attention のみ（FFN を触らない）
  qwen35_load_model_level_quantized(...)       ... embed / lm_head / final norm
  qwen35_open_quantized_reader(dir, names, crc)
  qwen35_layer_is_gdn(config, layer)

dense loader    : body 部品 + 自分の dense FFN 名 3 本と FFN load
qwen35moe loader: body 部品 + 自分の expert / router / shared 名と expert load
```

FFN は**両家とも自分の名前を読む**という対称な形にしたので、dense API に MoE 由来の
引数は残らない。部品化で dense 側の挙動は変わっていない
（`test_weight_load` / `test_qwen35_mtp_weight_load` / `test_qwen35_tp_weight_load` が
そのまま PASS）。

## 2. 判断: MoE FFN の保持は `shelf::qwen35moe`（`Qwen35LayerWeights` を拡張しない）

`Qwen35LayerWeights` に expert 字段（w13 / w2 / router / shared）を足す案は不採用。
qwen35 の struct が MoE の語彙を持つことになり、module 境界が消える。代わりに

```cpp
// shelf::qwen35moe
struct MoeLayerWeights { router, expert_w13, expert_w2, shared_gate, shared_*_proj };
struct MoeModelWeights { shelf::qwen35::Qwen35ModelWeights body;
                         std::vector<MoeLayerWeights> layers; };
```

`body.layers[l]` は layernorm + attention だけ（dense FFN 字段は空）、同じ layer の
MoE FFN は `moe.layers[l]` に入る。index 対応は型ではなく contract で書き、
`test_qwen35moe_model_load` は「dense FFN が空のまま」「expert 平面が plan と一致」で
固定する。

## 3. 判断: 呼び分けは `shelf_models` factory（apps が 1 箇所で済む）

`shelf_qwen35moe` は `shelf_qwen35` に依存する（`Qwen35MoeTextConfig` が base を継承）ため、
qwen35 から MoE loader を呼ぶと循環する。両方を link する上位ターゲット `shelf_models`
を置き、`num_experts` だけで分流する（`read_declared_num_experts` は config.json の
`num_experts` だけを読む probe。他の MoE field の妥当性は loader 側が検証する）。

apps の入口切り替えは行っていない。lowering が通らない段階で切り替えると 20 GB load の
後に失敗してデバッグしづらいため。残りは §7 に書いた。

境界は機械的に守らせる。`test_architecture_boundaries` に
`include|src/shelf/models/qwen35/**` から `qwen35moe` / `Qwen35Moe` / `MoE` / `moe`
を禁止する check を追加した（check 数 588 → 676）。

## 4. 実測: 実 checkpoint の load（`test_qwen35moe_model_load_real`）

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`、gfx1201 1 枚、`shelf::models::load_model_weights`
経由（`SHELF_BUILD_OPTIONAL_TESTS=ON`、label `gpu1;optional;external_files`）:

```text
config  hidden 2048 / moe_inter 512 / shared 512 / experts 256 / top_k 8 / layers 40
load        30.5 s
arena 使用  20.527 GiB
expert 平面 15.967 GiB（plan 合計 = codes + scales + row_ref = 17,144,217,600 B）
peak RSS    18.98 GiB
fold-exact 平面 80 / 80
```

- **expert 平面は 15.967 GiB**（Gate B が書いた payload 17,112,760,320 B に row_ref
  31,457,280 B を足した値）。`moe_expert_group_bytes(plan)` の 40 層合計と実 load の
  allocation が一致することをテストで固定した。
- **dense 部分も含めた GPU 常駐は 20.527 GiB**。embed 248320×2048 BF16 が 0.94 GiB、
  残りは 40 層の attention / norm / shared expert。32 GiB 環境では KV / workspace と
  競合するので、`qwen35_executor_reserve_bytes` の MoE 項が効いてくる（§7）。
- **peak RSS 18.98 GiB** は host の private copy ではなく mmap した checkpoint の
  page cache が大半（`ru_maxrss` は reclaimable な mapped file を数える）。layer loop は
  `PlaneDraft`（1 layer codes 384 MiB + scales 24 MiB + native 408 MiB）を layer ごとに
  捨てる形になっており、private 分は 1 layer 分に収まる設計のまま。実測 30.5 s は
  shard read + 17 GB の host 側 preshuffle + upload。
- **fold-exact 80/80**: 実 checkpoint の全 expert 平面で行単位の E8M0 幅が
  `kMxfp4FoldExactMaxDistance`(=8) 以内。つまり grouped linear は w13 / w2 とも fold 形
  （weight E8M0 を E4M3 operand に畳み、WMMA を K 全体 1 accumulator に chain する）を
  使える。dense で「3,904,000 行のうち 5 行だけ帯域の外」だった傾向と同じ。dispatch は
  `row_ref_exact` を見る contract なので、実測で fold が選ばれる見込み（kernel 配線で確認）。

## 5. 実測: 合成 model-level fixture（`test_qwen35moe_model_load`）

`tests/unit/moe_expert_fixture.h` に model-level 形（embed / lm_head / final norm +
層 0 は GDN・層 1 は full attention）を追加し、`gpu1;required` で load を通した。

```text
arena 使用 2,593,792 B、expert 平面 421,888 B（210,944 B × 2 層）
```

固定した contract: factory の family 判定、`body` の embed / lm_head / norm / attention
が load され dense FFN 字段は空、`layers[l]` の router・rank-3 平面・shared expert が
plan と一致（strides・row_ref 形状込み）、`moe_layer_logical_names` が `_scale` を含まない、
`moe_expert_group_bytes` が plan の byte 合計と一致、tamper した stride や router 形状を
`moe_layer_weights_match_plan` が弾く、dense loader が MoE を `logical tensor not found`
で拒否、MoE loader が dense config を `num_experts is required` で拒否、MTP /
partition plan / Shelf bundle を `unsupported` で拒否。

required acceptance は **146/146 PASS**（cpu 88 / gpu1 58）。

## 6. 見つかった既存 contract の粗（今回は直していない）

`validate_qwen35_weights` は gdn / full_attention topology を
**`full_attention_interval` からだけ**導出し、config が明示する `layer_types` を無視する。
一方 `qwen35_layer_is_gdn` は `layer_types` が完全なときにそれを優先する。
`layer_types` だけを書いて `full_attention_interval` を省略した checkpoint は、load は
通るのに model-level 検証で `unexpected layer topology` になる。実 MoE checkpoint は
両方持つ（`full_attention_interval: 4` + `layer_types`）ので現在は踏まないが、合成
fixture は両方書く必要があった。正本をどちらにするかは別の決定として扱う。

## 7. Gate D の残り

`docs/developer/qwen35moe.md` §8 に現在形で書いた: MoE lowering を qwen35moe 側に置く、
`PrimitiveKind` 3 種と `KernelId` 3 種、`WeightSlot` の experts 化、selector allowlist、
program workspace の group row 容量、apps 入口の factory 化、`shelf-bench` の計測口、
40 層 E2E 実測。

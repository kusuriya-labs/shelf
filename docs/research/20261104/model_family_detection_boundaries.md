# 「model_type で分岐しない」契約の境界調査

`num_experts` を正本とし `model_type` 文字列で分岐しない、という契約
（`docs/developer/qwen35moe.md` §2、`docs/developer/architecture.md`）が、
**全く別の構造の transformer checkpoint をロードしたときに破綻するか**の調査記録。
コード読みで確認した事実を主体に、実行再現したものはそれを明記する。

## 1. 結論

- `model_type` は `src/` と `include/` のどこにも**読まれていない**（grep で確認。
  現れるのはコメントのみ）。よって「model_type を見ない」こと自体が破綻の原因に
  なる箇所は無い。
- 破綻するのは次の 4 群で、いずれも model_type とは無関係に**語彙が固定**されている問題。
  1. config.json の構造（`text_config` で括られていること）
  2. tensor 名（`model.language_model.layers.<l>.` と backbone / FFN の名前集合）
  3. topology 規則（`layer_types` → `full_attention_interval` → `l % 4 < 3`）
  4. lowering 契約（attention output gate 必須 = `q_proj` が 2 倍幅、QK-norm 必須、
     GDN の q|k|v 分割順）
- ただし契約と**緊張する例外が 1 つある**: Shelf は `architecture` 文字列で分岐している。
  `qwen35` の dense 量子化 load と DFlash2 drafter load の 2 系統で、model_type の代わりに
  Shelf 自身の artifact 名を見ている（§4）。

## 2. 破綻箇所（コードで確認できたもの）

| 論点 | 該当 | 前提 | 別構造の checkpoint で起きること |
| --- | --- | --- | --- |
| `text_config` 必須 | `src/shelf/models/common/qwen_hybrid/config.cpp`（`read_qwen_hybrid_config` と `open_text_config`）、`src/shelf/models/qwen35/model/rope_profile.cpp` | 3 つの reader が `root["text_config"]` を要求 | flat config（Llama / Qwen2 系）では `config.json missing text_config` で即失敗。factory の `probe_model_dir` も同じ経路なので family 判定に到達しない |
| FFN key の相互排他 | `src/shelf/models/qwen35/model/qwen35_config.cpp`、`src/shelf/models/qwen35moe/model/qwen35moe_config.cpp` | dense は `intermediate_size` 必須 / `num_experts` 宣言禁止、MoE は `intermediate_size` 併記禁止 | dense FFN 1 本 + experts のような混成 FFN は拒否。shared expert を持たない MoE（`shared_expert_intermediate_size` 欠落）も拒否 |
| `head_dim` を config から読む | `config.cpp` | `hidden_size / num_attention_heads` へは**導出しない** | 未宣言モデルは `attention_head_dim = 0`。rope 解決が `invalid head_dim` で失敗 |
| rope key の位置 | `config.cpp`（`rope_parameters` のみ）、`rope_profile.cpp` | `text_config.rope_parameters.{rope_theta,partial_rotary_factor}`。`rope_scaling` は読まない | HF 標準の `text_config.rope_theta` 直置きは読まれず既定 1e7 に落ちる。**エラーwithout 警告で数値だけ変わる**（silent class） |
| `partial_rotary_factor` 既定 0.25 | `common/qwen_hybrid/config.h`、`lower_to_primitives.cpp`（0 → 0.25） | key 欠落 = 1/4 rotary | full rotary のモデルで rotary_dim が head_dim/4 になる（silent） |
| tensor 名 prefix | `common/qwen_hybrid/weight_loader.cpp`（`layer_tensor_prefix`）と、それを使う全 load 経路 | `model.language_model.layers.<l>.` | `model.layers.<l>.` では全 tensor が `tensor not found` / `logical tensor not found` |
| backbone 名前列挙 | 同上 `hf_quantized_body_names` / `load_hybrid_layer_body_quantized` | GDN 9 名 + full attention 6 名に完全一致 | 融合 QKV（`c_attn`）は q/k/v 名が無く失敗。QK-norm を持たないモデルは `q_norm` / `k_norm` が無く失敗 |
| FFN 名 | dense `mlp.{gate,up,down}_proj.weight`、MoE `mlp.experts.<E>.*`（split 形） | 名前集合と split 表現 | 融合 `gate_up_proj` や `[E, 2I, H]` の expert 融合形は解決できない（`docs/research/20261103/qwen36_35b_a3b_moe_structure.md` §6 に 2 表現の実測比較） |
| topology 規則 | `qhy::layer_is_gdn` / `qhy::count_layer_topology` | ① `layer_types` が层数分あれば它、② interval から導出、③ interval も無ければ `l % 4 < 3` | GDN を持たないモデルで ②③ が発火し、GDN tensor を要求して load 失敗（quantized path）または validation で topology 不一致 |
| `layer_types` の未知語 | `config.cpp`（未知 → `-1`）、`lower_to_primitives.cpp` | 語彙は `linear_attention` / `full_attention` の 2 つ | `sliding_attention` 等は full attention として load され、lowering まで進んで `unknown Qwen3.5 layer type`（エラーが遅れる） |
| attention output gate 必須 | `common/qwen_hybrid/attention_lowering.h`（`q_proj_features = 2 * q_features`）、`attention_geometry.h` | gate 込みの q_proj。`attn_output_gate` key は**どこにも無い** | gate 無しモデルは load は通るが `validate_gated_attention_geometry` で失敗 |
| GDN の特徴分割 | `common/qwen_hybrid/gdn_lowering.h`、`gdn_geometry.h` | `in_proj_qkv = [2*qk + value, hidden]`、q\|k\|v 順固定 | 分割順や gate 混入が違う GDN 系（RWKV / Mamba2 相当）は shape が合っても意味がずれる（silent 危険） |
| 量子化 dir 認識 | `src/shelf/quantization/bundle/hf_manifest.cpp` | `quant_method ∈ {"fp8","quark"}`、fp8 は `e4m3` + `[128,128]`、quark は `fp4` + `per_group` + `group 32` + `e8m0` + `real_quantized` | GPTQ / AWQ / compressed-tensors / modelopt は量子化 dir と認識されず、plain safetensors 経路に落ちて tensor 名で失敗（`docs/research/20261104/moe_gate_b_weights.md` §8.1 に実測） |
| BF16 raw path の dtype 未検証 | `src/shelf/weights/weight_loader.cpp`（`load_bf16_matrix_raw`） | rank-2 ことだけを見て raw upload | F32 / F16 checkpoint を BF16 として**そのまま再解釈**できる（silent corruption の可能性）。HF 量子化 path には dtype チェックがある |
| kernel selector allowlist | `src/shelf/inference/execution/selectors/*`、optimized rope 変種（`head_dim == 256 && rotary_dim == 64`） | 測定済み形状 | 別の head 分割では correctness fallback に落ちる。**性能のみ**で正确性は保たれる |

## 3. 契約が守っているもの（実測・テストで確認できる実例）

- `tests/unit/test_moe_config.cpp` の全 fixture は **model_type を `qwen3_5_moe` に固定したまま**、
  dense / MoE を `intermediate_size` と `num_experts` だけで切り分けている。
  - `dense_with_experts`（model_type は moe）→ dense reader が拒否
  - `dense_zero_experts`（同じ model_type）→ dense として受理
  - `moe_with_dense_ffn` → MoE reader が拒否
  ここを model_type で分岐していたら、`dense_with_experts` は dense として load を開始し、
  checkpoint に無い dense FFN を読みに行く。
- drafter と target は model_type を比較しない。`tests/unit/test_dflash2_config.cpp` の
  drafter fixture は `"model_type": "qwen3"`、target fixture は `"qwen3_5"`
  （`tests/support/qwen35_tp_fixture.h`）で、組合せは成立している。
- family 違いの誤 load は config 段で二重に止まる（dense reader / MoE reader 双方の拒否）。
  probe が誤っても GPU 転送前に失敗する。

## 4. 契約と緊張する例外: `architecture` 文字列分岐

`model_type` は読まないが、Shelf は**自分が書いた artifact の `architecture` 文字列**で分岐する。

| 箇所 | 比較 |
| --- | --- |
| `src/shelf/models/qwen35/weights/model_weights.cpp` | `manifest.architecture != "qwen35_dense"` → `unsupported quantized model architecture` |
| `src/shelf/quantization/bundle/hf_manifest.cpp` | HF real-quantized から合成する manifest に**常に `"qwen35_dense"` を書く**（MoE でも） |
| `src/shelf/models/qwen35/dflash2/weights.cpp` | bundle の `architecture != "dflash2_draft"` → 拒否 |
| `src/shelf/models/qwen35/dflash2/config.cpp` | `architectures[0] != "DFlash2DraftModel"` → 拒否 |

- HF path では dense loader の `architecture` 検査が**常に通過する形式上のチェック**に
  なっている（合成元が MoE でも `"qwen35_dense"` が入るため）。MoE loader は
  `architecture` を見ないので矛盾は表面化しない。
- Shelf bundle はリポジトリ内で生成しない（`docs/developer/quantization.md` に「読込専用」）。
  よって別モデル族の bundle を `"qwen35_dense"` 以外で書くと dense loader が拒否する。
  ここが「文字列で分岐しない」契約を実際に曲げている唯一の箇所なので、
  モデル族を増やすなら **(a) family 許可集合にする**、**(b) 検査を loader から format 層へ移す**
  のどちらかを決める必要がある（推測を含む設計判断）。

## 4.1 `architecture` 文字列を決めるのに `model_type` は要るか

結論: **要らない**。`architecture` が語っているのは checkpoint の出自ではなく、
**その artifact を読む loader（= Shelf の model family）**だから、family はすでに
`num_experts` で決まっている。

- Shelf bundle を書く側（外部 tool）が `architecture` に何を書くべきかの正本は、
  config.json の `model_type` ではなく「どの loader が読むか」である。
  現状 repo 内に bundle 生成 tool は無い（`docs/developer/quantization.md` に「読込専用」）。
  よって `model_type` を写す運用をしているとしても、それは契約ではなく運用の偶然である。
- HF real-quantized path では `hf_manifest.cpp` が合成 manifest の `architecture` に
  **常に `"qwen35_dense"` を書く**。呼び出し側は既に家族を知っている
  （dense loader か MoE loader か）ので、ここで `model_type` は不要なのに、
  結果として MoE checkpoint にも dense ラベルが付いている。
  dense loader の `manifest.architecture != "qwen35_dense"` 検査は HF path では
  **常に成立する形式検査**に落ちている。MoE loader は `architecture` を見ていない。
- 整合させるなら選択肢は 2 つ（いずれも `model_type` は使わない）:
  1. `open_quantized_model_reader` / `synthesize_hf_quantized_source` に
     **呼び出し側が expected architecture を渡す**（dense なら `"qwen35_dense"`、
     MoE なら `"qwen35_moe"`）。合成 manifest はその値を書き、loader は自分の値と照合する。
     MoE 側にも検査が増え、HF path の形式検査が実検になる
  2. `architecture` 検査を loader から外し、**format 层の contract 検証**
     （`validate_quantized_manifest_contract` は現在 `architecture` を見ない）に任せる
- 外部 bundle 作者向けには「Shelf model family 名を書く」ことを docs に明文化する必要がある
  （現状 `architecture` の値域を定めた developer doc が無い）。

## 5. 実際に到達を確認した破綻例

- `docs/research/20261104/moe_gate_b_weights.md` §8.1:
  MoE checkpoint を `shelf-compute`（dense 入口）で読んだとき
  `quantized_model_reader.cpp: logical tensor not found`。原因は dense FFN 名前列挙と
  `architecture` 検査。family 分岐が production 入口に未配線であることを示す実測。
- 本次（`docs/research/20261104/moe_ffn_architecture_split.md` §11）:
  BF16 checkpoint が `lm_head.weight`（prefix なし）を持つケースで dense BF16 path が
  失敗。名前の别名 tolerance が `lm_head` にしかないことを示す実測。

それ以外の破綻経路（flat config、融合 QKV、`layer_types` 欠落 + interval のみ、
GPTQ 系、F32 BF16 誤解釈など）は**コード読みによる静的判断で、実行再現はしていない**。
該当 checkpoint が手元に存在しない。

## 6. 今回この調査で実際に修正した箇所

- **层数計算の分散**: `layer_types` を直接巡回していた 5 箇所
  （`gdn_state_layout.h`、`runtime/executor.hip` の KV pool 照合、
  `compute_common.h`、`compute_runtime.hip`、`runtime/tensor_parallel.cpp`、
  さらに `apps/bench/{tg,pp,batch,quality}.hip`）を
  `qhy::count_layer_topology` に統合した。`layer_types` を宣言しない checkpoint では
  load 側は interval 規則で GDN 层を決めるのに GDN state pool は 0（→ clamp 1）、
  KV pool は `num_attention_layers == 0` を拒否して起動失敗、という不一致が起きていた
  （`src/shelf/inference/state/paged_kv_pool.cpp` の受理条件）。
  `tensor_parallel_plan.cpp` も同じ規則の写しをやめて `layer_is_gdn` を使うようにした。
- **prefix の分散**: `tensor_parallel_plan.cpp` が独自に `snprintf` していた checkpoint
  prefix を `qhy::layer_tensor_prefix` に寄せた。層 prefix を組む箇所は 1 箇所になった
  （`src/apps/compute/private_target_weights.h` は model-level 名のみで層 prefix は作らない）。
- **BF16 lm_head 别名**: §5 の実測破綻を修正。

## 7. 残る境界（別モデル族を追加する場合に触る箇所）

推測を含む。ファイル単位で、上の表の実測済み・静的判断を区別したまま置く。

1. config 語彙: `common/qwen_hybrid/config.cpp`（`text_config` 前提、`layer_is_gdn`、
   `output_gate_type` 語彙、rope key 位置）
2. tensor 名: `common/qwen_hybrid/weight_loader.cpp` の prefix と名前列挙
3. weight 型: `common/qwen_hybrid/layer_weights.h`（GDN 字段と full attention 字段を
   同一構造体に並置している。attention 構造が違う族は共有できない）
4. lowering: `common/qwen_hybrid/attention_lowering.h` / `gdn_lowering.h` と
   `qwen35/model/lower_to_primitives.cpp`
5. 量子化受理条件: `quantization/bundle/hf_manifest.cpp`、
   `quantization/manifest/runtime_resolution.h`、`architecture` 文字列検査 3 箇所
6. load 入口: apps を `shelf::models` factory へ切り替え、options 型を family 中立に
7. kernel selector allowlist（性能域）

# Dense / MoE の weight・loader 分離（アーキテクチャ再設計）記録

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4` の production E2E に入る前に、
Config / Weights / Loader / Validation / Factory / CMake の責務を整理した作業の記録。
現在形の contract は `docs/developer/architecture.md`、
`docs/developer/qwen35.md` §10、`docs/developer/qwen35moe.md` §1 / §2 / §5。

前提になった Gate A〜D の記録:
[moe_gate_b_weights.md](moe_gate_b_weights.md)、
[moe_gate_c_kernels.md](moe_gate_c_kernels.md)、
[moe_gate_d_load_wiring.md](moe_gate_d_load_wiring.md)。

## 1. ARCH-1: 共通部は plain-data 基底の継承、FFN は composition（指示書のイメージから逸脱）

作業指示書の型イメージは `Qwen35TextConfig { QwenHybridConfig hybrid; DenseFfnConfig ffn; }`
のような**完全 nesting**だった。採用したのは

```text
Qwen35TextConfig      : qwen_hybrid::QwenHybridConfig { DenseFfnConfig ffn; }
Qwen35MoeTextConfig   : qwen_hybrid::QwenHybridConfig { MoeFfnConfig ffn; }
Qwen35LayerWeights    : qwen_hybrid::HybridLayerWeights { DenseFfnWeights ffn; }
Qwen35MoeLayerWeights : qwen_hybrid::HybridLayerWeights { MoeFfnWeights ffn; }
Qwen35ModelWeights    : qwen_hybrid::HybridModelLevelWeights { layers; mtp; }
Qwen35MoeModelWeights : qwen_hybrid::HybridModelLevelWeights { layers; }
```

- **禁止された形は全て排除した**: MoE config の dense config 継承、MoE model の dense
  model 内包、空の dense FFN、二重 layer 配列、`body()` による偽 dense view。
  これらは nestingでも継承でも防げる。区別を作っているのは FFN 字段の**所有**側である。
- nesting を採用しなかった理由は移行コストと実害: config の共有字段 access は
  `config.hidden_size` / `c.head_dim` の形で 500 箇所近くに及ぶ。`head_dim` 等は
  `DFlash2Config`・GDN geometry struct・kernel 側 param にも別定義があり、
  `\.head_dim` の機械置換は別 struct を壊す。実際に dflash2 系 5 ファイルは
  置換対象から除外する必要があった。
- 継承は data 型（virtual なし・slicing しない使い方）であり、ホットパスの動的ディスパッチ
  は発生しない。`Qwen35ModelWeights` → `HybridModelLevelWeights&` の implicit upcast は
  loader が参照で受け取る箇所で使い、逆変換はしない。
- **副次的な得失**: upcast できるため共通 loader / 共通 validation は concrete model 型を
  知らなくて済む。一方 `Qwen35TextConfig` と `Qwen35MoeTextConfig` は同じ基底を持つので、
  dense 専用 API に MoE config を渡せる（逆も同様）が、`read_qwen35_text_config` が
  `num_experts` 宣言を拒否し、`read_qwen35_moe_text_config` が `intermediate_size` 宣言を
  拒否するため、実行時に家族違いの config を渡す経路は塞がっている（ARCH-7）。

## 2. ARCH-2: `common/ffn/{dense,moe}/` directory は作らない（REJECT）

指示書の配置案には `models/common/ffn/dense/weights.h` と `common/ffn/moe/weights.h` が
あった。作成時点で見ると、FFN 型は**片方の family しか使わない**ので、共通 module に置くと
「共通だが単一消費者」の型になり、`test_architecture_boundaries` が要求する
`common -> 具体モデル禁止` の意味と噛み合わない（共通 FFN 型に expert 幾何や
`intermediate_size` を置く正当な理由がない）。

- `DenseFfnConfig` / `DenseFfnWeights` → `shelf::qwen35`
- `MoeFfnConfig` / `MoeFfnWeights` → `shelf::qwen35moe`
- 共通に置いたのは FFN **非依存**の部材だけ: `QwenHybridConfig`、`HybridLayerWeights`、
  `HybridModelLevelWeights`、`weight_loader.h`、`weight_validation.h`、`stop_tokens.h`

将来 FFN 横断の共通型が本当に必要になったときだけ作る（仮定の将来機能軸で設計しない）。

## 3. ARCH-3: MoE の MTP 拒否は型で表現する（ACCEPT）

旧実装は MoE load が `shelf::qwen35::Qwen35LoadOptions` を受け、
`options.load_mtp_layers == true` を実行時に `unsupported` で弾いていた。
これは dense 固有 option を MoE API が公開している形で、MoE → dense の依存そのものだった。

現在の形:

- `qwen_hybrid::HybridLoadOptions` は `verify_quantized_payload_crc` と
  `shelf::weights::WeightLoadOptions` のみ。MTP 字段が存在しないので
  `load_qwen35moe_weights` に MTP を要求できない（`test_qwen35moe_model_load` の
  `static_assert`）。
- `Qwen35LoadOptions : HybridLoadOptions { load_mtp_layers; vocab_parallel_lm_head; }`。
  dense 固有字段はここだけ。
- apps / factory の入口は dense options を取る。MoE checkpoint に対して
  `load_mtp_layers == true` が渡されたときは**無視せず** factory が `unsupported` を返す
  （`test_qwen35moe_model_load` がメッセージで固定）。
- TP も同様に、MoE loader は `partition_plan != nullptr` を `unsupported` で弾く
  （rank-3 の partition は bundle contract 外。Gate B の contract を維持）。

## 4. ARCH-4: BF16 単一 GPU path の backbone load は共通化しなかった（意図的に残す）

共通 loader（`qwen_hybrid::weight_loader`）は **quantized path** の backbone 読み取りだけを持つ。
dense 側の BF16 checkpoint path（`load_layer_bf16`）は `shelf::qwen35` の内部に残した。

- MoE は HF real-quantized dir しか読まず、Shelf quantized bundle も BF16 dir も
  `unsupported` なので、BF16 path を通らない。共通化しても**実際に共有する処理にならない**。
- 両 path は topology 判定の手段も違う（BF16 path は checkpoint に `self_attn.q_proj` が
  有る無しさで絞り、quantized path は config の `layer_types` / `interval` で決める）。
  片方の規則へ無理に揃えると、BF16 path の既存受理条件が変わる。
- 残った duplication（层 layernorm / attention の tensor 名列挙が 2 箇所）は、
  quantized path 側の名前列挙が `qhy::hf_quantized_body_names` に集約済みなので、
  BF16 path が独自に持つ 1 箇所のみ。将来 BF16 MoE checkpoint を扱う時点で共通化の
  価値が決まる（現時点では仮定）。

## 5. ARCH-5: topology 判定を `layer_is_gdn` へ一元化（ACCEPT）

旧 `validate_qwen35_weights` は `full_attention_interval` からのみ expected topology を導き、
`layer_types` を無視していた。一方 loader 側は `layer_types` を優先していた
（Gate D の段階から）ので、両者が食い違う checkpoint では load は成功し validation は
失敗し得た。実 checkpoint と fixture では `layer_types` と interval が一致するため
これまで顕在化していなかった。

共通化後は loader・validation・lowering が同じ `qhy::layer_is_gdn` を使う。
挙動変更は required acceptance で検出していない（146 / 146 PASS）。

## 6. ARCH-6: shared expert 形状検証が routed expert 幅を使っていた（REJECT、修正）

Gate B〜D の `moe_layer_weights_match_plan` は shared expert の rows を
`plan.w13_gate_rows` から取っていた。`w13_gate_rows` は routed expert の
`moe_intermediate_size` であり、`shared_expert_intermediate_size` ではない。
当時の fixture は両者が同じ値（`moe_intermediate_size ==
shared_expert_intermediate_size == 128`）だったので、この混同は検出できなかった。

- 検証を `moe_ffn_weights_match(ffn, plan, ffn_config)` に分け、expert 平面は plan、
  router / shared expert は `MoeFfnConfig` で見るようにした。
- plan と config の食い違いもここで弾く（`plan.experts` vs `ffn.num_experts`、
  `plan.w13_gate_rows` vs `ffn.moe_intermediate_size`）。
- 回帰テストとして `shared_expert_intermediate_size = 192`（routed は 128）の
  fixture を追加し、(a) 読み込み成功、(b) shared expert を routed 幅へ伪造すると拒否、
  (c) shared down_proj の cols を伪造すると拒否、を確認する。
  旧実装なら (b) を通していた。
- fixture 側でも config.json と tensor shape が同じ値から出るよう `shared_inter` を
  引数化し、`write_fixture` が config.json に同じ値を書くようにした
  （当初 config だけ 128 のまま tensor が 192 という fixture 不整合でテストが失敗した。
  検出したのは production 側の検証であり、fixture の穴だった）。

## 7. ARCH-7: dense config の受理条件を強化（ACCEPT）

`read_qwen35_text_config` は `intermediate_size == 0` を拒否し、
`num_experts > 0` を宣言する config を `dense config must not declare num_experts` で拒否する。
`num_experts: 0` は dense のまま。対称に MoE reader は `intermediate_size` 併記を拒否する
（Gate A の contract を維持）。

副作用: dense config の最小 fixture を使っていた `test_qwen35_config_tie` は、
`intermediate_size` を fixture に追加した。この test の主題は tie 解決であり FFN ではない。

効果: 旧契約では「dense loader が MoE checkpoint を拒否する」のは FFN tensor 名が
解決できないから（`logical tensor not found`）で、checkpoint を開いた後の失敗だった。
今は config 段で失敗するので、MoE checkpoint を dense として読む試みは
GPU 転送の前に拒否される。

## 8. ARCH-8: `LoadedModelWeights` を variant に（ACCEPT）

旧実装は `dense` と `moe` を同時に保持し、`body()` が MoE からも
`Qwen35ModelWeights` を返していた（dense FFN が空の偽 model view）。
`LoadedModelWeights` は `std::variant<Qwen35ModelWeights, Qwen35MoeModelWeights>` と
`model_family()` に置き、access は `std::get_if` で明示する。
load 入口（非ホットパス）だけで分岐するので、実行時の型検索は GPU path に現れない。

apps 側（`shelf-compute` / `shelf-server` / `shelf-bench`）は dense 入口を直接使うまま。
factory へ切り替える工作是 MoE lowering と同じ単位で残す（`qwen35moe.md` §8）。

## 9. 実測 A/B: 構造変更による GPU / host メモリ変化なし

`test_qwen35moe_model_load_real`（`SHELF_MODEL_DIR_MOE=models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`、
gfx1201、arena は free 31 GiB、`-DSHELF_BUILD_OPTIONAL_TESTS=ON`）を同一機械で実行した。
base は Gate D の構造（commit `28fed13`）、refactor は `refactor/model-ffn-architecture`。

| build | load 時間 | arena 使用 | expert 平面（plan 合計） | peak RSS | fold-exact 平面 |
| --- | --- | --- | --- | --- | --- |
| `28fed13`（Gate D 構造）run1 | 23.8 s | 20.527 GiB | 15.967 GiB | 21.40 GiB | 80 / 80 |
| `28fed13`（Gate D 構造）run2 | 23.8 s | 20.527 GiB | 15.967 GiB | 21.40 GiB | 80 / 80 |
| refactor run1 | 23.8 s | 20.527 GiB | 15.967 GiB | 21.40 GiB | 80 / 80 |
| refactor run2 | 23.8 s | 20.527 GiB | 15.967 GiB | 21.40 GiB | 80 / 80 |

- GPU 側（arena 使用量・常駐 expert 平面 byte）は**完全に一致**。weight の二重保持も
  追加 allocation も発生していない。
- peak RSS も一致。[moe_gate_d_load_wiring.md](moe_gate_d_load_wiring.md) に書いた
  18.98 GiB との差は再現条件の差であり、構造の差ではない。この test の host 使用量は
  checkpoint safetensors の mmap page に支配されるため、peak RSS は page cache の状態で
  動く（Gate D の値は当時の実測としてそのまま残す。今回の run で再検証はしていない）。
- 追加された host 側 work は validation が layer 分の plan を作り直すこと
  （40 层 × 768 source 名 = 30,720 文字列）。これは load 全体の 0.1 s 未満で、
  RSS にも表れていない（上表の一致）。

## 10. 受け入れ

`python3 tests/run_required_acceptance.py --build-dir build`（gfx1201、GPU あり）で
**146 / 146 PASS、SKIP 0**。optional 込みの全 test target（163 件登録）も
compile は通っている（`test_qwen35moe_model_load_real` を含む）。
個別テストの受理条件は `docs/developer/qwen35moe.md` §7 に現在形で書く。

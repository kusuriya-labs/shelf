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

## 6.1 ARCH-12: layer body の行列幅検査は撤回した（REJECT、自分の実装を差し替え）

共通 layer body 検証で `self_attn.q/k/v_proj` の入力幅と `out_proj` の出力幅が
`hidden_size` と一致することを要求した（partition を考慮して模型次元へ戻して比較）。
**`test_weight_load` の BF16 fixture がこれを拒否した**ため、契約自体を調べ直した:

- fixture は `k_padded` の非一致経路を検証するために `self_attn.k_proj` を
  `cols = 24` / `hidden = 32` の**意図的な非一致形状**にしている（既存の意図であり、
  fixture の誤りではない）
- lowering は head 幾何を **weight から**取り、config の head 字段から再導出しない
  （`common/qwen_hybrid/attention_geometry.h` は config 側も見るが、行列幅の正典は weight）
- よって「attention 行列の入力幅 = hidden」はこの境界で保証されていない主張だった

撤回後の layer body 検証は 3 点: (a) 2 つの layernorm が `[hidden]`、
(b) その topology が要求する tensor が全てロード済み、(c) 逆 topology の tensor が無い。
幅の検査は、契約が幅を定めている箇所だけに残した: model level の vocab / hidden、
dense FFN の `[intermediate, hidden]` / `[hidden, intermediate]`（実 checkpoint 27B FP8・
MXFP4・9B BF16 で通過を確認）。教訓: **fixture を通すために契約を緩めたのではなく、
契約が元から及んでいない箇所へ検査を伸ばしていた**。

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

## 9.1 実測 A/B: dense production path（`shelf-compute`）も変化なし

`shelf-compute --model-dir models/Qwen3.8-27B-FP8 --input-ids-file <5 tokens>
--max-new-tokens 8` を同一機械で base と refactor の binary から実行した。

| run | arena 使用 | load | decode | sampled ids |
| --- | --- | --- | --- | --- |
| `28fed13` tp 1 | 30.777 GiB | 2.57 s | 18.45 tok/s | `11,401,52280,84857,369,8093,3762,26861` |
| refactor tp 1 | 30.777 GiB | 3.95 s | 18.22 tok/s | 同上 |
| `28fed13` tp 2（device 0,1） | 29.838 GiB | 20.43 s | 29.33 tok/s | 同上 |
| refactor tp 2（device 0,1） | 29.838 GiB | 20.59 s | 29.29 tok/s | 同上 |

- `validate_qwen35_weights` は 27B dense checkpoint（単一 GPU / TP rank 双方）を
  通る。新規の形状検査（backbone tensor の充足、dense FFN の `[intermediate, hidden]`、
  partition を考慮した次元復元）は実 checkpoint を拒否しない。
- sampled ids は 4 run で一致。load 時間の差（tp 1 の 2.57 s / 3.95 s）は page cache の
  warm 状態の差で、TP2 では 20.43 s / 20.59 s と差がない。decode 速率の差は run 間噪声の
  範囲（1.5%）。**性能改善の主張はしない**（構造変更なので利得も想定していない）。

## 10. 受け入れ

`python3 tests/run_required_acceptance.py --build-dir build`（gfx1201、GPU あり）で
**146 / 146 PASS、SKIP 0**。optional 込みの全 test target（163 件登録）も
compile は通っている（`test_qwen35moe_model_load_real` を含む）。
個別テストの受理条件は `docs/developer/qwen35moe.md` §7 に現在形で書く。

## 11. 同時に生じた既存の穴（修正済み）と、残した実行系の穴

- **BF16 path は `lm_head.weight` を解決できなかった（ARCH-9、修正済み）**。
  `models/Qwen3.5-9B` は `tie_word_embeddings: false` で tensor 名が `lm_head.weight`
  （prefix なし）だが、dense の BF16 model-level load は
  `model.language_model.lm_head.weight` しか引かなかった。quantized path には
  prefix なしへの fallback が既にあった（非対称）。base `28fed13` の `shelf-compute`
  でも同じメッセージで失敗することを確認した（今回導入した現象ではない）。
  BF16 側にも同じ别名 fallback を入れ、`test_weight_load` に bare `lm_head.weight`
  の BF16 fixture case を追加した。
- **`test_qwen35_mtp_weight_real` は Shelf quantized bundle を要求していた（ARCH-10、修正済み）**。
  HF real-quantized dir では load と MTP geometry 検証まで通り、後の CRC 比較段が
  `QuantizedModelReader::open`（bundle 専用）を直接呼ぶため `shelf_quantization.json not found`
  になった。base でも同じ。production と同じ `qhy::open_quantized_model_reader` を使うようにし、
  arena も 24 GiB 固定から free VRAM 基準（`SHELF_MTP_REAL_ARENA_GIB` で上書き可）へ変えた。
  実測（今回、すべて PASS・全 MTP tensor の CRC32 MATCH）:
  `Qwen3.5-9B`（BF16）/ `Qwen3.8-27B-Quark-AWQ-MXFP4` / `Qwen3.8-27B-FP8`。
- **層数計算が位相規則と分散していた（ARCH-11、修正済み）**。
  `layer_is_gdn` は `layer_types` → `full_attention_interval` → `l % 4 < 3` の順で位相を決めるが、
  层数を数える側（`gdn_state_layout.h`、`executor.hip` の KV pool 照合、`compute_common.h`、
  `compute_runtime.hip`、`runtime/tensor_parallel.cpp`、`apps/bench/{tg,pp,batch,quality}.hip`）は
  `layer_types` を直接巡回していた。`layer_types` を宣言しない checkpoint では
  GDN state pool が 0（→ clamp 1）、KV pool が `num_attention_layers == 0` を拒否して
  起動失敗する（`paged_kv_pool.cpp` の受理条件）。`qhy::count_layer_topology` に統合した。
  該当 checkpoint は手元に無いため、この破綻経路自体は**静的判断**（統合後の規則一致は
  `test_qwen35_tp_context` で検証）。
- **残した穴: BF16 dense の実行結果が HF 参照と一致しない**。
  lm_head 别名を直したことで `models/Qwen3.5-9B`（BF16）がロードできるようになったが、
  `shelf-compute` の greedy 出力が HF transformers の参照と食い違う。
  prompt token ids `[3710,369,279,6511,314,9338,30,21134,25]`
  （"What is the capital of France? Answer:"）に対する 24 token:

  ```text
  HF      (torch 2.13+rocm10, transformers 5.20.2, bf16, greedy):
          11751,13,271,3710,...  -> " Paris.\n\nWhat is the capital of Germany? Answer: Berlin..."
  Shelf   (本次 refactor, tp 1, optimized):
          95839,98900,88792,15729,3709,67,13,...  -> 無意味列
  Shelf   (SHELF_QWEN35_KERNEL_MODE=correctness): 上記と同一 ids
  ```

  weight のロード自体は `test_qwen35_mtp_weight_real` の CRC32 MATCH で一致している。
  `SHELF_QWEN35_KERNEL_TRACE=1` では `LINEAR_BF16` 996 件・`EMBEDDING_LOOKUP` 4 件・
  `RESIDUAL_ADD` 256 件・`SWIGLU` 128 件が correctness fallback（optimized BF16 linear が
  この形状で allowlist 外）、GDN / attention / rope / rmsnorm は optimized kernel。
  optimized と correctness で ids が同一なので、原因は kernel 選択ではなく
  load / lowering / state 構成側の可能性が高い（**未調査・推測**）。
  本次の refactor とは無関係（BF16 9B は refactor 前にそもそもロードできなかった）。
  追跡の入口としては `--hidden-taps` の layer 出力と HF の layer 出力比較が有効（未実施）。

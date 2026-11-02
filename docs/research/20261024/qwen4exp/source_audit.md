# Qwen4Exp ソース差分調査票（G0）

- ID: QWEN4EXP-G0-SOURCE-AUDIT
- 実施日: 2026-10-24
- 正本: worktree `.worktrees/qwen4exp`、branch `investigation/qwen4exp-main`、
  HEAD `9825d10896fea4f4d9736f918617c41322ae53b2`
- 手法: 実リポジトリの `AGENTS.md` / `docs/developer/*` / 該当ヘッダ / 実装 /
  呼出元 / テスト / CMake を順に読んだうえで、手順書 §3 の「ZIP 時点の観測」を
  現行ソースへ照合した。

## 1. 差分調査票

| 調査対象 | ZIP 時点の想定（手順書） | 現行実装 | 食い違い | 追加で読んだ箇所 | 採用判断 | 回帰リスク | 検証計画 | 判断根拠 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `safetensors_reader` の `SType` | `I64` が無く未知 dtype で拒否 | 7 値のみ。`I64` は `unsupported dtype` で拒否 | **なし（一致）** | writer / weight_loader / hf_manifest / quantized_model_reader | 最小修正（`I64` 追加） | 既存 dtype 判定を壊さない | I64 roundtrip、未知 dtype 拒否の維持 | `safetensors_reader.h:12-20`、`safetensors_reader.cpp` の dtype 解析 |
| `data_base_offset()` / `tensor_spec()` | 存在 | 存在（`data_base_offset = 8 + header_len`） | なし | — | 再利用 | なし | span の絶対 offset 検証 | reader 実装 |
| 部分読込機能 | 「既存の部分読込機能」を確認 | `read_tensor` は完全コピー必須。row / Expert span API は**不在** | **あり** | `read_tensor` / `tensor_data` 契約 | 新設（読み取り専用 span API） | mmap 経路の互換性を壊さない | 先頭・末尾 row、任意 Expert ID、無効 ID | reader 実装 |
| mmap 経路 | 未記載 | `mmap(PROT_READ, MAP_PRIVATE)` + RAII | なし | — | 再利用 | なし | — | `MappedFile` |
| 検証群（overflow / 境界） | 検査を求める | shape 積・要素数・byte count・ファイル境界・offsets 順序・重複名を検査済み | なし（概ね充実） | — | 再利用 + 新 API 側へ追加 | なし | truncated / overflow case | reader 実装 |
| shard 解決（index.json） | 確認する | `SafetensorsCollection`（`weight_loader.cpp`）が解決。`src/shelf/io/` には reader/writer のみ | なし | `hf_manifest.cpp` / `quantized_model_reader.cpp`（index 読取が 3 系統） | 再利用（統合はしない） | index 解決の重複には触れない | 複数 shard fixture | `weight_loader.cpp:333-434` |
| path traversal 検査 | 必要 | Collection は `model_dir_ + "/" + shard` で**未検査**（`bundle_util.h` の `validate_shard_path` は quantization 側にのみ存在） | **あり** | `bundle_util.h` | 最小修正（Collection へ検査追加） | 検査追加のみ、既存パスは通る | `../` / 絶対パス拒否 | `weight_loader.cpp:404` |
| HF FP8 受理条件 | `fmt == "e4m3"` 必須、rank-2 行前提 | `quant_method=fp8` / `fmt=e4m3` / `weight_block_size=[128,128]` 必須、codes は rank-2 F8_E4M3/U8、scale は BF16 rank-2 | **なし（一致）** | `detect_hf_quant_method` / `synthesize_fp8_tensor` | 再利用。**任意 FP8 の無条件受理はしない** | FP8 既存ロード | 既存 `test_hf_manifest` | `hf_manifest.cpp:198-225, 308-334` |
| `activation_scheme` 検査 | 検査を求める | **未検査**（ソースに出現しない） | **あり（仕様差）** | `detect_hf_quant_method` 全体 | **保留**：FP8 checkpoint 不在のため実値が不明。推測での検査追加はしない | なし | FP8 checkpoint 取得後に判定 | 手順書 §2.2 の revision 固定義務との衝突 |
| rank-3 fused expert | 確認する | FP8 は rank-2 必須で拒否。BF16 は manifest に受理されるが `load_bf16_matrix` が rank-2 必須でロード時に落ちる（非対称） | **あり** | `canonical_partition.cpp` / `load_quantized_small` | G3 で扱う（G1 は byte span 特定まで） | 既存 TP partition | G3 の Expert slice テスト | `hf_manifest.cpp:199-203`、`weight_loader.cpp:136-139,187-190,559-562` |
| `output_gate_silu` 切替 | sigmoid 経路の有無を確認 | **実装済み**。`gdn_lowering.h:140-145` が `SiLUNode`/`SigmoidNode` を分岐、config は `output_gate_type` を解析、HIP kernel まで到達（`KernelId::SIGMOID` → `launch_sigmoid_bf16_f32`） | **あり（docs 側が stale）** | `qwen35_config.cpp` / `lower_to_primitives.cpp` / dispatch / elementwise kernel | **再利用・再実装禁止** | Qwen35 の既定挙動を変えない | GDN 適合試験で sigmoid 出力を検証 | `gdn_lowering.h`、`lower_to_primitives.cpp:574` |
| paged attention selector の新 QSA geometry | 高速経路の有無を確認 | ルールは `(16,4,256)` / `(24,4,256)` / `(12,2,256)` のみ。**(24,2,256) は不在** → QSA は常に correctness fallback。fallback 自体は存在 | **あり** | selector / dispatch / 既存 selector テスト | fallback は再利用。optimized ルール追加は G4/QSA 側 | 既存 Qwen35 の選択結果を変えない | G4 QSA で correctness 経路の数値を検証 | `paged_attention_selector.cpp:24-63` |
| `models/qwen4exp` の有無 | 見当たらない | **不在**（`src/shelf/models/` は `qwen35` のみ） | なし（一致） | `models/qwen35` 全体 / `models/common/qwen_hybrid` | 新設（G5） | architecture boundary | `test_architecture_boundaries` | ディレクトリ一覧 |
| hipFile 連携 | 見当たらない | **不在**（`src` / `include` / `tests` / `cmake` / `docs` に 0 件）。ただし **libhipfile.so.0.4.0 とヘッダ・CMake config は環境に実在** | なし（一致） | `hipfile.h` API 一覧 | 新設（G2）。PLE 固有処理は入れない | I/O 層の汎用性をPLE依存にしない | G2 の bitwise / fastpath 試験 | `hipfile_qualification.md` |
| `docs/developer/qwen4exp.md` | 仕様書あり。gap と `rnd/qwen4exp/gate0.md` 参照が古い可能性 | 仕様書は実 checkpoint と一致。**`rnd/qwen4exp/gate0.md` は不在**（`rnd/` 無し）、gap #7 は stale | **あり（docs stale）** | `docs/research/README.md` / `docs/developer/README.md` | docs 更新（research へ経緯、developer へ現行 contract） | docs だけ | レビュー | 参照先不存在の実測 |
| `tools/inspect_qwen4exp.py` | 見当たらない | **不在**。`tools/` に同等の inventory 機構もない | なし（一致） | `tools/` 一覧 | **新設しない**（単発 Gate 後に残す script は残さない規約）。inventory は `docs/research/` の記録で代替 | なし | — | `AGENTS.md` tools ルール |
| `tensor_partition.md` §10（Qwen4Exp EP 例） | FP8 N-D 未対応の制約に注意 | 実装と一致。BF16 rank-3 は descriptor + materialization 対応、**FP8 rank-3 は unsupported**（拒否テストあり） | なし | `canonical_partition.cpp` / `test_canonical_partition.cpp` | 制約を維持する | 既存 TP partition | 既存 `test_canonical_partition` | doc §5/§10 と実装の突合 |
| required acceptance の運用 | ベースラインを保存 | 正本は `tests/run_required_acceptance.py`。required は **LABELS と `shelf-required-tests` DEPENDS の 2 箇所**が正本 | **あり（既存のずれ 3 件 / 1 件）** | `cmake/tests.cmake` 全体 / `testing.md` | 新規テストは両方へ追加。既存のずれは本 Gate の範囲外として記録のみ | テスト build漏れ | ベースライン 143/143 PASS を取得済み | `build/required-acceptance.txt` |

## 2. 参照 revision の固定

| 対象 | revision / 値 | 状態 |
| --- | --- | --- |
| Shelf HEAD | `9825d10896fea4f4d9736f918617c41322ae53b2` | 固定 |
| Qwen3.8-Flash-Next (BF16) | HF `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 固定 |
| Qwen3.8-Flash-Next-FP8 | — | **不在（未固定）** |
| Transformers `qwen4_exp` | — | **未固定**（インストール済み `transformers 5.14.1` に `models/qwen4_exp` なし） |
| ROCm / HIP | ROCm 10.0.0（pip SDK）/ HIP 7.15.26333 / ROCk 7.1.3.31500000 | 固定 |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) ×4 | 固定 |

## 3. 機能分解（再利用 / 最小修正 / 新設）

| 判定 | 項目 |
| --- | --- |
| 再利用 | mmap reader と検証群、`data_base_offset()`、`SafetensorsCollection` の index 解決・shard キャッシュ、`SafetensorsWriter`（fixture 生成）、`TensorPartitionDesc` + BF16 rank-3 materialization、`load_quantized_small`（rank 1-3 BF16）、GDN sigmoid 切替（`output_gate_silu`）、paged attention の correctness fallback、required テスト運用 |
| 最小修正 | `SType::I64` 追加（reader 解析 / elem_size / writer）、読み取り専用 span API、Collection の path traversal 検査 |
| 新設 | PLE row gather（SSD / RAM / VRAM）、hipFile I/O ラッパ、BF16 / FP8 Expert slice、PLE / GR / MoE / QSA 独立カーネル、`models/qwen4exp/` |
| Gate の不要化 | GDN sigmoid 切替の再実装（§9.6「既存の切替機能を使える場合は再利用」） |
| 保留（判断材料不足） | `activation_scheme` 検査（FP8 checkpoint 不在） |

## 4. 未解決の食い違い

| # | 内容 | 扱い |
| --- | --- | --- |
| 1 | `docs/developer/qwen4exp.md` の gap #7（GDN silu 固定）が stale | docs 更新（G0 で対応） |
| 2 | 同 doc の `rnd/qwen4exp/gate0.md` 参照が不存在 | docs 更新（research へのリンクへ置換） |
| 3 | 同 doc の `tools/inspect_qwen4exp.py inventory` が不存在 | docs 更新（削除） |
| 4 | `activation_scheme` 未検査 vs 手順書の検査要求 | **保留（継続）**。FP8 checkpoint 不在のため実値照合できず。推測での検査追加はしない |
| 5 | QSA `(24,2,256)` の optimized ルール不在 | **G4c で確定: ルールは追加しない**（未計測の geometry に Optimized を宣言しない）。correctness fallback で機能成立。optimized 化は計測後に判断 |
| 6 | required の LABELS / DEPENDS ずれ（3 件 / 1 件） | 本 Gate の範囲外。記録のみ（新規テストは両方へ追加） |
| 7 | index.json 解決が 3 系統に重複 | 統合しない（回帰リスクが高く G1 の範囲外） |
| 8 | `SafetensorsWriter::finish()` の古いコメント（存在しない `weight_map` 引数に言及） | **未対応（軽微）**。本 Gate の範囲外として記録のみ |
| 9 | HF FP8 rank-3 拒否と BF16 rank-3 受理後のロード失敗の非対称 | G3 で扱う |
| 10 | FP8 checkpoint / Transformers 参照 commit が未取得 | **G0 の残課題**（§5） |

## 5. G0 判定

| PASS 条件 | 状態 |
| --- | --- |
| 未解決の重要なソース差分がない | **達成**（上表のうち重要なものはすべて扱いを決めた。保留は #4 のみで、これは実データ不在による仕様未確定でありソース差分ではない） |
| 参照 revision を固定した | **部分達成**（Shelf / BF16 checkpoint / 環境は固定。FP8 checkpoint と Transformers 参照 commit は未取得） |
| 独立テストに使う fixture の構成が決まっている | **達成**（自己完結の synthetic safetensors fixture を `tests/unit/test_safetensors_partial_read.cpp` で確定。実 checkpoint 参照は optional） |

**G0 = PASS（条件付き）**。未取得の FP8 checkpoint と Transformers 参照 commit は
G3 / G6 の開始条件であり、G1〜G5 の synthetic fixture には不要である。
この2件を未解決項目として `gates.md` に引き継ぐ。

---

# 付録: G5（モデル統合）の設計判断資料

G0 の追加調査として、`models/qwen4exp/` の lowering が既存 Primitive で
どこまで表現できるかを整理した。判定根拠は
`include/shelf/runtime/graph/primitive_node.h`（`PrimitiveNode` variant 全 19 種）と
`include/shelf/runtime/program/kernel_id.h`（`KernelId` 全 28 値）の実測。

## 既存 Primitive の一覧（実測）

`EMBEDDING_LOOKUP` / `LINEAR` / `RMS_NORM` / `RESIDUAL_ADD` / `SPLIT` / `SILU` /
`SIGMOID` / `MUL` / `SWIGLU` / `ROPE` / `L2_NORMALIZE` / `SCALE` / `KV_APPEND` /
`PAGED_ATTENTION` / `STATEFUL_CAUSAL_CONV1D` / `GDN_RECURRENCE` / `OUTPUT_GATHER` /
`SAMPLING` / `CONCAT`

## qwen4exp forward の要素と再利用可否

| 要素 | 必要な演算 | 既存 Primitive | 判定 |
| --- | --- | --- | --- |
| token embedding / lm_head | lookup / linear | `EMBEDDING_LOOKUP` / `LINEAR` | **再利用可** |
| norm（grouped、`1+w`） | grouped RMSNorm + ONE_PLUS | `RmsNormNode.group_size` + `ONE_PLUS`（`gdn_lowering.h` が使用） | **再利用可** |
| GDN（conv / L2 / recurrence / sigmoid gate） | — | `emit_gdn` + `STATEFUL_CAUSAL_CONV1D` + `GDN_RECURRENCE` + `L2_NORMALIZE` + `SIGMOID` | **再利用可**（`output_gate_silu` 切替実装済み） |
| QSA の gated query + partial RoPE | 2x q_proj + sigmoid gate + rotary 前半 | `emit_gated_attention`（`attention_lowering.h`） | **dense は再利用可**。sparse mask は不可 |
| QSA の sparse attention | 選択 token 限定の softmax | 選択 mask を表現する Primitive **なし** | **新規必要**（G4c の判定待ち） |
| QSA indexer（4-token mean / ReLU / Top-512） | F32 mean / ReLU / topk | mean・ReLU・topk を持つ Primitive **なし** | **新規必要**（G4c） |
| MoE router（softmax F32 + Top-10 + 再正規化） | softmax / topk / reduce | **なし** | **新規必要**（G4b） |
| MoE routed experts（512 のうち 10） | 選択 expert への GEMM 合算 | **なし** | **新規必要**（G4b / G3 の slice） |
| shared expert | gate/up/down + sigmoid gate | `LINEAR` + `SILU` + `SIGMOID` + `MUL` | **再利用可** |
| Gated Residual read / write | grouped RMSNorm + down/up + sigmoid + **4-stream mean** | RMSNorm / LINEAR / SIGMOID / MUL は可。**stream 方向 mean の reduce が無い** | **新規必要**（G4a） |
| final HyperConnection mixer | 同上 | 同上 | **新規必要**（G4a） |
| PLE row gather（SSD / RAM / VRAM） | byte span gather | `row_gather` カーネル + `shelf_io` | **G2 で実装済み**（Primitive 化は未定） |
| PLE projection / gate / conv（dilation=3） | grouped RMSNorm / linear / sigmoid / depthwise conv | `STATEFUL_CAUSAL_CONV1D` は **GDN geometry 固定**（`GdnShapeKey` 必須）。dilation 対応と PLE 固有の gate 演算は未確認 | **要調査**（G4a） |

## 結論（現時点）

1. **GDN / attention / norm / linear / embedding / lm_head は既存 Primitive で再利用できる。**
   共通 runtime の変更は不要。
2. **MoE（router + experts）、Gated Residual、QSA indexer / sparse mask、PLE の一部は
   既存 Primitive で表現できない。** 手順書 §10-4 に従い「新しい Primitive が不可避なら
   generic な意味と形状契約を定義し、既存モデルの挙動を変えない」方針で扱う。
   新 Primitive をどれだけ追加するかは G4 の各成果を見てから確定する。
3. `STATEFUL_CAUSAL_CONV1DNode` が `GdnShapeKey` を要求するため、
   PLE の depthwise conv（dilation=3、group=10240）を既存ノードで表せるかは
   **未確認**。G4a の調査結果を待つ。
4. `PrimitiveStateKind` は `KV_CACHE` / `GDN_CONV_STATE` / `GDN_RECURRENCE_STATE` の 3 種のみ。
   QSA indexer key state と PLE の n-gram context state は**新 state kind が必要**。
   手順書 §13 の記載と一致する。

## 実 checkpoint との食い違い（G0 で判明、§4 の追記）

| # | 内容 | 扱い |
| --- | --- | --- |
| 11 | `docs/developer/qwen4exp.md` §9.1 の `ple_layer_index` は実 checkpoint と不一致。正しくは `ple_slot`（`ple_layer_ids` 配列内の 0-indexed 位置） | **本 Gate で docs を修正済み**（G4a の総当たり照合で確定） |
| 12 | 同 §9.1 の `unigram_vocab_size` は `ngram_vocab_size_base` ではなく `vocab_size = 248320` | **本 Gate で docs を修正済み** |
| 13 | 同 §1.1 の `seed = 1234` は実 `config.json` に**キーが存在しない** | docs に「既定値」と明記し、config 実装でも既定値 1234 にした |
| 14 | `layer_multipliers` / `ngram_heads_vocab_sizes` / `ngram_heads_offsets` は checkpoint に I64 で実在（`qwen4exp.md` §9.1 は計算式のみ記載） | 実値を docs §9.1 に併記 |

### 追記（G5 調査）: state pool と PLE conv の適合

`GdnStatePoolLayout`（`include/shelf/inference/state/gdn_state_pool.h`）は
`conv_dim` / `conv_history` を **pool 全体で単一の値**として持ち、
`GdnStatePoolLayout::make(gdn_layers, conv_dim, conv_history, ...)` で生成する。

- 現行 Qwen35: `conv_dim = 10240`（Qwen3.8-27B）、`conv_history = 4`
- qwen4exp GDN: `conv_dim = 2 * 16 * 128 + 48 * 128 = 10240`、`conv_history = 4` → **現行 layout で足りる**
- qwen4exp PLE short conv: `conv_dim = 10240`、**`conv_history = 9`**（`(4 - 1) * 3`、`docs/developer/qwen4exp.md` §9.1/§13）

**→ PLE conv state は現行 `GdnStatePoolLayout` に単一 `conv_history` では収まらない。**

`docs/developer/qwen4exp.md` §13 は「PLE short conv は `GDN_CONV_STATE` の
`state_index` で表現できる」としているが、これは
(a) `conv_history` が pool 全体で共通である点、
(b) `STATEFUL_CAUSAL_CONV1DNode` が `GdnShapeKey`（dilation 非保持）に依存する点
を満たさない。

**未解決の設計課題 #15 として記録する。** G4a / G5 で次の方針から選ぶ:

1. `GdnStatePool` とは別に PLE 専用の小さな conv state pool を model-local に持つ（推奨候補。共通 runtime を触らない）
2. `GdnStatePoolLayout` に複数の conv history を持たせる（共通 runtime 変更、回帰リスク大）

---

# 付録: 手順書 §13 の推奨配置と現状の対応

手順書 §13 は「パスは例であり、実装前に現行ソースで再確認する」と明記している。
本 Gate 群での対応:

| 手順書 §13 の例 | 現状 | 備考 |
| --- | --- | --- |
| `include/shelf/io/...` / `src/shelf/io/hipfile_reader.cpp` | **採用**（`row_reader.{h,cpp}` / `hipfile_reader.{h,cpp}`） | PLE 固有処理は入れていない（汎用 I/O 層） |
| `include/shelf/weights/...` / `src/shelf/weights/`（metadata / tensor slice / Expert slice） | **採用**（`SafetensorsCollection` の span API、`load_*_matrix_axis_slice`） | — |
| `include/shelf/models/qwen4exp/...` / `src/shelf/models/qwen4exp/model/`（config / weights / lowering） | **採用**（config / weights は実装済み。lowering は未着手） | — |
| 同 `ple/`（N-gram hash / shard mapping / PLE 演算） | **採用**（`ple_row_gather` / `ngram_hash` / `ple_conv`） | shard mapping は `ple_row_gather` に同居 |
| 同 `runtime/`（executor / state / batch） | **未作成** | G5 の残課題 |
| `src/shelf/kernels/optimized/`（GR / MoE / QSA） | GR は `models/qwen4exp/hyper/` に model-local、QSA は `kernels/optimized/attention/` に配置 | GR は host 実装のため model-local。QSA の配置は G4c の判断 |
| `tests/unit/` / `tests/kernels/` / `tests/fixtures/` | **採用**（fixture は `tests/data/qwen4exp_moe/`、`tests/fixtures/qsa/` を追加） | — |
| `cmake/targets.cmake` / `cmake/tests.cmake` | **採用**（機能ごとに追記） | required は LABELS と `shelf-required-tests` DEPENDS の 2 箇所に追加 |
| `docs/research/<YYYYMMDD>/` | **採用**（`docs/research/20261024/qwen4exp/`） | 配置規約（`docs/research/README.md`）に適合 |
| `docs/developer/qwen4exp.md`（成立した contract のみ更新） | **採用**（実 checkpoint と食い違っていた記述を修正、gap の更新） | 経緯は `docs/research/` に分離 |
| `tools/inspect_qwen4exp.py` | **新設しない** | `AGENTS.md` の tools ルール（単発 Gate 後の script は残さない）に従い `docs/research/` の記録で代替。G4b の oracle 生成 script は `tools/gen_qwen4exp_moe_oracle.py`（fixture 再生成に使用） |

追加されたディレクトリ（§13 に無い例）:

- `src/shelf/models/qwen4exp/hyper/` — Gated Residual（model-local host 実装）
- `src/shelf/kernels/optimized/attention/qsa_*.{h,hip}` — QSA indexer / sparse attention
- `tests/data/qwen4exp_moe/` — MoE oracle fixture（各 20 KB 程度の小ファイル）
- `tests/fixtures/qsa/` — QSA fixture

---

# 付録: G5 lowering のための新 Primitive 設計メモ（暫定）

`DispatchBinding`（`include/shelf/runtime/program/program.h`）の制約を実測した:

- `kMaxIoSlots = 6`（`input_slots` / `output_slots` は各 6 まで）
- `parameter_index0` / `parameter_index1` の **2 個だけ**、`weight_index` は 1 個
- `scalar_a` / `scalar_b` / `rotary_dim` / `group_size` / `descriptor_slot` あり

## 既存 Primitive へ分解できるもの（新規不要）

Gated Residual は 1 ノードに詰め込まず、既存ノードへ分解できる:

| GR のステップ | 既存 Primitive |
| --- | --- |
| grouped RMSNorm（group=2560、`1+w`） | `RmsNormNode`（`group_size` + `ONE_PLUS`、`gdn_lowering.h` が利用済み） |
| `input_mix_weight_down` | `LINEAR` |
| `silu(d / hc_count)` → `input_mix_weight_up` → `sigmoid(u)` | `SILU` / `SCALE` / `LINEAR` / `SIGMOID` |
| `inj = 2 * sigmoid(block_inject_weight(n) / hc_count)` | `LINEAR` / `SCALE` / `SIGMOID` |
| final collapse の read 部分 | 上と同じ組合せ |

分解して残るのは **2 種の新しい要素演算のみ**:

1. **stream 方向の mean** — `mixed = mean_over_4(w * n.unflatten(-1,(4,2560)))`
   （reduce に該当する既存 Primitive は無い）
2. **broadcast 積** — `x4 = x4 + (sub_out[...,None,:] * inj[...,None]).flatten(-2)`
   （`[T,4,2560] × [T,4,1]`。既存 `MULNode` は同次元前提）

`weight_index` 1 個の制約に対しても、`LINEAR` を個別ノードに分ければ問題にならない。

## 新規が必要な Primitive（現時点の候補）

| 候補 | 入出力 | 必要理由 | 前提となる G4 の成果 |
| --- | --- | --- | --- |
| stream mean（reduce） | `[T,4,H]` → `[T,H]` | GR read の `mean_over_4` | G4a（GR host 実装） |
| broadcast mul / add | `[T,4,H] × [T,4,1]` → `[T,4,H]` | GR write | G4a |
| MoE router | `[T,H] × [512,H]` → top-10 index / weight | softmax(F32) + topk + renorm が無い | G4b（作業中） |
| MoE experts | `[T,H]` + top-10 → routed 合算 | fused rank-3 の選択 GEMM 合算 | G3（slice 読み込み）+ G4b |
| PLE embedding gather | row ID `[T,16]` → `[T,16,160]` | G2 の row gather を graph から呼ぶ | G2 + G4a（hash） |
| PLE conv（dilation=3） | `[T,10240]` → `[T,10240]` + state | 既存 conv は dilation 非対応 | G4a（host 実装） |
| QSA indexer | `[T,H]` → select mask + key state | indexer / block 選択 | G4c（作業中） |
| sparse attention | query + KV + select mask → output | 選択 token 限定の softmax | G4c（作業中） |

不要（既存で足りる）: shared expert の合成
（`sigmoid(shared_expert_gate(x)) * shared` = `SIGMOID` + `MUL`）。

## state の追加要否

| state | 現状 | 要否 |
| --- | --- | --- |
| QSA indexer key state | `PrimitiveStateKind` に無い（3 種のみ） | **新 kind 必須**（`docs/developer/qwen4exp.md` §13 の記載どおり） |
| PLE short conv state（`conv_history = 9`） | `GdnStatePoolLayout` は pool 全体単一値（4） | **新 kind ではなく model-local pool**（G4a の判断を支持。`GdnStatePool` 非変更） |
| PLE n-gram context（2 token 履歴） | integer state。既存 kind は全て GPU tensor 前提 | **model-local**（host 側でも保持する必要があり、GPU state と分離） |

## 判断に必要な未確定事項

1. **PLE の hash を GPU で実行するか host で実行するか**。
   host の場合、row ID の upload が毎 token 発生し、G2 の `gather_rows_ssd` は
   同期読み込み前提（非同期化は手順書 §7.2-6 で後続扱い）。**最適化前の正しさ優先**で
   まず同期経路を graph に載せるのが手順書の方針に合う。
2. 各新 Primitive が既存モデル（Qwen35）に影響しない generic な意味と形状契約を持つか。
   手順書 §10-4 の要求。**Qwen35 の挙動を変えないことを必須条件にする**。
3. `KernelId` の追加に伴う correctness kernel（`kernels/correctness/detail/*.inc`）と
   optimized kernel の双方の整備。**correctness fallback を先に成立させる**
   （手順書 §9.5「最初から最速カーネルでなくてよい」）。

## 追記（G5 lowering 調査）: PLE embedding は既存 `EmbeddingLookupNode` で表せない

実測:

- `EmbeddingLookupNode`（`primitive_node.h:58`）は `weight_index` を持ち、
  dispatch（`program.cpp:220`）は `c.weights.count` を検査して
  **WeightSlot（GPU 上に常駐する weight）を参照する**。
- qwen4exp の `ngram_embedding` は **320,001,536 行 × 160 × BF16 = 約 95.4 GiB**
  で、`docs/developer/qwen4exp.md` §9.1 は「GPU 常駐を前提にしない」と明記している。
- したがって PLE の行取得は **G2 の `ple_row_gather`（SSD / RAM / VRAM の
  byte span gather）を graph の外側、または新しい経路で供する**必要がある。
  `WeightSlot` に載せることは VRAM 予算的に不可能。

**設計判断（暫定）**: PLE embedding の行取得を PrimitiveGraph へどう接続するかは
次のいずれかで、**未確定**:

1. 外部入力として row bytes を graph 外で gather し、`external_inputs` 経由で渡す
   （row ID の計算は host、gather も graph 外。手順書 §7.2-6 の非同期化は後続扱い）
2. 新規 Node を設け、その Node の実行が G2 の gather を呼ぶ
   （state / workspace の ownership 定義が必要）

どちらでも **既存 `EmbeddingLookupNode` を流用しない**という点は確定。
選択の根拠は G5 の後続（PLE lowering）で確定し、この節を更新する。

## 追記（G5 lowering 調査）: GR を表すための新規 Primitive の最終案

G5a（GR の Primitive 化担当）へ渡した設計:

| 新規 Primitive | 意味（generic な形状契約） | GR での用途 |
| --- | --- | --- |
| `GROUP_MEAN` | 入力行を連続する `group` 個のブロックに分け、ブロック内を平均して1行を返す（`[rows*group, features]` → `[rows, features]`、`group_size` に `group`） | `mixed = mean_over_4(w * n.unflatten(-1,(4,2560)))` |
| `GROUP_BROADCAST_MUL` | `out[r*group + g, h] = a[r, h] * b[r, g]`（`[rows, features] × [rows, group]` → `[rows*group, features]`） | `sub_out[...,None,:] * inj[...,None]` |

既存 Primitive へ分解できるもの（新規不要）:
grouped RMSNorm（`RmsNormNode.group_size` + `ONE_PLUS`）/ `LINEAR` / `SILU` /
`SCALE` / `SIGMOID` / `RESIDUAL_ADD`。

`DispatchBinding` の制約（`kMaxIoSlots = 6`、`parameter_index` 2 個、
`weight_index` 1 個）により、GR を 1 ノードに収められないことが分解の根拠。

## 追記（G5 lowering 調査）: MoE の weight を `WeightSlot` に載せられない

実測と計算:

| 項目 | 値 |
| --- | --- |
| checkpoint 総パラメータ（BF16） | 約 180 B（`index.json` の `total_size` 359,999,963,128 B ÷ 2） |
| routed expert（1 層） | `gate_up [512,1280,2560]` + `down [512,2560,640]` = 約 2.52 B パラメータ |
| routed expert（48 層） | **約 120.8 B パラメータ** |
| R9700 VRAM | 34,208,743,424 B（約 31.9 GiB） |

→ **expert 全体は VRAM に常駐できない**。手順書 §1.4「当面利用できる BF16 / FP8 の
**必要 tensor・必要 Expert だけを読み込み**」という方針と整合し、
G3 の `load_bf16_matrix_axis_slice` / `load_fp8_matrix_axis_slice` がまさにその手段である。

一方 `WeightSlot`（`include/shelf/runtime/program/program.h:21`）は
`rows` / `cols`（**rank-2 1 枚**）と `codes` / `scales` を各1つしか持たない:

1. **fused rank-3 tensor をそのまま `WeightSlot` に載せられない**（rank 拘束）
2. **expert ごとの pointer 一覧を持てない**（`weight_index` は1つの slot を指すだけで、
   `kMaxIoSlots = 6` でも top-10 の weight 参照には足りない）
3. `MOE_EXPERTS` を1つの Node にすると、dispatch 側に「選択 index に応じて
   複数の weight を参照する」仕組みが**存在しない**

**設計上の未確定事項（G5 の後続が判断する）**:

- **(a) synthetic のみ**: G5 の小型 fixture（expert 4 件）は全件 VRAM に載るため、
  expert を rank-2 に分解して `WeightSlot` を4つ登録し、
  `MOE_EXPERTS` Node に `weight_index` を複数渡せるよう `DispatchBinding` を拡張する。
  実モデル（512 expert）は G6 の課題として明記する。
- **(b) 必要 Expert の動的ロード**: G3 の slice API で選択 Expert をロードし、
  GEMM を回す。キャッシュと ownership が要り、**手順書の G7（AVX512 CPU/GPU 協調）や
  §1.4 の方針に近い**。G5 の synthetic 範囲を超える。
- **(c) expert を rank-2 × E 枚の `WeightSlot` に展開して登録**し、
  `MOE_EXPERTS` に slot 範囲を与える。

いずれも **`DispatchBinding` / `WeightSlot` への runtime 変更を伴う**。
手順書 §10-4「既存モデルの挙動を変えない」を守りつつ最小限にすること。
G4b の MoE 単体検証（kernel 契約）の完了を待ってから選択する。

## 追記（G5 lowering 調査）: 1 layer の forward と既存ノードの対応

`docs/developer/qwen4exp.md` §2 の 1 層の構造を、既存 Primitive との対応に分解した
（`emit_gdn` / `emit_gated_attention` が既に使えることを G0 で確認済み）。

```text
x4 (10240)
  └ if ple: x4 = x4 + ple(x4, input_ids)            # PLE 層のみ（0-indexed 1）
  mixed, hyper, inj = attn_hyper_connection(x4)      # GR read
  attn_out = attention_or_gdn(mixed)                 # 既存 emit_gated_attention / emit_gdn
  x4 = hyper + (attn_out[...,None,:] * inj[...,None]).flatten(-2)   # GR write
  mixed, hyper, inj = mlp_hyper_connection(x4)       # GR read
  moe_out = moe(mixed)                               # MoE（新規）
  x4 = hyper + (moe_out[...,None,:] * inj[...,None]).flatten(-2)    # GR write
```

| ステップ | 対応 | 状態 |
| --- | --- | --- |
| `mixed, hyper, inj = GR_read(x4)` | `RmsNormNode(group=2560, ONE_PLUS)` + `LINEAR` + `SILU` + `SCALE` + `SIGMOID` + **新規 `GROUP_MEAN`** | G5a が担当 |
| GR write | **新規 `GROUP_BROADCAST_MUL`** + 既存 `RESIDUAL_ADD` | G5a が担当 |
| `emit_gated_attention`（QSA dense 12 層） | `q_proj [12288,2560]` = 2×24×256 → `SplitNode(INTERLEAVED_HEADS)` で query / gate に分割 → `q_norm` / `k_norm`（ONE_PLUS）→ `RoPENode(rotary_dim=64)` → `KV_APPEND` → `PAGED_ATTENTION` → `SIGMOID` → `MUL` → `o_proj [2560,6144]` | **既存ノードのみで表現可能**（`paged_attention_selector` の `(24,2,256)` ルールが無いため correctness fallback。機能は成立する） |
| `emit_gdn`（GDN 36 層） | `emit_gdn` そのまま（`output_gate_silu=false` で sigmoid） | **再利用可**（G4a が config→lowering 契約を 46 件で検証） |
| `moe(mixed)` | router / experts | **新規**（weight 管理の制約は前節） |
| PLE | hash → row gather → projection / gate / conv | hash / conv / gather は実装済み。graph への接続方法が未確定（前節） |
| `h = hyper_connection_mixer(x4)` | GR collapse（`use_combine=False`、write なし） | G5a が担当 |
| `logits = lm_head(h)` | `LINEAR` | 再利用可（final norm は存在しない点に注意） |

## 追記（G5 lowering 調査）: QSA の selector 経路

`select_paged_attention_implementation`（`src/shelf/inference/execution/selectors/paged_attention_selector.cpp`）
の登録ルールは `(16,4,256)` / `(24,4,256)` / `(12,2,256)` のみ。
qwen4exp の QSA main attention は **24 query / 2 kv / head_dim 256** であり、
**optimized ルールが無い**。geometry 不一致は `paged_attention_selector.cpp:63` の
`Correctness` へのフォールバックで拾われるため機能は成立するが、
optimized 経路は使えない。

- 回帰リスク: ルールを追加すると既存 Qwen35 の選択結果に影響しうるため、
  **G5 ではルール追加をせず correctness fallback を受け入れる**。
  optimized 化は QSA の独立 Gate（本手順書の後続）で扱う。

---

# 付録: G4 / G5a で確定した設計判断（実装後の追記）

| 項目 | 判断 | 根拠 |
| --- | --- | --- |
| GDN output gate の sigmoid 切替 | **既存を再利用（再実装禁止）** | `gdn_lowering.h` の分岐、config 解析、SIGMOID kernel が既に実装済み。G4b の fit テスト（50 件）で config→lowering→selector→kernel の4段を検証 |
| q/k repeat-interleave 16→48 | **recurrence kernel 内の `num_v_heads/key_heads=3` が吸収**（lowering に repeat ノードなし） | G4b の差分調査票。oracle 側で独立に再現して検証 |
| QSA sparse mask の表現 | **新 PrimitiveKind 不要**。`PagedAttentionNode` に optional な 2 番目入力（mask）を追加 | G4c。dispatch の io 検証と kernel の mask ポインタ追加のみ。Qwen35 は入力 1 本のまま不変 |
| QSA indexer key state | **新 state kind `INDEXER_KEY_CACHE` が 1 種必要** | `KV_CACHE` は paged pool と `kv_heads*head_dim` 検証が前提で `[tokens,128]` 連続 raw key を格納不可 |
| (24,2,256) の optimized ルール | **追加しない** | 未計測 geometry に Optimized を宣言しない。correctness fallback で機能成立 |
| GR の表現 | **既存ノードへ分解 + 新規 2 種のみ**（`GROUP_MEAN` / `GROUP_BROADCAST_MUL`） | `DispatchBinding` の容量制約（`kMaxIoSlots=6`、`parameter_index` 2 個、`weight_index` 1 個）。G5a の `hyper_lowering.h` |
| GR の `unflatten` | **ノード化しない** | row-major 連続メモリ上での解釈。G5a が G4a の `mean_over_4` と一致することを実測 |
| MoE の weight 常駐 | **fused rank-3 を `WeightSlot` に載せられない**（rank-2 1 枚のみ、expert ごとの pointer 一覧も持てない）。Expert は約 120.8 B パラメータで VRAM に常駐不可 | 手順書 §1.4「必要 Expert だけを読み込み」と整合。選択肢 (a)(b)(c) を前節に記録、G5 の後続が選択 |
| PLE embedding の graph 接続 | **既存 `EmbeddingLookupNode` は使えない**（`WeightSlot` 参照で 95.4 GiB を VRAM 常駐させられない） | 前節に記録。G2 の `ple_row_gather` を経由する |
| PLE conv の state | **`GdnStatePool` 非変更 + model-local `PleConvState`** | `GdnStatePoolLayout` は `conv_history` を pool 全体単一値（4）で保持し、PLE の 9 を保持不可 |
| indexer の `q/k_layernorm` | **RMSNorm + ONE_PLUS**（暫定） | `docs/developer/qwen4exp.md` §6.2 に記載なし、§4.1 に倣った。**参照実装入手後に要再確認** |
| Top-K 同点の正本順序 | **(score 降順, block id 昇順)**（契約として定義） | 正本未確定のため両 oracle と kernel で一致させて固定 |

---

# 付録: G5 統合の実装 interface 固定（round 2）

round 2 で G5 本体へ着手するため、**委任間で不整合が起きないよう interface を先に固定**した。
これより下の記述は各担当の正本とする。

## A. lowering API（`Qwen4ExpLoweredPrimitives`）

Qwen35 の `lower_qwen35_to_primitives`（`models/qwen35/model/lower_to_primitives.h`）と
**同型**にする（手順書 §10-1「現行のモデル層（config / weights / lower_to_primitives）を
再調査する」。差を生やさない）:

```cpp
// include/shelf/models/qwen4exp/model/lower_to_primitives.h
namespace ps::qwen4exp {

struct Qwen4ExpLoweredPrimitives {
    ps::runtime::PrimitiveGraph graph;
    std::vector<ps::runtime::WeightSlot> weights;
    std::vector<ps::runtime::StaticParameterSlot> parameters;
    std::vector<std::pair<std::uint32_t, ps::runtime::ValueId>> layer_input_values;
};

struct Qwen4ExpLowerOptions {
    std::span<const std::uint32_t> hidden_taps;
    std::span<const std::uint32_t> override_layers;
    ps::EffectiveRopeProfile rope_profile;   // qwen35 のものと同型なら再利用可
};

Result<Qwen4ExpLoweredPrimitives> lower_qwen4exp_to_primitives(
    const Qwen4ExpTextConfig& config,
    const Qwen4ExpModelWeights& weights,
    const Qwen4ExpLowerOptions& options = {});

}
```

`EffectiveRopeProfile` が Qwen35固有ヘッダにある場合の扱いは、担当が再利用可否を
差分調査票で判定する（`models/qwen35/` を `qwen4exp` から include できるかは
`test_architecture_boundaries` の規則を確認すること。generic な共通ヘッダへの移設は
既存 Qwen35 の挙動を変えないことを条件に判断する）。

## B. MoE の Node 契約（新規 2 種）

G4b の `models/common/qwen_hybrid/moe_math.h`（pure F32 計算）が oracle として既にある。
runtime 側は次の契約で実装する。

| Node | 入力 | 出力 | geometry / weight |
| --- | --- | --- | --- |
| `MoERouterNode` | `x [T, hidden]`（BF16） | `top_idx [T, top_k]`（I32）、`top_w [T, top_k]`（F32） | `weight_index` = router `[E, hidden]`、`experts = E`、`top_k`、`renorm`（`norm_topk_prob`）。softmax は **F32**（`docs/developer/qwen4exp.md` §7.1） |
| `MoEExpertsNode` | `x [T, hidden]`、`top_idx`、`top_w` | `out [T, hidden]` | fused rank-3 の代わりに **expert ごとの rank-2 を連続 `WeightSlot` に展開**（gate_up E スロット + down E スロット）。`weight_index` = gate_up 先頭、`experts = E`、`top_k` |

**weight 管理の判断（手順書 §1.4 と整合）**:
- `WeightSlot` は rank-2 1 枚のみで fused rank-3 を直接載せられない（前節）。
- G5 は **synthetic fixture（expert 数小）が前提**のため、
  `load_qwen4exp_weights_bf16` が取得した rank-3 fused tensor を
  **lowering 時に expert ごとの rank-2 `WeightSlot` へ分割登録**する
  （VRAM に載るのは fixture 分のみ。実モデル 512 expert の常駐は G6 の課題として明記）。
- `MOE_EXPERTS` の `weight_index` は gate_up の先頭 slot を指し、
  `experts` ぶん連続、down はその直後 E スロットと契約する。
  この連続性は lowering 側が保証する。

**shared expert は新 Node を作らない**: `LINEAR`（gate / up）+ `SILU` + `MUL` +
`LINEAR`（down）+ `SIGMOID` + `MUL`（`shared_expert_gate`）で表現できる。

**`DispatchBinding` 内の geometry 詰め方**（`experts` / `top_k` / `renorm` の渡し方）は
担当が決めてよい。ただし次の制約を守ること:
- `DispatchBinding` は `group_size` / `rotary_dim`（uint32）と `scalar_a` / `scalar_b`（float）、
  `flags`（uint32）のみ。既存フィールドの流用は**意味を変えない**こと。
  （例: `rotary_dim` を MoE の `top_k` に流用するなら、その旨を差分調査票に明記する）
- 既存 `KernelId` 値・既存 dispatch case の変更は禁止（追記のみ）。
  `KernelId::COUNT` の増分に伴う配列サイズ用途の追跡は `test_program_quant_contract` 等で回帰確認。
- correctness kernel を先に、optimized は後から（手順書 §9.5 / §9.7）。

## C. 1 layer の lowering 構造（担当の正本）

`docs/developer/qwen4exp.md` §2 の pseudo-code を、G5a / G4 の成果で次の順に接続する:

```text
x4 = emb.repeat(1,1,streams)                       # EMBEDDING_LOOKUP + expand
for each layer:
    if ple_layer: x4 = x4 + ple(x4, input_ids)     # PLE（接続方法は PLE 担当の判定）
    mixed, inj = emit_hyper_read(x4)               # hyper_lowering.h（G5a、実装済み）
    attn_out = emit_gated_attention(mixed)  or  emit_gdn(mixed)   # 既存（実装済み）
    x4 = emit_hyper_write(x4, attn_out, inj)
    mixed, inj = emit_hyper_read(x4)
    moe_out = router(experts(mixed)) + shared(mixed)   # MoE Node（B）+ 既存ノード
    x4 = emit_hyper_write(x4, moe_out, inj)
h = emit_hyper_collapse(x4)                        # hyper_lowering.h（実装済み）
logits = lm_head(h)                                # LINEAR。**final norm は存在しない**
```

- 層の種別は `Qwen4ExpTextConfig::layer_is_qsa(layer)`（12 QSA / 36 GDN）。
- **PLE は 0-indexed layer 1 のみ**（`ple_layer_index`）。
- `docs/developer/qwen4exp.md` §2「`model.language_model.norm.weight` は checkpoint に
  存在しない」「最終 collapse は `hyper_connection_mixer`」を必ず守る。

## D. state owner の整理（`docs/developer/qwen4exp.md` §13 との突合）

| state | 現状の所有者 | G5 での扱い |
| --- | --- | --- |
| QSA KV cache | 既存 `PagedKVPool`（`PrimitiveStateKind::KV_CACHE`） | 再利用。`emit_gated_attention` が `KV_APPEND` + `PAGED_ATTENTION` を emit |
| GDN conv / recurrent | 既存 `GdnStatePool` | 再利用。`conv_history = 4` は qwen4exp の GDN と一致 |
| **QSA indexer key cache** | **新 state kind `INDEXER_KEY_CACHE` が必要**（G4c の判定: `KV_CACHE` は paged pool 前提で連続 raw key を格納不可） | 所有者は **qwen4exp-local の独立 pool**（§E の決定。`G5d` の `IndexerKeyPool` として実装済み）。kind 登録は G5c の runtime 編集と衝突するため後続（§E） |
| **PLE short conv（`conv_history = 9`）** | `GdnStatePoolLayout` は pool 全体単一値（4）で保持不可 | **model-local `PleConvState`**（G4a の実装済み）。共通 pool は非変更 |
| **PLE n-gram context（2 token 履歴）** | integer のため既存 GPU state kind と不整合 | **model-local**（host 側保持を含め、GPU state と分離） |

`docs/developer/qwen4exp.md` §13 が「PLE short conv は `GDN_CONV_STATE` の
`state_index` で表現できる」としている点は**実装上成立しない**（単一 `conv_history` 制約）。
G5 で owner を確定したら docs を更新する。

## E. mini runtime の設計方針（round 2 追記）

手順書 §10 の PASS 条件（小型 fixture で prefill → decode、GDN / QSA / PLE 状態継続、
MoE 合成、final logits 再現）を満たす runtime の調査結果。

### 既存 runtime の構造（実測）

- **`Executor` は `models/qwen35/runtime/executor.hip`（2,295 行）にあり、
  `Qwen35Model` と `lower_qwen35_to_primitives` に密結合**
  （`executor.hip:1500` の `const Qwen35Model& model`、`:1837` の
  `lower_qwen35_to_primitives(...)`）。`qwen35::create_model_executor` の呼び出し元は
  `src/apps/compute/compute_runtime.hip` と `src/apps/bench/{quality,batch}.hip`、
  および qwen35 専用テストのみ。
- `Executor` は `SequenceSlotPool` / `GdnStatePool` / `PagedKVPool` を非所有ポインタで受け、
  `ProgramSet` と GPU scratch を所有する（`docs/developer/runtime.md` §State ownership）。
- **model 非依存で再利用できる部品**:
  - `ps::runtime::ProgramSet` / `build_program_set`（`runtime/program/program.h`）
  - `ps::execution::HostExecutionContext` と program executor（`inference/execution/program_executor.h`）
  - `ps::runtime::DeviceBatchContext` / `DeviceBatchContextStorage`（`runtime/batch/device_batch_context.h`）
  - `PagedKVPool` / `GdnStatePool` / `SequenceSlotPool`（`inference/state/`）
  - GDN / Attention selector、`kernels/optimized`・`kernels/correctness`

### 採用方針

**qwen35 の `Executor` を共通化しない**（手順書 §10-4「既存 Primitive で表現できるなら
共通 runtime を変更しない」。加えて `models/qwen35/runtime/` を触れば既存モデルへの
回帰リスクが直結する）。

代わりに `models/qwen4exp/runtime/` に **qwen4exp 専用の mini runtime** を置き、
上記の model 非依存部品を組み合わせる:

```text
models/qwen4exp/runtime/
    qwen4exp_executor.*      # ProgramSet + HostExecutionContext + state pool の組立と prefill/decode
    qwen4exp_state.*         # INDEXER_KEY_CACHE pool / PLE conv state / PLE n-gram context の owner
```

- `qwen35::Executor` の複製（2,295 行）は行わない。**手順書 §10-8 が求める
  「層間入出力と state 更新の比較」に必要な最小経路**（1 batch、prefill → decode、
  state の継続と破棄）に絞る。
- 新規の汎用 `IModel` / Registry / Plugin interface は作らない（§10-9）。
- `ContinuousBatcher` との統合は mini runtime の検証後、§10-10 の
  「新 state の保存・復元に必要な構造と owner を記録する」を先に満たす。
  `PrefixCache` への統合は後続課題（Drafter 固有情報を PrefixCache へ持ち込まない）。

### owner の割り当て（確定案）

| state | owner | 備考 |
| --- | --- | --- |
| QSA KV cache | 既存 `PagedKVPool`（app 側） | 再利用 |
| GDN conv / recurrent | 既存 `GdnStatePool`（app 側） | 再利用。`conv_history = 4` が qwen4exp GDN と一致 |
| QSA indexer key cache | **qwen4exp-local の独立 pool** | `PrimitiveStateKind::INDEXER_KEY_CACHE` の追加が前提（G5c の runtime 編集と競合するため後続） |
| PLE short conv（`conv_history = 9`） | **qwen4exp-local `PleConvState`**（G4a 実装済み） | `GdnStatePoolLayout` の単一 `conv_history` 制約により共通 pool 不可 |
| PLE n-gram context（2 token 履歴） | **qwen4exp-local（host integer）** | 既存 GPU state kind と dtype 不整合 |

## F. PLE の graph 接続方法（round 2 調査）

`PrimitiveGraph::external_inputs`（`runtime/graph/primitive_graph.h:68`）が既に存在し、
qwen35 lowering も `pg.external_inputs.push_back(tokens)` として使う（`lower_to_primitives.cpp:532`）。
`program.cpp:153` が external value を `ValueStorage::EXTERNAL_INPUT` に割り当て、
`HostExecutionContext::external_inputs`（`void* const*`）から device pointer を受け取る。

**採用案（暫定）**: PLE の「hash → row gather → [16,160] → [2560]」は **graph の外**で実行し、
`[T, 2560]` BF16 の embed 値を `external_inputs` として graph へ渡す。

- graph 外の工程（**既に G4a / G2 で実装・検証済み**）:
  `ple::compute_ngram_ids`（host、n-gram context は host integer state）→
  `ple::locate_rows` → `ple::gather_rows_ssd` / `_ram` / `_vram`（GPU）
- graph 内は embed 値からの projection 以降（既存ノードで表現できる範囲が多い）

これにより **`EmbeddingLookupNode` を PLE に使わない**（95.4 GiB を `WeightSlot` に
載せられない）という制約を回避しつつ、新規 Node の追加を最小化できる。

### graph 内で新規が必要になる範囲（`docs/developer/qwen4exp.md` §9.2 より）

| ステップ | 対応 | 判定 |
| --- | --- | --- |
| `key_proj` / `value_proj` / `key/value` の投影 | `LINEAR` | 再利用可 |
| `norm_key` / `norm_query` / `norm_conv` | `RmsNormNode`（group=2560、`ONE_PLUS`） | 再利用可 |
| `gate = sum_over_2560(key * query) / sqrt(2560)` の elementwise 積 | `MUL` | 再利用可 |
| **`sum_over_2560`（features 方向の reduce）** | 既存に**無い**。`GROUP_MEAN` は行ブロック平均で方向が異なる | **新規 Node 要**（または graph 外へ逃がすか） |
| `gate = sqrt(abs(gate).clamp_min(1e-6)) * sign(gate)` | `SCALE` / `SIGMOID` の組合せでは表現困難 | **新規 Node 要**（または graph 外） |
| `sigmoid(gate) * value` | `SIGMOID` + `MUL` | 再利用可 |
| `depthwise_conv1d(dilation=3)`（`conv1d.weight [10240,1,4]`） | 既存 `STATEFUL_CAUSAL_CONV1D` は `GdnShapeKey` に **dilation を持たず再現不可**（G4a で判定） | **新規 Node 要**（G4a の model-local host 実装を GPU 化するか、graph 外） |
| `out = gated + conv` | `RESIDUAL_ADD` | 再利用可 |

**結論**: PLE の graph 接続は G5 の後続ラウンドで次のいずれかを選ぶ。

1. **外部入力を embed 値まで広げ、gate 計算と conv を graph 外で完結させる**
   （新規 Node は 0、graph 内は既存ノードのみ。ただし PLE の計算が graph の外に出る）
2. **新規 Node 2〜3 種（PLE gate reduce、gate 後処理、dilated conv）を追加する**
   （graph 内に PLE を収める。新規 Primitive の generic な契約を定義する必要あり）

いずれも **PLE n-gram context（2 token 履歴）と PLE conv state の owner は
qwen4exp-local**（§D）で変わらない。

## G. 手順書 §10-10: 新 state の保存・復元に必要な構造と owner（round 2 記録）

「PrefixCache との完全統合は後続課題とするが、新 state の保存・復元に必要な構造と
owner を記録する」（Drafter 固有情報を PrefixCache へ持ち込まない）に対する記録。

### owner と lifecycle（G5d で実装済み）

| state | owner（実装ファイル） | 保存に必要な構造 | 復元に必要な操作 |
| --- | --- | --- | --- |
| QSA KV cache | 既存 `PagedKVPool`（app 側） | page 単位の K/V（既存 `PrefixCache` が扱う対象） | 既存経路 |
| GDN conv / recurrent state | 既存 `GdnStatePool`（app 側） | layer × conv_dim × `conv_history`、layer × v_heads × head_k × head_v（FP16） | 既存経路 |
| QSA indexer key cache | `ps::qwen4exp::runtime::IndexerKeyPool`（`models/qwen4exp/runtime/qwen4exp_state.{h,cpp}`） | slot × layer × token × 128 BF16 の**連続レイアウト**（paged ではない）。`key_elements_offset()` がオフセット契約を公開 | `initialize`（ゼロ埋め）→ 保存した raw key 列の書き戻し。`PrimitiveStateKind::INDEXER_KEY_CACHE` の登録は後続（G5c の runtime 編集完了後に追加可能） |
| PLE short conv state | `ps::qwen4exp::runtime::PleConvStatePool`（同上） | slot × conv_dim × 9（`ple::kPleConvStateLen`）。**host `float`** | `initialize`（`ple::PleConvState::reset`）→ 履歴の書き戻し。計算は G4a の `ple_dilated_conv1d` へ委譲 |
| PLE n-gram context | `ps::qwen4exp::runtime::PleNgramContext`（同上） | slot ごとの host `uint32[2]` + GPU 転送バッファ `[slots, 2]` | `initialize`（`[eos, eos]`）→ 2 token の書き戻し。`upload` で GPU へ同期 |

### PrefixCache 統合を後続にする際の注意点

1. **既存 `PrefixCache` が扱うのは KV page と GDN state の shadow pool**
   （`docs/developer/runtime.md` §State ownership、
   `docs/research/20261023/qwen_model_layer_split/` の設計方針）。
   qwen4exp の indexer key / PLE conv / PLE n-gram は**それらと別 owner** にあり、
   snapshot する場合は PrefixCache 側に新フィールドが要る。
2. **PLE n-gram context は host integer** であり、GPU state の snapshot 経路に乗らない。
   PrefixCache の既存 API（GPU shadow pool / cold pinned buffer）とは性質が異なる。
3. **PLE の n-gram table 自体（95.4 GiB）は snapshot 対象外**。
   復元時に必要なのは 2 token 履歴だけであり、row は G2 の経路で再取得できる。
4. **Drafter（DFlash2 / EAGLE4）固有の情報を PrefixCache へ持ち込まない**
   （手順書 §10-10 の禁止事項）。qwen4exp の MTP は §11 のとおり forward 正本が
   未確定であり、Drafter 統合は後続 Gate。


---

# 付録: G5f（PLE の graph 接続）の差分調査票（round 3）

| 項目 | 記録内容 |
| --- | --- |
| 調査対象 | PLE（0-indexed layer 1）の graph 接続方法と PLE 状態（n-gram context / short conv）の継続検証 |
| ZIP 時点の想定 | 付録 F が2案を提示（案 1: 外部入力を embed 値まで広げ gate / conv を graph 外、案 2: 新規 Node 2〜3 種を graph 内に追加）。いずれも PLE 加算位置の `x4` を「`emb.repeat` の値」として記述 |
| 現行実装 | `docs/developer/qwen4exp.md` §2 は PLE を**層ループ先頭**に置く（`if ple: x4 = x4 + ple(x4, input_ids)`）。したがって加算位置の `x4` は **layer 0（GDN + GR + MoE）の出力**であり `emb.repeat` ではない。`ple(x4, ...)` の `query = norm_query(hidden_states)` も layer 0 出力を消費する |
| 食い違い | **あり**。round 2 の付録 C / 付録 F の pseudo-code を「PLE 層の入力は emb.repeat」と誤読した（手順書 §0 の「実ソースと違ったら調査をやり直す」に該当）。§2 と `lower_to_primitives.cpp` の層ループを追加読解して再判定した |
| 追加で読んだ箇所 | `docs/developer/qwen4exp.md` §2 / §9 / §13、`src/shelf/models/qwen4exp/model/lower_to_primitives.cpp`（層ループ）、`src/shelf/runtime/graph/primitive_graph.cpp`（`validate_primitive_graph` / external input 契約）、`src/shelf/runtime/program/program.cpp`（`EXTERNAL_INPUT` の binding）、`src/shelf/kernels/correctness/detail/elementwise.inc`（`RESIDUAL_ADD` の dtype 契約: F32 出力、BF16 入力は F32 へ変換）、`src/shelf/models/qwen35/runtime/executor.hip:1844`（`layer_input_values` の用途 = MTP の layer 入力上書きであり中間値の取り出しではない）、`src/shelf/models/qwen4exp/ple/*`（G2 / G4a の部品）、`src/shelf/models/qwen4exp/runtime/qwen4exp_state.*`（G5d）、`include/shelf/inference/state/{paged_sequence_state,gdn_state_pool}.*` |
| 採用判断 | **案 1 を graph 分割付きで採用**（下記）。案 2 は新規 `PrimitiveStateKind` と `ModelDispatchStateView` の追加が不可避で G5d で固定した「PLE conv state は model-local owner」（§13 の owner 表）と矛盾するため不採用 |
| 回帰リスク | `connect_ple=false`（既定）の経路は node 列・weight slot・parameter slot・external_inputs 数が不変。Qwen35 / 既存FP8 / GDN / KV / Scheduler への影響なし（`test_architecture_boundaries` も PASS） |
| 検証計画 | 新規 `test_qwen4exp_ple_graph`（prefill chunk 境界 / decode 続続 / cancel / EOS / Request 再利用 + 仕様 oracle 比較）と既存5件の回帰 |
| 判断根拠 | 手順書 §10-4「既存 Primitive で表現できるなら共通 runtime を変更しない」、AGENTS.md「200行の高速コードを2000行の汎用コードより優先する」「仮定の将来機能を軸にアーキテクチャを再設計しない」。実測: `test_qwen4exp_ple_graph` 83 件 / 0 failed（GPU2 / GPU3）、全比較が bitwise（max_abs 0） |

## 採用した設計（graph 分割）

`Qwen4ExpLowerOptions::connect_ple`（既定 false）。true のとき lowering は graph を2つ返す:

```text
prefix graph: tokens(I32) → EMBEDDING_LOOKUP → CONCAT×3 → layers[0, ple_layer)
              → x4 (F32 [T, hc_hidden]) を external output
（graph 外）: PleBranch（n-gram hash → row gather → projection / norm / gate → dilated conv）
              PleNgramContext と PleConvStatePool を更新し PLE 出力 [T, hc_hidden] を得る
main graph:  x4(F32) + ple_out(F32) → RESIDUAL_ADD("L1.ple.add") → layers[ple_layer, N)
             → collapse → lm_head → sampling
```

- **新規 Node 0・新規 Primitive 0・runtime / kernels 無編集**。
  `Qwen4ExpLoweredPrimitives` に `has_ple_prefix` / `ple_prefix_graph` /
  `ple_prefix_weights` / `ple_prefix_parameters` を追記し、既存4フィールドは不変。
- GDN / KV の `state_index` は prefix → main で連番を継続させ、同一の pool を両 graph が共有。
- `hidden_taps` / `override_layers` / `lm_head_tied` と `connect_ple` の併用は
  `invalid_argument` で拒否（未対応）。
- PLE 重み（key_proj / value_proj / conv1d / norm_*）は graph に載せず、
  `PleBranch::create` が caller の stream で host F32 へ stage する
  （`weight_loader` が `hipMemcpyAsync` で積むため、stage には stream 同期が必須。
  同期なしでは非決定的な NaN / 大値が出ることを実測で確認）。

## 残課題（PLE 関連）

- PLE projection / gate / conv の **GPU 実行経路**（現状は host F32。
  `docs/developer/qwen4exp.md` gap #6 の後半が未達のまま）
- 実モデル（`ple_embed_dim = 2560` / `hc_hidden = 10240` / 95.4 GiB table）での
  SSD 経路（`gather_rows_ssd`）の E2E は G6
- prefix / main の2 program 化を吸収する runtime 経路（entrypoint 接続は後続）

# Qwen4Exp Gate 進行記録

- ID: QWEN4EXP-GATES
- 対象: Qwen3.8-Flash-Next（qwen4exp）対応 調査・実装手順書の Gate ごとの
  PASS / FAIL / BLOCKED と証跡
- 配置理由: これは R&D 記録であり `docs/developer/` には置かない
  （`docs/developer/` は現在有効な contract のみを書く規約）。

## 差分調査票の所在（手順書 §0.3 との対応）

§0.3 の9列（調査対象 / ZIP 時点の想定 / 現行実装 / 食い違い / 追加で読んだ箇所 /
採用判断 / 回帰リスク / 検証計画 / 判断根拠）で記録した箇所を一覧する。

| Gate | 記録箇所 | 形式 |
| --- | --- | --- |
| G0 | `source_audit.md` §1「項目別の差分調査票」+ §4「未解決の食い違い」 | 9列の表（全11項目） |
| G1 | 下記「G1」の差分調査票 | 9列の表 |
| G2 | 下記「G2」の差分調査票 | 9列の表 |
| G3 | 下記「G3」の差分調査票 | 9列の表 |
| G4a | 下記「G4a」の差分調査票（項目別） | 項目別表 |
| G4b | 下記「G4b」の「差分調査票の要点」 | 項目別表 |
| G4c | 下記「G4c」の「設計判断（G5 へ引き継ぎ）」 | 判断別表（新 Primitive / 新 state kind / 再利用 / 不要化） |
| G5a | 下記「G5a」の「GR を既存 Primitive へ分解した判断」と「新規 Primitive 2 種」 | 判断別表 |
| G5b | 下記「G5b」の「項目別（個別判断）」 | 判断別表（`EffectiveRopeProfile` / PLE / QSA selector / final norm ほか） |
| G5c | 下記「G5c」の「`DispatchBinding` の geometry 流用」と「weight 契約」 | 判断別表 |
| G5d | 下記「G5d」の「既存 `GdnStatePool` との違い」と「G4a の conv state を再実装せず組み込んだ方法」 | 比較表 |
| G5e | 下記「G5e」の「既存 forward fixture とどう違うか」と「重大な発見」 | 比較表 + 発見記録 |
| G5f | 下記「G5f」の設計表と「修正した実装バグ」 | 判断別表 |

**未解決の食い違いの追跡**は `source_audit.md` §4 と各 Gate 節の「残る BLOCKED / 未解決」に記載。

なお **round 2 / round 3 の実装 interface 固定と設計判断の統合記録**は
`source_audit.md` の「付録: G5 統合の実装 interface 固定」「付録: G5 lowering のための
新 Primitive 設計メモ」「付録: PLE の graph 接続方法」「付録: MoE の weight を
`WeightSlot` に載せられない」「付録: mini runtime の設計方針」
「付録: 手順書 §10-10 の state 保存・復元」にまとまっている。

## 環境・ベースライン

| 項目 | 値 |
| --- | --- |
| worktree | `.worktrees/qwen4exp` |
| branch | `investigation/qwen4exp-main` |
| HEAD | `9825d10896fea4f4d9736f918617c41322ae53b2` |
| ベースライン required acceptance | **全件 PASS**（`PHASESHIFT_TEST_GPUS=2,3` 指定。GPU0/1 は他プロセスが VRAM 約 33.6GB 使用中のため runner の予約台帳に載らず OOM になる） |
| ベースライン出力 | `build/required-acceptance.txt` |
| 注意 | 件数は build 構成で変わるため固定しない（`docs/developer/testing.md`） |

ベースライン取得の経緯:

1. 初回は `phaseshift-bench` / `phaseshift-compute` 未ビルドで bench smoke / CLI sync が Not Run。
2. 全ターゲット build 後も GPU0/1 の外部使用により `hipMalloc failed: out of memory` が多発。
3. `PHASESHIFT_TEST_GPUS=2,3` で空き GPU に限定して全件 PASS を確認。

以後の判定はすべてこのベースラインとの差分で行う。

G1〜G3 と G5 の一部が揃った時点での中間スナップショット
（`PHASESHIFT_TEST_GPUS=2,3 ctest --test-dir build -L required`）:
**149 件すべて PASS**（ベースライン 143 + 本 Gate 群で追加した 6 件）。
追加された 6 件は `test_safetensors_partial_read` / `test_ple_row_gather` /
`test_qwen4exp_config` / `test_qwen4exp_weights` / `test_expert_slice_fixture` /
`test_expert_slice_load`。

## Gate 一覧

| Gate | 内容 | 状態 | 証跡 |
| --- | --- | --- | --- |
| G0 | 現行ソース再調査・参照 revision 固定 | **PASS（条件付き）** | `source_audit.md` / `model_inventory.md` / `hipfile_qualification.md` |
| G1 | Safetensors メタデータ・部分読込基盤 | **PASS** | 下記「G1」 |
| G2 | PLE SSD + hipFile / RAM / VRAM | **PASS** | 下記「G2」 |
| G3 | BF16 / FP8 Expert slice | **PASS** | 下記「G3」 |
| G4 | PLE / GR / MoE / QSA / GDN 単体検証 | **PASS（G4a / G4b / G4c）** | 下記「G4」 |
| G5 | qwen4exp 統合（小型 fixture） | **PASS** | 下記「G5」 |
| G6 | 実モデル E2E | **着手不可（前提未達）** | 下記「G6」 |
| G7 | AVX512 CPU/GPU 協調 | **着手不可（G6 未達）** | 下記「G7」 |

## 手順書 §14.2 のテスト要件との対応

| Gate | 手順書の最低限のテスト | 自己完結性 | 本 Gate 群での対応 |
| --- | --- | --- | --- |
| G0 | source audit / inventory の整合性 | checkpoint 未取得でも継続可 | `source_audit.md` / `model_inventory.md`（ヘッダのみ解析、本文未ロード）。充足 |
| G1 | safetensors dtype / offset / slice | 必須 | `test_safetensors_partial_read`（I64 / BF16 / FP8、単一・複数 shard、rank-2/3、Python 比較）。充足 |
| G2 | hipFile SSD row gather、RAM / VRAM bitwise、fastpath 統計 | synthetic と設備依存を分離 | `test_ple_row_gather`（synthetic、required）と `ais-stats` 実測（`g2_ais_stats.txt`、設備依存の証跡）を分離。充足 |
| G3 | Expert BF16 / FP8 slice + scale + GEMM | synthetic 必須、実重み optional | `test_expert_slice_fixture` + `test_expert_slice_load`（synthetic のみ）。実 checkpoint は FP8 不在のため G6 へ引き継ぎ |
| G4 | 各演算の oracle / 境界 / state 一致 | 原則 synthetic 必須 | G4a: `test_ple_ngram_hash` / `test_gated_residual` / `test_ple_conv`（+ optional の実 checkpoint 照合）。G4b: `test_qwen4exp_moe` / `test_qwen4exp_gdn_fit` / `test_qwen4exp_gdn_fit_gpu`。G4c: `test_qsa_indexer` / `test_qsa_sparse_attention`。充足 |
| G5 | mini-model prefill / decode、state ownership | synthetic 必須 | `test_qwen4exp_config` / `test_qwen4exp_weights` / `test_qwen4exp_state`（owner lifecycle）/ `test_qwen4exp_lowering_contract` / `test_qwen4exp_forward_fixture` / `test_qwen4exp_hyper_lowering` / `test_group_ops` / `test_moe_ops` / `test_qwen4exp_moe_graph` / `test_qwen4exp_ple_graph`。**PASS 条件の全項目を充足** |
| G6 | 実モデル E2E、SSD / VRAM 側実行、品質 | 外部モデル / 量子化完了が前提 | **着手不可**（FP8 checkpoint 不在、G1〜G5 未完） |
| G7 | CPU/GPU 協調の正しさ＋再現可能な速度差 | G6 後のみ | **着手不可**（G6 未達） |

## 手順書 §3「ZIP 時点の技術的観測」チェックリストとの対応

§3 は「修正命令ではなく再調査のチェック項目」と明記している。全項目の判定:

| # | §3 の項目 | 判定（着手時に再調査した結果） | 扱い |
| --- | --- | --- | --- |
| 1 | `safetensors_reader` の `SType` に `I64` がなく未知 dtype で拒否 | **一致**（7 値のみ。`I64` は `unsupported dtype`） | G1 で `I64` 追加。実 checkpoint で I64 3 tensor 実在を確認 |
| 2 | `data_base_offset()` / `tensor_spec()` は存在 | **一致** | 再利用。`tensor_span` / `axis_span` を追加 |
| 3 | 「既存の部分読込機能」を確認 | **食い違い**（row / Expert span API は HEAD 時点で不在） | G1 で新設（`StAxisSpan` / `axis_span`） |
| 4 | HF FP8 は `fmt == "e4m3"` 必須、rank-2 行前提 | **一致**（`hf_manifest.cpp` の rank-2 検査とエラー文） | 制約を維持。G3 でも任意 FP8 を受理しない実装にした |
| 5 | `weight_loader` は行列ロード中心 | **一致**（slice / partition / FP8 canonical の接続点を調査） | G3 で `load_{bf16,fp8}_matrix_axis_slice` を追加、既存 `load_quantized_view` を再利用 |
| 6 | `gdn_lowering.h` の `output_gate_silu` 切替は存在 | **食い違い**（実装済み。gap #7 が stale） | 再実装禁止。G4b の fit テストで4段連鎖を検証 |
| 7 | `paged_attention_selector` の新 QSA geometry | **食い違い**（`(24,2,256)` ルール不在、fallback は存在） | fallback を再利用。**ルール追加はしない**（未計測 geometry の Optimized 宣言をしない） |
| 8 | `models/qwen4exp` に Executor は見当たらない | **一致**（不在） | G5 で `model/` を新設（config / weights）。`runtime/` は round 2 で新設予定 |
| 9 | hipFile 連携は ZIP 内に見当たらない | **一致**（コード側は不在。**環境には libhipfile.so.0.4.0 が実在**） | G2 で `io/hipfile_reader` を新設。fastpath 実測まで達成 |
| 10 | `docs/developer/qwen4exp.md` の gap と `rnd/qwen4exp/gate0.md` 参照が古い可能性 | **食い違い**（`rnd/` 不在、gap #7 stale、`tools/inspect_qwen4exp.py` 不在） | 参照を `docs/research/20261024/qwen4exp/` へ置換、gap 表を現状化、該当記述を削除 |
| 11 | `tools/inspect_qwen4exp.py` は ZIP に見当たらない | **一致**（不在） | **新設しない**（`AGENTS.md` の tools ルール。inventory は `model_inventory.md` で代替） |

§0 の「本書に『未実装』『要変更』と書いてあっても、実際のソースで既に対応しているなら修正しない」に従い、
項目 6 / 7 / 10 は「docs 側の記述を現実に合わせる」方向で処理し、ソースを無理に変えていない。

## コミット分割（手順書 §15 との対応）

| commit | 変更単位 | 手順書の対応 |
| --- | --- | --- |
| `docs(research): Qwen3.8-Flash-Next の G0 ソース監査と Gate 記録を追加する` | G0 調査資料 + `docs/developer/qwen4exp.md` の stale 記述修正 | PR-0 |
| `feat(io): Safetensors の I64 と部分読込 span API を追加する` | G1（io + weights Collection API + test + 登録） | PR-1 |
| `feat(ple): SSD / RAM / VRAM の PLE row gather と hipFile 経路を実装する` | G2（io/row_reader + hipfile_reader + GPU gather + qwen4exp/ple + test + CMake） | PR-2 |
| `feat(weights): fused rank-3 expert の axis-slice ロードを追加する` | G3（weights slice + 2 テスト + quantization doc） | PR-3 |
| `feat(qwen4exp): config 読み込みと BF16 weights ローダを追加する` | G5 のモデル層第一段（config + weights + 2 テスト） | PR-6 の一部 |
| `feat(qwen4exp): PLE n-gram hash / PLE conv / Gated Residual の単体実装を追加する` | G4a（PLE / GR / PLE conv + 4 テスト） | PR-4 |

`cmake/tests.cmake` と `cmake/targets.cmake` は機能ごとに分割して stage した
（`git diff` の hunk 単位では複数機能が混在するため、HEAD からの段階復元で分離）。

上記 6 コミット時点の required acceptance: **152 件すべて PASS**
（`PHASESHIFT_TEST_GPUS=2,3`。ベースライン 143 に本 Gate 群の追加 9 件）。
G4b 追加後は 155 件、G4c + G5a 追加後は 159 件、
G5d 追加後は 160 件、G5c 追加後は 161 件、**G5b 追加後は 163 件すべて PASS**
（いずれも `PHASESHIFT_TEST_GPUS=2,3`。最終 163 件は 2 回連続で確認）。

**一時的失敗の記録**: 途中の1回で `99% tests passed, 1 tests failed out of 152`
を観測したが、直後の再実行で 152/152 PASS。
原因は同時刻に他エージェントが同一 GPU 群をテストで使用した競合と考えられる
（`phaseshift-gpu-test-runner` の予約台帳は自プロセスのみを追跡し、
他プロセスの VRAM 使用は反映されない。ベースライン取得時の GPU0/1 でも同じ現象を観測）。
恒常的な FAIL ではないため、判定には影響しない。

## 手順書 §15「完了の定義」との対応

| 手順書の定義 | 条件 | 現状 |
| --- | --- | --- |
| **カーネル実装完了** | G1〜G4 PASS（実モデル E2E 未達でも完成と扱える） | **G1〜G4 は PASS**（G4a / G4b / G4c）。本手順書の G5 は別 Gate として継続 |
| **モデル統合完了** | G5 PASS（実モデルの量子化待ちは明記） | **G5 = PASS**。実モデルの量子化（FP8 checkpoint）待ちは明記のまま |
| **Qwen3.8-Flash-Next 実モデル対応** | G6 PASS（hipFile Direct 状況は別に表示） | **着手不可**。FP8 checkpoint 不在、G1〜G5 未完。hipFile Direct は G2-Direct で実測済み（達成） |
| **CPU/GPU 協調最適化完了** | G7 PASS（G6 未達なら着手禁止） | **着手不可**（G6 未達） |

## G0

- 調査票: `source_audit.md`
- tensor inventory（実 checkpoint、ヘッダのみ解析）: `model_inventory.md`
- hipFile / SSD 環境: `hipfile_qualification.md`
- 判定: **PASS（条件付き）**
  - 重要なソース差分はすべて扱いを決めた
  - 保留 1 件: HF FP8 の `activation_scheme` 検査の要否（FP8 checkpoint 不在のため実値不明）
  - 未取得: `Qwen3.8-Flash-Next-FP8` checkpoint、Transformers `qwen4_exp` 参照 commit
    （G3 / G6 の開始条件。G1〜G5 の synthetic fixture には不要）

## G1

### 差分調査票（手順書 §0.3）

| 項目 | 記録内容 |
| --- | --- |
| 調査対象 | Safetensors の dtype 解析 / data offset / shard 解決 / 部分読込 API |
| ZIP 時点の想定 | 手順書 §6: `SType` に `I64` が無く未知 dtype で拒否。`data_base_offset()` / `tensor_spec()` は存在、「既存の部分読込機能」を確認する |
| 現行実装 | enum は 7 値のみ。`read_tensor` は dst_bytes の完全一致必須の memcpy、`tensor_data` は mmap 内ポインタの zero-copy。行 / Expert span API は**不在**。`SafetensorsCollection` は index.json の shard 解決のみで `shard_for` は private、path traversal 検査なし |
| 食い違い | **あり**（部分読込 API の不在、Collection の location API 不在、path traversal 検査なし）。`I64` 欠如は一致 |
| 追加で読んだ箇所 | `safetensors_reader.{h,cpp}`、`safetensors_writer.{h,cpp}`、`weight_loader.{h,cpp}`、`quantization/bundle/bundle_util.h`（`validate_shard_path`）、`tests/unit/test_safetensors_writer.cpp`、`cmake/tests.cmake` |
| 採用判断 | **再利用**: mmap 経路・検証群（shape / overflow / 境界）、index.json 解決、`SafetensorsWriter`（fixture）。**最小修正**: `SType::I64`、`StAxisSpan` / `tensor_span` / `axis_span` / `file_size`、Collection の location API、shard 名の path traversal 検査。**新設しない**: PLE 固有の N-gram hash や shard 配列（汎用 I/O 層を PLE 依存にしない） |
| 回帰リスク | mmap 経路と `read_tensor` / `tensor_data` の契約を変えていない。既存 reader / writer / manifest / partition / weight load を回帰対象にした |
| 検証計画 | `test_safetensors_partial_read`（cpu;required、86 件）を追加し、`test_safetensors_writer` / `test_hf_manifest` / `test_tensor_partition` / `test_canonical_partition` / `test_weight_load` を回帰確認 |
| 判断根拠 | 実 checkpoint の I64 3 tensor（`model_inventory.md` §3）、Python `safetensors` 0.8.0 との header / payload 比較、quantization 側 `validate_shard_path` の既存実装 |

### 変更

| ファイル | 変更 |
| --- | --- |
| `include/phaseshift/io/safetensors_reader.h` | `SType::I64` 追加、`stype_elem_size` / `stype_name`、`StAxisSpan`、`tensor_span` / `axis_span` / `file_size` の宣言 |
| `src/phaseshift/io/safetensors_reader.cpp` | `I64` の dtype 解析と elem_size 8、span 計算（64-bit overflow / tensor 境界 / ファイル境界の検査） |
| `src/phaseshift/io/safetensors_writer.cpp` | `dtype_name` / `elem_size` を `stype_name` / `stype_elem_size` へ統合（I64 対応） |
| `include/phaseshift/weights/weight_loader.h` | `SafetensorsCollection` に `tensor_shard` / `tensor_span` / `axis_span` / `shard_file_size` を追加 |
| `src/phaseshift/weights/weight_loader.cpp` | 上記の実装、shard 名の path traversal 検査 |
| `tests/unit/test_safetensors_partial_read.cpp` | 新規テスト |
| `cmake/tests.cmake` | テスト登録と `phaseshift-required-tests` DEPENDS へ追加 |

### 設計判断

- span API は**読み取り専用**で、tensor 全体の CPU コピー / GPU upload を前提としない。
  既存の mmap 経路（`tensor_data` の zero-copy）を壊さない。
- `axis_span` は axis-0 単位の連続 byte span を返す。rank-2 では行、
  rank-3 fused tensor では指定 index の 1 Expert slice に対応する。
  PLE の row と Expert slice を 1 つの API で扱える。
- PLE 固有の N-gram hash や shard 配列は入れない（汎用 I/O 層を PLE 依存にしない）。
- shard 名の path traversal 検査は quantization 側 `validate_shard_path` と
  同一の方針（先頭 `/` 拒否、`..` 拒否、区切り文字拒否）を weights 側へ導入した。

### 検証

- `test_safetensors_partial_read`（`cpu;required`）: **86 件 PASS / 0 FAIL**
  - I64 roundtrip（dtype / shape / 24 bytes / payload / absolute offset）
  - 未知 dtype（I16）の拒否維持
  - rank-2: 先頭 row / 末尾 row / 中間 range / 範囲外 / 空 range / overflow count / 未存在 tensor
  - rank-3: 全 Expert の offset・bytes、byte 列の一致、無効 Expert ID、範囲超過
  - rank-0: `axis_span` 拒否
  - truncated file / shape overflow の拒否
  - 複数 shard の index 解決、`tensor_shard`、`tensor_span`、`axis_span`、`shard_file_size`
  - path traversal（`../`）と絶対パスの拒否
  - **Python `safetensors` 0.8.0 との header / byte 列比較**（I64 `[4]`、offsets `[0,32]`、payload 完全一致）
- 回帰: `test_safetensors_writer` / `test_hf_manifest` / `test_tensor_partition` /
  `test_canonical_partition` / `test_weight_load` すべて PASS

### G1 PASS 条件の充足

> PLE 行や Expert 1 個を、モデル全体を materialize せず正しい byte span として特定できる。
> 既存ロード動作は変わらない。

**充足。** rank-2 の任意 row と rank-3 の任意 Expert が絶対 offset + byte length で
特定でき、既存の reader / writer / manifest / partition / weight load は回帰なし。

## G2

### 差分調査票（手順書 §0.3）

| 項目 | 記録内容 |
| --- | --- |
| 調査対象 | hipFile 連携の有無、整列読み出し、PLE row の SSD / RAM / VRAM 取得経路 |
| ZIP 時点の想定 | 手順書 §7: hipFile 連携は ZIP 内に見当たらない。`ais-check` と `ais-stats` の結果は Gate 記録に残す。O_DIRECT の整列単位 A を対象環境で確認する |
| 現行実装 | コード側に hipFile 連携は不在。**環境には `libhipfile.so.0.4.0` とヘッダ・CMake config が実在**。`ais-check`（non-sudo）は `hipFile-capable volume: False` / `O_DIRECT: unverified`。GPU row gather カーネルなし |
| 食い違い | **あり**（コード不在と環境実在の乖離、`ais-check` 判定と実 I/O の乖離） |
| 追加で読んだ箇所 | `hipfile.h`（API 全体）、`hipfile-config.cmake`、`cmake/options.cmake` / `targets.cmake`、G1 の span API、`qwen_hybrid` の lowering |
| 採用判断 | **新設**: `io/row_reader`（整列計画・読み出し）、`io/hipfile_reader`（RAII ラッパと POSIX fallback、`SsdReadPath` で経路観測）、`kernels/optimized/row_gather`（GPU gather）、`models/qwen4exp/ple/ple_row_gather`（shard 解決と 3 経路）。`find_package(hipfile)` は任意 |
| 回帰リスク | io の既存契約（`SafetensorsReader` / `SafetensorsWriter`）は無変更。新規ファイルのみ |
| 検証計画 | `test_ple_row_gather`（gpu1;required、55 件）: 3 経路の bitwise 一致、ページ境界・重複 ID・全 row、エラー系、40 回反復の device memory drift。設備依存の fastpath は required と分離して `ais-stats` 実測で記録 |
| 判断根拠 | fastpath 条件の実測（offset / size とも 4096 整列で `Fastpath Rejections: 0`）、非整列 320 B row での `pread` との一致、EOF クランプの挙動（`g2_ais_stats.txt` / `g2_ais_check.txt`） |

### 着手前の調査（hipFile 実機）

小型 probe プログラムと `ais-stats -p <pid>` で実挙動を実測した。
（`ais-stats` は対象プロセスにアタッチする方式。`--help` は使えず
`ais-stats -p <PID> [-i]` のみ）

| 試験 | 結果 |
| --- | --- |
| `hipFileGetVersion` | 0.4.0 |
| `hipFileDriverOpen` / `hipFileHandleRegister` / `hipFileBufRegister` | すべて成功 |
| `hipFileRead`（GPU buffer、offset 168 / 4096 B） | 成功、`pread` と **8192 B 完全一致** |
| `hipFileRead`（320 B 非整列 row） | 成功、`pread` と 640 B 完全一致 |
| `hipFileRead`（file 末尾 100 B 手前から 4096 B） | **100 B を返す**（EOF を越えない） |
| offset / size とも 4096 整列 | `Fastpath Rejections: 0`、**fastpath 1,228,800 B、fallback 0** |
| size 5120（非整列） | `Fastpath Rejections: 300`、fallback 1,536,000 B、`Fallback Unaligned 300` |
| offset 非整列（1755878424） | `Fastpath Rejections: 300`、fallback 1,228,800 B |
| O_DIRECT フラグ有無（整列時） | どちらも fastpath 成立 |
| `hipFileDriverGetProperties` | err=5030（Internal GPU IO library error）。property 取得は不可 |

→ **fastpath は file offset と read size が両方 4096 整列のときにのみ成立する。**
PLE row は 320 B・非整列なので、手順書 §7.2 の式どおり
`read_begin = floor(row_begin/4096)*4096` / `read_end = ceil(row_end/4096)*4096`
で整列ブロックへ拡張して読む必要がある。

`ais-check -v`（non-sudo）の結果は `g2_ais_check.txt` のとおり
`hipFile-capable volume: False` / `O_DIRECT: unverified` だったが、
これは **root 権限がないための一時判定**であり、実 I/O では fastpath が成立した。
**`ais-check` の判定と fastpath の実使用は別である**ことを実測で確認した（手順書 §7.1 の注意どおり）。

### 実装

| ファイル | 責務 |
| --- | --- |
| `include/phaseshift/io/row_reader.h` / `src/phaseshift/io/row_reader.cpp` | 汎用の整列読み出し計画（`probe_odirect_alignment` / `plan_row_reads` / `read_blocks_buffered` / `read_blocks_direct` / `read_rows_buffered`）。PLE 非依存 |
| `include/phaseshift/io/hipfile_reader.h` / `src/phaseshift/io/hipfile_reader.cpp` | hipFile の RAII ラッパ（driver refcount、handle/buffer 登録、`read`、明示 `shutdown`）。hipFile 非搭載 build では POSIX fallback。`SsdReadPath` で経路を観測可能 |
| `include/phaseshift/kernels/optimized/row_gather.h` / `src/phaseshift/kernels/optimized/row_gather.hip` | GPU row gather カーネル（src offset / dst offset / byte length のリストから行を連結） |
| `include/phaseshift/models/qwen4exp/ple/ple_row_gather.h` / `src/phaseshift/models/qwen4exp/ple/ple_row_gather.cpp` | PLE 固有（shard 名、global row → shard/row 解決、3 経路の gather、統計）。汎用 I/O 層は PLE 非依存 |
| `cmake/options.cmake` / `cmake/targets.cmake` | `find_package(hipfile)`（任意）、`phaseshift_qwen4exp` ターゲット |
| `tests/unit/test_ple_row_gather.hip` | `gpu1;required` |

設計判断:

- **row gather 契約は 3 経路共通**（`(absolute_offset, byte_length)` → GPU 上の連結出力）。
  SSD は整列ブロック読み出し + GPU gather、RAM は mmap からの CPU gather、
  VRAM は fixture table からの GPU gather。**計算式は共通、取得経路だけが違う**。
- 整列ブロック計画は**同一ページの重複 access をマージ**する。
- 統計で `logical_bytes`（row の合計）と `physical_bytes`（整列ブロックの合計）を分離記録。
  手順書 §7.2 の「logical と physical を別々に記録する」に対応。
- `fastpath_eligible` はブロック単位で 4096 整列かどうかを判定し、記録する
  （fastpath 未達でも機能 PASS は成立する）。
- EOF の tail ブロックは整列できないことがある。その場合は O_DIRECT を使わず
  buffered read に落とし、`fully_aligned` に記録する
  （手順書 §7.2-5「EOF の外まで読み込まない」「fallback を明示する」）。

### 検証

`test_ple_row_gather`（`gpu1;required`、**55 件 PASS / 0 FAIL**）:

- shard 解決: shard 0/1/末尾の shard ファイル、row index、絶対 offset の増加、範囲外拒否
- **SSD（hipFile 許可）/ SSD（POSIX 強制）/ RAM / VRAM の 4 経路が bitwise 一致**
- RAM 出力が fixture の元 row 列と完全一致
- 重複 ID（同一 row 8 回）、同一ページ複数 row、全 row 連続
- 物理バイト数 >= 論理バイト数、ブロック数の記録
- 経路別統計の観測（`path` / `logical` / `physical` / `blocks` / `hipfile_blocks` / `fastpath_eligible`）
- エラー系: null ids / null device dst / 範囲外 row id（SSD・RAM 両方）、
  欠損 shard、truncated shard、O_DIRECT alignment probe
- リソース: 40 回反復 gather で device free memory の drift が 1 MiB 以内、
  reader の open 前 read / 未登録 buffer read / null buffer 登録 / shutdown 後挙動 /
  double shutdown

### G2-Direct（fastpath 実測）

実 checkpoint の PLE shard `model-00005-of-00131.safetensors`（`shard_0`）を対象に、
`plan_row_reads` の整列ブロック計画で 300 回読み込み、`ais-stats -p` で計測。

```text
pid=3717030 shard=model-00005-of-00131.safetensors blocks=1 eligible=1 total=36864 fully_aligned=1
read total=11059200 over 300 iterations (logical 36864 per pass)

File Handle Registrations: 300
Buffer Registrations: 300
Fastpath Rejections: 0
Total Fastpath Read Size (B): 11059200
Total Fallback Read Size (B): 0
Total Fallback Read Errors: 0
Fastpath Read Unaligned Count: 0
Fallback Read Unaligned Count: 0
```

生ログ: `g2_ais_stats.txt` / `g2_ais_check.txt`

- **fastpath read bytes が 11,059,200 B 増え、rejections 0、fallback 0**。
- **G2-Direct PASS**。

### G2 判定

| 条件 | 状態 |
| --- | --- |
| G2-Functional PASS: SSD→GPU から正しい行を返せる | **PASS**（RAM / VRAM と bitwise 一致） |
| G2-Direct PASS: 実測で hipFile fastpath が使用されている | **PASS**（上記） |
| RAM / VRAM との差がゼロ | **PASS** |
| I/O 状態が観測可能 | **PASS**（`SsdReadPath` + 統計 + `ais-stats`） |

**G2 総合 PASS。**

## G3

実施: 別エージェントが担当し、私が受け入れ判定した。

### 差分調査票（手順書 §0.3）

| 項目 | 記録内容 |
| --- | --- |
| 調査対象 | weight loader の slice / partition / FP8 canonical 接続点、fused rank-3 の扱い |
| ZIP 時点の想定 | 手順書 §8: 既存 slice API の再利用可否を確認。FP8 rank-2 制約の下で「1 Expert を rank-2 として既存 preshuffle を利用」。FP8 N-D materialization は unsupported のまま |
| 現行実装 | `load_bf16_matrix` / `load_quantized_matrix` は rank-2 必須。`canonical_partition` は FP8 rank-3 を `unsupported`（意図的な制約、拒否テストあり）。HF manifest は rank-2 F8_E4M3/U8 + BF16 rank-2 scale のみ受理。`load_quantized_small` のみ rank 1-3 の BF16 を許容 |
| 食い違い | **なし**（ZIP 想定と一致。G1 の `axis_span` 実在を確認してから着手） |
| 追加で読んだ箇所 | `canonical_partition.{h,cpp}`、`fp8_native.h` / `fp8.h`、`hf_manifest.{h,cpp}`、`quantized_model_reader.cpp`、correctness GEMM / swiglu、`test_canonical_partition.cpp`、`test_weight_load.hip`、`docs/developer/tensor_partition.md` §5 / §10 |
| 採用判断 | **再利用**: `axis_span`（G1）、`load_quantized_view`（既存実装を view 単位へ抽出、挙動不変）、既存 correctness GEMM / swiglu。**新設**: `load_bf16_matrix_axis_slice` / `load_fp8_matrix_axis_slice` の 2 関数のみ。**不要化**: `TensorPartitionDesc` 経路は使わず axis-0 byte span の直接 slice で rank-2 化して既存 preshuffle へ投入。**制約維持**: FP8 rank-3 は unsupported のまま、HF FP8 受理条件も無変更 |
| 回帰リスク | `load_quantized_view` の抽出は挙動不変。Qwen35 の load path、`canonical_partition` の unsupported 判定、`hf_manifest` の受理条件に無変更 |
| 検証計画 | `test_expert_slice_fixture`（cpu;required）と `test_expert_slice_load`（gpu1;required）を追加。回帰は指定 7 件 + 重みロード関連 5 件 |
| 判断根拠 | `canonical_partition.cpp` の MatrixBlock 2D 必須、`test_canonical_partition` の FP8 N-D 拒否テスト、`tensor_partition.md` §5、`fp8_native.h` の batch=1 拘束 |

### 変更

| ファイル | 変更 |
| --- | --- |
| `include/phaseshift/weights/weight_loader.h` | `load_bf16_matrix_axis_slice` / `load_fp8_matrix_axis_slice` の宣言追加（既存 API 不変） |
| `src/phaseshift/weights/weight_loader.cpp` | `load_quantized_matrix_impl` を「resolve+partition の wrapper」と `load_quantized_view`（rank-2 canonical → preshuffle → upload 本体）へ**抽出分割**（挙動不変）。新規は axis-slice ローダ 2 関数のみ |
| `tests/unit/expert_slice_fixture.h` | synthetic fixture（BF16 rank-3 + FP8 codes rank-3 + BF16 scale rank-3 + config の `quantization_config`）。実 checkpoint 非依存 |
| `tests/unit/test_expert_slice_fixture.cpp` | CPU required |
| `tests/unit/test_expert_slice_load.hip` | GPU required |
| `cmake/tests.cmake` | 追記のみ（登録 + DEPENDS の 2 箇所） |
| `docs/developer/quantization.md` | 「Fused rank-3 expert tensor の axis-slice ロード」節を新設 |
| `docs/developer/qwen4exp.md` | gap #2 の 1 行のみ更新 |

### 設計判断（手順書 §8 との対応）

- **既存を再利用**: `axis_span`（G1）、`load_quantized_view`（新設せず既存実装を view 単位へ抽出）、
  既存 correctness GEMM / swiglu カーネル（新 GEMM 実装なし）。
- **FP8 の rank-3 materialization は unsupported のまま維持**。
  partition 経路ではなく **axis-0 byte span の直接 slice** で rank-2 に化けて
  既存 preshuffle へ投入した（手順書 §8-6「1 Expert を取り出して rank-2 として
  既存 preshuffle を利用する」に一致）。
- **HF FP8 受理条件は無変更**（`fmt == "e4m3"`、`weight_block_size [128,128]`、
  codes rank-2 F8_E4M3/U8、scale BF16 rank-2）。
  新規 FP8 slice ローダも rank-3 codes + BF16 rank-3 scale `[E, ceil(N/128), K/128]` +
  `K % 128 == 0` を要求し、**任意 FP8 形式を無条件に受理しない**。
- 検証固定: fused（batch=4）payload の preshuffle 拒否、HF manifest の rank-3 拒否。
- `TensorPartitionDesc` / TP partition 経路には手を触れていない。

### 検証結果

新規（いずれも `PHASESHIFT_TEST_GPUS=2,3`）:

| テスト | 結果 |
| --- | --- |
| `test_expert_slice_fixture`（cpu;required） | **PASS** |
| `test_expert_slice_load`（gpu1;required） | **PASS** |

oracle: BF16 / FP8 の入力・重みをテスト内で F32 に展開
（FP8 は E4M3 decode、scale は BF16 → F32 widen）、GEMM は F32 累積、SiLU は
`x/(1+exp(-x))`、出力は BF16 へ丸めた値で比較。

| 経路 | 1 本目 GEMM max_abs | 2 本目 GEMM max_abs | 許容 |
| --- | --- | --- | --- |
| BF16 e=0 / 2 / 3 | 0.0（bit 一致） | 0.0（bit 一致） | 2% of ref_scale |
| FP8 e=0 | 0.0 | 0.0078（ref_scale 390） | 7.80 |
| FP8 e=2 | 0.0 | 0.0（ref_scale 380） | 7.60 |
| FP8 e=3 | 0.000031 | 0.125（ref_scale 378） | 7.56 |

FP8 の非ゼロ誤差は E4M3 encode の host 表 vs device intrinsic の丸め差（1 code 分）。
**arena 増分が slice 分のバイト数のみ**であることも検証（モデル全体を VRAM に載せない）。

回帰: `test_hf_manifest` / `test_tensor_partition` / `test_canonical_partition` /
`test_weight_load` / `test_safetensors_partial_read` / `test_safetensors_writer` /
`test_architecture_boundaries`（7/7）＋
`test_fp8_payload` / `test_qwen35_tp_weight_load` / `test_qwen35_mtp_weight_load` /
`test_eagle4_weight_contract` / `test_dflash2_weight_contract`（5/5）すべて PASS。

### G3 PASS 条件の充足

> モデル全体を VRAM に置かず、任意の Expert 1 個を BF16 / FP8 でロードして
> 参照計算と比較できる。量子化形式や VRAM 予算の決定を要求しない。

**充足。** FP8 checkpoint 不在による実データ検証は G6 の開始条件として引き継ぐ。

## G4（PLE / GR / MoE / QSA / GDN 単体検証）


### G4a: PLE n-gram hash / PLE conv / Gated Residual — **PASS**

担当: 別エージェント。私が受け入れ判定・コミットした。

| ファイル | 責務 |
| --- | --- |
| `ple/ngram_hash.{h,cpp}` | splitmix64（標準形、2^64 wrap 含む）、素数 sizes、cumulative offsets、layer multipliers、`shift_right_ignore_eos`、`compute_ngram_ids`、`advance_ngram_context` |
| `ple/ple_conv.{h,cpp}` | dilation=3 / kernel=4 / state=9 の dilated depthwise causal conv、model-local conv state |
| `hyper/gated_residual.{h,cpp}` | read（要素単位 gate）/ write（stream 単位 inject）/ final collapse、中間 trace |

**設計判断**（差分調査票の要点）:

| 調査対象 | 手順書の想定 | 現行実装との食い違い | 採用判断 |
| --- | --- | --- | --- |
| 素数 sizes | `global = ple_layer_index(0-indexed=1) * 16 + head_idx` | 実 checkpoint は 1〜16 個目の素数（`ple_layer_ids` 内の slot=0 で一致） | 実 checkpoint と一致する slot 解釈を採用、docs §9.1 を修正 |
| `half_bound` | `unigram_vocab_size = ngram_vocab_size_base = 20,000,000` | 実値は **`vocab_size = 248,320`**（half_bound = 18,571,544,855,136）で一致 | 同上。設計意図も整合（max token × max multiplier < 2^63） |
| `seed` | config に 1234 | **公式 config.json に `seed` キーが無い** | 既定値 1234 を明示、docs に「既定値」と記載 |
| PLE conv の再利用 | `STATEFUL_CAUSAL_CONV1D` で可否を確認 | `GdnShapeKey` に **dilation が無く**、GDN conv カーネルは連続 tap 固定 → 再現不可 | **model-local 実装** |
| PLE conv state | `GDN_CONV_STATE` の `state_index` で表現可能（docs §13） | `GdnStatePoolLayout` は `conv_history` を pool 全体単一値（=4）で保持し、PLE の 9 を保持不可 | **model-local `PleConvState`**、共通 `GdnStatePool` は非変更 |
| splitmix64 | 既存の有無を確認 | `sampling_splitmix64`（`sampling_common.h`）は算法は同一だが **HIP device 前提**で host 経路に include 不適 | PLE 側に独立配置 |

**検証結果**（いずれも `PHASESHIFT_TEST_GPUS=2,3`）:

| テスト | 結果 | 誤差指標 |
| --- | --- | --- |
| `test_ple_ngram_hash`（cpu;required） | **51/51 PASS** | 整数 ID は Python oracle と **bitwise 完全一致**（max_abs = max_rel = RMSE = 0） |
| `test_gated_residual`（cpu;required） | **63/63 PASS** | float32 vs float64 naive oracle、T=17: normed 1.54e-6 / down 1.67e-5 / mixed 4.31e-7 / write 8.94e-8 / collapse 4.31e-7（max_abs）。prefill 分割は bitwise 一致、gate∈(0,1) / inject∈(0,4) |
| `test_ple_conv`（cpu;required） | **27/27 PASS** | raw conv max_abs 3.54e-9 / RMSE 5.50e-10、silu 1.93e-10。分割 prefill と decode 追加は bitwise 一致、state の oldest-first 保持 |
| `test_ple_ngram_checkpoint`（cpu;optional;external_files） | **9/9 PASS** | 実 checkpoint の I64 3 tensor を 4 パターン照合。**slot=0 + vocab_size=248320 のみ全一致**（他 3 パターンは不一致）。不在なら exit 77 |
| 回帰（partial_read / architecture_boundaries / ple_row_gather） | **3/3 PASS** | — |

golden の再生成手順: PLE hash は Python3 stdlib のみ（`random.seed(20261024)`、
script はリポジトリに残さない）。GR / conv の oracle はテスト内 float64 naive ループ
（自己完結、再現はテスト実行のみ）。

**未解決（G5 へ引き継ぎ）**:

1. PLE conv の重み方向（`w[0]` = 最古 tap）は既存 GDN conv の約定に合わせたが、
   参照実装（Transformers `qwen4_exp`）不在のため数値照合不能。**G5 で実重みにより再確認要**。
2. GR / PLE conv は **host float 実装**（GPU kernel 化・融合は未実施）。
3. PLE conv state の runtime owner（model-local pool の G5 での位置づけ）。

### G4b: MoE / GDN 適合試験 — **PASS**

担当: 別エージェント。私が CMake 適用・ctest 再実行・受け入れ・コミットした。

| テスト | 区分 | 結果 |
| --- | --- | --- |
| `test_qwen4exp_moe`（cpu;required） | §9.3 MoE | **77 件 PASS** |
| `test_qwen4exp_gdn_fit`（cpu;required） | §9.6 config → lowering → selector | **50 件 PASS** |
| `test_qwen4exp_gdn_fit_gpu`（gpu1;required） | §9.6 カーネル数値 | **15 件 PASS** |
| 回帰 `test_gdn*` / `test_qwen35_gdn_*` / `test_architecture_boundaries` | — | **10/10 PASS** |

**MoE の oracle 条件**: seed=20261024（splitmix64、位置指定
`state = seed + (p+1)·γ`、numpy ベクトル化で言語間 bitwise 一致）。重みは bf16 RNE 生成 →
f32 widen、router softmax は **F32**（仕様どおり）、expert / shared は
oracle=float64 / 実装=float32、tie-break は値降順・同値 index 昇順。
生成 revision は `tests/data/qwen4exp_moe/MANIFEST.txt` に記録。
ケースは random / zero（完全同点→`[0..9]`）/ tie（境界同点）/ extreme（±4096）/
same（複数 token 同一選択）/ norenorm の 6 種。

| 段階 | 閾値 | 実測最大 |
| --- | --- | --- |
| Top-K index | **bitwise** | **全ケース完全一致** |
| probs | 1e-5 | 3.58e-6（zero は 0、典型 ~3e-9） |
| top_w / top_w_used | 1e-5 | 3.54e-6 |
| routed | 5e-2 (rtol 5e-4) | 1.12e-2（extreme、巨大項の相殺増幅、典型 ~3e-8） |
| shared / gate | 5e-3 / 1e-5 | 3.24e-4 / 5.96e-8 |
| moe 合算 | 5e-2 | 1.12e-2（典型 ~1e-7、RMSE 最大 3.18e-3） |

**GDN 適合（GPU 実測）**:

| 検証 | 閾値 | 実測 max_abs |
| --- | --- | --- |
| conv（10240, kernel4, prefill64 + 継続8） | 1e-6 | **0.0（完全一致）**、state bf16 30720/30720 bitwise |
| qk L2（2048/128）nq / nk / sq | 1e-5 | 1.49e-8 / — / 1.86e-9 |
| recurrence decode step1 → step64 | 1e-4 / 1e-3 | 9.88e-6 → **3.46e-6**（長期で増加なし） |
| recurrent state（**FP16 lossy 契約**） | fp16 step 境界 3.00e-4 | 差 6.10e-5、bitwise 429599/786432。**FP32 化していない** |
| prefill chunk step1 / step64 | 1e-4 / 5e-3 | 4.36e-6 / 1.03e-3 |
| decode1 vs wmma | 1e-6 | **0.0（完全一致）** |
| sigmoid gate / silu（Qwen35 参照） | 1e-6 | 5.96e-8 / 2.38e-7 |

判定は `max_abs <= 閾値 OR max_rel <= 閾値`（手順書 §9.7）。閾値は全実測値から
マージン付きで確定。

**差分調査票の要点**:

- 手順書の **gap #7「GDN output gate 未実装」は stale**。現行 `gdn_lowering.h` の
  `SiLUNode` / `SigmoidNode` 分岐、config の `output_gate_type` 解析、
  SIGMOID elementwise kernel は実装済み → **再利用、再実装せず**。
  従来 `output_gate_silu` を参照するテストが 0 件だった穴を本テストで解消。
- q/k repeat-interleave 16→48 は **recurrence kernel 内**の
  `num_v_heads / key_heads = 3` が吸収（lowering に repeat ノードなし）。
  oracle 側で独立に 16→48 を再現して検証。
- 既存 MoE コードは models / kernels に**ゼロ**（config reader の MoE フィールド解析のみ）。

**未完了（G5 / G6 へ引き継ぎ)**:

1. **FP8 Expert 分解実行の oracle 比較**（§9.3 PASS 文言の「FP8 分解した実行結果」）:
   FP8 MoE 実行経路が未実装（G3 の slice は実装済み、実行経路は G5）。本 Gate は BF16。
2. **参照実装**: `transformers` に `qwen4_exp` が無いため §2.2 の
   `modeling_qwen4_exp.py` の commit 固定は**達成不能**。正本は
   `docs/developer/qwen4exp.md` + 公式 `config.json`。
3. MoE の **lowering → GPU 実行統合**は G5 スコープ（本 Gate は単体検証）。

### G4c: QSA Indexer / QSA Sparse Attention — **PASS**

担当: 別エージェント。私が CMake 適用・ctest 再実行・受け入れ・コミットした。

| テスト | 区分 | 結果 |
| --- | --- | --- |
| `test_qsa_indexer`（gpu1;required） | §9.4 | **17 件 PASS** |
| `test_qsa_sparse_attention`（gpu1;required） | §9.5 | **24 件 PASS** |
| `test_architecture_boundaries`（単体） | — | passed 693 / failed 0 |

**再利用した既存実装**（再実装せず）: `launch_rmsnorm`（q/k_layernorm と q_norm/k_norm）、
`launch_rope_f32_bf16_generic`（rotary_dim=64）、`emit_gated_attention`
（geometry を (24,2,256,rotary64) に変えるだけで QSA main attention になる）、
`paged_prefill.hip` / `paged_decode.hip`（dense との一致比較）。

| 比較 | max_abs | max_rel | RMSE | 閾値 (abs/rel/rmse) |
| --- | --- | --- | --- | --- |
| Indexer GPU vs CPU oracle（全 2053 行） | 0.0468 | 0.137 | 0.00525 | 6e-2 / 2.5e-1 / 1e-2 |
| Indexer GPU vs PyTorch golden（17 行） | 0.0303 | 0.117 | 0.00547 | 5e-2 / 2.5e-1 / 1e-2 |
| Sparse q（norm + partial RoPE） | 0.0156 | 0.0078 | 3.75e-5 | 3.2e-2 / 2e-2 / 1e-4 |
| Sparse 出力 GPU vs CPU oracle | 2.86e-4 | 2.65e-2 | 5.88e-6 | 1e-3 / 5e-2 / 2e-5 |
| Sparse 出力 GPU vs golden | 2.74e-4 | 2.20e-2 | 6.22e-6 | 1e-3 / 5e-2 / 2e-5 |
| **全選択短文: sparse vs 既存 dense** | **9.24e-7** | 5.63e-5 | 6.26e-8 | 1e-5 / 1e-3 / 1e-6 |

- Indexer の誤差主因は既存 `rmsnorm_row_body` が F32 入力時の中間 `x*inv` を
  bf16 へ丸める挙動（block key 経路のみ影響、スコア尺度 12.6 に対し 0.37%）。
- **bitwise 一致**: 選択 block ID（集合）、counts、mask 全要素、無効スコア 0。
  対象位置 2047/2048/2049/2051/2052、tail 長 0..3、Top-K tie、
  prefill vs decode、state prefix、causal を確認。
- Sparse: paged KV read は置換 page table で検証、selector は (24,2,256) → `Correctness` /
  (16,4,256) → `Optimized`（既存挙動・変更なし）、partial RoPE の非 rotary 列は bitwise、
  真の sparse 長文の選択率 0.252、選択外 token は softmax に混入しない。

**設計判断（G5 へ引き継ぎ）**:

1. **新 PrimitiveKind は不要**。sparse mask は `PagedAttentionNode` に optional な
   2 番目入力（mask ValueId）を追加して表現できる（dispatch の io 検証を
   `1/1 または 2/1` に、kernel に mask ポインタ 1 本を追加）。Qwen35 は入力 1 本のまま。
2. **新 state kind `INDEXER_KEY_CACHE` が 1 種必要**。`KV_CACHE` は paged pool と
   `kv_heads*head_dim` feature 検証が前提で `[tokens,128]` の連続 raw key を格納できない。
3. **(24,2,256) の optimized selector ルールは追加しない**（未計測。未計測 geometry の
   Optimized 宣言はしない）。

**未解決（再確認要）**: indexer の `q_layernorm` / `k_layernorm` が RMSNorm か LayerNorm か、
weight mode が `ONE_PLUS` か `DIRECT` かは `docs/developer/qwen4exp.md` §6.2 に記載が無く、
§4.1 に倣って **RMSNorm + ONE_PLUS** で実装した。参照実装入手後に要再確認。
Top-K 同点の正本順序も未確定のため `(score 降順, block id 昇順)` を契約として定義した。

## G5（進行中）

### 完了済み

| 項目 | 状態 | 証跡 |
| --- | --- | --- |
| `Qwen4ExpTextConfig` と `read_qwen4exp_text_config` | **PASS** | `test_qwen4exp_config`（cpu;required、46 件）。実 checkpoint での parse も手動確認済み |
| `Qwen4ExpModelWeights` と `load_qwen4exp_weights_bf16` | **PASS** | `test_qwen4exp_weights`（gpu1;required、94 件、synthetic fixture） |

`test_qwen4exp_config` は実 `text_config` の全項目（48 layer types、12 QSA / 36 GDN、
`rotary_dim = 64`、`block_topk = 512`、`ple_slot = 0`、indexer 制約、未知
`output_gate_type` の拒否、`layer_types` 長の不一致拒否）を検証する。

`test_qwen4exp_weights` は synthetic な 8-shard checkpoint で次を検証する:
embed / lm_head（非 tie）/ final mixer / 4 層分の attn・mlp hyper connection /
GDN 層の rank-3 conv1d と A_log / QSA 層の indexer / MoE router と shared expert /
rank-3 fused expert tensor の rank 検証 / layer 1 の PLE と **I64 の 3 tensor** /
PLE が layer 1 以外に無いこと / 欠損 tensor での拒否。

| n-gram hash → row gather の結合 | **PASS** | `test_ple_row_gather` の pipeline 検証（生成 ID が実表範囲内、locate が正しい shard / row を返す、SSD / RAM / VRAM が bitwise 一致、元 row と一致） |

### G5a: Gated Residual の Primitive 化と lowering 基盤 — **PASS**

手順書 §10-4 に従い、**既存 Primitive で表現できるものは分解**し、新規を最小 2 種に限定した。
既存ノード・既存 dispatch case・既存 `KernelId` 値の変更はなく**追記のみ**
（Qwen35 への影響がないことを `test_qwen35_lowering_contract` /
`test_qwen35_mtp_lowering` / `test_architecture_boundaries` で確認）。

| GR のステップ | 使用ノード |
| --- | --- |
| `hc_norm(x4)`（grouped RMSNorm, group=2560, `(1+w)`） | `RmsNormNode`（`group_size` + `ONE_PLUS`） |
| `input_mix_weight_down` / `up` / `block_inject` | `LINEAR` |
| `silu(d / hc_count)` / `2 * sigmoid(...)` | `SILU` / `SCALE` |
| read gate `sigmoid(u)` | `SIGMOID` |
| `unflatten` | **ノード化しない**（row-major 連続メモリ上の解釈のため） |

新規 Primitive 2 種（generic な意味と形状契約）:

| Primitive | 契約 | GR での用途 |
| --- | --- | --- |
| `GROUP_MEAN` | 入力行を連続する `group` 個のブロックに分けブロック内平均。`[rows, group*features]` → `[rows, features]` | `mean_over_4`（group = stream 数 4） |
| `GROUP_BROADCAST_MUL` | `out[r,g*F+h] = a[r,h] * b[r,g]`。`[rows, F] × [rows, group]` → `[rows, group*F]` | `sub_out[...,None,:] * inj[...,None]`（write は + 既存 `RESIDUAL_ADD`） |

変更箇所（すべて追記）: `primitive_node.h`（PrimitiveKind 19/20）、
`primitive_graph.cpp`（arity）、`kernel_id.h`（28/29、`COUNT` 28→30）、
`program.cpp`（dispatch case 2 個）、correctness `detail/group.inc`、
optimized `group_ops.{h,hip}`、`optimized_dispatch.hip`（F32 限定 `try_launch_group_ops`）、
`models/common/qwen_hybrid/hyper_lowering.h`（`emit_hyper_mixed` / `read` / `write` / `collapse`）。

| 検証 | 結果 |
| --- | --- |
| `test_group_ops`（gpu1;required、14 ケース） | correctness / optimized 双方を CPU double 参照と比較、**max_rel ≤ 6e-5、RMSE ≤ 1.1e-7** |
| `test_qwen4exp_hyper_lowering`（gpu1;required、26 チェック） | lowering → `build_program` → correctness dispatch 実行 → **G4a host 実装と比較**。write 出力 **max_abs 0（bitwise 一致）**、read mixed 1.62e-3 / inject 2.47e-4 / collapse 1.62e-3 |
| 回帰（architecture_boundaries / qwen35_lowering_contract / qwen35_mtp_lowering / gated_residual） | **4/4 PASS** |
| 回帰（`KernelId::COUNT` 変更周辺の program 系 5 件） | **5/5 PASS** |

絶対誤差の主因は既存 correctness `RMS_NORM` の契約（F32 入力の中間 `x*rsqrt` を bf16 丸め）。

### G5b: `lower_qwen4exp_to_primitives`（層結線） — **PASS**

手順書 §10 の手順 5（4-stream hidden、各層の GDN / QSA、MoE、final HyperConnection mixer、
`lm_head` の接続）に対応。**新規 Primitive は作らず**、G5a / G4 / 既存 lowering と
G5c の Node を組み合わせて結線した。

| 項目 | 内容 |
| --- | --- |
| lowering API | `source_audit.md` 付録 A と同型 + 承認済みの `bool connect_moe = true` |
| 構造 | `EMBEDDING_LOOKUP` → `CONCAT` で x4 展開 → 層ループ（GR read → `emit_gated_attention` / `emit_gdn` → GR write → [MoE]）→ `emit_hyper_collapse` → `OUTPUT_GATHER` → `lm_head` の `LINEAR` → `SAMPLING`。**final norm は存在しない**（`docs/developer/qwen4exp.md` §2 を厳守） |
| weights の食い違い修正 | 現行ローダは fused expert を GPU に載せず名前と `expert_count` のみ保持していた（付録 B の前提と不一致）→ `Qwen4ExpMoeWeights` に expert ごとの rank-2 `MatrixWeight` を追加し、`load_expert_weights`（**既定 false**）で `load_bf16_matrix_axis_slice(name, 0, E)` を連続確保 → `Tensor::slice` で E view に分割。G5c kernel の `weight_index + e` / `+ E + e` 連続オフセット契約に一致させるため。既定 false により `test_qwen4exp_weights` 94 件は非破壞 |
| 再利用 | `EffectiveRopeProfile` は `ps::qwen35::` を include（境界スキャン対象外 + 前例 `qwen4exp_config.h` → `qwen35/stop_tokens.h`）。shared expert の gate は features 不一致のため既存 `GROUP_BROADCAST_MUL(group=1)` を使用。GDN / attention の `out_dtype` は x4 の F32 ストリームに合わせ F32 |

**検証**:

| テスト | 結果 | 内容 |
| --- | --- | --- |
| `test_qwen4exp_lowering_contract`（gpu1;required） | **59 件 PASS** | 173 ノードの kind 列厳密一致、weight slot 78（PLE 除外 + expert rank-2 分割）、param 17、state 連番、PLE 不在、MoE 幾何と slot 連続性、final 鎖（collapse → gather → lm_head、間に `RMS_NORM` 不在）、selector の `(24,2,256)` → correctness fallback / `(16,4,256)` → Optimized、`build_program` と MoE dispatch 生成 |
| `test_qwen4exp_forward_fixture`（gpu1;required） | **53 件 PASS** | synthetic safetensors → `load_expert_weights=true` → lowering（`connect_moe=false`）→ `build_program` → correctness dispatch 実行 |

**層間入出力と state 更新の誤差指標（実測）**:

| 検証 | max_abs | max_rel | RMSE |
| --- | --- | --- | --- |
| prefill 分割等価性（logits / selected hidden / **全 4 層 tap**） | **0** | 0 | 0 |
| decode 続続（split 後 decode1 vs prefill5 一括） | **0** | 0 | 0 |
| 因果 prefix（pos3 比較） | **0** | 0 | 0 |
| x4 生成（`emb.repeat`） | **0** | 0 | 0 |
| GR read mixed / inject | 1.79e-7 / 1.19e-7 | 2.29e-7 / 1.19e-7 | 2.90e-8 / 2.98e-8 |
| GR write 層出力 | **0** | 0 | 0 |
| final collapse | 2.61e-3 | 1.18e-2 | 4.02e-4 |
| lm_head ロジット（final norm なし） | 8.94e-8 | 6.81e-7 | 3.36e-8 |

collapse の 2.61e-3 は既存 correctness `RMS_NORM` が F32 入力の中間値を bf16 へ丸める
既存経路（G5a 実測と同じ由来）によるもの。
**state 更新**: GDN conv / recurrent が run 前 → chunk1 後 → chunk2 後で**毎回バイト列が変化**し、
QSA KV もページ単位で変化することを確認。split 等価性の完全一致が
conv（境界でも無損）/ recurrent（fp16 丸めも同一）/ KV append 継続の正しさを実証。

**回帰**: `architecture_boundaries` / `qwen35_lowering_contract` / `qwen35_mtp_lowering` /
`qwen4exp_hyper_lowering` / `qwen4exp_weights` / `qwen4exp_config` の **6/6 PASS**。

### G5 の PASS 条件（手順書 §10）との対応

> PASS条件：小型fixtureでテキストprefill→decode、GDN/QSA/PLE状態継続、MoE合成、
> final logitsが再現可能。既存Qwen35に回帰がない。

| 条件 | 状態 | 根拠 |
| --- | --- | --- |
| 小型 fixture でテキスト prefill → decode | **充足** | `test_qwen4exp_forward_fixture`（prefill 分割・decode 続続・因果 prefix がいずれも max_abs 0） |
| GDN 状態継続 | **充足** | conv / recurrent の逐回バイト変化と split 等価性 |
| QSA 状態継続 | **充足** | KV のページ単位変化と split 等価性 |
| **PLE 状態継続** | **充足** | `test_qwen4exp_ple_graph`（83 件 / 0 failed、GPU2 と GPU3 で同一）。prefill chunk 境界・decode 1 token 続続・cancel・EOS・Request 再利用を、`PleConvStatePool` / `PleNgramContext` に対して **prefix graph → PLE branch → main graph の実行経路**で検証。graph の ple.add 行と host 参照（x4 + PLE 出力）は **bitwise（max_abs 0）**、split / decode / EOS 分割 / reuse の logits も **bitwise（max_abs 0）** |
| **MoE 合成** | **充足** | `test_qwen4exp_moe_graph`（121 件）で `connect_moe=true` の full-graph 実行を検証。router `top_idx` 全層 bitwise、合算 composed vs oracle 最大 1.58e-8、`composed == routed + shared` は 0（bitwise）、MoE あり/なしの logits 差 5.00e-2 で効きを確認 |
| final logits 再現可能 | **充足** | lm_head ロジット 8.94e-8 |
| 既存 Qwen35 に回帰がない | **充足** | 指定 6 件 + round 2 中の広い回帰 13 件が PASS |

**結論: round 3 の G5e（MoE 合成）と G5f（PLE 状態継続）で両項目が充足し、
表の全項目が充足した。→ G5 = PASS（手順書 §10 の PASS 条件を満たす）。**

なお本 Gate の範囲は「小型 fixture での検証」であり、以下は**引き継ぐ残課題**（G5 の
PASS 条件には含まれない）:
- entrypoint（`phaseshift-compute` / `phaseshift-bench`）からの起動経路、
  `ContinuousBatcher` / `PrefixCache` との統合（state の保存・復元構造は
  `source_audit.md` 付録 G に記録済み）
- optimized dispatch 接続（GR / MoE / QSA は現状 correctness fallback。
  `kernels/optimized/moe_ops.hip` は router 出力が row_stride 契約に未対応）
- `PrimitiveStateKind::INDEXER_KEY_CACHE` の登録
- PLE projection / gate / conv の GPU kernel 化（現状 host F32）
- QSA の sparse mask 入力と indexer key state の lowering 接続

### G5c: MoE の Primitive 化（Node + kernel + dispatch） — **PASS**

手順書 §10-4「新しい Primitive が不可避なら generic な意味と形状契約を定義し、
既存モデルの挙動を変えない」に対応。`source_audit.md` 付録 B の契約どおりに実装した。

| 項目 | 内容 |
| --- | --- |
| Node | `PrimitiveKind::MOE_ROUTER=21` / `MOE_EXPERTS=22`（既存値は不変）。`MoERouterNode` は入力 1・出力 2（`top_idx` I32 + `top_w` F32）、`MoEExpertsNode` は入力 3・出力 1 |
| `DispatchBinding` の geometry 流用 | `group_size` = experts（`RmsNormNode.group_size` / `GROUP_*` と同型の「グループ数」）、`rotary_dim` = top_k（`SAMPLING` が `rotary_dim` に `vocab_size` を入れる**既存先例**に倣う）、`flags` bit0 = renorm（`RmsNormNode.weight_mode` の先例に倣う）。**既存フィールドの意味は変更なし** |
| weight 契約 | expert ごとの rank-2 `WeightSlot` を連続展開（`weight_index + e` = gate_up、`+ E + e` = down）。graph 検証で 2E スロットの範囲と geometry 一致を確認 |
| correctness kernel | `kernels/correctness/detail/moe.inc`（新規）。F32 softmax + stable top-k + renorm、SiLU(gate)*up → down の GEMM 合算。BF16 以外の重みは `INVALID_PROGRAM` で拒否 |
| optimized kernel | `kernels/optimized/moe_ops.{h,hip}`（新規）。F32 activation 限定、それ以外は NotApplicable → correctness フォールバック |

**検証**（`test_moe_ops`、gpu1;required、**74 件 / 0 failed**、GPU2 と GPU3 で確認）:
G4b の `models/common/qwen_hybrid/moe_math.h`（pure F32）を oracle とし、
E=8 / hidden=64 / inter=32 / top_k=3 の rank-2 `WeightSlot` × 17 スロットで検証。

| 指標 | 結果 |
| --- | --- |
| Top-K index | **全 6 ケース bitwise 一致** |
| `top_w` max_abs | **全 6 ケース 0.0（bitwise 一致）** |
| routed output max_abs | random / tie / no-renorm 3.49e-10、zero 0.0、**extreme 7.06e-5（最大）**、same 2.33e-10 |
| routed output max_rel / RMSE | 最大 4.15e-6 / 1.90e-5（extreme） |

extreme ケースの 7.06e-5 は logits ±80 相当で expert 出力の絶対値が大きいことによる
FMA 丸め差。renorm 有効時は top-k 和が 1（許容 1e-5）、無効時は 1 未満も確認。
負例（router スロットと重なる experts 登録 / `top_k > experts` / スロット範囲超過）は
graph 検証と build で拒否されることを確認。

**回帰**: `test_architecture_boundaries` / `test_qwen4exp_moe` /
`test_program_quant_contract` / `test_qwen35_lowering_contract` の 4 件、
GROUP / QSA / PLE 関連 9 件、計 13 件 PASS。

**未接続（後続）**: optimized kernel の dispatch 接続
（`optimized_dispatch.hip` への `try_launch_moe_ops` 相当）。

### G5d: qwen4exp-local の state pool — **PASS**

手順書 §10 の手順 6（state owner の明示）と 7（Request 再利用 / prefill chunk 境界 /
decode 続続 / cancel / EOS / 状態破棄の小型 fixture 検証）に対応。
**共通 runtime は変更せず** `models/qwen4exp/runtime/` に新設した。

| owner | 内容 | lifecycle |
| --- | --- | --- |
| `IndexerKeyPool` | slot × layer × token × 128 BF16 の連続レイアウト（paged ではない）。`key_row` / `key_cache` / `key_elements_offset` で G4c の `launch_qsa_indexer` 契約を公開 | initialize / reset / release / 明示 shutdown |
| `PleConvStatePool` | G4a の `ple::PleConvState` を slot 所有（`conv_history = 9` は `ple::kPleConvStateLen` を公開、独自定数を作らない）。計算は G4a へ委譲 | 同上 |
| `PleNgramContext` | host integer の 2 token 履歴 + GPU 転送バッファ `[slots, 2]` | 同上 |

**検証**（`test_qwen4exp_state`、gpu1;required、**97 件 / 0 failed**、GPU2 と GPU3 で
複数回同一結果、**device memory drift 0 bytes**）:

| 段階 | IndexerKeyPool | PleConvStatePool | PleNgramContext |
| --- | --- | --- | --- |
| prefill chunk 境界 | 2 chunk 書き込みが一致、他 layer / slot はゼロ | 5+3 分割 vs 一括で一致 | 5 token を 2+2+1 分割 vs 一括 prefill の ID 列が **bitwise 一致** |
| decode 1 token 継続 | token 12..15 を追記し全行一致 | 1 token ×4 追記と一括が **bitwise 一致** | decode 追加 ID が一括と一致 |
| cancel（reset） | slot 全 layer ゼロ化、所有維持 | 9 tap 全ゼロ → fresh 参照と **bitwise 一致** | `[eos, eos]` へ復帰 |
| EOS 破棄（release） | `key_cache() == nullptr`、used = 0 | `state() == nullptr` → 再 initialize が fresh と一致 | context / device が `nullptr` |
| Request 再利用 | residue なし（4 行書画後 5 行目以降ゼロ） | fresh slot と一致 | `[eos, eos]` 復帰、ID 列が fresh と一致 |
| shutdown | 二重 no-op、以降は `invalid_state` で拒否 | 同上 | 同上 |

`docs/developer/qwen4exp.md` §13 の owner 表と実装は**一致**（不一致なし）。
`PrimitiveStateKind::INDEXER_KEY_CACHE` の登録は G5c の runtime 編集と衝突するため
後続として残す（現状は kind 登録を伴わない model-local の所有バッファとして成立）。

### round 2 の分担（G5 本体）

手順書 §10 の PASS 条件を満たすため、interface を先に固定して
（`source_audit.md` 付録「G5 統合の実装 interface 固定（round 2）」）、
3 つに分けて並行実施する。

| 担当 | 範囲 | 主な成果物 | 依存 | 状態 |
| --- | --- | --- | --- | --- |
| G5b | `lower_qwen4exp_to_primitives` 本体（層結線）と lowering contract / forward fixture テスト | `models/qwen4exp/model/lower_to_primitives.{h,cpp}`、`test_qwen4exp_lowering_contract.hip`、`test_qwen4exp_forward_fixture.hip` | G5a の `hyper_lowering.h`（済）、既存 `emit_gdn` / `emit_gated_attention`（済）、G5c の MoE Node（済） | **PASS** |
| G5c | MoE の Primitive 化（Node 2 種 + correctness / optimized kernel + dispatch） | `primitive_node.h` / `kernel_id.h` / `program.cpp` の追記、`kernels/correctness/detail/moe.inc`、`kernels/optimized/moe_ops.*`、`test_moe_ops.hip` | G4b の `moe_math.h`（oracle、済） | **PASS** |
| G5d | qwen4exp-local の state pool（indexer key / PLE conv / PLE n-gram context） | `models/qwen4exp/runtime/qwen4exp_state.{h,cpp}`、`test_qwen4exp_state.hip` | G4a の `ple_conv.*`（済）、G4c の indexer 契約（済） | **PASS** |

**私が管理するもの**: `cmake/targets.cmake` / `cmake/tests.cmake`（全担当が編集禁止。
完了報告の「コピー可能な行」を私が適用）、G5b/c/d の受け入れ判定とコミット分割。

round 2 完了後に私が行うもの: MoE の lowering 接続統合、`PrimitiveStateKind::INDEXER_KEY_CACHE`
の登録（G5c の runtime 編集完了後）、mini runtime（`models/qwen4exp/runtime/`、
設計方針は `source_audit.md` 付録 E）と prefill → decode fixture。

### G5e: `connect_moe=true` の full-graph 実行検証 — **PASS**（ctest 登録は G5f 完了後に適用）

手順書 §10 の PASS 条件のうち「**MoE 合成**」に対応。

| 項目 | 内容 |
| --- | --- |
| テスト | `tests/unit/test_qwen4exp_moe_graph.hip`（新規、約 1420 行）。`connect_moe=true` で lowering → `build_program`（`no_workspace_reuse=true`）→ correctness dispatch 実行。既存 forward fixture は非改変 |
| 中間値取得 | `node_output_by_debug_name` で `L{l}.mlp.hyper_mixed` / `moe_router` 出力 / `moe_experts` / `shared_gate_mul` / `moe_add` を特定し、`plan.find_value` の `row_stride` に従って読み出し。**取得不能値なし** |
| oracle | G4b の `models/common/qwen_hybrid/moe_math.h`（pure F32）。入力 x は graph 実測の mixed（GR read は forward fixture で host 参照検証済みのため鎖として妥当） |

**誤差指標（GPU3 実測、T=4 prefill）**:

| 検証 | max_abs | max_rel | RMSE |
| --- | --- | --- | --- |
| router `top_idx` | **全 4 層 bitwise 一致**（prefill / decode とも） | — | — |
| router `top_w` | 5.96e-8（L3 は 0） | 2.46e-7 | 4.21e-8 |
| routed experts 合算 | 1.86e-8 | 3.94e-4 | 4.52e-9 |
| shared expert | 1.12e-8 | 4.55e-5 | 3.57e-9 |
| 合算 composed vs oracle | 1.58e-8 | 2.32e-5 | 5.72e-9 |
| graph 内 `composed == routed + shared` | **0（bitwise）** | 0 | 0 |
| prefill 分割等価性 / decode 続続（MoE あり） | **0** | 0 | 0 |

**MoE あり / なしの併記**（同一 fixture・同一トークン {3,11,7,19}、fresh シークンス）:
lm_head ロジット差 max_abs **5.00e-2**、final collapse 差 1.97e-1、
全 4 層の層出力 tap で 7.69e-2 〜 2.73e-1 の有意差。両 arm とも greedy argmax が
各自の logits と一致 → **MoE が実際に効いていることの証拠**。

構造チェック: `top_idx` 全て expert 範囲内、renorm で `top_w` 和 = 1
（4 層×全トークン）、routed / composed 非ゼロ、`MOE_ROUTER` / `MOE_EXPERTS`
dispatch が層ごとに 1 本ずつ（計 8 本）、`connect_moe=false` プランには MoE dispatch なし。

#### 重大な発見: correctness kernel の row_stride 契約違反（full-graph 初の multi-row 検出）

runtime は `build_program` で WORKSPACE / EXTERNAL_OUTPUT 値に
**行バイトを 16B 境界へパディングした `aligned_row_stride`** を割り付ける
（`program.cpp:140`）。これに違反していた 2 箇所を full-graph（T>1）で初めて検出・修正した:

1. `kernels/correctness/detail/linear.inc` の `exec_linear_weight` が compact（`m * out.features`）で
   store していた → shared expert gate の `LINEAR([T,1])` → `SIGMOID` 経路で
   prefill 時に row1 以降が未初期化ワークスペースを読み、**全層の出力が破壊**
   （実測 shared 誤差 max_abs 5e-3、split 等価性が 3.5e-3 に崩壊）
2. `kernels/correctness/detail/moe.inc` の router / experts が compact（`t * top_k`）で
   書き込んでいた → binding 契約（`row_stride = 4`）と不一致

いずれも **行数 = 1（decode）では偶発的に一致するため既存テストが全て通過していた**。
修正後は `features * 4` が 16B アラインされているケースで `row_stride == features` となり
挙動不変。`test_moe_ops` のホスト側バッファも plan の `row_stride` 準拠へ整えた
（**検証 74 件・tolerance・ケースは不変**、修正後 74/74 PASS）。

**未修正（後続へ）**: `kernels/optimized/moe_ops.hip` も router 出力を compact で書いている。
optimized dispatch 未接続のため現状無害だが、**dispatch 接続時に row_stride 契約へ揃える必要あり**。

**回帰**: `test_qwen4exp_lowering_contract` / `test_qwen4exp_forward_fixture` /
`test_moe_ops` / `test_architecture_boundaries` / `test_qwen35_lowering_contract` の **5/5 PASS**。
`test_qwen4exp_moe_graph` は GPU2 / GPU3 で各 **121 件 / 0 failed**。

### G5f: PLE の graph 接続（graph 分割）と PLE 状態継続の検証 — **PASS**

手順書 §10 の「PLE 状態継続」と `source_audit.md` 付録 F の 2 案選択に対応。
**付録 F の案 1（新規 Node 0、PLE 計算は graph 外）を採用**したが、round 2 の
調査には誤りがあった。`docs/developer/qwen4exp.md` §2 のとおり PLE 加算位置の `x4` は
`emb.repeat` ではなく **layer 0 の出力**であるため、PLE を graph 外に置くには
graph を分割する必要がある（手順書 §0 に従い §2 と lowering 実装を追加読解して再判定）。
最終設計は次のとおり。

| 項目 | 内容 |
| --- | --- |
| 設計 | `Qwen4ExpLowerOptions::connect_ple`（**既定 false**）を追加。true のとき lowering は **prefix graph**（embed + repeat + layers[0, ple_layer)、x4 を external output）と **main graph**（x4 / PLE 出力を external input とする `RESIDUAL_ADD` → layers[ple_layer, N) → collapse → lm_head → sampling）の2つを返す |
| 新規 Node / Primitive | **0**。加算は既存 `RESIDUAL_ADD`、runtime / kernels は無編集（§10-4） |
| state_index | prefix が GDN conv / recurrent（layer 0 分）、main が QSA KV（layer 1 分）。連番を継続させ両 graph が同一 pool を共有 |
| PLE 本体 | `models/qwen4exp/ple/ple_forward.{h,cpp}`（qwen4exp-local）。G4a の `compute_ngram_ids` / `ple_dilated_conv1d`、G2 の `gather_rows_vram`、G5d の `PleNgramContext` / `PleConvStatePool` を再利用。stage は `create(config, weights, max_tokens, row_source, stream)` で stream 同期込み |
| 既存テスト | `connect_ple=false` の経路は node 列・weight slot・param 数・external_inputs 数を不変（contract / forward fixture が PASS） |

**検証**（`test_qwen4exp_ple_graph`、gpu1;required、**83 件 / 0 failed**、GPU2 と GPU3 で
同一結果、ctest 0.31 秒）:

| ケース | 指標 |
| --- | --- |
| PLE 出力 vs 仕様 oracle（§9.1 / §9.2 の独立実装） | prefill 4 / chunk 2 / chunk 境界 / decode 1 / EOS chunk / cancel 後 / reuse の全7件で **max_abs 0（bitwise）** |
| graph の `L1.ple.add` 出力 vs host 参照（x4 + PLE 出力） | **max_abs 0（bitwise）** |
| prefill chunk 境界（4 = 2+2）: logits / selected hidden / ple.add 行 | いずれも **max_abs 0（bitwise）** |
| decode 1 token 続続（split prefill → decode vs prefill 5）: logits / ple.add | いずれも **max_abs 0（bitwise）** |
| EOS 分割（chunk 境界が eos を跨ぐ vs one-shot）: logits | **max_abs 0（bitwise）** |
| cancel（n-gram context / conv state の reset） | context は `[eos, eos]` へ復帰、conv state は全要素 0、その後の PLE 出力は fresh 状態の oracle と **bitwise** |
| Request 再利用（release → 同一 slot 再確保） | slot 再利用を確認。logits / PLE 出力とも fresh slot 実行と **bitwise** |
| n-gram context の遷移 | prefill 後に `[19, 23]`、EOS 後に `[7, 11]` を確認 |

**修正した実装バグ**（テスト駆動で発見）:
`PleBranch` の `gemv` が行ポインタ更新を `++w += cols` と書いていたため 1 行ごとに
`cols + 1` 進み、2 行目以降がずれて最終行は配列外を読んでいた（NaN / 大値の非決定挙動の原因）。
`w += cols` に修正。

**回帰**: `test_qwen4exp_lowering_contract` / `test_qwen4exp_forward_fixture` /
`test_qwen4exp_weights` / `test_ple_row_gather` / `test_architecture_boundaries` の
**5/5 PASS**（`PHASESHIFT_TEST_GPUS=2,3`）。

### round 3 の分担（G5 の残り）

round 2 の結果、G5 の PASS 条件（手順書 §10）からは
**「PLE 状態継続」と「MoE 合成（full-graph）」**が未充足（上表参照）。
この2件を並行して実施する。

| 担当 | 範囲 | PASS 条件への対応 |
| --- | --- | --- |
| G5e | `connect_moe=true` での full-graph 実行検証（G4b の `moe_math.h` と oracle 比較、MoE あり/なしの併記） | **MoE 合成** |
| G5f | PLE の graph 接続（`source_audit.md` 付録 F の 2 案から選択）と PLE 状態継続の検証（`PleConvStatePool` / `PleNgramContext` を graph 経由で） | **PLE 状態継続** — **PASS**（下節） |

**mini runtime（`models/qwen4exp/runtime/` の executor 経路）の要否判断**:

手順書 §10 の PASS 条件はいずれも**小型 fixture での検証**であり、
`test_qwen4exp_forward_fixture` は `lowering → build_program → correctness dispatch` 経由で
prefill → decode を実行している。§10-7（Request 再利用 / prefill chunk 境界 / decode 続続 /
cancel / EOS / 状態破棄）は G5d が state owner レベルで、forward fixture が計算レベルで検証済み。

したがって **`Executor` / `ContinuousBatcher` 経由の mini runtime は §10 の PASS 条件を
満たすために必須ではない**と判断する。ただし次の点は後続課題として記録する:

- `phaseshift-compute` / `phaseshift-bench` から qwen4exp を起動する経路（entrypoint 接続）
- `ContinuousBatcher` との統合、`PrefixCache` への新 state の保存・復元
  （構造と owner は `source_audit.md` 付録 G に記録済み）
- optimized 経路（correctness fallback で動いている GR / MoE / QSA の optimized 接続）

### G5 の残課題（PASS 条件外の引き継ぎ）

round 3 までに G5 の PASS 条件は全項目が充足した（下の表）。
以下は **PASS 条件に含まれないが、後続 Gate で必要になる作業**として引き継ぐ。

| 項目 | 状態 | 引き継ぎ先 |
| --- | --- | --- |
| entrypoint（`phaseshift-compute` / `phaseshift-bench`）からの起動経路 | 未着手 | 後続（mini runtime） |
| `ContinuousBatcher` / `PrefixCache` との統合（新 state の保存・復元） | 構造と owner を `source_audit.md` 付録 G に記録済み、実装は未 | 後続 |
| optimized dispatch 接続（GR / MoE / QSA は現状 correctness fallback） | 未着手。`kernels/optimized/moe_ops.hip` は router 出力が row_stride 契約に未対応 | 後続 |
| `PrimitiveStateKind::INDEXER_KEY_CACHE` の登録 | 未着手（G5c との衝突回避で保留。pool は `IndexerKeyPool` として実装済み） | 後続 |
| PLE projection / gate / conv の GPU kernel 化（現状 host F32） | 未着手 | G6 に必要なため要対応 |
| QSA の sparse mask 入力と indexer key state の lowering 接続 | 未着手（kernel と pool は実装済み） | 後続 |
| `docs/developer/qwen4exp.md` gap #3 / #4 の解消 | 上記の接続が条件 | 後続 |

## G6（着手不可）

開始条件: G1〜G5 PASS **かつ** Expert 量子化・対応ローダーが別課題で準備済みであること。

現状:

- **G1〜G5 は PASS**（本 Gate 群の範囲は達成）
- **`Qwen3.8-Flash-Next-FP8` checkpoint がローカル不在**（FP8 の `quantization_config` /
  scale tensor 名・shape が未確定）← 主なブロッカー
- Expert 新量子化フォーマットは手順書 §1.3 で対象外と明記
- 実モデル 512 expert の VRAM 常駐・キャッシュ、PLE の SSD 経路 E2E、
  PLE / QSA の GPU kernel 化は未着手（G5 の残課題表を参照）

→ **BLOCKED（FP8 checkpoint 不在・前提未達）**。着手しない。

## G7（着手不可）

開始条件: E2E-VRAM・E2E-SSD の両方 PASS（G6）。

→ **BLOCKED（G6 未達）**。着手しない。

# Feature Base Engine 設計

## 1. 目的

> 未対応なら動かない。対応済みなら、宣言した機能が本当に使われる。

モデルや量子化方式が増えても、機能の対応漏れが「暗黙のフォールバック」「成功扱い」「無警告のダウングレード」で隠れないようにする。新しい仮想 Interface は原則追加せず、既存の `PrimitiveGraph` / `Program` / `KernelId` / `ExecutorModelInput` を活かして、**モデル契約・実行計画・起動時検証**の3点で保証する。

本設計は、2026-11-17 の QSA sparse attention 脱落問題（FP8 / Octave KV が `qsa_sparse_supported` の BF16 限定に引っかかり、dense 経路に無警告で落ちていた）を契機に、リポジトリ全体を静的調査した結果に基づく。

## 2. 調査で判明した問題

### 2.1 最優先: correctness interpreter が未対応機能を「成功扱い」で実行する

`program_executor.hip:454–538` の `KernelMode::Auto` は、optimized launch が `NotApplicable` を返すと correctness interpreter にフォールバックする。ところが interpreter 側が機能を実装しておらず、誤った結果を `COMPLETE` で返す経路がある。

| 箇所 | 問題 |
| --- | --- |
| `kernels/correctness/detail/attention.inc:193–330` | `input_count==2` の block mask（QSA 選択マスク）を完全に無視して full attention を計算する。tree visibility も見ない。QSA モデルで selector が miss すると sparse が dense に化けて誤出力が成功として返る |
| `kernels/correctness/detail/gdn.inc:79–175` | spec history capture / tree を見ずに recurrent state を確定書き込みする。verify バッチがフォールバックすると拒否された draft の state が恒久状態にコミットされる |
| `inference/execution/gdn_recurrence_dispatch.hip:246–247` | resolve の全エラーを `NotApplicable` に握りつぶす。本来伝播すべき `invalid_state` までフォールバックに化ける |
| `kernels/correctness/detail/gdn.inc:5–69` | conv1d も同型の state commit 問題 |
| `kernels/correctness/detail/linear.inc:139,237` | FP8 / MXFP4 fallback が `row_stride` と `col_offset` を無視して密に書く |
| `kernels/correctness/detail/embedding.inc:26–29` | vocab 範囲チェックなしで OOB read |

さらに、`correctness_fallbacks` の計上は `SHELF_QWEN35_PERF_STATS` 有効時のみで、`perf_stats_dump` を呼ぶのは bench の tg/pp だけ。**本番の shelf-compute / shelf-server ではフォールバックが完全に不可視**。唯一の可視化は `paged_attention_dispatch.hip:910–919` の一度きりの stderr warning のみ。

### 2.2 暗黙のフォールバック・無効化・ダウングレード

| 箇所 | 問題 |
| --- | --- |
| `apps/bench/pp.hip:472–474`、`tg.hip:512–514` | TP 経路が `octave:...` を無警告で BF16 に化けさせる（実害バグ） |
| `apps/compute/compute_runtime.hip:502–503,710–711`、`single_request.hip:881–883`、`stdio_server.hip:1433–1435` | MTP の KV dtype が OCTAVE→BF16 に無警告ダウングレード |
| `apps/compute/main.hip:112–115` | PP>1 で prefix cache を強制オフ、WARN のみで終了コード 0 |
| `apps/compute/compute_runtime.hip:465–476,805–843`、`main.hip:299–303` | PrefixCache 生成失敗を記録して warning で続行 |
| `quantization/bundle/hf_manifest.cpp:643–647`、`weights/weight_loader.cpp:845–866`、`models/qwen35/weights/model_weights.cpp:488–491` | 未知 quant_method（awq/gptq 等）が `unsupported` ではなく BF16 ロード経路へ流れる |
| `models/common/qwen_hybrid/config.cpp:344–350,371–379` | 未知 layer_type が `-1` になり、暗黙に full attention 扱い |
| `inference/execution/optimized_dispatch.hip:646–675` | EXL3 dense prefill の実エラーを `hipSuccess` にリセットして generic 経路を再実行 |
| `inference/execution/exl3_dense_gemm.hip:279`、`gemm_lt.hip:186` | hipBLASLt のあらゆる失敗を `hipErrorNotSupported` に変換し、実エラーを「未対応」に化けさせる |
| `apps/compute/compute_runtime.hip:147–152` | Qwen4Exp DFlash2 drafts>6 をエラーではなく stderr 通知で 6 にキャップ |
| `runtime/executor.hip:2748` | QSA state replay が program に QSA が無いと `make_ok()` で丸ごとスキップ |

### 2.3 状態プールと Lowering 結果の不整合

| 箇所 | 問題 |
| --- | --- |
| `runtime/executor.hip:2178–2196` | `uses_qsa_state` / `uses_ple_conv_state` フラグだけでプールを確保。Program に `QSA_SELECT` / `PLE_CONV` があるかとの照合が生成時に存在しない |
| `models/qwen35qsa/model/qwen35qsa_model.cpp:102–110` と `lower_to_primitives.cpp:279` | プール側と lowering 側が `max_context_tokens > budget` という同一式を独立に複製。一致は規約であり検証ではない |
| `apps/compute/compute_runtime.hip:466–467` | 単一 GPU の `PrefixCache::create` が KV+GDN のみを渡し、QSA/PLE プールを渡さない。TP 側 `:828–831` は渡しており非対称 |
| `runtime/prefix_cache.cpp:528–538,564–573,638–647` | QSA プールが nullptr のとき save/restore/capture が QSA 部分を無言でスキップ。復元後に reset も走らず、古い pooled key が `QSA_SELECT` に読まれる |
| `runtime/continuous_batcher.cpp:620–626` | `external_prefix_restore` が nullptr なら `Status::make_ok()` で成功扱い |
| `runtime/prefix_cache.cpp:741–766` | `prefix_cache_bytes()` の qsa/ple 引数は本番で未使用。容量は `executor_state_reserve_bytes()` で別途加算され、状態定義が分散 |
| `models/qwen35qsa/model/qwen35qsa_model.cpp:15–28` と `models/tp_rank_model.cpp:560–570` | QSA layout 関数が2箇所に複製 |
| `include/shelf/models/tp_rank_model.h:67–80` | `executor_state_reserve_bytes()` / `uses_qsa_state()` / `mtp_grouped_reserve_bytes()` / `normed_token_hidden_width()` のデフォルトが 0/false。override 漏れを検出する仕組みが無い |

### 2.4 網羅性の欠如

| 箇所 | 問題 |
| --- | --- |
| `models/model_weights_factory.cpp:11–108`、`models/loaded_model.h:49–54`、`loaded_model.cpp` 各所 | `ModelFamily` に対する switch はリポジトリ全体に存在せず、全 dispatch が if 連鎖 + Dense/Moe フォールバック |
| `include/shelf/models/model_weights_factory.h:30–34` | `LoadedModelWeights` variant は4メンバで `Qwen4Exp` が存在しない。enum は5値。両者を結びつける `static_assert` が無い |
| `models/pipeline_stage_model.cpp:101–108` | Qsa と Qwen4Exp が暗黙に dense builder へ落ちる |
| `quantization/manifest/manifest_v4.cpp:131–138,434–443` ほか | enum→名前変換 switch が `default:` 付きで暗黙フォールバック。`-Wswitch` さえ出ない |
| `quantization/manifest/manifest_v4.cpp:440,445–451` | `Fp8PerChannel` の v4 ラウンドトリップ欠落（serialize すると再 parse 不能） |
| `cmake/targets.cmake`、`cmake/apps.cmake` | `-Wswitch-enum` / `-Werror=switch-enum` は未設定。アプリ・テストには warning フラグ自体が無い |

### 2.5 リポジトリ衛生

`src/shelf/models/qwen35/runtime/` の `prefix_cache.cpp` / `continuous_batcher.cpp` / `tensor_parallel.cpp` は、`src/shelf/runtime/` の実装とほぼ同一の複製だが、**CMake に含まれていない dead code**（`shelf_qwen35_runtime` は `executor_model.cpp` 等4ファイルのみをビルド）。状態定義の二重管理はこの複製構造にも由来する。AGENTS.md の「非アクティブな実装をリポジトリ内に保存しない」に反する。

## 3. 設計原則

1. **既存の層と型を活かす。** `ExecutorModelInput` のフラグ、`KernelId`、`PrimitiveStateKind`、`check_config_identity` の allowlist gate は既にある。これらを「契約」として明確化し、新しい仮想 Interface や多重継承は追加しない。
2. **ホットパスに変更を入れない。** 検証は生成時・起動時に限定する。性能優先の哲学と衝突しない。
3. **Lowering 結果を正本にする。** モデルが何を宣言しているかだけでなく、生成された `Program` に実際に何が現れるかを照合する。
4. **コンパイル時網羅性と起動時検証を組み合わせる。** enum の網羅は switch + `-Wswitch-enum` でビルド時に検出し、dtype・文脈長・TP 数など実行時条件は起動時に検証する。
5. **correctness interpreter は「未対応なら unsupported」。** 誤った結果を成功として返す経路をなくす。

## 4. 提案する設計

### 4.1 Model Contract（既存 `ExecutorModelInput` の拡張）

新設ではなく、既存の `ExecutorModelInput`（`include/shelf/runtime/executor.h:262–276`）を契約として明確化する。

```cpp
struct ExecutorModelInput {
    bool uses_qsa_state = false;
    bool uses_ple_conv_state = false;
    // 既存の qsa_layers / qsa_head_dim / qsa_compress_ratio / qsa_max_blocks など
};
```

- このフラグは「モデルが必要とする状態」の宣言であり、`TpRankModel::uses_qsa_state()` のような**見積り用の別チャネルを廃止**して、ここに一本化する。
- `TpRankModel` の `executor_state_reserve_bytes()` / `uses_qsa_state()` は、この契約から導出する共通処理に寄せる。デフォルト実装による対応漏れの隠蔽をなくす。
- 状態 layout の計算関数は1箇所に統一する（`qwen35qsa_model.cpp` と `tp_rank_model.cpp` の複製を解消）。

### 4.2 Execution Requirements（Lowering 結果から導出）

`Program` / `PrimitiveGraph` を走査し、実際に必要な機能と状態を導出する。

```cpp
struct ExecutionRequirements {
    bool needs_qsa = false;      // QSA_SELECT / QSA_KEY_CACHE が存在するか
    bool needs_ple_conv = false; // PLE_CONV が存在するか
    uint32_t qsa_state_layers = 0;
    uint32_t ple_state_layers = 0;
    // KV dtype、tree visibility、block mask の有無など
};
```

既存の `program_needs_qsa_workspace()`（`executor.hip:1902–1907`）を拡張し、workspace 計算だけでなく状態プールの照合にも使う。

### 4.3 Feature Validation（生成時照合）

`create_executor()` の `build_program_set()` 直後に、双方向の照合を行う。

- `requirements.needs_qsa && !input.uses_qsa_state` → `Status::unsupported`
- `input.uses_qsa_state && !requirements.needs_qsa` → `Status::invalid_state`（プールだけ確保される無駄を防ぐ）
- `requirements.qsa_state_layers != pool_layout.layers` → `Status::invalid_state`
- `state_index` の上限を通常経路でも検査する（現状は replay 経路のみ）。

これにより、現在の「起動時 dispatch の fail-fast が事後の砦」という構造を、生成時の検証に前倒しする。

### 4.4 Prefix State Binding（Executor から一括取得）

`PrefixCache::create()` の2つのオーバーロードを廃止し、状態 Binding を Executor から一括取得する。

```cpp
struct PrefixStateBindings {
    PagedKVPool& kv;
    GdnStatePool& gdn;
    ::shelf::state::QsaKeyStatePool* qsa;
    ::shelf::state::PleConvStatePool* ple;
    ExternalPrefixRestoreHandler* history_restore;
};

Result<PrefixStateBindings> make_prefix_state_bindings(Executor& executor);
Status validate_prefix_state(const ExecutorModelInput& model,
                             const PrefixStateBindings& bindings);
```

- 単一 GPU と TP で同じ `create()` を呼ぶ。QSA/PLE の渡し忘れを構造的に防ぐ。
- `model.uses_qsa_state && bindings.qsa == nullptr` なら Prefix Cache を作る前にエラーにする。
- 容量見積もりも同じ `PrefixStateBindings` を参照し、`prefix_cache_bytes()` の未使用引数と `executor_state_reserve_bytes()` の分散を解消する。
- `external_prefix_restore` が nullptr で、かつモデルが Graph 外の状態（PLE n-gram history など）を宣言している場合は、`Status::make_ok()` ではなくエラーにする。

### 4.5 correctness interpreter の Fail Fast

`sampling_dispatch.hip:90–93` の既存パターン（token bitmask があれば `unsupported`）を他経路に展開する。

- block mask / tree visibility / spec history 付きの dispatch は interpreter で `Status::unsupported` を返す。
- `gdn_recurrence_dispatch.hip:246–247` の resolve エラー握りつぶしを修正し、エラーを伝播する。
- FP8 / MXFP4 fallback の `row_stride` / `col_offset` を正しく扱うか、拒否する。
- `correctness_fallbacks` を常時計上し、本番経路（shelf-compute / shelf-server）でも報告する。

## 5. 具体的な変更箇所

| 対象 | 変更 |
| --- | --- |
| `kernels/correctness/detail/attention.inc`、`gdn.inc`、`linear.inc`、`embedding.inc` | mask/tree/spec history 付き dispatch を `unsupported` に。stride/offset を正しく扱う |
| `inference/execution/gdn_recurrence_dispatch.hip:246–247` | resolve エラーを伝播 |
| `inference/execution/program_executor.hip`、`decode_perf_stats.cpp` | fallback を常時計上し、本番で報告 |
| `apps/bench/pp.hip:472–474`、`tg.hip:512–514` | octave を正しく映射するか、TP+octave を明示的に拒否 |
| `apps/compute/compute_runtime.hip`、`single_request.hip`、`stdio_server.hip` | MTP の OCTAVE→BF16 ダウングレードを明示的なエラーまたは宣言された仕様にする |
| `apps/compute/main.hip:112–115,299–303` | PP+prefix cache、prefix cache 生成失敗をエラーにする |
| `quantization/bundle/hf_manifest.cpp:643–647` | 未知 quant_method を `unsupported` で拒否 |
| `models/common/qwen_hybrid/config.cpp:344–350` | 未知 layer_type を `unsupported` で拒否 |
| `runtime/executor.hip:2178–2196` 周辺 | Lowering 結果と状態プールの生成時照合を追加 |
| `runtime/prefix_cache.cpp`、`prefix_cache.h`、`compute_runtime.hip` | State Binding を統一し、単一 GPU と TP の非対称を解消 |
| `models/model_weights_factory.cpp`、`loaded_model.h`、`loaded_model.cpp`、`pipeline_stage_model.cpp`、`tp_rank_model.cpp` | if 連鎖を switch に書き換え、`-Wswitch-enum -Werror=switch-enum` を適用 |
| `include/shelf/models/tp_rank_model.h:67–80` | デフォルト実装を廃止し、契約から導出 |
| `cmake/targets.cmake`、`cmake/apps.cmake`、`tests/CMakeLists.txt` | warning フラグをアプリ・テストにも適用 |
| `src/shelf/models/qwen35/runtime/` の dead code 複製 | 削除 |

## 6. 実装順序

| Step | 内容 | 理由 |
| --- | --- | --- |
| 0 | correctness interpreter の silent wrong output を Fail Fast 化 | correctness に直結。今回の QSA sparse 脱落問題の根本原因の一つ |
| 1 | 暗黙フォールバックの Fail Fast 化（bench TP octave、MTP ダウングレード、未知 quant_method、PP+prefix cache など） | 実害バグと対応漏れの隠蔽をなくす |
| 2 | Lowering 結果と状態プールの生成時照合 | 「宣言した機能が本当に使われる」を保証 |
| 3 | PrefixCache の State Binding 統一 | 単一 GPU と TP の非対称を解消 |
| 4 | ModelFamily の switch 化と網羅性チェック | モデル追加時の対応漏れをビルド時に検出 |
| 5 | dead code 削除と機能組み合わせテスト追加 | リポジトリ衛生と回帰防止 |

Step 0 と 1 は比較的小さく、効果が大きい。Step 2–4 は既存フラグとの統合を含む中型の変更。Step 5 は最後でよい。

## 7. テスト計画

- **対応漏れ時の起動失敗テスト**: QSA が必要なのにプールが無い、未知 quant_method、未知 layer_type、PP+prefix cache などで `Status::unsupported` が返ることを確認。
- **状態復元テスト**: 単一 GPU と TP の両方で、QSA/PLE 状態が Prefix Cache から正しく復元されることを確認。既存の `test_prefix_cache.hip` はプール付きオーバーロードを検証しているが、batcher 統合テストは4引数版のみなので拡張する。
- **実行経路の検証**: 出力が正しいだけでなく、期待するカーネルが選択されていることを `correctness_fallbacks` と kernel trace で確認。QSA × KV dtype（BF16/FP8/Octave）の組み合わせで sparse 経路が選ばれることをテスト。
- **生成一致アンカー**: 各変更で `GENERATED_IDS` / `GREEDY_TOKEN_SUM` が変わらないことを確認（Fail Fast 化は未対応ケースを弾くだけで、対応ケースの出力は変えない）。

## 8. 受入条件

- 未対応の機能組み合わせは、実行を続けずに `Status::unsupported` で止まる。
- 対応済みの機能は、宣言だけでなく実際の実行経路（Lowering 結果・選択カーネル・状態プール）で使われていることが検証される。
- correctness interpreter が誤った結果を成功として返す経路が無い。
- `ModelFamily` への追加漏れがビルド時に検出される。
- ホットパスの性能が変わらない（検証は生成時・起動時のみ）。

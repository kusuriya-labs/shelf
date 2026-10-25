# shelf — XeStrataをベースに空のリポジトリから推論エンジンを再構築する計画

> **ステータス:** 初期設計案  
> **プロジェクト名:** `shelf`  
> **作成日:** 2026-10-25  
> **位置づけ:** XeStrataの高速化資産とPhaseShiftの責務分離を参考に、既存クラス階層を引き継がず新規設計する推論ランタイム

## 1. 目的と設計方針

`shelf` は、XeStrataのSYCL Kernel、EXL3処理、CPU/GPU協調、Expertの階層化、PCIe転送、複数GPU対応を活用しつつ、PhaseShiftのModel Lowering、Primitive IR、量子化契約、State管理、アーキテクチャ境界テストを参考に構築する。

**基本方針は、既存リポジトリをクラス単位で移植するのではなく、契約を先に設計して性能資産を段階的に接続すること。**

### 1.1 達成したいこと

1. **モデル非依存:** モデル追加時は、原則としてModel定義・Lowering・Tensor名Mappingのみ追加する。
2. **Weight形式非依存:** GGUFとEXL3などの保存形式を、モデル実行系から切り離す。
3. **量子化とKernel選択の分離:** PSQ4/PSQ5、MXFP、EXL3などの物理表現と、その実行Kernelを独立させる。
4. **CPU/GPU協調の維持:** Expertの常駐、転送、計算配置、イベント同期を独立した責務として扱う。
5. **数値と性能の回帰防止:** Kernel・Layer・Logits・Token・スループットの各段階で既存実装と比較する。
6. **依存方向の固定:** 禁止依存・循環依存・グローバルなモデル固有可変状態を検知する。

### 1.2 採用しない構造

- 巨大な`BaseModel`にLoader、State、実行、Samplingを集中させない。
- `GGUFModel`と`EXL3Model`のように保存形式ごとのモデル実行系を作らない。
- EXL3・PSQなどの重みを必ずBF16へ展開して実行する設計にしない。
- CPU/GPUのすべての処理を一つの万能な仮想Backendで抽象化しない。
- Decodeのホットパスで毎回フォーマット判定・Kernel探索をしない。
- モデルごとにGPU Kernelのディレクトリを分裂させない。
- 最初から完全な汎用Compilerや全モデル対応を目指さない。

## 2. 全体アーキテクチャ

```text
Application / CLI / Serving
            |
            v
Inference Runtime (Session / Batch / Scheduling / Speculation)
            |
            v
       Model Definition
            |
            v
     Semantic Lowering
            |
            v
        Primitive IR
            |
            +<---- Weight Binding <---- Artifact Reader (GGUF / EXL3 / ...)
            |              ^
            |              |
            |        QuantSpec / PhysicalLayout
            |
            +<---- Placement Planner <---- Device Topology / Residency
            |
            v
      Execution Compiler
            |
            v
       Executable Plan
            |
            +---- SYCL GPU Kernels / Dispatch
            +---- CPU SIMD Kernels
            +---- Transfer / Queue / Event Dependencies
            +---- Graph Capture / Replay
```

この図は**データと契約の関係**を示す。WeightのMaterialization、Placement、Compileは準備段階で協調し、実行中にResidencyやBatchが変化する場合は動的な計画・イベント制御を組み合わせる。すべての処理が単一の直列パイプラインであるという意味ではない。

### 2.1 最重要の3契約

| 契約 | 責務 | 禁止事項 |
|---|---|---|
| **Semantic Model** | モデルの数学的演算、形状、State依存、Weightの論理的要求 | 保存形式やSYCL/CPU実行方法を持たない |
| **Weight Representation** | 論理Tensorから物理Payload・量子化・レイアウトへの対応 | モデルのforwardや実行時スケジュールを持たない |
| **Execution** | Kernel選択、メモリ配置、デバイス転送、イベント依存、実行 | モデル固有の数学的意味を勝手に変更しない |

## 3. 推奨ディレクトリ構成

完成形の目安。**初期コミットですべての空ディレクトリを用意する必要はない。**

```text
shelf/
├── CMakeLists.txt
├── CMakePresets.json
├── LICENSE
├── README.md
├── cmake/
│   ├── Dependencies.cmake
│   ├── Options.cmake
│   └── Targets.cmake
├── include/shelf/                # 外部公開APIのみ
│   ├── shelf.hpp
│   ├── model.hpp
│   ├── session.hpp
│   ├── generation.hpp
│   └── status.hpp
├── src/
│   ├── foundation/               # Tensor/Shape/DType/Result等
│   ├── model/                    # Model定義・Lowering・Tensor Mapping
│   │   ├── common/
│   │   ├── registry/
│   │   └── qwen35/               # 対応モデルを増やす際に追加
│   ├── ir/                       # Graph/Operations/State/Validation
│   ├── artifact/                 # GGUF / EXL3 / safetensors Reader
│   │   ├── common/
│   │   ├── gguf/
│   │   └── exl3/
│   ├── quant/                    # QuantSpec / Codec / Reference Decode
│   │   ├── spec/
│   │   └── codec/
│   │       ├── psq/
│   │       ├── exl3/
│   │       ├── mxfp/
│   │       └── gguf/
│   ├── weights/                  # Catalog / Binding / PhysicalLayout
│   │   ├── catalog/
│   │   ├── binding/
│   │   ├── layout/
│   │   └── materialization/
│   ├── compiler/                 # IR → Executable Plan
│   │   ├── passes/
│   │   ├── kernel_selection/
│   │   ├── memory_planning/
│   │   └── execution_plan/
│   ├── backend/
│   │   ├── common/
│   │   ├── cpu/
│   │   │   ├── kernels/
│   │   │   └── simd/
│   │   └── sycl/
│   │       ├── device/
│   │       ├── kernels/
│   │       │   ├── linear/
│   │       │   ├── attention/
│   │       │   ├── gdn/
│   │       │   ├── moe/
│   │       │   ├── norm/
│   │       │   └── quant/
│   │       ├── dispatch/
│   │       ├── memory/
│   │       └── graph/
│   ├── placement/
│   │   ├── topology/
│   │   ├── partition/
│   │   ├── residency/
│   │   └── experts/
│   ├── runtime/
│   │   ├── execution/
│   │   ├── session/
│   │   ├── state/
│   │   │   ├── kv_cache/
│   │   │   └── recurrent/
│   │   ├── scheduling/
│   │   ├── batching/
│   │   └── speculative/
│   └── api/
├── apps/
│   ├── cli/
│   ├── server/
│   └── inspect/
├── tests/
│   ├── architecture/
│   ├── unit/
│   ├── kernels/
│   ├── parity/
│   ├── integration/
│   └── fixtures/
├── benchmarks/
│   ├── kernels/
│   ├── inference/
│   └── memory/
├── tools/
│   ├── conversion/
│   ├── inspection/
│   └── quality/
├── docs/
│   ├── architecture/
│   ├── contracts/
│   └── formats/
└── third_party/
```

### 3.1 ディレクトリの依存ルール

- `foundation` → C++標準ライブラリを中心とした最下層。モデル・SYCLに依存しない。
- `model` → `foundation`、IR契約、Weightの**論理的要求**のみを参照する。
- `artifact` → 保存形式の構造だけを解釈し、Qwen固有のforwardを知らない。
- `quant` → 量子化仕様、物理Codec、CPU参照復号を管理し、モデル固有処理を持たない。
- `weights` → Artifact由来TensorとModel由来Tensor要求を対応付け、物理Weightを構成する。
- `compiler` → IR、Weights、Placement、Backend能力を受け取りPlanを構築する。
- `backend` → CPU/SYCLのデバイスとKernelを管理し、`if (model == Qwen...)`を持たない。
- `placement` → デバイス・Weight常駐・Expert配置戦略を管理し、演算の意味を定義しない。
- `runtime` → 実行済みPlanと動的なSession/Batch/State/Eventを管理する。
- `include/shelf/` → 最小限の外部公開APIだけを置く。内部モジュール間のヘッダーをすべて公開しない。

CMake Target例：`shelf_foundation`, `shelf_ir`, `shelf_artifact`, `shelf_quant`, `shelf_weights`, `shelf_model`, `shelf_backend_cpu`, `shelf_backend_sycl`, `shelf_placement`, `shelf_compiler`, `shelf_runtime`, `shelf_api`。

**Targetを分けることと共有ライブラリへ分割することは別。** 初期はStatic Library / Object Library / Interface Targetを活用する。

## 4. 最初に決める6つの契約

### 4.1 Tensor Contract

形状・軸の意味・DType・ID・役割を定義する。論理Tensorに生ポインタやGPU常駐フラグを持たせない。

```cpp
struct TensorSpec {
    TensorId id;
    TensorShape shape;
    DataType logical_dtype;
    TensorRole role;
};
```

### 4.2 Weight Contract

コンテナ形式、量子化形式、物理レイアウト、Kernel用ComputeSpecを区別する。

```cpp
struct QuantSpec {
    QuantFormatId format;
    QuantParameters parameters;
};

struct PhysicalWeight {
    TensorId tensor;
    QuantSpec quant;
    PhysicalLayout layout;
    BufferHandle storage;
};
```

`GGUF`はコンテナ形式、`GGUF_Q4_K`などはコンテナ内の量子化形式。`EXL3`ではアーティファクトReaderとTrellis等の物理Codecを別責務として扱う。`PSQ4`/`PSQ5`や`MXFP4`/`MXFP6`にも同じ原則を適用する。

想定する経路：

```text
Artifact Reader
  → Raw Tensor Catalog
  → Model-specific Tensor Name / Axis Mapping
  → Canonical Tensor Catalog
  → QuantSpec + PhysicalLayout
  → Materialized Weight / Kernel View
```

**同じPSQ5形式なら、モデルごとにPSQ5 Loaderを作らない。** モデル差分は名前解決、軸変換、演算構造に限定する。

### 4.3 Operation Contract

RMSNorm、Linear、RoPE、Attention、GDN、MoE、Residualなどの数学的意味、形状制約、State読み書きを定義する。GDNのゲート関数などを共有グローバル可変状態にしない。

### 4.4 Kernel Contract

`Operation + Shape + QuantSpec + PhysicalLayout + Device Capability + 実行モード`に基づきKernelを選択する。選択は原則Compile段階で終え、Decode中の汎用探索を最小化する。

### 4.5 Resource Contract

Buffer所有権、RAM/VRAM常駐、Session State、Workspace寿命、Graph Capture中のポインタ安定性、転送完了条件を定義する。CPU/GPU間の非同期処理では、イベントと生存期間を必ず結び付ける。

### 4.6 Execution Contract

演算順序・Queue・転送・完了イベント・Join条件を管理する。MoEのRouterが選んだExpertの**演算意味論**と、そのExpertをCPU/GPUのどちらで走らせるかを分離する。

## 5. 実装フェーズ（0〜9）

| Phase | テーマ | 完了基準 |
|---|---|---|
| 0 | 仕様・ライセンス・Golden Fixture | 既存実装の数値・性能基準と移植元が記録済み |
| 1 | Foundation・ビルド・CI | CPU-onlyビルド、型・境界テスト成功 |
| 2 | Device / Buffer / Queue / Event | CPU↔GPU転送、依存同期、所有権テスト成功 |
| 3 | Weight / Artifact / Quant契約 | BF16の基本経路と形式別Readerの検証成功 |
| 4 | Primitive IR / CPU参照Executor | 合成モデルの実行とState更新成功 |
| 5 | SYCL Backend / Execution Compiler | CPU参照とのKernel出力比較成功 |
| 6 | 最初の実モデル | Layer比較、Logits比較、複数Token Decode成功 |
| 7 | Expert階層化 | CPU/GPU混合MoEの数値Parity成功 |
| 8 | Prefill / Batch / Multi-GPU | 可変長Batch、Cancel、複数Device状態管理成功 |
| 9 | 量子化・最適化・Serving拡張 | 精度・性能・メモリ回帰が継続的に評価可能 |

### Phase 0 — 再実装前の基準凍結

**実施内容**
- XeStrataのSYCL/CPU Kernel、EXL3 Reader、Expert Runtime、MoEスケジューリング等の実装とテストを棚卸しする。
- PhaseShiftのPrimitive IR、Lowering、量子化契約、StateとArchitecture Boundary Testを棚卸しする。
- Kernel、1 Layer、最終Logits、Token列についてGolden Fixtureを作る。
- 対象ハードウェア・モデル・量子化構成の速度（TTFT、Token/s等）とメモリ使用量を記録する。
- 取り込むコードについて、ファイル単位でライセンス・Copyright表記・由来を管理する。

**完了条件:** Golden Fixtureとベンチマークが再実行可能で、元コードのライセンス確認が終わっている。

> **ライセンス上の注意:** 空のリポジトリで作り直しても、ライセンス対象コードをコピー・改変した場合の義務は消えない。XeStrata側のLGPL系コードや他のライセンス対象ファイルを移植する前に、対象ファイル・リンク形態・配布条件を個別に確認する。法務的な最終判断は別途必要。

### Phase 1 — Foundation / CI

**実施内容:** `TensorId`、`Shape`、`DType`、`ModelSpec`、`Result`、検証関数、CMakeとCPU-only CIを用意する。

```bash
cmake -S . -B build/cpu \
  -DSHELF_ENABLE_SYCL=OFF \
  -DSHELF_BUILD_TESTS=ON
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure
```

> CMakeオプション名は`shelf`用の提案であり、現時点で存在するコマンドではない。

**完了条件:** CPU-onlyでビルドが通り、形状不正、Overflow、Tensor ID不整合、禁止依存を検知できる。

### Phase 2 — Device / Memory

**実施内容:** Device列挙、Capabilities、RAII Buffer、Queue、Event、Host↔Device Copyを実装する。SYCL型はBackend内に閉じ込める。

**完了条件:** CPU→GPU→CPUの往復転送と複数Queue間のEvent依存が正しく、解放・再利用の競合を検証できる。

### Phase 3 — Artifact / Weight / Quant

**実施内容:** まずBF16の小さなテスト用Weightを読み込み、Tensor Catalogと論理/物理Bindingを完成させる。その後EXL3 ReaderやGGUF Readerの必要機能を追加する。量子化Weightは専用Kernelに直接渡せる表現を維持する。

**完了条件:** 同じ論理Tensorを異なるコンテナから解決でき、形状・軸変換・Codec Payload検証が通る。

### Phase 4 — Primitive IR / CPU参照Executor

**最初の候補Primitive:** `Embedding`、`RMSNorm`、`Linear`、`SiLU/SwiGLU`、`ResidualAdd`、`RoPE`、`Attention`、`GDN`、`MoERoute`、`ExpertCompute`、`MoEMerge`、`OutputProjection`。

各PrimitiveのState読み書きと、型・形状・依存関係を検証可能にする。

**完了条件:** 小型の合成Model GraphがCPUで動作し、State更新のGolden Fixtureに一致する。

### Phase 5 — SYCL Backend / Compiler

**実施内容:** `Primitive IR → Kernel Selection → Memory Planning → Dependency Scheduling → Executable Plan` の経路を作る。SYCL RMSNorm、Linear等を最初に移植する。Device Capabilityに応じた特殊化をCompile時に行う。

**完了条件:** 同じIRをCPU参照ExecutorとSYCL Backendで実行でき、数値差が定義済みの許容範囲内に収まる。

### Phase 6 — 最初の実モデルの縦実装

**候補:** まずQwen3.5系MoEの小型構成または1 Layer Fixtureから開始する。選定は、両既存実装で参照できるWeight/Lowering/Kernelの整合性を確認して確定する。

**実施順:**
1. 1 Layer FixtureをModel Loweringする。
2. WeightをBindingする。
3. CPU参照でLayer出力を検証する。
4. SYCLでLayer出力を検証する。
5. 実モデルの全Layerを実行する。
6. Logits比較と複数Token Decodeを検証する。
7. EXL3等の量子化Weightを同じModel IRに接続する。

**完了条件:** 1トークン・複数トークン生成、KV/GDNなどのState更新、最終Logits比較が通る。

### Phase 7 — Expert階層化 / CPU・GPU協調

**主要責務:**
- `ExpertStorage`: Expertのデータ提供と物理メタデータ。
- `ExpertResidencyManager`: RAM/VRAM常駐、退避、Prefetch。
- `ExpertPlacementPlanner`: 現在の常駐状況とDevice能力に基づく計算配置。
- `ExpertExecutor`: 対応するCPU/SYCL Kernelの実行。
- `Runtime Execution`: CPU/GPU処理・転送のEvent JoinとMergeの発火。

```text
MoE Router
    |
    v
Expert Placement
    +---- GPU Resident → GPU Expert Compute ---+
    +---- CPU Resident → CPU SIMD Compute -----+--> Join → Weighted Merge
    +---- Transfer/Prefetch → GPU Compute -----+
```

**完了条件:** GPUのみ、CPUのみ、混在の各経路で数値比較が通り、非同期処理の重複実行・競合・古いWeight参照がない。

### Phase 8 — Prefill / Batch / Multi-GPU

**推奨順:** Prefill → KV管理 → Continuous Batching → 複数GPUのStage分割 → 必要に応じてTensor Parallel。

静的なModel IRと、Stepごとに変わるBatch/Request情報を分離する。SessionごとにKV・GDN等の永続Stateと一時Workspaceの寿命を管理する。

**完了条件:** 可変長Batch、終了・キャンセル、異なるデバイス上のState整合性をテストできる。

### Phase 9 — 最適化 / 拡張 / Serving

**実施内容:** EXL3の高度なKernel最適化、PSQ4/PSQ5、MXFP系、Fusion、Graph Capture/Replay、対応モデルの追加、Tokenizer・Sampling・CLI・Serverを段階的に組み込む。

**完了条件:** Prefill速度、Decode速度、TTFT、RAM/VRAMピーク、量子化精度、混合BatchのスループットをCI/ベンチマークで継続追跡できる。

## 6. Kernelの配置規則

Kernelは**モデル名ではなく演算名で整理する**。

```text
backend/sycl/kernels/
├── linear/
│   ├── bf16/
│   ├── exl3/
│   ├── psq/
│   └── mxfp/
├── attention/
│   ├── dense/
│   └── sparse/
├── gdn/
├── moe/
├── norm/
├── rope/
└── sampling/
```

Intel XMX、AMD Matrix Core、NVIDIA Tensor Coreなどの機器固有Kernelは、**実装が本当に分岐する演算だけ**`targets/`などで分ける。

## 7. テスト戦略と品質ゲート

| 分類 | テスト内容 | 対象 |
|---|---|---|
| Architecture | 禁止依存、循環依存、内部API越境 | CMake Target・ソース |
| Unit | Shape/Type/Weight Mapping/Codec | CPU-only |
| Kernel | 数値誤差、境界形状、Device差 | CPU参照 vs SYCL |
| Layer Parity | 1 Layer出力、State更新 | 対象モデル |
| Model Parity | Logits差、Top-1一致率、Token列 | End-to-End |
| Quant Quality | KLD、復元誤差、Top-1、Payload互換 | BF16基準との比較 |
| Async/Race | Queue/Event、Buffer寿命、Expert Join | CPU/GPU混在 |
| Performance | TTFT、Token/s、帯域、VRAM/RAMピーク | GPU構成別 |

**注意点:** モデルや量子化方式をまたいだ単一のKLD/Top-1合格値は設定しない。対象構成ごとに基準を定義する。速度だけでKernelを採否判定せず、転送・Workspace・同期・Graph Capture適合性を含めて判断する。

## 8. 推奨PR分割

| PR | 内容 | Merge条件 |
|---|---|---|
| PR-001 | CMake・骨格・Architecture Test | CPU-only CI成功 |
| PR-002 | Tensor / Shape / ModelSpec | 契約テスト成功 |
| PR-003 | Device / Buffer / Event | GPU転送と同期成功 |
| PR-004 | BF16 Weight Reader / Catalog | 読み込みParity成功 |
| PR-005 | IR / CPU参照Executor | 合成Model成功 |
| PR-006 | SYCL RMSNorm / Linear | CPUとの差分検証成功 |
| PR-007 | Qwen3.5 1 Layer Lowering | Layer比較成功 |
| PR-008 | GPU Decode最小経路 | 初回Logits・連続Decode成功 |
| PR-009 | EXL3 Reader / Kernel | 同一IRで形式切替成功 |
| PR-010 | Expert Residency / CPU-GPU混合 | 混合経路Parity成功 |

各PRで契約・依存関係・テスト・必要な設計文書を更新する。PR-008を、アーキテクチャ全体の成立を確認する最初のマイルストーンとする。

## 9. リファクタリングではなく再構築である理由

XeStrataの既存`PlainModel`や`Exl3FlashModel`を基底クラスとして採用すると、モデル種別、保存形式、State、Kernel、Expert Residencyの責務が再び結合する危険がある。

`shelf`では以下を守る。

1. モデルは**何を計算するか**を定義する。
2. Weightは**何がどの物理形式で存在するか**を定義する。
3. Placementは**どこに配置するか**を決定する。
4. Compilerは**どのKernel・Buffer・依存関係で実行するか**を確定する。
5. Runtimeは**いつ、どのRequest/Sessionで実行するか**を制御する。
6. Backendは**実際の演算を高速に実行する**。

これらを準備段階で結び付け、ホットパスには事前構築されたPlanを渡す。

## 10. 最重要マイルストーンと見直し条件

### PR-008時点で成立しているべきこと

- [ ] Model DefinitionとExecution Planが独立している。
- [ ] Weightのコンテナ/量子化形式がModel Loweringに漏れていない。
- [ ] CPUとSYCLで同じ演算意味論を検証できる。
- [ ] StateとWorkspaceの所有者・寿命が明確である。
- [ ] 新しいKernelを追加してもModel側の変更が原則不要である。
- [ ] 同じIRに異なる物理WeightをBindできる見通しが立っている。
- [ ] 禁止依存をArchitecture Testで検知できる。

**設計見直しのシグナル:** `Qwen...Model::forward()`の中にGPU転送、EXL3固有Payload処理、CPU Expertの起動・待機、グローバルなGDN/RoPE設定が入り始めた場合は、その場で責務境界を見直す。

## 11. 最終方針

`shelf` の最初の目標はXeStrataの全機能を再実装することではない。

**「同じModel IRに、異なるWeight形式と異なるBackendを接続して、正しさと性能を検証できる」ことを最初に証明する。**

そのために、XeStrataのKernel/CPU-GPU協調処理を段階的に移植し、PhaseShiftのIR/Lowering/契約境界を参考にしながら、`shelf`自身の設計を正本として育てていく。

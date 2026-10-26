# Shelf HIPバックエンドのTile API導入・カーネル関数分割手順書

- **作成・更新日:** 2026-10-26
- **対象リポジトリ:** 添付 `shelf-main(2).zip` の `shelf-main/`
- **想定格納先:** `docs/architecture/hip-tile-api-refactor-procedure.md`
- **対象技術:** HIP / C++20 / AMD gfx1201（RDNA4、wave32）
- **作業の種類:** 既存カーネルの構造的リファクタリング。アルゴリズム・バッチング・実行計画の改造ではない。
- **目的:** Tritonに似た「タイル単位で処理を組み合わせる」記述スタイルを、**JITなしのHIP/C++デバイス関数**で実現する。

> この文書は以前の「PhaseShift GDN WMMA専用手順書」を、**Shelfの実際のソースに合わせて全面改訂したもの**である。以下のシンボル・パス・テストは添付ZIPから確認した。サンプルの新APIは設計例であり、実装済みコードではない。

---

## 0. 最初に確認する事実と変更対象

前回文書の前提をそのまま移植しないこと。特に**Shelfには `gdn_recurrence_wmma_body` が存在しない**。

| 現在のShelfの実装 | 確認できた状態 | 今回の扱い |
|---|---|---|
| `src/backend/hip/kernels/gemm/linear_bf16.hip` | 約701行。`bf16_wmma_body<kRowTiles>`、`bf16_splitk_wmma_body`、`gemm_bf16_exact_rows_body<kRows,kDot2>`、複数のカーネル入口、launch分岐 | **第一対象**。WMMAとTile APIの最小実証に使う |
| `src/backend/hip/kernels/attention/attention_paged_decode.hip` | 約1,682行。BF16通常decode / split / reduce / FP8通常decode / FP8 split の複数実装 | **第二対象**。タイル・wave演算を再利用できるか検証する |
| `src/backend/hip/kernels/gdn/gdn_recurrence.hip` | 約139行。`gdn_recurrence_f32_impl`。Recurrent StateはFP32。**GDNのWMMA高速版は未移植** | 最初は触らない。GDN用WMMAを新規実装する作業ではない |
| `src/backend/hip/include/shelf/backend/hip/kernels/detail/` | `vector_io.h`、`l2_normalize_device.h` 等、既存のdevice helper群 | 同じ機能を二重実装せず、再利用候補として調査 |
| `src/backend/hip/step_executor.cpp` | GEMMでは現状 `rows=1 → ExactRows`、`rows>=2 → ExactRowsDot2` を選択する箇所がある | **入口APIと選択結果を固定**。WMMAが既定の全経路に使われると誤認しない |
| `cmake/Targets.cmake` | HIP `.hip` を `GLOB_RECURSE` で収集。既存 `shelf_backend_hip` ターゲットに追加 | header-onlyの抽出なら原則CMake変更不要 |

### 0.1 変更しない境界（必須）

1. 公開HIP契約: `src/backend/hip/include/shelf/backend/hip/kernels/` 内の既存`*Args`、`launch_*`、config列挙、既存の`extern "C" __global__`シンボルとその名前。
2. GEMMの振る舞い: WMMA / Wide / KPartition(Split-K) / ExactRows / ExactRowsDot2、`BF16/F32`出力、stride、tail mask、row分割、grid・block・stream、GPU Launch回数。
3. 計算・同期: 現行の積和・キャスト順、WMMA命令系列、Split-Kのreduce順、Shared Memoryの寿命、`__syncthreads()`の到達条件と順序。
4. Attentionの振る舞い: BF16/FP8、split/reduceの既存二段処理、paged KV addressing、部分RoPEとの連携、tree ancestor表現、invalid-page処理、出力dtype。
5. アーキテクチャ: HIP型は`src/backend/hip`に閉じ込め、`foundation`、`ir`、`model`、`backend/common`へ伝播させない。依存ルールと`tests/architecture/test_architecture_boundaries.cpp`を維持。
6. 実行制御の既定方針: **PrefillとDecode+Verifyを別実行モードとし、active sessionは1つ、残りはQueue管理。Mixed BatchingとContinuous Batchingは導入しない。** 本リファクタリングでscheduler・PCIe転送・CPU/GPU時間配分を変更しない。
7. SYCLについて: `src/backend/sycl/`は現状スケルトン。HIPのTile APIを中立APIとして見せかけたり、SYCL実装を同時に追加したりしない。
8. PhaseShift由来の実装について: `docs/architecture/porting-provenance.md`と既存コメントに記載された移植由来を維持する。

**禁止:** Triton導入、JIT/動的コンパイル、ランタイムTensor式評価、追加のデバイスメモリ確保、汎用の自動Layout変換、カーネルを細切れにしてGPU Launchを増やすこと、性能チューニング目的の計算順序変更、同時にRuntime/IRを再設計すること。

---

## 1. 目標のアーキテクチャ

### 1.1 三層に分離する

```text
[Host / Dispatch / ABI]             既存契約を維持
  shelf::hipk::launch_gemm_bf16(...)
       |                    ↑ C++ホスト側（変えない）
======= GPU Launch の境界 ======================================
[Algorithm Layer / op固有]
  gemm::load_operands() / gemm::compute() / gemm::store()
  attention::load_q() / attention::accumulate_online() / merge_waves()
       |
[Tile Primitive Layer / HIP内の小さな共通部品]
  tile::Wave32 / tile::WmmaBf16 / tile::load_masked()
  tile::warp_sum() / tile::warp_max() / tile::RegTile
       |
[Hardware Intrinsic Layer / gfx1201]
  __builtin_amdgcn_wmma_... / __shfl_xor / LDS / __syncthreads
```

- **Host ABI**は既存のまま。Kernel選択、バッファ所有、CUDA/HIPモジュール境界等に変更を加えない。
- **Algorithm Layer**は「何をするか」「どの順番か」を記述する。GEMM、Attention、GDNで共有しない数学的処理を無理に抽象化しない。
- **Tile Layer**は「どのlaneが何を持つか」「どうload/storeするか」「どの行列命令を実行するか」を実装する。まずは**gfx1201固定の小さいAPI**で十分。
- **Hardware Layer**はHIP builtin・vector型・LDS等を保持する。`__device__ __forceinline__`およびコンパイル時テンプレートで解決する。

### 1.2 Tritonから採用する部分・採用しない部分

| Tritonに近づけたい要素 | Shelfでの実現方法 | 初期スコープ |
|---|---|---|
| tile単位のload/store | masked vector load、store helper、静的範囲 | 必要な形だけ |
| 明示的なlayout | `WmmaLinearMap`、`PagedAttentionWaveMap`のような固定mapping型 | 必須 |
| 行列積 `dot` | gfx1201 BF16 WMMAの薄いラッパー | 必須 |
| warp/wave reduce | 既存のshuffle順を保った専用関数 | Attention段階で必要 |
| tensor形状 | `constexpr`なTileサイズとレジスタfragment型 | 必須 |
| Layoutの自動変換・融合最適化 | 実装しない | 対象外 |
| Triton言語、JIT、ランタイムIR生成 | 導入しない | 対象外 |

「Tritonと同じ名前を並べる」より、**GPU上のレイアウトを関数の外部契約として表現する**ことを重視する。`threadIdx.x`/`lane & 15`などが演算本体のあちこちに現れないことを目標とする。

### 1.3 提案するファイル配置

初期状態では既存の`src/backend/hip/include/shelf/backend/hip/kernels/detail/`を活用する。これらは**HIPバックエンド内部でのみ使用するprivate実装ヘッダ**として扱う（ビルドのincludeパスにあることと、外部APIとして安定保証することは別）。

```text
src/backend/hip/
  include/shelf/backend/hip/kernels/
    detail/
      vector_io.h                  # 既存
      tile/
        wave32.h                   # 新規: lane/wave/warp reduce
        bf16_wmma_gfx12.h          # 新規: reg fragmentとWMMA builtin
        masked_io.h                # 必要が出た場合だけ
    gemm/
      linear_bf16.h                # 既存: 公開ランチャー/Argsを維持
      linear_bf16_config.h         # 既存: config ABIを維持
      detail/
        linear_bf16_device.h       # 新規: GEMM固有のlayout・処理関数
    attention/
      paged_attention.h            # 既存: 公開契約を維持
      detail/
        paged_decode_device.h      # 第二段階以降で追加。分解できた時点で分割可
  kernels/
    gemm/linear_bf16.hip           # 薄いentry + 段階処理 + host launcher
    attention/attention_paged_decode.hip
    gdn/gdn_recurrence.hip         # 当面変更しない
```

**ファイル数を増やすこと自体は目的ではない。** 最初は同じ`.hip`内のprivate関数で抽出し、コンパイル・性能を確認してからheaderへ移してよい。`tile/`はGEMM側で「少なくとも1つの明確な責務」が安定した段階で作る。

---

## 2. Phase 0 — 変更前のBaselineと不足テストを確保

### 2.1 正本と環境を固定する

1. ZIP展開物は保管し、実作業は別ブランチ/別worktreeで行う。
2. `git rev-parse HEAD`（Git cloneの場合）、ROCmバージョン、HIP compiler、実GPU、CMake preset、コンパイルオプションを記録する。
3. 既存文書 `docs/architecture/dependency-rules.md`、`docs/architecture/paged-kv-abi.md`、`docs/architecture/rebuild-plan.md`、`docs/architecture/porting-provenance.md`を読む。
4. 比較用ログは例えば`docs/engineering/hip_tile_refactor/`またはGit管理外のbaselineディレクトリに保存し、測定条件と紐付ける。

CPU-onlyチェック:

```bash
cmake --preset cpu
cmake --build --preset cpu
ctest --test-dir build/cpu --output-on-failure
```

HIPのDebug/Release（ROCmが利用できる実機で実施）:

```bash
# 次のいずれかでROCm rootを解決できること:
#   export ROCM_PATH=/path/to/rocm
#   または cmake --preset hip -DSHELF_ROCM_ROOT=/path/to/rocm
cmake --preset hip
cmake --build --preset hip
ctest --test-dir build/hip --output-on-failure

cmake --preset hip-rel
cmake --build --preset hip-rel
ctest --test-dir build/hip-rel --output-on-failure
```

`cmake/RocmRoot.cmake`はROCm rootからClangを設定し、`cmake/Dependencies.cmake`がHIPを有効化する。`CMAKE_HIP_ARCHITECTURES=gfx1201`はpresetで指定済み。**性能測定は`hip-rel`を用いる**（HIPの`-O3`はDependencies.cmakeで付加される）。

主要CTest名は`unit.hip_gemm_bf16`、`unit.hip_attention`、`unit.hip_gdn`、`unit.hip_step_executor`、`unit.hip_elementwise_kernels`、`unit.hip_device`、`architecture.boundaries`。CMake定義で確かめること。

```bash
ctest --test-dir build/hip --output-on-failure -R \
  '^(unit\.hip_gemm_bf16|unit\.hip_attention|unit\.hip_gdn|unit\.hip_step_executor|architecture\.boundaries)$'
```

**環境制約:** ROCmやGPUを利用できない環境ではCPUテストしか評価できない。GPU未実施を「合格」として扱わない。

### 2.2 GPUのベンチマークを取る

既存 `tools/bench_gemv_hip.cpp` は `bench_gemv_hip <gpu> <rows>` 形式で、5つのGEMM configを実形状で計測する。

```bash
./build/hip-rel/tools/bench_gemv_hip 0 1 | tee baseline_gemm_rows1.txt
./build/hip-rel/tools/bench_gemv_hip 0 4 | tee baseline_gemm_rows4.txt
```

- 既存ベンチには大きな`lm_head`行列が含まれる。GPUメモリ不足なら小さな形状だけを測る別fixtureを追加し、測れなかった形状を明記する。
- `rows=1`と`rows>=2`では実際に選択されるconfigが異なる。**WMMA単体の結果とStepExecutorの実選択経路の結果を別々に記録する**。
- 実行前warmup、同一stream・同一GPU・同一shape、A/B交互複数回、p50とp95を記録する。CPU launch overheadを含む値とkernel単体を混同しない。
- VGPR、SGPR、LDS、scratch/spill、barrier、WMMA命令、カーネル数、実行時間を取得する。利用可能なROCm profiler/コードオブジェクト解析ツールのバージョンとコマンドも記録する。

### 2.3 先に埋めるべきテスト穴

現行の`tests/unit/test_hip_gemm_bf16.cpp`は主に`ExactRows`と`Wmma`を比較する。**`WmmaWide`、`WmmaKPartition`、`ExactRowsDot2`の十分な正確性確認は別途必要**。現行`tests/unit/test_hip_attention.cpp`はBF16 decode/splitを中心に検証し、FP8全経路はカバーしていない。

差分を開始する前に、少なくとも次の入力をgolden fixtureとして追加する（最初は既存APIを直接呼ぶテストでよい）。

| 対象 | 必須の検証軸 | 既存仕様を保つための注意 |
|---|---|---|
| BF16 GEMM WMMA | `rows=1,2,4,15,16,17,63,64,65`、`N`の16端数、`K`の16端数 | `input_row_stride`・`output_row_stride`を含める |
| GEMM Wide | 1/2/3/4 row tileが実際に使われるケース | `active`のswitch分岐をすべて通す |
| GEMM KPartition | 8 waveによるK分割とreduce、Kが分割粒度より短いケース | shared reduceの加算順序と唯一のbarrierを固定 |
| ExactRows / Dot2 | 行数1〜16、16超のchunk分割、K tail、BF16/F32出力 | Dot2と通常版の演算差は意図的。両者の出力をbyte一致とは要求しない |
| BF16 paged attention | head_dim=64/256、非full/full、通常/split/reduce、page境界、tree祖先 | 物理ABI・invalid pageフラグの共有に注意 |
| FP8 paged attention | スケール・LUT、通常/split、head_dim端数、ページ境界 | BF16と同じ数値経路としてまとめない |

既存の独立CPU/f64参照との数値チェックに加えて、**変更前HIP出力の固定入力スナップショット**を保存する。目的は同一variant内でのbefore/afterのbitwise比較。数値が変わったときは、原因究明なしに許容誤差を緩めない。

**Phase 0完了条件:** CPU/HIPテストのbaseline、全対象configのgolden、性能・資源記録を保存。未検証variantを一覧化できた。

---

## 3. Phase 1 — Kernelごとの「実行契約」を凍結する

コードをまだ動かさず、1ページの対応表を作る。Triton風にするうえで、**Thread/Wave/Memory配置の情報をアルゴリズム本体から追い出すための前提**である。

### 3.1 `linear_bf16.hip` の不変量

- `bf16_wmma_body<kRowTiles>`（現行110〜195行付近）は**wave32の32 threads/block**。WMMAの入力fragmentは`short8_t`、accumulatorは`float8_t`。1演算は16×16×16、`kRowTiles=1`または`4`。
- レーン配置は`lane_low4=lane&15`、`k_half=lane>>4`、`a_row`と`xrow`に直結。ここを `WmmaLinearMap`として固定する。
- `bf16_splitk_wmma_body`（213〜294行付近）は**8 wave / 256 threads**。`kb_begin/end`でK範囲を分担し、`__shared__ float red[8][8][32]`へ書き、**全blockのbarrierの後、wave0がwave順で加算**する。ここは別の`SplitKMap`。
- `gemm_bf16_exact_rows_body<kRows,kDot2>`（311〜388行付近）はWMMAでない経路。`__builtin_amdgcn_fdot2_f32_bf16`版とスカラ演算版を区別する。**WMMA用layoutに無理に載せない**。
- `launch_gemm_bf16()`（639〜699行付近）の`config.id`選択、各block/grid、streamを変更しない。
- `phaseshift_gemm_bf16_*`という名前の公開`extern "C"`カーネルが現存する。名称が旧プロジェクト由来でも、**今回の整理でリネームしない**。ヘッダの文字列定数と実体の一致を維持。

### 3.2 `attention_paged_decode.hip` の不変量

- 通常BF16 decode（約49〜281行）、BF16 split・reduce（約343〜657行）、head grouping付きreduce（約667〜802行）、通常FP8（約1057〜1311行）、FP8 split（約1357〜1610行）を**独立したアルゴリズムvariant**とみなす。
- 通常decodeは8 waves / 256 threads、split系は4 waves / 128 threads。これらに同じblock mappingを押し付けない。
- Q/K/Vのload、online softmax（`run_max`、`run_sum`、`acc`）、page traversal、wave間merge、出力書込みを段階として記録。
- `__shfl_xor`とshared mergeのreduction順序、waveごとのpage/token分担、barrier後のblock一様なinvalid/early-returnを記録。
- `PagedAttentionCommonArgs`等のサイズ/offsetに`static_assert`がある。**ArgsのABIは変更しない**。
- `docs/architecture/paged-kv-abi.md`の**rev2 SeparatePools**が正準。現行カーネルのK/V別poolを保持し、packing/ABIを変更する作業は別PRにする。実装・テスト・文書の間で差異が見つかった場合は、本作業中に静かに修正せず別課題として明記。

### 3.3 同期・共有メモリ契約表

最低限、各bodyについて次を列挙する:

```text
入口条件 / 全thread一様なreturnか / lane別のreturnか
threadIdx.x -> lane/wave/tile index
shared配列の宣言 / 書き手 / 最後の読み手
barrierの前後で可視化すべき領域
演算順（WMMA、split reduce、online softmax）
分岐（output dtype、head mask、Tail、Config、FullHead）
```

**重要:** barrierは最初からhelper内へ隠さない。カーネルbodyまたはトップレベルのアルゴリズム関数に**明示的に残す**。barrier前後で一部laneだけがreturnする構造を作らない。

**Phase 1完了条件:** `WmmaLinearMap`、`SplitKMap`、`PagedAttentionWaveMap`の計算式、各`__syncthreads()`、Shared/Regの寿命がレビュー可能。

---

## 4. Phase 2 — 最小Tile Primitiveの抽出

**一度に1個ずつ、意味を変えずに移す。** 抽出した関数はすべてHIPデバイス用で、ランタイムの抽象クラスやvirtual関数を導入しない。

### 4.1 最初に必要な3要素

**A. 既存WMMA命令のラッパー**（`detail/tile/bf16_wmma_gfx12.h`）:

```cpp
// APIイメージ。実装前にHIPコンパイラの型・ABIを検証する。
namespace shelf::hipk::tile {
using Bf16x8 = short __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ F32x8 dot_bf16_16x16x16(
    Bf16x8 a, Bf16x8 b, F32x8 acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
}
} // namespace shelf::hipk::tile
```

- 現在の`wide_tile()`とSplit-KのWMMA呼び出しを**同じ薄い関数**に差し替える。
- 型を包むだけでfragmentの余計なcopyを生成しないことを確認する。
- 命令内訳が変化した場合は抽象化を戻し、関数シグネチャから見直す。

**B. 静的lane/wave mapping**（まず`gemm/detail/linear_bf16_device.h`内）:

```cpp
// 例示。実コードで使用しているビット式をそのまま移す。
struct WmmaLinearMap {
    uint32_t lane;
    uint32_t output_lane;  // lane & 15
    uint32_t k_half;       // lane >> 4
    __device__ __forceinline__ static WmmaLinearMap current();
};

struct SplitKMap {
    uint32_t lane, wave;
    __device__ __forceinline__ static SplitKMap current();
};
```

- `WmmaLinearMap`と`SplitKMap`を無理に1型へ統合しない。
- `Tile<dtype,shape,layout>`の万能テンプレートは、具体的に2種類以上の再利用先が必要になるまで保留する。

**C. Masked load**（必要に応じて既存`detail/vector_io.h`も検討）:

- 現行`load_bf16x4_masked()`を**同じ境界条件・同じaligned path**のまま抽出。pointer演算やtail maskの意味を変更しない。
- 型幅は既存に合わせ`uint64_t`とし、抽象Tensorによる余計なレジスタ複製を避ける。
- まず`wide_load_a()`で使用し、他演算への適用は後段で判断する。

### 4.2 まだ作らないAPI

- `tile::autotune()`、`tile::reorder()`、`tile::broadcast_to_any_layout()`のような汎用オペレータ。
- GPU共有メモリの自動確保・暗黙コピー・暗黙barrierをする`Tile`。
- 動的`dtype`/`shape`で分岐する巨大なdevice dispatcher。
- backend/commonに置く、HIP/SYCL共用を名乗るGPU device関数。

**Phase 2完了条件:** WMMA builtin・lane mapping・masked loadの配置が一箇所に集約され、命令・数値・資源の比較が通過。

---

## 5. Phase 3 — `bf16_wmma_body`をTriton風に読む

まず通常WMMAだけを対象にし、wide・splitはそのままにする。**関数抽出のたびに1コミット、1テスト、短いA/B**を推奨する。

### 5.1 望ましい責務分割

| 段階関数（例） | 現行の責務 | 変更してはいけない点 |
|---|---|---|
| `gemm::make_tile_context()` | `blockIdx`、`lane`、K半分、tile先頭、tail判定 | uniform return条件とstride |
| `gemm::init_accumulators<kRowTiles>()` | `float8_t acc[kRowTiles]`初期化 | 全要素のゼロ・レジスタ使用量 |
| `gemm::prepare_row_views<kRowTiles>()` | `wrow`、`xrow[]`、`row_valid[]` | pointerの実配列・mask |
| `gemm::accumulate_wmma<kRowTiles>()` | 既存K tileループ、A/B fragment load、WMMA | `active`の1〜4分岐、K進行順 |
| `gemm::store_output<kRowTiles>()` | F32/BF16への書込み | レーンが担当するout index、丸め順、tail |

### 5.2 カーネル内の到達イメージ

```cpp
// アルゴリズムAPIの擬似コード。新しい __global__ の追加ではない。
template<uint32_t RowTiles>
__device__ __forceinline__ void bf16_wmma_body(/* 既存と同じargs */) {
    const auto ctx = gemm::make_tile_context<RowTiles>(/* args */);
    if (!ctx.block_active) return;          // block全体で一様

    auto acc = gemm::init_accumulators<RowTiles>();
    const auto rows = gemm::prepare_row_views<RowTiles>(ctx);

    gemm::accumulate_wmma<RowTiles>(ctx, rows, acc);
    gemm::store_output<RowTiles>(ctx, rows, acc);
}
```

実際の`accumulate_wmma`では既存の`gemm_bf16_wmma_ktiles<1/2/3/4>`を活かし、`tile::dot_bf16_16x16x16()`を呼ぶ。`active`のswitchを廃止して全RowTilesを計算するなどの変更はしない。

### 5.3 抽出順

1. `store_output`だけを抽出し検証。
2. `make_tile_context`と`prepare_row_views`を抽出し検証。
3. `init_accumulators`を抽出し検証。
4. 既存`gemm_bf16_wmma_ktiles`を`accumulate_wmma`の下に移し、WMMA命令ラッパーで実装し検証。
5. すべてPASS後、必要なら`gemm/detail/linear_bf16_device.h`へ移動する。

**Phase 3完了条件:** `bf16_wmma_body`を読むと「tile決定→load→dot反復→store」の流れがわかり、kernel symbol/launch/ISA/性能に意図しない変化がない。

---

## 6. Phase 4 — Wide / Split-K / ExactRowsへ展開

### 6.1 Wide: 同一algorithm、別の静的RowTiles

- `kRowTiles=4`のfragment配列、`active=1,2,3,4`の分岐を保持。
- Wide専用の巨大な別bodyを増やさない。**同じphase関数のコンパイル時variant**にする。
- Wideの入力row配列やaccumulatorを一括コピーする汎用`Tile`オブジェクトを作らない。

### 6.2 Split-K: `compute → stage → barrier → reduce → store`

例示:

```cpp
// 同じ256-thread kernel内の流れ。実際の引数は設計時に確定。
auto ctx = gemm::make_splitk_context(args);
if (!ctx.block_active) return;

auto partial = gemm::accumulate_splitk(ctx);        // waveごと
__shared__ float red[8][8][32];                 // 既存の形を保持
gemm::write_splitk_partials(ctx, partial, red);
__syncthreads();                               // 既存の同期位置を維持
if (ctx.wave == 0u)
    gemm::reduce_and_store_splitk(ctx, red);       // wave 0のみ
```

- `warp==0`の内側へbarrierを移動しない。
- `red[8][8][32]`とK領域分割の式は変更しない。
- Split-Kを複数GPU kernelに分けない（元は1 kernelであり、別種のattention split/reduceと混同しない）。
- 初期段階でreduceを汎用`tile::reduce`に統一しない。順序の維持を優先する。

### 6.3 ExactRows: WMMAでないので、別の算法関数として扱う

- `gemm_bf16_exact_rows_body<kRows,kDot2>`を、load / partial-dot / shuffle reduce / storeに分ける。
- FMA・`__builtin_amdgcn_fdot2_f32_bf16`の差、K tail、`__shfl_down`の順序を固定する。
- `phaseshift_gemm_bf16_exact_rows_1..16`という既存entry、`PS_BF16_EXACT_ROWS_ENTRY`マクロ、config数を変更しない。
- 「カーネル宣言が長く並ぶ」問題は、**各entryを薄いone-line wrapperに保つ**ことで解決する。entryの数を減らすためにランタイム特殊化へ変えない。

**Phase 4完了条件:** 5つのConfigIdがすべて同一公開launcherから利用可能で、各variantの分解が共通演算と固有演算に整理されている。

---

## 7. Phase 5 — 1,682行のPaged Attentionを段階的に分解

ここで初めてGEMM以外の利用者にTile APIを適用する。**最初はBF16の通常decodeだけ**。一度にBF16・FP8・split全体を統合しない。

### 7.1 段階関数（BF16通常decode）

| 順序 | 関数例 | 役割 |
|---|---|---|
| 0 | `attention::map_decode_task()` | `task`→`row/q_head/kv_head`、wave/lane |
| 1 | `attention::initialize_visible_range()` | `slot`・page table・ancestor処理、shared metadata初期化 |
| 2 | `attention::load_query_tile()` | Qをレジスタへ、`kFullHead` mask |
| 3 | `attention::visit_pages_bf16()` | page walk、KVタイルロード、score計算 |
| 4 | `attention::online_softmax_step()` | `run_max/run_sum/acc`の更新 |
| 5 | `attention::visit_ancestors_bf16()` | tree ancestorの位置指定アクセス |
| 6 | `attention::stage_wave_partials()` | LDSへのmax/sum/acc書込み |
| 7 | `attention::merge_wave_partials()` | 既存reduction順でwave間集約 |
| 8 | `attention::store_output()` | BF16/F32へ最終出力 |

**注意:** `initialize_visible_range()`と`stage_wave_partials()`の直後にあるbarrierはトップレベルで明示する。global invalidでreturnするときは、全blockが同じ判定を参照できる同期点の後に限る。

### 7.2 Attentionで再利用できるもの

- `tile::Wave32`のlane/wave表現はGEMMの`Wave32`と共有候補。
- `warp_sum_all`と`warp_max_all`は**shuffle順を保った**薄いhelperとして`tile/wave32.h`にまとめる候補。
- `load_query_tile`と`load_kv_tile`はmask付きloadの考え方を共有できるが、BF16とFP8の物理表現は別実装を残す。
- online softmaxの`run_max/run_sum/acc`はAttention固有。**GEMMの汎用dotに統合しない**。

### 7.3 BF16 split/reduce → FP8の順に展開

1. 通常BF16 decodeを分割・テスト・性能比較。
2. BF16 split（`kQPerKV`、4 waves）を分割する。`partials`のレイアウト・第一段の出力内容を変えない。
3. BF16 reduceとhead grouping付きreduceを分割する。**既存の二段GPU Launchを維持**し、workspaceの解釈を固定。
4. 通常FP8 decodeのLUT/dequant・per-token scale処理を保持したまま分割。
5. FP8 splitとBF16 splitに同じ責務の処理がある場合に限り、**実測で利益がある極小helperだけ**共通化する。
6. `PagedAttentionCommonArgs`、`PagedAttentionSplitArgs`、`PagedAttentionFp8SplitArgs`等の`sizeof/offsetof`を既存のまま維持する。

**Phase 5完了条件:** 旧1ファイルの巨大なbodyが、page traversal・online softmax・merge・storeという流れで追える。BF16/FP8/通常/split/reduceに機能欠落がなく、既存GPU Launch数を変えない。

---

## 8. Phase 6 — 真に共有できたAPIだけ固定する

GEMMとAttentionの両方で必要性を検証した部品のみ、HIPバックエンド内の`detail/tile/`に昇格する。

| 候補 | 昇格の判定 | 昇格しない条件 |
|---|---|---|
| `Wave32` / warp reduce | lane mapping・shuffle順を明示できる | 利用側の演算順が変わる |
| BF16 WMMA `dot` | GEMM内で1種類以上のvariantに使え、命令同一 | 型ラップでVGPRが増える |
| masked vector load | メモリ・alignment・mask意味が共通 | 条件を隠すため追加分岐や分岐発散が起きる |
| `RegTile` | コピーを生成しない固定小構造として有用 | 大規模なgenericテンソル化が必要 |
| block/barrier helper | 単なる名前付けだけなら不要 | barrierがbodyから見えなくなる |

**固定対象は「Triton互換API」ではなく「ShelfのHIP Tile API」である。** `tile::dot()`という命名を使っても、対応dtype・tile shape・layoutが実装から明らかになるようにする。例えばBF16 WMMAとFP8 decodeのscore内積を無理に同じ関数で処理しない。

将来、SYCLバックエンドを実装する際は、**アルゴリズム段階の名前と責務**を参考にするが、HIPのdevice型・builtin・LDS wrapperをSYCL側がincludeする構造にしない。

---

## 9. 各段階で実行する検証ゲート

### 9.1 ビルド・テスト

```bash
# 日常の境界検査（CPU-onlyでも実行可能）
cmake --build --preset cpu
ctest --test-dir build/cpu --output-on-failure

# HIP実機（変更のたびに）
cmake --build --preset hip --target shelf_backend_hip -j
ctest --test-dir build/hip --output-on-failure -R \
  '^(unit\.hip_gemm_bf16|unit\.hip_attention|unit\.hip_gdn|unit\.hip_step_executor|architecture\.boundaries)$'

# PR/Phaseの完了時はHIP全件およびReleaseで再テスト
cmake --build --preset hip-rel
ctest --test-dir build/hip-rel --output-on-failure
```

### 9.2 確認対象（before vs after）

| 観点 | 合格条件 |
|---|---|
| 公開API / ABI | `*Args`、config enum、`extern "C"` symbol、launcherの署名が不変 |
| 分岐・境界 | 全variant / row / K / N / head_dim / page / tailを検証 |
| 数値 | 固定入力で同一variantのbefore/after bitwise一致を第一目標。差分は理由を説明し承認を要する |
| Dispatch | `step_executor`の実選択config、kernel数、stream、grid/blockが不変 |
| 同期 | barrier数・到達条件・LDS寿命が不変。race/hangなし |
| GPU命令 | WMMA・dot2・shuffle・reduce順に意図しない差なし |
| 資源 | 新規spillなし。VGPR/SGPR/LDS増加を説明・許容判断 |
| 性能 | 同一条件のA/B交互測定で一貫した劣化なし。Kernel単体とStepExecutor両方 |
| 境界テスト | `architecture.boundaries`に違反なし。HIP型はbackend/hip内のみ |

性能が偶然速くなっても、**正確性・同期・ABIのゲートを省略しない**。逆に単なるrename・移動で性能低下を起こした場合は、説明なしに受け入れない。

### 9.3 切り戻し条件

- GPU hang / race / 新規NaN / 必須テストの失敗。
- barrier位置・レーン分岐が不明瞭になった。
- 新規spill、意味のあるVGPR増加、継続的なlatency悪化。
- 汎用抽象化を理解するために、元の実装以上に複雑な知識が必要になった。
- 既存`launch_*` APIや実際の処理経路を変えないと成立しなくなった。

発生した場合は**最後に加えた1抽象化/1関数抽出だけを戻す**。反射的にGPUアルゴリズムの最適化を追加して性能を埋め合わせない。

---

## 10. 推奨PR構成と完了定義

| PR | 変更範囲 | 完了ゲート |
|---|---|---|
| PR-0 | Baselineログ、実行契約表、足りないGEMM/Attentionテスト | 全variantのテスト計画と比較基準が存在 |
| PR-1 | GEMMのWMMA命令ラッパー、layout、小さなmasked-load抽出 | 数値・ISA・資源・速度の維持 |
| PR-2 | `bf16_wmma_body`を段階関数へ分割 | 通常/Wideで読みやすいbody、完全回帰なし |
| PR-3 | Split-K・ExactRows/Dot2の処理関数分割 | config分岐・barrier・公開symbol維持 |
| PR-4 | Attention BF16通常decodeの分割 | 通常decode/BF16 parity、page・tree境界 |
| PR-5 | Attention BF16 split/reduce、FP8の順次分割 | workspace・Launch数・FP8 scale semantics維持 |
| PR-6（任意） | 実証できた`tile/` primitiveの共通化 | 第二利用者と性能根拠が揃う |

**各PRには**コード差分、変更前後の固定入力出力比較、CTest結果、Release性能、GPU資源・命令差分、残る未検証範囲を添付する。

### 最終的な「読みやすさ」の基準

`gemm/linear_bf16.hip`のWMMA bodyでは、次の流れを上から読めること。

```text
タイル位置確定 → 行列fragment読み込み → WMMA dot反復
→ (必要なvariantのみreduce) → dtype別store
```

`attention/attention_paged_decode.hip`では次の流れを上から読めること。

```text
担当row/headと可視範囲決定 → Queryをレジスタへload
→ page/KV走査＋online softmax → wave間merge → output store
```

そして、**処理が分割されていてもGPU実行は従来の単一カーネルまたは従来から存在するsplit/reduce構成のまま**であること。これがShelfで目指す「Tritonのような書き方・HIPネイティブの実行」の完成条件である。

---

## 補足: ZIP確認時点の検証状況

- 添付ZIPの`cmake --preset cpu` → `cmake --build --preset cpu` → `ctest --test-dir build/cpu --output-on-failure`を実行し、**既存CPUテスト7件が全件PASS**した。
- この文書の作成環境ではROCm/HIP compilerとGPUが使えないため、**HIPのビルド・GPU数値比較・性能・ISA検証は未実施**。上記GPU工程は実機での実施手順であり、合格済みとは記さない。
- ソースコードへの変更は行っていない。本書はリファクタリング実装のための手順書である。

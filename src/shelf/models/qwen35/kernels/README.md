# Qwen3.5 kernels

Qwen35 固有 kernel（correctness / dflash2）の配置と分割の規約。
model 非依存の optimized kernel は `src/shelf/kernels/optimized/` を正本とする。

```text
kernels/
├── correctness/
│   ├── model_dispatch_correctness.hip   # 唯一の __global__ dispatch + KernelId switch + launcher
│   ├── detail/
│   │   ├── dispatch_env.h               # ModelDispatchEnv / ResolvedValue / resolve_* / load-store helper
│   │   ├── elementwise.inc              # SILU / SIGMOID / SCALE / MUL / RESIDUAL_ADD / SWIGLU / SPLIT / CONCAT
│   │   ├── normalization.inc            # RMS_NORM / L2_NORMALIZE
│   │   ├── linear.inc                   # LINEAR_BF16 / LINEAR_FP8_FINEGRAINED / ACTIVATION_QUANTIZE_FP8_FINEGRAINED
│   │   ├── attention.inc                # ROPE / KV_APPEND / PAGED_ATTENTION
│   │   ├── gdn.inc                      # STATEFUL_CAUSAL_CONV1D / GDN_RECURRENCE
│   │   ├── embedding.inc                # EMBEDDING_LOOKUP
│   │   └── output.inc                   # OUTPUT_GATHER / SAMPLING
│   └── standalone/                      # kernel 単体テスト用の参照実装（E2E correctness とは別系統）
optimized kernel（model 非依存）は `src/shelf/kernels/optimized/` に
移動した。Qwen35 が使う最適化済み実装はここから参照する。
```

linear の family-specific KernelConfig 型は
`include/shelf/kernels/optimized/linear/config.h` に置く。

## 分割ルール

- `KernelId` は runtime が「何を実行するか」を表す論理IDであり、ファイル構成を決めない。
- ソースファイルは「一緒に理解・変更・最適化される実装単位」で分ける。
- 1 ファイルに複数の `__global__` と複数の launcher を置いてよい。
- dtype はソース所有境界ではなく variant。同じ最適化問題の BF16 / FP8 は同じファイルに置く。
- `1 KernelId = 1 file`、`1 __global__ = 1 file`、`1 dtype = 1 file` は原則として避ける。
- 基本単位は `1 optimization problem ≒ 1 implementation family ≒ 1 source file`。
- correctness だけは `1 model reference executor = 1 translation unit` とし、
  `model_dispatch_correctness.hip` が `detail/*.inc` を include して単一 TU を構成する
  （RDC を要求せず、device helper の共有と KernelId switch の一箇所化を両立する）。
- `common/` に operator 実装を逃がさない。共有部品だけを `optimized/detail/` に置く。
- fused kernel・fusion matcher は置かない（現在 1 dispatch = 1 kernel）。

## KernelConfig

linear optimized path の physical variant 選択は次の 4 層で分担する。

```text
KernelId
    = semantic operation（何を計算するか）
KernelConfig
    = compile 済み physical implementation の指定（どの実装を起動するか）
Selector
    = runtime facts -> KernelConfig（どの Config を使うか）
Launcher
    = KernelConfig -> template specialization（Config に対応する __global__ を起動する）
__global__
    = algorithmic family（計算そのもの）
```

- `KernelConfig` は family-specific な小さな POD descriptor とし、`ConfigId` で実在の
  specialization を指定する。任意の tile サイズ等を並べた「自由入力型」にはしない。
  これにより binary に存在しない組み合わせを表現できない。
- `KernelConfig` は launch 時の transient value である。`Program` / `DispatchBinding` /
  `KernelId` などの runtime Program ABI には保存しない。`actual_rows`、batch topology、
  pointer alignment などは runtime で変わるため。
- Selector が config 選択の single source of truth である。launcher は shape policy を
  再判定しない。launcher に残してよいのは引数の整合性チェックと Config から
  specialization への dispatch だけである。
- Selector は optimized 可否を `optional<KernelConfig>`（無ければ fallback）で表す。
- `1 KernelId = 1 file` ではない。`1 Config = 1 source __global__` でもない。
- 同じ処理の compile-time variant は `__global__` をコピーせず、template specialization に
  まとめる。別 algorithmic stage（例: K partition と reduction、exact-rows GEMV）は
  別 `__global__` family を許可する。
- 依存方向は `runtime selector -> kernel config -> kernel launcher` であり、kernel 側から
  runtime selector を include しない。

## public launcher

optimized 側の外部境界。semantic operator / optimization family ごとに置く。
launcher は Selector が渡した `KernelConfig` に従って specialization を起動するだけで、
shape / dtype / algorithm variant の選択は Selector（runtime）で行う。

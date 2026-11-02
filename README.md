# Shelf

AMD RDNA4 GPU向けの単一ハードウェア推論ランタイムの研究プロジェクト。

目的は一つ:

> 単一ハードウェア（AMD RDNA4 / ROCm / HIP / Linux）で、最速の推論ランタイムを作る。

汎用性は目的ではない。性能が目的。

## Current scope

- ハードウェア: AMD RDNA4（gfx1201、R9700）/ ROCm / HIP / Linux
- Runtime family: Qwen3.5 系 runtime（コード上は `src/shelf/models/qwen35/`）
- 主要 performance target: **Qwen3.8-27B の HF FP8（`models/Qwen3.8-27B-FP8`）と BF16**
- Speculative decode: **DFlash2** drafter（Qwen3.8-27B target + DFlash2）
- KVキャッシュ: paged KV、BF16 / FP8_E4M3
- 対応 dtype: BF16 / FP8 E4M3 のみ（weight・KV ともにこの 2 つ）

GPU-MCU backend の実装は `exp/gpu-mcu` ブランチへ退避しており、この branch には
含まれない。

runtime の namespace は `qwen35` のままである。Qwen3.8-27B を「別 family へ rename
済み」ではない。コード上の Qwen3.5 runtime family を基盤として、現在の主要
performance target に Qwen3.8-27B の HF FP8 / BF16 model を使っている。

機能追加や性能値の正本は README ではなく各 docs を参照する。

## Current execution

- correctness mode + Qwen-local optimized auto dispatch
- `PrimitiveGraph` → `ProgramSet` → Qwen35 Executor
- continuous batching（prefill / decode mixed）
- DFlash2 speculative decode
- pre-quantized model loading（FP8 E4M3 / self-contained quantized safetensors）

## Server

`shelf-server` は aiohttp による OpenAI Chat Completions adapter である。
利用方法は [docs/user/server.md](docs/user/server.md)、Server の責務境界は
[docs/developer/architecture.md](docs/developer/architecture.md)、developer 向け入口は
[docs/developer/server_status.md](docs/developer/server_status.md) を参照。prefix cache の
protocol contract は [docs/developer/prefix_cache.md](docs/developer/prefix_cache.md) を参照。

## Entrypoints

| 名 | 用途 |
|---|---|
| `shelf-compute` | 単一request推論（BF16 / 量子化model、greedy、DFlash2 speculative decode） |
| `shelf-server` | OpenAI互換chat serving（[docs/user/server.md](docs/user/server.md)） |
| `shelf-bench` | prefill/decode E2E + kernel/operator micro-benchmarks（[docs/user/bench.md](docs/user/bench.md)） |

## Build

標準 build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSHELF_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --parallel
```

HIP graph execution mode を使う場合は別 option で指定する。

```bash
cmake -S . -B build -DSHELF_HIP_GRAPH=ON
```

要件・option 一覧: [docs/user/build.md](docs/user/build.md)。

量子化 model の format と読み込み経路は [docs/developer/quantization.md](docs/developer/quantization.md) を正本とする。

## Tests

required（自己完結: repository + ROCm + 1 GPUのみ）:

```bash
ctest --test-dir build -L required --output-on-failure
```

optional: external model を要する heavy test（DFlash2 gate /
TP / MTP weight contract / FP8 KV quality gate）。`SHELF_BUILD_OPTIONAL_TESTS=ON`
で build。品質 gate は `ctest --test-dir build -L quality` で実行する。
詳細: [docs/developer/testing.md](docs/developer/testing.md)。

## Documentation map

- [docs/user/](docs/user/README.md) — 現在ユーザーが使用する操作方法
- [docs/developer/](docs/developer/README.md) — 現在実装されている architecture / runtime / server / kernel contract
- [docs/perf/current.md](docs/perf/current.md) — 現在の性能値と再現条件
- [docs/perf/methodology.md](docs/perf/methodology.md) — どう測るか
- 過去に何を試してどう判断したか（PoC / Gate / 採否判断）は Git history を参照

## Contributors

- **jyohukuchan** — DFlash2 固定語彙 profile の pack / install tool、
  host / GPU テストと利用文書の追加
  （[docs/developer/dflash2.md](docs/developer/dflash2.md)）

## License

MIT — [LICENSE](LICENSE)

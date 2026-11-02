# Build

## 要件

- OS: Linux
- ROCm toolchain（clang++ / hipcc）。GCC・libstdc++は使用不可
- host libc++ 23: `/usr/lib/llvm-23/include/c++/v1`（`libc++-23-dev` / `libc++abi-23-dev`）。
  別箇所なら `-DPHASESHIFT_LIBCXX_LLVM_ROOT=<root>` で指定
- GPU: AMD Radeon AI PRO R9700（gfx1201）
- CMake >= 3.24
- Ninja
- OpenSSL（libcrypto）
- Python 3（`phaseshift-launcher`、`phaseshift-server`、E2E）
  - `phaseshift-server`: `aiohttp`、`transformers==5.14.1`
  - `phaseshift-launcher`: 標準ライブラリのみ
  - server test: `openai`（OpenAI Python client E2E）

`phaseshift-compute` 単体は Python 非依存。

server runtime dependency は次の 2 package のみである。

```bash
pip install 'aiohttp' 'transformers==5.14.1'
```

`transformers==5.14.1` は chat template / Qwen response parser の regression を避けるため固定する。

`rocm-sdk` CLIがある環境では `rocm-sdk path --root` でROCm rootを解決する。
なければ `-DPHASESHIFT_ROCM_ROOT=<path>` か環境変数 `ROCM_PATH` を使う。

## 手順

```bash
export ROCM_PATH="$(rocm-sdk path --root)"
ARCH=gfx1201

cmake -S . -B build-${ARCH} -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPHASESHIFT_ROCM_ROOT="${ROCM_PATH}" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_PATH}" \
  -DCMAKE_HIP_ARCHITECTURES="${ARCH}"
cmake --build build-${ARCH} --parallel
```

PhaseShiftが対象とするのは `gfx1201`（AMD Radeon AI PRO R9700）のみである。
`CMAKE_HIP_ARCHITECTURES` に他のarchを指定した場合、configureで `FATAL_ERROR` になる。

## options

| option | 既定 | 説明 |
|---|---|---|
| `PHASESHIFT_BUILD_TESTS` | ON | test suite build |
| `PHASESHIFT_BUILD_OPTIONAL_TESTS` | OFF | external model を要する optional / heavy test |
| `PHASESHIFT_BUILD_BENCHMARKS` | ON | benchmark executables（bench + E2E前提） |
| `PHASESHIFT_HIP_GRAPH` | OFF | HIP graph execution mode |
| `PHASESHIFT_LIBCXX_LLVM_ROOT` | `/usr/lib/llvm-23` | host libc++ 23 root |
| `CMAKE_HIP_ARCHITECTURES` | 必須 | 対象GPUアーキテクチャ（`gfx1201` のみ） |

外部 model を要する optional / external test の model directory は cmake option
ではなく環境変数で指定する。Qwen3.5 系の test は `PHASESHIFT_MODEL_DIR`、
MTP は `PHASESHIFT_MODEL_DIR_MTP`、DFlash2 draft は
`PHASESHIFT_MODEL_DIR_DFLASH2`、DFlash2 target は
`PHASESHIFT_MODEL_DIR_DFLASH2_TARGET`、TP は `PHASESHIFT_TP_MODEL_DIR` を使う。
指定が無い test は exit 77 で skip する。

`PHASESHIFT_HIP_GRAPH` は graph capture / replay による execution mode である。
有効にする場合:

```bash
cmake -S . -B build -DPHASESHIFT_HIP_GRAPH=ON
```

## 成果物

```text
build/phaseshift-compute
build/phaseshift-launcher
build/phaseshift-server
build/phaseshift-server-lib/
build/phaseshift-bench
build/phaseshift-model-host
```

`phaseshift-server` は Python 製の HTTP server である。起動時に model processor を load し、
`phaseshift-compute --serve-stdio` を subprocess として起動してから HTTP port を開く。

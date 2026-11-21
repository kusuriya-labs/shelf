# Build

## 要件

- OS: Linux
- ROCm toolchain（clang++ / hipcc）。GCC・libstdc++は使用不可
- host libc++ 23: `/usr/lib/llvm-23/include/c++/v1`（`libc++-23-dev` / `libc++abi-23-dev`）。
  別箇所なら `-DSHELF_LIBCXX_LLVM_ROOT=<root>` で指定
- GPU: AMD Radeon AI PRO R9700（gfx1201）
- CMake >= 3.24
- Ninja
- OpenSSL（libcrypto）
- Python 3（`shelf-server`、E2E）
  - `shelf-server`: `aiohttp`、`transformers==5.14.1`
  - server test: `openai`（OpenAI Python client E2E）

`shelf-compute` 単体は Python 非依存。

server の通常実行に必要な package は次のとおり。構造化出力には別途 `xgrammar==0.2.9` を Python 環境にインストールする。未導入の場合、通常 request は動作し、構造化出力 request は HTTP 400 で拒否する。

```bash
pip install 'aiohttp' 'transformers==5.14.1'
# 構造化出力を利用する場合
pip install 'xgrammar==0.2.9'
```

`transformers==5.14.1` は chat template / Qwen response parser の regression を避けるため固定する。

`rocm-sdk` CLIがある環境では `rocm-sdk path --root` でROCm rootを解決する。
なければ `-DSHELF_ROCM_ROOT=<path>` か環境変数 `ROCM_PATH` を使う。

## 手順

```bash
export ROCM_PATH="$(rocm-sdk path --root)"
ARCH=gfx1201

cmake -S . -B build-${ARCH} -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSHELF_ROCM_ROOT="${ROCM_PATH}" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_PATH}" \
  -DCMAKE_HIP_ARCHITECTURES="${ARCH}"
cmake --build build-${ARCH} --parallel
```

Shelfが対象とするのは `gfx1201`（AMD Radeon AI PRO R9700）のみである。
`CMAKE_HIP_ARCHITECTURES` に他のarchを指定した場合、configureで `FATAL_ERROR` になる。

## options

| option | 既定 | 説明 |
|---|---|---|
| `SHELF_BUILD_TESTS` | ON | test suite build |
| `SHELF_BUILD_OPTIONAL_TESTS` | OFF | external model を要する optional / heavy test |
| `SHELF_BUILD_BENCHMARKS` | ON | benchmark executables（bench + E2E前提） |
| `SHELF_HIP_GRAPH` | ON | HIP graph execution mode |
| `SHELF_LIBCXX_LLVM_ROOT` | `/usr/lib/llvm-23` | host libc++ 23 root |
| `SHELF_LOAD_LIBSTDCXX_FIRST` | ROCm 10 以上で ON、それ未満・不明時は OFF | ROCm C++ ライブラリ互換のため `libstdc++.so.6` を `libc++abi.so` より先にリンクする |
| `CMAKE_HIP_ARCHITECTURES` | 必須 | 対象GPUアーキテクチャ（`gfx1201` のみ） |

外部 model を要する optional / external test の model directory は CMake option ではなく環境変数（`SHELF_MODEL_DIR` 等）で指定する。環境変数の一覧は [env_vars.md](env_vars.md) を参照する。指定が無い test は exit 77 で skip する。

`SHELF_HIP_GRAPH` は graph capture / replay による execution mode で、既定で有効である。
無効にする場合:

```bash
cmake -S . -B build -DSHELF_HIP_GRAPH=OFF
```

実行時に verify graph や PLaMo 3 の graph のみを無効化する環境変数（`SHELF_GRAPH_VERIFY=0`、`SHELF_PLAMO_HIP_GRAPH=0`）は [env_vars.md](env_vars.md#4-hip-graphgdnoctave-kvplamo-3-制御) を参照する。Qwen3.5 系の DFlash2 target では、条件を満たす単一 request の 8 行または 10 行の greedy verify（tree verify を含む）を既定で graph 実行する。複数 request、stochastic verify、QSA / PLE history、TP schedule、および対応する metadata が揃わない tree verify は通常経路へ戻る。Octave KV cache の append と decode / verify attention も同じ graph に capture される。

## ROCm バージョンごとの注意とトラブルシュート

### ROCm 10.2 と hipBLASLt C++ ABI（`libstdc++` 先行読み込み）

Shelf 本体は Clang + libc++ でビルドするが、ROCm の C++ ライブラリ（`libhipblaslt` / `libtensilelite-host.so.1`）は実行時に `libstdc++.so.6` に依存する。`libc++abi.so` が `libstdc++.so.6` より先に RTTI シンボルを束縛すると、hipBLASLt 初期化時に ABI 不一致が発生する。

- **ROCm 10 以降（ROCm 10.2 等）**: `cmake/options.cmake` が ROCm のメジャーバージョンを自動検出し、`SHELF_LOAD_LIBSTDCXX_FIRST=ON` を既定値として `libstdc++.so.6` を `libc++abi.so` より先に直接リンクする。
- **既存の CMake cache を使い回した場合**: cache に古い `SHELF_LOAD_LIBSTDCXX_FIRST=OFF` が残っていると、起動時に stderr へ次のようなメッセージが出て hipBLASLt 経路（EXL3-INT8 prefill の A 経路等）が初期化エラーになる。
  ```text
  shelf: hipBLASLt C++ ABI 検査に失敗: ... libstdc++ を libc++abi より先に読み込んでください
  ```
  この場合は `-DSHELF_LOAD_LIBSTDCXX_FIRST=ON` を明示して再 configure・再ビルドする。
- **一時的な回避策**: 再ビルド前に動作確認だけ行いたい場合は、起動時に `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` を指定して `libstdc++` を先行読み込みできる（パスはディストリビューションの配置に合わせる）。
- **ROCm 10 未満 / バージョン不明環境**: `SHELF_LOAD_LIBSTDCXX_FIRST` の既定値は `OFF` である。hipBLASLt の INT8 heuristic が使えない場合や `HIPBLASLT_USE_ROCROLLER=1` が設定されている場合、EXL3-INT8 prefill は自動的に B 経路（自前 gfx12 iu8 WMMA）へフォールバックする。

### 起動時のその他の注意点

- **`libhipblaslt.so.1` が見つからない**: 実行ファイルは `libhipblaslt.so.1` に直接リンクされているため、欠落すると起動前に動的リンカエラーになる。`LD_LIBRARY_PATH` に `${ROCM_PATH}/lib` と `${SHELF_LIBCXX_LLVM_ROOT}/lib` が含まれているか、ビルド時の RPATH が有効か確認する。
- **デスクトップ表示と GPU を共有する環境（NixOS 等）**: 表示コンポジタが VRAM を消費するため、`--arena-gib auto` では 16 GiB カード等でメモリ上限に当たりやすい。`--arena-gib 14` のように明示指定する（[recipes/qwen3.8-27b-128k-16gib.md](recipes/qwen3.8-27b-128k-16gib.md)）。
- **複数 GPU 環境での device 選択**: `rocm-smi` と ROCr の番号が一致しないことがある。使用する GPU を固定したい場合は `rocm-smi --showproductname --showuniqueid` で UUID を確認し、`ROCR_VISIBLE_DEVICES=GPU-<UUID> HIP_VISIBLE_DEVICES=0` で絞り込む。

## 成果物

```text
build/shelf-compute
build/shelf-server
build/shelf-server-lib/
build/shelf-bench
```

`shelf-server` は Python 製の HTTP server である。起動時に model processor を load し、
`shelf-compute --serve-stdio` を subprocess として起動してから HTTP port を開く。

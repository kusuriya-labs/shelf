# Shelf

[日本語](README.md) | **English**

A lightweight inference engine for AMD RDNA4 GPUs (R9700).

Highlights:
- Decent accuracy
- Decent speed
- Decent concurrency
- Minimal dependencies

Practicality is the primary goal.

## Quick Start

Build:
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSHELF_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --parallel
```

Run:
```bash
./build/shelf-server \
    --model-dir models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2-FP8 \
    --prefix-cache-strategy radix \
    --max-seq-len 131072 \
    --kv-cache-dtype octave:c5v5 \
    --port 8123
```

- Fits 4 concurrent requests with a 131,072-token context window on a single R9700
- Runs at practical speeds with DFlash2 enabled

## Documentation

> Note: The linked documents under `docs/` and `AGENTS.md` are currently written in Japanese.

The canonical references for supported features, design specifications, and performance numbers live in [docs/](docs/README.md). See [AGENTS.md](AGENTS.md) for development philosophy and coding conventions.

### User Guides ([docs/user/](docs/user/README.md))

- [build.md](docs/user/build.md) — Requirements, build steps, CMake options, and ROCm troubleshooting
- [quantize.md](docs/user/quantize.md) — EXL3-INT8 (`mul24`) and Ngram4 / Ngram5 conversion, verification, and quality checks
- [compute.md](docs/user/compute.md) — `shelf-compute` (single-shot inference, `--serve-stdio`, DFlash2 / MTP / Prefix Cache)
- [server.md](docs/user/server.md) — `shelf-server` (OpenAI-compatible API, agent integration, structured outputs)
- [multi-gpu.md](docs/user/multi-gpu.md) — Multi-GPU execution via PP / EP / TP and per-model support
- [bench.md](docs/user/bench.md) — `shelf-bench` (`pp` / `tg` / `quality` / microbench)
- [env_vars.md](docs/user/env_vars.md) — `SHELF_*` environment variable reference (canonical)
- [recipes/](docs/user/recipes/) — Recipe index

### Developer, Performance, and R&D

- [docs/developer/README.md](docs/developer/README.md) — Developer documentation (architecture, runtime, quantization, kernels, and test contracts)
- [docs/perf/README.md](docs/perf/README.md) — Performance and benchmarks (current numbers, methodology, and quality evaluation)
- [docs/research/README.md](docs/research/README.md) — R&D records (experiment logs, adoption decisions, and rejection rationale)

## Contributors

- **jyohukuchan** — EXL3-INT8 development / EXL3-INT8 kernel optimization / structured outputs, etc.

## License

MIT — [LICENSE](LICENSE)

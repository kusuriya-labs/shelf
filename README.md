# shelf

Model非依存・Weight形式非依存・Hardware非依存の演算契約を持ち、各Backendで高速実行する推論ランタイム

## ディレクトリ構成

```
shelf/
├── include/shelf/
├── src/
│   ├── foundation/
│   ├── model/
│   ├── ir/
│   ├── artifact/
│   ├── quant/
│   ├── weights/
│   ├── compiler/
│   ├── api/
│   ├── backend/
│   │   ├── common/
│   │   ├── cpu/
│   │   ├── hip/
│   │   └── sycl/
│   ├── placement/
│   └── runtime/
├── apps/
├── tests/
├── benchmarks/
├── tools/
└── docs/
```

## ビルド

```bash
# CPU-only（CI・スケルトン）
cmake --preset cpu
cmake --build --preset cpu
ctest --test-dir build/cpu --output-on-failure

# HIP バックエンド（AMD ROCm / gfx1201・R9700 x4）
cmake --preset hip
cmake --build --preset hip
ctest --test-dir build/hip --output-on-failure
```

Phase 1–2 完了分: `foundation`（Status/Result, TensorShape, DataType, TensorId,
TensorSpec, bf16/fp16変換）+ `backend/common`（Device/BufferView 等の中立契約）+
`backend/hip`（Device/Stream/Event/DeviceBuffer・転送・依存同期）。
`backend/sycl` は SYCL ランタイム導入まで空（意図的に HIP で代替しない）。

- 依存ルール（機械検査される契約表): [docs/architecture/dependency-rules.md](docs/architecture/dependency-rules.md)
- 移植由来・ライセンス台帳（PhaseShift は自己制作のため表記不要 / XeStrata LGPL 方針）: [docs/architecture/porting-provenance.md](docs/architecture/porting-provenance.md)

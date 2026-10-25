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
│   │   ├── cpu/
│   │   └── sycl/
│   ├── placement/
│   └── runtime/
├── apps/
├── tests/
├── benchmarks/
├── tools/
└── docs/
```

## ビルド（スケルトン / CPU-only）

```bash
cmake --preset cpu          # configure: build/cpu, Ninja, SHELF_ENABLE_SYCL=OFF
cmake --build --preset cpu
ctest --test-dir build/cpu --output-on-failure
```

Phase 1 スケルトン: `foundation`（Status/Result, TensorShape, DataType, TensorId,
TensorSpec, bf16/fp16変換）+ 単位テスト + Architecture境界テスト。

- 依存ルール（機械検査される契約表): [docs/architecture/dependency-rules.md](docs/architecture/dependency-rules.md)
- 移植由来・ライセンス台帳（PhaseShift は自己制作のため表記不要 / XeStrata LGPL 方針）: [docs/architecture/porting-provenance.md](docs/architecture/porting-provenance.md)

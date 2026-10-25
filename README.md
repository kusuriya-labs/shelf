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

Phase 1–2 完了 + PR-004/PR-005: `foundation`（Status/Result, TensorShape, DataType,
TensorId, TensorSpec, bf16/fp16変換）+ `backend/common`（Device/BufferView 等の中立契約）+
`backend/hip`（Device/Stream/Event/DeviceBuffer・転送・依存同期）+
`artifact/safetensors`（BF16 Reader・Raw Tensor Catalog）+ `weights/catalog`
（Canonical Tensor Catalog・マルチシャード解決）+ `ir`（Primitive 契約と Graph 検証:
SSA・shape 代数・state 宣言整合）+ `backend/cpu`（CPU 参照 Executor・合成モデル
Golden Fixture・KV state 更新検証）。
`backend/sycl` は SYCL ランタイム導入まで空（意図的に HIP で代替しない）。

### Qwen3.5-4B（BF16）が shelf で動作（CPU 参照路径・遅いが正確）

```bash
./build/app/apps/shelf-run --model ~/work/models/Qwen3.5-4B \
    --ids 760,6511,314,9338,369 --steps 16 --out shelf-out
```

- `model/qwen35`: config.json → per-layer Graph lowering（hybrid 3:1 linear/full、
  GDN recurrent・conv state・GQA+sigmoid gate・partial RoPE・tied embeddings）
- 検証: f64 oracle（`~/work/references/qwen35-f64-oracle`、transformers op-for-op）に対し
  prefill 最終位置 logits top-1 一致・最大差 <5e-5（f32 CPU）
- `tools/parity_logits.py` が parity チェック

- 依存ルール（機械検査される契約表): [docs/architecture/dependency-rules.md](docs/architecture/dependency-rules.md)
- 移植由来・ライセンス台帳（PhaseShift は自己制作のため表記不要 / XeStrata LGPL 方針）: [docs/architecture/porting-provenance.md](docs/architecture/porting-provenance.md)

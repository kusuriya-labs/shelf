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

# SYCL バックエンド（AdaptiveCpp 経由 / gfx1201 GPU 向け AOT）
ACPP_PATH=/opt/acpp cmake --preset sycl -DSHELF_SYCL_ROOT=$ACPP_PATH
cmake --build --preset sycl
ctest --test-dir build/sycl --output-on-failure

# SYCL バックエンド（OpenMP デバイス = CPU 上での API/カーネル統合検証）
cmake --preset sycl-cpu -DSHELF_SYCL_ROOT=<AdaptiveCpp prefix>
cmake --build --preset sycl-cpu
ctest --test-dir build/sycl-cpu --output-on-failure
```

Phase 1–2 完了 + PR-004/PR-005: `foundation`（Status/Result, TensorShape, DataType,
TensorId, TensorSpec, bf16/fp16変換）+ `backend/common`（Device/BufferView 等の中立契約）+
`backend/hip`（Device/Stream/Event/DeviceBuffer・転送・依存同期）+
`artifact/safetensors`（BF16 Reader・Raw Tensor Catalog）+ `weights/catalog`
（Canonical Tensor Catalog・マルチシャード解決）+ `ir`（Primitive 契約と Graph 検証:
SSA・shape 代数・state 宣言整合）+ `backend/cpu`（CPU 参照 Executor・合成モデル
Golden Fixture・KV state 更新検証）。
`backend/sycl` は `backend/hip` の SYCL 移植（AdaptiveCpp でビルド）。
hip_api→sycl_api（in-order queue/USM）、hipk→syclk（nd_range/nd_item +
local memory + sub_group collective）、WMMA/gfx12 組み込み命令はスカラー
FMA タイルへフォールバック済み。実行は 32 レーン sub_group（AMD wave32）
必須のカーネル群と、CPU でも動く純 elementwise 群に分かれる。
GPU AOT（`hip:gfx1201`）は AdaptiveCpp がビルドされた LLVM と ROCm
device library の LLVM メジャーが一致する環境が必要（このマシン:
AdaptiveCpp 25.10/LLVM 21 vs pip ROCm 10.0/LLVM 23 → 不一致で AOT/JIT
不可。`sycl-cpu` プリセットは動作）。

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
- Paged KV / prefix 共有（radix 相当）ABI: [docs/architecture/paged-kv-abi.md](docs/architecture/paged-kv-abi.md)
- 移植由来・ライセンス台帳（PhaseShift は自己制作のため表記不要 / XeStrata LGPL 方針）: [docs/architecture/porting-provenance.md](docs/architecture/porting-provenance.md)

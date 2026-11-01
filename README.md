# shelf

現状 > AMD RDNA4 GPU向けの単一ハードウェア推論ランタイム

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
│   │   └── hip/
│   ├── placement/
│   └── runtime/
├── apps/
├── tests/
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

Python 面（`apps/server`・OpenAI 互換 HTTP）は任意コンポーネントで、実行ファイルの
ビルドには結線しない。テストは ctest の `serve.python` から実行する
（`pytest` / `aiohttp` が無い環境は skip）。

```bash
python3 -m pytest apps/server/tests -q
# または
ctest --test-dir build/cpu -R serve.python
```

# Shelf

AMD RDNA4 GPU（R9700）向けの軽量推論エンジン

特徴
- まあまあの精度
- まあまあの速度
- まあまあの並列性
- 依存が少ない

実用性を第一目標とする

## クイックスタート

ビルド
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSHELF_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$(rocm-sdk path --root)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build --parallel
```

起動
```bash
./build/shelf-server \
    --model-dir models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw \
    --dflash2-model-dir models/Qwen3.8-27B-DFlash2-FP8 \
    --prefix-cache-strategy radix \
    --max-seq-len 131072 \
    --kv-cache-dtype octave:c5v5 \
    --port 8123
```

- R9700 1枚で 4並列 コンテキストサイズ 131072 を確保
- DFlash2 有りで実用的な速度で動作

## Contributors

- **jyohukuchan** — EXL3-INT8 開発 / EXL3-INT8 カーネル最適化 / 構造化出力 etc

## License

MIT — [LICENSE](LICENSE)

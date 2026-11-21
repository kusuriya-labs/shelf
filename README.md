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

## ドキュメント

対応機能・設計仕様・性能値の正本は [docs/](docs/README.md) に置いている。開発方針やコーディング規約は [AGENTS.md](AGENTS.md) を参照。

### 利用者ガイド（[docs/user/](docs/user/README.md)）

- [build.md](docs/user/build.md) — 要件・ビルド手順・CMake オプション・ROCm トラブルシュート
- [quantize.md](docs/user/quantize.md) — EXL3-INT8（mul24）・Ngram4 / Ngram5 の変換・検証・品質確認
- [compute.md](docs/user/compute.md) — `shelf-compute`（単発推論・`--serve-stdio`・DFlash2 / MTP / Prefix Cache）
- [server.md](docs/user/server.md) — `shelf-server`（OpenAI 互換 API・エージェント接続・構造化出力）
- [multi-gpu.md](docs/user/multi-gpu.md) — PP / EP / TP によるマルチ GPU 実行とモデル別対応
- [bench.md](docs/user/bench.md) — `shelf-bench`（`pp` / `tg` / `quality` / microbench）
- [env_vars.md](docs/user/env_vars.md) — `SHELF_*` 環境変数リファレンス（正本）
- [recipes/](docs/user/recipes/) — レシピ一覧

### 開発・性能・R&D

- [docs/developer/README.md](docs/developer/README.md) — 開発者向け（アーキテクチャ・ランタイム・量子化・カーネル・テスト契約）
- [docs/perf/README.md](docs/perf/README.md) — 性能・ベンチマーク（現在の性能値・測定手順・品質評価）
- [docs/research/README.md](docs/research/README.md) — R&D 記録（実験経緯・採否判断・棄却根拠）

## Contributors

- **jyohukuchan** — EXL3-INT8 開発 / EXL3-INT8 カーネル最適化 / 構造化出力 etc

## License

MIT — [LICENSE](LICENSE)

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
- [recipes/qwen3.8-27b-128k-16gib.md](docs/user/recipes/qwen3.8-27b-128k-16gib.md) — 16 GiB VRAM での 128Ki コンテキスト運用例

### 開発者向け（[docs/developer/](docs/developer/README.md)）

- [architecture.md](docs/developer/architecture.md) — レイヤー構成・ネームスペース・CMake ターゲット・責務境界
- [runtime.md](docs/developer/runtime.md) — 実行経路と state ownership
- [quantization.md](docs/developer/quantization.md) / [ngram_quantization.md](docs/developer/ngram_quantization.md) / [canonical_soa_payload.md](docs/developer/canonical_soa_payload.md) — 量子化フォーマット・Ngram4 / Ngram5・binary layout 契約
- [multi-gpu.md](docs/developer/multi-gpu.md) / [tensor_partition.md](docs/developer/tensor_partition.md) — PP / EP / TP 設計と TP weight materialization 契約
- [kernels.md](docs/developer/kernels.md) / [dflash2.md](docs/developer/dflash2.md) / [prefix_cache.md](docs/developer/prefix_cache.md) / [testing.md](docs/developer/testing.md) — カーネル契約・投機デコード・Prefix Cache・テスト構成

### 性能・ベンチマーク（[docs/perf/](docs/perf/README.md)）

- [current.md](docs/perf/current.md) — 現在再現可能な性能ベースライン
- [methodology.md](docs/perf/methodology.md) — 測定原則・GPU 排他・品質比較手順
- [exl3-int8-prefill-quality.md](docs/perf/exl3-int8-prefill-quality.md) / [quality-bf16/README.md](docs/perf/quality-bf16/README.md) / [kv-octave.md](docs/perf/kv-octave.md) — EXL3-INT8・BF16 基準品質・Octave KV の実測

### R&D 記録（[docs/research/](docs/research/README.md)）

- 日付ごとの実験経緯・採否判断・棄却根拠の正本

## ディレクトリ構造

| ディレクトリ | 役割 |
| --- | --- |
| `include/shelf/` | パブリックヘッダ（`core/`, `inference/`, `io/`, `kernels/`, `models/`, `quantization/`, `runtime/`, `weights/`） |
| `src/shelf/` | ランタイム・モデル・量子化・HIP カーネルの実装（`include/shelf/` と同一階層） |
| `src/apps/` | 3 つのエントリポイント（`compute/`, `server/`, `bench/`）および共通コード（`common/`） |
| `cmake/` | CMake のオプション・ターゲット・アプリ・テスト定義（`options.cmake`, `rocm_root.cmake`, `targets.cmake`, `apps.cmake`, `tests.cmake`） |
| `tests/` | テストスイート（`unit/`, `correctness/`, `kernels/`, `quality/`, `server/`, `benchmarks/`, `fixtures/`, `data/`, `support/`） |
| `tools/` | 量子化変換（`quantize/`）・品質評価（`quality/`, `quality_gate.py`）・カーネル検証およびフィクスチャ生成スクリプト |
| `docs/` | ドキュメント一式（`user/`, `developer/`, `perf/`, `research/`, `idea/`） |
| `vendor/` | 同梱する外部依存（`nlohmann/`, `BetterBench/`） |

## Contributors

- **jyohukuchan** — EXL3-INT8 開発 / EXL3-INT8 カーネル最適化 / 構造化出力 etc

## License

MIT — [LICENSE](LICENSE)

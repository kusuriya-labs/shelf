# Shelf 利用者ガイド（docs/user）

Shelf は AMD RDNA4 GPU（`gfx1201`）向けの推論ランタイムである。初めて利用する場合は、以下の順に進める。

## 導入フロー

1. **要件確認とビルド** — [build.md](build.md)
   - Linux・ROCm clang/hipcc・host libc++ 23・`gfx1201` GPU を準備し、`shelf-compute` / `shelf-server` / `shelf-bench` をビルドする。
   - ROCm 10 以降（ROCm 10.2 等）の hipBLASLt / `libstdc++` ABI の注意点も [build.md](build.md#rocm-バージョンごとの注意とトラブルシュート) に記載している。
2. **モデルの入手または量子化変換** — [quantize.md](quantize.md)
   - BF16・HF FP8・Quark MXFP4 に加え、Shelf の **EXL3-INT8（mul24、`0x00DCD12D`）** checkpoint をそのまま読み込める。
   - 公開済み EXL3-INT8 モデル（例: [Qwen3.8-27B-EXL3-INT8-3.05bpw](https://huggingface.co/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-3.05bpw)）を使うか、`tools/quantize/shelf_quantize.py` で BF16 checkpoint から変換する。N-gram table を持つモデルでは `--ngram ngram4` / `ngram5` も指定できる。
3. **メモリ設定の確認** — 下節「対応ハードウェアとメモリ見積もり」および [recipes/qwen3.8-27b-128k-16gib.md](recipes/qwen3.8-27b-128k-16gib.md)
   - GPU の VRAM 容量（16 GiB / 32 GiB / マルチ GPU）に応じて `--arena-gib`、`--kv-cache-dtype`、`--prefill-chunk`、`--max-seq-len` を決める。
4. **推論の起動** — [compute.md](compute.md) / [server.md](server.md) / [multi-gpu.md](multi-gpu.md)
   - 単発実行・標準入出力エンジンは `shelf-compute`、OpenAI 互換 API（Chat Completions / Responses API、OpenCode / Codex CLI 等からの接続）は `shelf-server` を使う。
5. **品質確認とベンチマーク** — [quantize.md](quantize.md#品質確認) / [bench.md](bench.md)
   - 変換したモデルや KV 量子化設定の精度は `shelf_quantize.py quality`・`tools/quality/kld-vs-bf16.sh`・`shelf-bench quality` で BF16 基準との KLD / top-1 一致率を確認する。速度は `shelf-bench pp` / `tg` で測る。
6. **環境変数とトラブルシュート** — [env_vars.md](env_vars.md) / [build.md](build.md#rocm-バージョンごとの注意とトラブルシュート)
   - EXL3-INT8 の prefill 経路切替（A: hipBLASLt / B: 自前 WMMA）や診断フラグはすべて [env_vars.md](env_vars.md) に一元化している。

---

## 対応ハードウェアとメモリ見積もり

### 対応ハードウェア

Shelf がビルド・実行対象とするのは **`gfx1201`（RDNA4）のみ**である（`cmake/options.cmake` で `CMAKE_HIP_ARCHITECTURES=gfx1201` 以外は configure エラーになる）。

| 構成の目安 | 想定 GPU | 推奨される使い方 |
| --- | --- | --- |
| **16 GiB VRAM**（表示共有時は実効約 15 GiB） | RX 9070 XT など | Qwen3.8-27B **EXL3-INT8 3.05bpw** + `--kv-cache-dtype octave:c5v5` + `--arena-gib 14` + `--prefill-chunk 512` で 128Ki コンテキスト（投機なし）。詳細は [recipes/qwen3.8-27b-128k-16gib.md](recipes/qwen3.8-27b-128k-16gib.md) |
| **32 GiB VRAM** | Radeon AI PRO R9700 | Qwen3.8-27B **EXL3-INT8 4bpw** + DFlash2 投機 + `octave:c5v5` で 128Ki コンテキスト・4 並列。BF16 / FP8 KV も余裕をもって利用可能 |
| **マルチ GPU（2〜8 枚）** | R9700 × 2 / 4 / 8 | 27B FP8（単一 32 GiB に収まらないため TP=2）や Qwen3.8-Flash-Next（MoE、TP / EP / PP 併用）。詳細は [multi-gpu.md](multi-gpu.md) |

> [!NOTE]
> **デスクトップ共有環境（NixOS デスクトップ等）での注意**: 表示（Wayland / X11 コンポジタ等）に同じ GPU を使っている場合、起動時の空き VRAM や実効上限が物理容量より 0.5〜1 GiB 以上少なくなる。`--arena-gib auto` は空き VRAM のほぼ全量を KV に割り当てるため、表示共有環境や 16 GiB カードでは `--arena-gib 14` のように固定値を指定する方が安定する。

### `--arena-gib` と KV キャッシュの消費量

- `--arena-gib auto`（既定）: 起動時の空き VRAM（`hipMemGetInfo`）から **0.5 GiB**（512 MiB）を引いた値を arena 容量とする。
- `--arena-gib -N`: 総 VRAM から `N` GiB を引いた値を使う。
- `--arena-gib N`: arena 容量を `N` GiB に固定する（再現測定や 16 GiB 運用向け）。
- arena 内では model weights → drafter（使用時）→ working set reserve（`--prefill-chunk` / `--max-num-batched-tokens` に応じて決まる）→ prefix cache capture slot を確保した**残りをすべて KV pool に割り当てる**。`embed_tokens` と prefix checkpoint 本体は pinned host RAM に置くため VRAM を消費しない。起動時の `MEMORY_PLAN`（stderr）で実際の内訳と収容可能 token 数を確認できる。

**Qwen3.8-27B（full-attention 16 層・KV heads 4・head_dim 256）の KV キャッシュ消費量目安:**

| `--kv-cache-dtype` | 1 token あたり | 32Ki (32,768) | 128Ki (131,072) | 備考 |
| --- | ---: | ---: | ---: | --- |
| `bf16` | 65,536 B | 2.00 GiB | 8.00 GiB | 既定。量子化なし |
| `fp8` | 33,280 B | 1.02 GiB | 4.06 GiB | BF16 の約半分。decode 速度は BF16 と同等 |
| `octave:c5v5` | 20,736 B | 0.63 GiB | 2.53 GiB | 16 GiB / 128Ki 構成の推奨（単一 GPU のみ。TP / PP / MoE は非対応） |
| `octave:c4v4` | 16,640 B | 0.51 GiB | 2.03 GiB | さらに省メモリな Octave 形式 |
| `octave:b4n3v3` | 13,568 B | 0.41 GiB | 1.66 GiB | 最小クラスの Octave 形式 |

詳細は [../perf/kv-octave.md](../perf/kv-octave.md) と [../perf/qwen-embed-host.md](../perf/qwen-embed-host.md) を参照する。

---

## ドキュメント一覧

- [build.md](build.md) — 要件・ビルド手順・CMake オプション・ROCm 10.2 / hipBLASLt ABI トラブルシュート
- [quantize.md](quantize.md) — EXL3-INT8（mul24）・Ngram4 / Ngram5 の変換・形式検証・品質評価（BF16 基準 KLD / top-1）
- [compute.md](compute.md) — `shelf-compute`（単発推論・`--serve-stdio`・EXL3-INT8 prefill A/B・DFlash2 / MTP / YaRN / Prefix Cache）
- [hy4_flash.md](hy4_flash.md) — Hy4-flash の expert 並列
- [server.md](server.md) — `shelf-server`（OpenAI Chat Completions / Responses API・エージェント接続・構造化出力）
- [bench.md](bench.md) — `shelf-bench`（`pp` / `tg` / `quality` / microbench）
- [multi-gpu.md](multi-gpu.md) — PP / EP / TP によるマルチ GPU 実行とモデル別対応
- [env_vars.md](env_vars.md) — `SHELF_*` 環境変数リファレンス（正本）
- [recipes/qwen3.8-27b-128k-16gib.md](recipes/qwen3.8-27b-128k-16gib.md) — 16 GiB VRAM で Qwen3.8-27B 128Ki コンテキストを動かす構成例

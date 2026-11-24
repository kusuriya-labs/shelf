# docs/developer

現在実装されている architecture / runtime / server / kernel contract のみを置く。
過去の PoC・Gate・採否判断は Git history、現在の性能値は `docs/perf/` を参照。

## roadmap

- [roadmap.md](roadmap.md) — 今後の作業予定

## architecture / runtime

- [architecture.md](architecture.md) — layer 構成と依存ルール、Server を含む責務境界
- [runtime.md](runtime.md) — execution path と state ownership
- [multi-gpu.md](multi-gpu.md) — PP / EP / TP の座標、所有構造、collective、比較条件
- [debug-ring.md](debug-ring.md) — GPU 側状態を同期なしで観測する診断 ring
- [qwen35.md](qwen35.md) — Qwen3.5 runtime family の配線（source path / KernelId / execution order）
- [qwen35moe.md](qwen35moe.md) — Qwen3.5 MoE の config と routed expert weight 集約 contract
- [qwen35qsa.md](qwen35qsa.md) — Qwen3.5 QSA（`qwen3_5_qsa`）の indexer / block mask contract
- [sampling.md](sampling.md) — target LM sampling（temperature / top-k / top-p / seed）
- [prefix_cache.md](prefix_cache.md) — prefix cache の radix tree / KV page sharing / state store と JSONL protocol contract

## speculative decode

- [dflash2.md](dflash2.md) — DFlash2 speculative decoding の現在の設計仕様
- [mtp.md](mtp.md) — checkpoint の MTP block を drafter にする compute / serve 経路

## server

- [server_status.md](server_status.md) — Server の開発者向け入口（surface と詳細 contract へのリンク）

## quantization / formats

- [quantization.md](quantization.md) — 量子化フォーマットと offline pipeline の overview
- [ngram_quantization.md](ngram_quantization.md) — Ngram4 / Ngram5 table の形式契約
- [canonical_soa_payload.md](canonical_soa_payload.md) — canonical payload の binary/layout 仕様
- [tensor_partition.md](tensor_partition.md) — TP tensor partition と canonical → partition → preshuffle の weight materialization contract

## kernels

- [kernels.md](kernels.md) — active kernel 一覧と variant ルール

## testing

- [testing.md](testing.md) — required acceptance / optional E2E / quality regression / server regression group
- [testing_f64_oracle.md](testing_f64_oracle.md) — f64 oracle による token 監査

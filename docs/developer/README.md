# docs/developer

現在実装されている architecture / runtime / server / kernel contract のみを置く。
過去の PoC・Gate・採否判断は Git history、現在の性能値は `docs/perf/` を参照。

## ゴールと roadmap

- [objective.md](objective.md) — 現行のゴールと性能目標
- [roadmap.md](roadmap.md) — 今後の作業予定

## architecture / runtime

- [architecture.md](architecture.md) — layer 構成と依存ルール、Server を含む責務境界
- [runtime.md](runtime.md) — execution path と state ownership
- [qwen35.md](qwen35.md) — Qwen3.5 runtime family の配線（source path / KernelId / execution order）
- [qwen4exp.md](qwen4exp.md) — Qwen3.8-Flash-Next (qwen4_exp) の architecture contract
- [sampling.md](sampling.md) — target LM sampling（temperature / top-k / top-p / seed）
- [prefix_cache.md](prefix_cache.md) — prefix cache の radix tree / 2-tier store と JSONL protocol contract

## speculative decode

- [dflash2.md](dflash2.md) — DFlash2 speculative decoding の現在の設計仕様

## server

- [server_status.md](server_status.md) — Server の開発者向け入口（surface と詳細 contract へのリンク）

## quantization / formats

- [quantization.md](quantization.md) — 量子化フォーマットと offline pipeline の overview
- [tensor_parallel_execution.md](tensor_parallel_execution.md) — Qwen3.8 Dense の Tensor Parallel execution（rank-local runtime / collective / transport）
- [canonical_soa_payload.md](canonical_soa_payload.md) — canonical payload の binary/layout 仕様
- [tensor_partition.md](tensor_partition.md) — TP tensor partition と canonical → partition → preshuffle の weight materialization contract

## kernels

- [kernels.md](kernels.md) — active kernel 一覧と variant ルール

## testing

- [testing.md](testing.md) — required acceptance / optional E2E / quality regression / server regression group
- [resident_model.md](resident_model.md) — Resident Model Infrastructure（model weight の lifetime 分離、ModelSource、protocol、budget、health check）
- [testing_f64_oracle.md](testing_f64_oracle.md) — f64 oracle による token 監査

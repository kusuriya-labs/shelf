# MTP のマルチリクエスト verify（2026-11-21）

実験 ID: `Q4EXP-MTP-BATCH-1`

対象: `models/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw`
（qwen4exp、48 層、512 routed experts、EXL3 mul24、MTP 1 層）。

結論:

- **ACCEPT**: TP / EP 起動で MTP block を rank0 に load し、複数 request を同時に保持して
  MTP round を進められるようにした。
- **ACCEPT**: qwen4exp の TP では rank0 に全語彙の private lm_head を追加 load し、
  target の語彙分割 head と drafter の全語彙 head を分離した。
- **ACCEPT**: mul24 の split-K と QSA sparse attention の split 数を verify の
  行容量で固定した。
- **ACCEPT**: QSA / PLE を持つ qwen4exp は target の 1 行 decode をそのまま使う安全経路で
  verify する。この経路の生成列は target-only と一致し、`test_mtp_parity` は 64 token で
  PASS する。8 request 同時の生成列も target-only の直列実行と一致した。
- **REJECT**: qwen4exp で K+1 行の融合 verify を既定にすること。near-tie が多く、
  行数に依存する kernel の差が 13 token 目以降の分岐として現れた。

## 1. 目的

checkpoint の MTP block を drafter にしたまま、`--serve-stdio` で複数 request を
同時に進める。dense など QSA / PLE を持たない target は 1 round の verify を
4 request / 32 行の chunk に分けて融合実行する。qwen4exp は QSA / PLE の per-row state を
巻き戻さずに済む 1 行検証経路を使う。

## 2. 修正した契約

- `TpRankRuntimeConfig::load_mtp_layers` を rank ごとの config に伝播する。
  これが無いと TP / EP 起動で rank0 の MTP block が load されず、
  `LoadedModel::mtp_drafter_input` が `unsupported` を返していた。
- qwen4exp の TP で target の lm_head が語彙分割されるとき、rank0 だけ
  `mtp_lm_head` として全語彙 head を追加 load する。target の head は分割されたまま
  使い、drafter graph と drafter executor の staging plane は全語彙 head を含める。
- mul24（EXL3-INT8）の split-K を `splitk_tuning_rows` に従わせる。従来は
  `exl3_mul24_gemm_splits` が実行行数から split を選んでいた。
- QSA sparse attention の split 数も verify の行容量で固定する。
- QSA / PLE を持つ target では、`[pending, draft0, ...]` を target の通常 `DECODE` で
  1 行ずつ流し、最初の不一致で止める。target の state は受理 prefix だけを進めるため、
  GDN snapshot / compact log replay は不要である。

`MTP_BATCH` の終了行に `chunks` と `max_active` を追加した。`chunks` は verify の
target forward 数、`max_active` は同時に進めた request 数の最大値である。

## 3. 検証

### 単一 request の target-only 一致（TP=2、2048 token prompt、drafts 7）

`test_mtp_parity`:

```text
MTP_PARITY: PASS (64 tokens, rounds=16, partial_accepts=11, gdn_restores=0)
```

### 8 request の同時実行（EP=4）

`tests/server/test_compute_mtp_batch.py` は target-only を request ごとに直列実行した
基準と、8 request 同時の MTP を比較する。同じ 8 request をもう一度流して再現性も見る。

```text
[PASS] compute-mtp-batch:completed
[PASS] compute-mtp-batch:matches-target
[PASS] compute-mtp-batch:deterministic
[PASS] compute-mtp-batch:mtp-batch-enabled
[PASS] compute-mtp-batch:mtp-batch-stats
[PASS] compute-mtp-batch:multi-request-round
[PASS] compute-mtp-batch:multi-batch-verify
compute-mtp-batch: passed=7 failed=0
```

### 回帰

`ctest -L required` は 226/226 PASS。

## 4. 制約

- qwen4exp の安全経路は 1 round あたり最大 `drafts + 1` 回の target decode を行う。
  融合 verify の 1 forward より target 実行回数が多く、投機の速度利得は小さい。
- 融合 verify を使う family は、EXL3 split-K と QSA split を固定しても MoE grouped path
  などの数値が完全な行数不変ではない。near-tie が多い checkpoint では target-only との
  bit 一致を無条件には保証しない。

# qwen35-f64-oracle（外部参照）

## 概要

- https://github.com/mjsabby/qwen35-f64-oracle
- Qwen3.5 / Qwen3.6 ファミリー（hybrid GDN + full attention、MoE、MTP 含む）を
  純粋 IEEE f64 で実行する自己完結 CPU 実装。tokenizer を除き依存なし
  (safetensors mmap reader / json native、AVX-512 / scalar が同一の固定 reduction 順序)。
- bf16 checkpoint を f64 へ正確に変換し、norm / RoPE / softmax / DeltaNet scan
  をすべて f64 で計算する。HF transformers に対して op-by-op 検証済み（同 repo VERIFICATION.md）。

## リポジトリに含めない理由

- ライセンスが自repoの配布と整合しないため、**本リポジトリへ vendor しない**。
  PhaseNonShift のした以外の場所（例: `/root/qwen35-f64-oracle`）に clone して利用する。
  依存として build に連結しない。あくまで独立の referee tool である。

## 取得と build

```sh
git clone --depth 1 https://github.com/mjsabby/qwen35-f64-oracle /root/qwen35-f64-oracle
cd /root/qwen35-f64-oracle && make
```

## 使い方（PhaseShift 側 wrapper）

`tools/qwen35_oracle_tokens.py` が入力 token ids / prompt / chat prompt から
greedy トークンを f64 oracle で計算し、bench の `GREEDY_FIRST_TOKENS` と同形式で出力する。

```sh
# token ids 直接（E2E fixture の input_ids を指定）
/root/.venv/bin/python3 tools/qwen35_oracle_tokens.py \
  --model-dir models/Qwen3.5-4B --ids-file ids.txt --max-new-tokens 2

# chat prompt から（transformers tokenizer で encode）
/root/.venv/bin/python3 tools/qwen35_oracle_tokens.py \
  --model-dir models/Qwen3.5-4B --prompt "What is my favorite color?" --chat
```

- `ORACLE_FIRST_TOKENS=<ids>` が greedy 生成の第一トークン列
- 監査に使えるモード: `--dtype bf16|f16`（低精度 storage 語義）、
  `--gdn recurrent`（chunked との algebraic 等価性確認）

## 用途（正しく見える壊れを防ぐ）

- E2E / bench の greedy トークンと照合し、
  「correctness path が壊れているのに、それが整合出力に見える」事故を検出する。
  実例: preshuffle 導入時に correctness mode が native layout を row-major 読みし、
  スクランブル GEMM の出力がしばらく正規参照と扱われていた（docs 7.44）。
  f64 oracle は preshuffle / GPU 正当実装のいかんを問わず独立に token を確定する。

## 確認結果（2026-09-17 / Qwen3.5-4B bf16）

| case | f64 oracle | Phaseshift auto (tile+新GDN) | Phaseshift correctness(修正後) |
| --- | --- | --- | --- |
| `ids = 1 x 256`, 3 step | 248046,198,248045 | 248046,198,248045 | 248046,198,248045 |
| chat turn2 (36 ids), 2 step | 27775,383 = "Based on" | 7525,383 = "Your favorite"(KNOWN-LOSSY) | 27775,383 = "Based on" |

turn2 の margin: top1 26.004508 vs top2 23.703756（HF oracle fixture の
26.0045166015625 / 23.703771591186523 と一致）。interactive multi-turn のみ
optimized 側が flip する（`docs/rnd/gdn/optimization_history.md` §7.43 /
`docs/rnd/quantization/kernel_optimization_history.md` §7.44 参照）。

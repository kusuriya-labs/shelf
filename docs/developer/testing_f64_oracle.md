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
  本リポジトリの worktree 以外の場所（例: `/root/qwen35-f64-oracle`）に clone して利用する。
  依存として build に連結しない。あくまで独立の referee tool である。

## 取得と build

```sh
git clone --depth 1 https://github.com/mjsabby/qwen35-f64-oracle /root/qwen35-f64-oracle
cd /root/qwen35-f64-oracle && make
```

## 使い方（Shelf 側 wrapper）

`tools/qwen35_oracle_tokens.py` が入力 token ids / prompt / chat prompt から
greedy トークンを f64 oracle で計算し、bench の `GREEDY_FIRST_TOKENS` と同形式で出力する。

```sh
# token ids 直接（E2E fixture の input_ids を指定）
/root/.venv/bin/python3 tools/qwen35_oracle_tokens.py \
  --model-dir <bf16-model-dir> --ids-file ids.txt --max-new-tokens 2

# chat prompt から（transformers tokenizer で encode）
/root/.venv/bin/python3 tools/qwen35_oracle_tokens.py \
  --model-dir <bf16-model-dir> --prompt "What is my favorite color?" --chat
```

FP8 checkpoint は oracle が直接読めないため、Qwen3.8-27B-FP8 では
`tools/qwen38_fp8_cpu_fixture.py` を使う（後述）。

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

| case | f64 oracle | Shelf auto (tile+新GDN) | Shelf correctness(修正後) |
| --- | --- | --- | --- |
| `ids = 1 x 256`, 3 step | 248046,198,248045 | 248046,198,248045 | 248046,198,248045 |
| chat turn2 (36 ids), 2 step | 27775,383 = "Based on" | 7525,383 = "Your favorite"(KNOWN-LOSSY) | 27775,383 = "Based on" |

turn2 の margin: top1 26.004508 vs top2 23.703756（HF oracle fixture の
26.0045166015625 / 23.703771591186523 と一致）。interactive multi-turn のみ
optimized 側が flip する。

## Qwen3.8-27B-FP8 の CPU fixture

Qwen3.8-27B-FP8 の HF checkpoint は FP8 E4M3 weight と BF16 block scale
（128x128）を持つ。oracle はこの形式を直接読まないため、fixture 生成時に
`weight = f64(fp8_code) * f64(scale)` で F32 へ dequantize する。この積は F32 で
厳密に表現できるので、F32 化による追加の丸めはない。F32 model を oracle の
`--model` に渡し、forward は従来どおり f64 で実行する。

生成器は `tools/qwen38_fp8_cpu_fixture.py` である。dequantize 済み model は
`.cache/` に置き、repository には commit しない。

```sh
tools/qwen38_fp8_cpu_fixture.py \
  --model-dir models/Qwen3.8-27B-FP8 \
  --oracle-dir ../references/qwen35-f64-oracle \
  --ids 814,20139,303,799,13901,3069,279,12515,7701,6105,2261,279,1834,321,2438,506,41564,13 \
  --max-new-tokens 64 --hidden-tokens 16 --hidden-scope both \
  --out-json tests/fixtures/qwen35/qwen38_27b_fp8_cpu_oracle.json \
  --out-hidden tests/fixtures/qwen35/qwen38_27b_fp8_cpu_oracle_hidden.safetensors \
  --out-hidden-generated tests/fixtures/qwen35/qwen38_27b_fp8_cpu_oracle_hidden_generated.safetensors
```

出力:

- JSON: input ids、greedy 生成 64 token、各 step の top5 logits と margin。
- `..._hidden.safetensors`: 入力 prefill の先頭 16 position の
  `hidden_00..hidden_64`（F32）。`hidden_00` は embedding、
  `hidden_01..hidden_63` は decoder layer 0..62 の残差出力
  （= 次の layer の入力）、`hidden_64` は final norm 出力である。
- `..._hidden_generated.safetensors`: 生成 token 1..16 の位置の
  `hidden_00..hidden_64`（F32）。`input_ids + generated_ids[:16]` の
  34 token prefill から position 18..33 を取り出す。causal なので
  incremental decode の同じ位置と一致する。

`hidden_XX` の並びは oracle の dump format に対応する。oracle は final norm を
`hidden_64` に書くため、layer 63 の残差出力は個別 key を持たない。
layer 0..63 の入力は `hidden_00..hidden_63` で揃う。

fixture の 64 token は、F32 dequant 版 oracle と、FP8 を f64 forward 内で
直接読む patch 版 oracle で一致することを確認している。hidden は両者で
bit 一致し、16 token prefill と 18 token prefill の先頭 16 position も
bit 一致する。

GPU の correctness path と比較するときは、同じ input ids を prefill し、
各 layer の残差出力をこの safetensors の同名 key と比較する。

GPU の correctness path は production と同じ W8A8（weight も activation も
E4M3、activation は 128 要素ごとの dynamic scale）で実行する。fixture の oracle は
weight だけを FP8 から復号し、activation は f64 のまま計算するため、次の差は
実装の誤りではなく数値 class の差である。

- hidden は bf16 の中間値と A8 量子化を含むため、f64 oracle と bit 一致しない。
  比較は相対 L2 と平均絶対差で行う。
- greedy token は top1-top2 margin が小さい step で入れ替わり得る。margin が
  十分大きい step は完全一致し、小さい step は fixture の top5 内の別 token を
  許容する。

自動検証は `test_qwen35_correctness_fixture`（`gpu1;optional;external_files`）が行う。
`SHELF_MODEL_DIR` を設定して実行する。

```sh
SHELF_MODEL_DIR=models/Qwen3.8-27B-FP8 SHELF_BUILD_DIR=build \
  python3 tests/correctness/test_qwen35_correctness_fixture.py
```

固定する契約:

- `hidden_01..hidden_08` の相対 L2 が `2e-2` 以下、平均絶対差が `2e-2` 以下
- greedy token は fixture と先頭から一致する
- 最初の不一致は、fixture の margin が `0.5` 以下かつ生成 token が fixture の
  top5 に含まれる場合だけ許容する
- 許容した不一致より前の token は完全一致する

## 合成 checkpoint Fixture（外部 checkpoint 不要）

`tests/fixtures/qwen35/qwen35_gdn_tiny/` は合成 Qwen3.5 checkpoint（2 層: 層 0 = GDN、
層 1 = full attention、hidden 512、GDN key 2×128 / value 4×128）と、その f64 参照
（`oracle.json` と `oracle_hidden.safetensors`）を commit した fixture である。
`test_qwen35_gdn_gate_reference`（`gpu1;required`）が層 0 の残差出力と prompt 続きの
argmax を比較し、GDN の gate 活性（silu 固定）を固定する。外部 checkpoint を必要としない
ため required acceptance で毎回走る。

再生成には oracle が必要:

```sh
python3 tools/qwen35_gdn_tiny_fixture.py \
  --oracle-dir ../references/qwen35-f64-oracle
```

fixture の config は `output_gate_type` を**宣言しない**。これが contract の一部で
（参照実装はこの key を読まない）、宣言した形で生成し直すと gate 契約の検証ではなくなる。

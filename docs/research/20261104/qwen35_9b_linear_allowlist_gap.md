# Qwen3.5-9B BF16 linear allowlist gap

## 症状

`models/Qwen3.5-9B`（BF16）を target-only で実行すると、201 token prompt で
prefill 12.3 s / decode 3.75 tok/s だった。MTP の有無に依存しない。

```text
PROMPT_TOKENS=201
PREFILL_MS=12346.6
DECODE_TOKENS_PER_SEC=3.75
```

## 原因

`SHELF_LINEAR_DEBUG=1` で確認したところ、9B の hidden 4096 系の shape が
`kLinearShapeRules` に無く、ほぼすべての linear が correctness path に落ちていた。

```text
64 LINEAR_SELECT_MISS family=0 out=12288 k=4096
56 LINEAR_SELECT_MISS family=0 out=4096  k=4096
48 LINEAR_SELECT_MISS family=0 out=32    k=4096
32 LINEAR_SELECT_MISS family=0 out=8192  k=4096
32 LINEAR_SELECT_MISS family=0 out=4096  k=12288
16 LINEAR_SELECT_MISS family=0 out=1024  k=4096
 2 LINEAR_SELECT_MISS family=0 out=248320 k=4096
```

対応する 9B geometry:

- MLP gate/up: `(12288, 4096)`
- MLP down: `(4096, 12288)`
- GDN qkv: `(8192, 4096)`
- GDN z / out projection / attention o: `(4096, 4096)`
- GDN b/a: `(32, 4096)`
- attention k/v: `(1024, 4096)`
- lm_head: `(248320, 4096)`

`kLinearShapeRules` には `(12288, 5120)` など 27B 用の shape はあるが、
hidden 4096 のこれらの shape は登録されていない。

## 影響

- 9B の target-only decode / prefill が correctness path 律速になる。
- MTP の 9B E2E も同じ理由で遅く、2048 token prompt の optional parity が
  実用的な時間に収まらない。
- 今回の MTP 実装（compact log replay）の不具合ではない。

## 次の作業

`linear_selector.cpp` の `kLinearShapeRules` に 9B の BF16 shape を追加し、
WMMA linear の正しさを required test で確認する。追加後、9B で次を再測する。

```bash
SHELF_QWEN35_KERNEL_MODE=auto \
  shelf-compute --model-dir models/Qwen3.5-9B \
  --input-ids-file <201 token prompt> --max-new-tokens 8 --max-seq-len 512
SHELF_LINEAR_DEBUG=1 ...
```

受入条件:

- `LINEAR_SELECT_MISS` が 9B の通常 decode で出ない
- 9B target-only の生成列が変更前と一致する
- MTP の 9B parity が実用的な時間で完了する

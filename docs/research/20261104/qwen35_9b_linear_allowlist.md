# Qwen3.5-9B BF16 linear allowlist

## 背景

`models/Qwen3.5-9B`（BF16、hidden 4096）は target-only でも prefill / decode が
correctness path 律速だった。`SHELF_LINEAR_DEBUG=1` で確認すると、hidden 4096 系の
linear shape が `kLinearShapeRules` に無く、ほぼすべての linear が fallback していた。

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

## 判断

ACCEPT。上記 7 shape を BF16 family の allowlist に追加した。FP8 / MXFP4 は追加しない。

## 検証

- `test_qwen35_linear_selector` に 9B の shape（rows=1/8/256）と、FP8 では未登録である
  ことを追加した。
- 9B の 201 token / 2048 token prompt で、変更前後および correctness mode の
  `GENERATED_IDS` が一致した。
- 変更後の通常 decode で `LINEAR_SELECT_MISS` が出ない。
- 速度の実測は [../../perf/current.md](../../perf/current.md) を参照する。

## 残る作業

9B の他の selector（attention / GDN / norm など）に fallback が残る場合は、
同様に shape を確認して個別に allowlist へ追加する。今回は BF16 linear に限定した。

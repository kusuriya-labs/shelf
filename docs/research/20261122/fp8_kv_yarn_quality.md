# FP8 KV 品質の再測（Static YaRN 2x 併用）

- 実施日: 2026-11-22
- 対象: `models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw`
- GPU: Radeon AI PRO R9700（gfx1201）× 1（`--device 0`）
- build: HEAD `438879e2` + 作業ツリー（`shelf-bench quality` の `--rope-factor`、
  `tools/quality_gate.py run` の `--rope-factor` 追加）
- corpus: BetterBench v1 を `tools/quality_gate.py corpus` で変換した 2,203 token
  （sha256 `99fe6aa9…`）

## 目的

旧 Gate では、2026-10-20 の FP8 KV 品質ゲート FAIL（HF FP8 で
ppl +28.12% / +29.96%）を前提に、現行 build での再測と YaRN 2x 併用時の劣化分離を
求めていた。2026-10-23 の [fp8_kv_prefill_quality_audit.md](../20261023/fp8_kv_prefill_quality_audit.md)
で fp8 paged prefill の D1〜D3 が修正済みであり、その状態を current build で
再確認し、factor 2 の寄与を分離する。

HF FP8 checkpoint（`Qwen3.8-27B-FP8`）はこの環境に無い。dense 27B の
EXL3-INT8-4bpw checkpoint を測定対象として採用する。KV 保存形式と attention kernel は
同じ経路を通り、weight 量子化が異なる点だけを適用範囲に含める。

## 方法

`shelf-bench quality` の teacher-forced dump を BF16 KV で取り、同じ条件の FP8 KV を
`--compare` で比較した。`tools/quality_gate.py run --skip-endtask --skip-gsm8k` の
数値ゲート（PPL 回帰 1% 以下、mean KL 0.05 以下、top-1 一致 0.95 以上）を使った。

```sh
python3 tools/quality_gate.py run \
  --model-dir models/kusuriya-labs/Qwen3.8-27B-EXL3-INT8-4bpw \
  --build-dir build --corpus /tmp/opencode/corpus.psktok --device 0 \
  --context <128|1600> --positions <2048|602> --rope-factor <1|2> \
  --skip-endtask --skip-gsm8k --json <out>.json
```

`--rope-factor 1` は Native、`--rope-factor 2` は Static YaRN factor 2 である。

## 数値ゲート

| factor | context | positions | PPL BF16 | PPL FP8 | PPL 差 | mean KL | top-1 | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 128 | 2048 | 15.663659 | 15.466097 | -1.26% | 0.004362 | 0.975586 | PASS |
| 2 | 128 | 2048 | 15.782134 | 15.670305 | -0.71% | 0.003905 | 0.972656 | PASS |
| 1 | 1600 | 602 | 8.783930 | 8.707198 | -0.87% | 0.003053 | 0.985050 | PASS |
| 2 | 1600 | 602 | 9.547516 | 9.597738 | +0.53% | 0.004487 | 0.955150 | PASS |

FP8 KV の KL は 0.0031〜0.0045、top-1 は 0.955〜0.986 で、旧 FAIL（KL 0.22〜0.78、
top-1 0.72〜0.82）とは桁が違う。2026-10-23 の修正が current build でも有効である。

YaRN 2x の寄与は、同じ採点条件の BF16 同士で分離できる。

| context | positions | BF16 Native PPL | BF16 YaRN 2x PPL | YaRN の差 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2048 | 15.663659 | 15.782134 | +0.76% |
| 1600 | 602 | 8.783930 | 9.547516 | +8.69% |

FP8 KV の差（上の表）は、この YaRN 自体の差より小さい。したがって YaRN 2x 併用時に
追加される FP8 KV 由来の劣化は、この分解能では BF16 の分割差と区別できない。

## 448 位置での確認

`--positions 448` でも同じ比較を行った（`--context 128` は採点 129〜576、
`--context 1600` は 1601〜2048）。

| factor | context | PPL 差 | mean KL | top-1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | +5.20% | 0.002462 | 0.991071 |
| 2 | 128 | -6.80% | 0.004874 | 0.979911 |
| 1 | 1600 | -0.25% | 0.003583 | 0.982143 |
| 2 | 1600 | -1.12% | 0.005297 | 0.953125 |

448 位置の PPL 差は符号も大きさも安定しない。採点集合が小さいため PPL を採否の
主指標にせず、KL と top-1 で判断する（2026-11-05 の MoE 記録と同じ扱い）。

## 結論

- 現行 build の FP8 KV は、Native / YaRN 2x のどちらでも数値ゲート PASS。
- YaRN 2x 併用による追加の FP8 KV 劣化は、mean KL 0.0039〜0.0045 で
  Native の 0.0031〜0.0044 と同じ帯に収まる。
- 2026-10-20 の FAIL を現在の既知の欠陥として扱わない。
- HF FP8 checkpoint は無いが、測定対象は EXL3-INT8-4bpw を採用する。KV 経路の比較は
  同一 checkpoint 内の BF16 / FP8 で行っており、この記録で足りる。

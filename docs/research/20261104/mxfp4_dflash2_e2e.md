# MXFP4 × DFlash2 speculative decode の E2E 実測

対象: `models/Qwen3.8-27B-Quark-AWQ-MXFP4`（W4A8、native layout）に `models/Qwen3.8-27B-DFlash2`
を drafter として組み合わせた場合の E2E。Gate A / A' の MXFP4 導入（`docs/research/20261103/mxfp4_gate_a_prime.md` §16, §17）が
speculative decode の経路でどう出るかを測った。

## 1. 条件

- GPU: R9700 (gfx1201)、device 0 のみ。
- 計測口: `shelf-compute`。`shelf-bench pp|tg` には DFlash2 経路が無く、spec decode の計測口は `shelf-compute` にしかない。
- prompt: 2048 tokens（`/tmp/opencode/ids_2048.txt`）、`--max-seq-len 4096`、greedy。
- spec decode: `--dflash2-model-dir models/Qwen3.8-27B-DFlash2 --dflash2-drafts 7`。
- 比較 base（MXFP4 導入前）: `6bf9ed3`。同一 prompt・同一 drafts・同一 1 GPU 条件。

```
./build/shelf-compute --model-dir models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
    [--dflash2-model-dir models/Qwen3.8-27B-DFlash2 --dflash2-drafts 7] \
    --input-ids-file /tmp/opencode/ids_2048.txt --max-seq-len 4096 \
    --max-new-tokens 1        # prefill 断面
    --max-new-tokens 128      # decode 断面
```

## 2. 実測

| 断面 | 導入前 `6bf9ed3` | MXFP4 導入後 | 変化 |
| --- | ---: | ---: | ---: |
| pp2048（spec decode 無効） | 1095.9 ms / 1869 tok/s | 831.9 ms / 2462 tok/s | **+31.7%** |
| pp2048（DFlash2 有効） | 1408.2 ms | 1140.7 ms | +23.4% |
| tg128（DFlash2 有効） | 1261.1 ms / 100.6 tok/s | 1105.5 ms / 115.0 tok/s | +14.3% |
| tg128（spec decode 無効） | 4716.2 ms / 26.92 tok/s | 4792.8 ms / 26.49 tok/s | −1.6% |

prefill は §17.2（prompt 1024、+34%）と同じ方向。spec decode を有効にすると prefill に
drafter の forward が乗るため、導入前後とも pp2048 は非 spec より 340 ms 級大きい。

## 3. tg128 の内訳（DFlash2 有効）

| 項目 | 導入前 | 導入後 |
| --- | ---: | ---: |
| verify rounds | 22 | 19 |
| mean accepted / round | 4.773 | 5.684 |
| full accept 率 | 0.545 | 0.737 |
| wall / round | 57.3 ms | 58.1 ms |
| verify GPU / round | 41.4 ms | 42.2 ms |
| 非 spec decode | 37.1 ms/step | 37.7 ms/step |

## 4. 結論（採否）

- tg128 の **+14.3% は accept 率が上がった見かけ**であり、カーネルの利得ではない。
  1 round あたりの実時間は 57.3 → 58.1 ms で **1.5% 悪化**している。verify の GPU 時間
  （41.4 → 42.2 ms）も同方向で、MXFP4 の per-launch 利得（§17.1 の rows=1 で GEMM
  −5.6 / −18.1 µs）は 1 step の launch 構成に埋もれて E2E の分解能上に現れない。
- rounds が 22 → 19 に減ったのは、量子化後の logits が BF16 の logits と違う列を greedy で
  選び、drafter の提案と verifier の一致が変わったため（意味論の変化の副作用）。
  §12 / §17 の生成列一致の記録と整合する。
- したがって **speculative decode の性能寄与として MXFP4 を評価しない**。docs/perf へ
  この tg128 数値を載せない。性能主張として残すのは prefill 断面（§17.2 と本実測 §2 の
  pp 行）だけ。
- 非 spec decode は 37.1 → 37.7 ms/step（+1.6%）。MXFP4 が decode を遅くする原因
  （quantize の追加 launch、rows=1 の GEMM 形状）は §17.2 の末尾に書いた通り未分解のまま。

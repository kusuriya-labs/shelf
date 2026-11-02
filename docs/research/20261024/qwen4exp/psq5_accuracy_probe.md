# PSQ5-CB28 の精度評価（重量空間の形式間比較とタスク精度の試行）

- ID: QWEN4EXP-PSQ5-ACC-01
- 実施日: 2026-10-24
- 目的: PSQ5-CB28（5.25 bpw）の精度が実用範囲かを判断する材料を集める。
- 関連: [quantization_profile.md](quantization_profile.md)、
  [quantization_artifact.md](quantization_artifact.md)、
  [docs/quant/PSQ5-CB28.md](../../../quant/PSQ5-CB28.md)

## 1. 重量空間の形式間比較（実測）

Flash-Next の routed expert（BF16）を入力に、同じ 5.25 bpw の一様 INT5 と
4bit 形式、FP8 を比較した。指標は `||W - Ŵ||F / ||W||F`。
4 tensor × 16,384 行を 4,096 行ずつ処理し、誤差とエネルギーの総和から算出した。

| 形式 | bpw | relFrob（平均） |
| --- | ---: | ---: |
| PSQ5-CB28 (ue4m4) | 5.25 | 0.0496 |
| INT5 一様 + ue4m4 scale | 5.25 | 0.0488 / 0.0468 |
| NF4 (per-32 absmax) | 4.25 | 0.0878 |
| MXFP4 (E2M1 + E8M0 per-32) | 4.25 | 0.1893 |
| FP8 K128N128 | 8.4 | 0.0265 |

PSQ5 の特徴（実測で確認したもの）:

- 同 bpw の一様 INT5 より **1.4〜6% 誤差が大きい**。CB28 のコードブックは
  小さい値に密で、上位レンジ（16, 20, 24, 28）の刻みが約 2 倍粗いためである。
  Frobenius 誤差は大きい重みに支配されるので、上位の粗さが効く。
- ブロック内の重みの **約 3.4% が厳密な 0 になる**（FP8 では 0.001% 未満）。
  `|w| < amax/56` が 0 に丸められるためで、コードブックの仕様上の性質である。
- 4bit 形式よりは明確に良く、FP8 の約 1.9 倍の誤差である。5.25 bpw の位置づけとしては
  想定内だが、コードブックを MSE 最適化すれば改善余地がある。

## 2. タスク精度の試行（測定不能と判断）

27B の BF16 checkpoint から PSQ5 fake-quant を作り、HF FP8 経路（W8A8、K128N128）で
KLD / top-1 を測る試みを行った。runtime が PSQ5 をロードできないため、
復元値を FP8 で格納して既存経路に載せる方式である。

### 結果

| 条件 | PPL | mean KL | top-1 |
| --- | ---: | ---: | ---: |
| FP8 再量子化（対照 A） | 9.348 | base | base |
| PSQ5 → FP8（B） | 5.915 | 0.196 | 0.818 |

B の PPL が A より 36% 低い。PSQ5 の重み誤差は A の約 2 倍（0.0566 vs 0.0265）であり、
量子化で PPL が改善する方向の結果は品質向上として解釈できない。

### 対照実験

- 同じ BF16 から作った 2 つの FP8 エンコード（重み差 0.26%、誤差の大きさは同一）で
  KL 0.0147、PPL 差 8.4%。
- 全 scale を 0.2% 拡大しただけの条件で KL 0.0133、PPL 差 7.3%。
- 同一 checkpoint の byte 同一コピーは KL 0（harness 自体は決定的）。

つまり、重みの 0.2〜0.3% の差（量子化ノイズの実現値の違い）で PPL が 7〜8% 動く。
2,048 位置の corpus PPL / KL は、PSQ5 の寄与ではなく**量子化ノイズの実現値**に
支配されており、この構成では PSQ5 のタスク精度を分離できない。

### 判断

この測定は **不採用**（数値を品質の根拠にしない）。PSQ5 のタスク精度は引き続き未確立である。

## 3. 次に必要な測定

1. BF16 のまま実行できる環境（または小さいモデル）で、BF16 / PSQ5 / FP8 を
   fake-quant 比較する。runtime の activation quantize を挟まないこと。
2. PPL / KLD だけでなく end-task（reasoning / gsm8k）と複数 seed を使う。
   2,048 位置では分解能が足りない。
3. ランタイムに PSQ5 のネイティブ経路を追加し、activation quantize の実現値を
   固定したまま weight 形式だけを変える。
4. PSQ5 と一様 INT5 を同じ条件で比較し、decode の単純化と 1.4〜6% の誤差差の
   どちらを取るかを決める。

## 4. 再現

```bash
python3 tools/psq5_fake_quant.py --model-dir models/Qwen3.8-27B \
  --output-dir models/Qwen3.8-27B-FP8-b --storage fp8 \
  --quantized-from models/Qwen3.8-27B-FP8
./build/phaseshift-bench quality --model-dir models/Qwen3.8-27B-FP8-b \
  --tokens-file /tmp/bb_corpus.psktok --device 0 --tp 1 \
  --kv-cache-dtype bf16 --compare /tmp/a.psq
```

`/tmp/a.psq` は同じ手順の `--no-psq5` 版で生成した baseline dump である。

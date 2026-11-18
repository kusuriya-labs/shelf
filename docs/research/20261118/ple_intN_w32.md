# PLE n-gram table の INT5/INT6 W32 FP16 オンライン量子化

## 目的

`models/Qwen3.8-Flash-Next-Quark-MXFP4` は PLE n-gram table を HF shard 内に BF16 で持つ。
320,001,536 行 × 160 値 × 2 byte = 95.37 GiB が host RAM に常駐し、メモリ圧が高い。
起動時に固定 bit の W32 形式へ変換し、table の常駐量を下げる。

最初に INT5 W32 FP16（5.5 bit/value）を実装し、精度改善の候補として回転を検討した。
回転が不採用になった後、INT6 W32 FP16（6.5 bit/value）を採用した。

## 採用形式: INT6 W32 FP16

- 32 値 = 1 group。group ごとに符号付き 6-bit code（-32..31）と FP16 scale を持つ。
- scale は両側を切らない値にする。`scale = max(max_positive / 31, -min_negative / 32)`。
  全 zero の group は scale 1、code 0。
- 格納は group ごとに 24 byte の bit-packed code と 2 byte の FP16 scale。
  1 group = 26 byte、160 値の 1 行 = 130 byte。
- 6-bit code は little-endian の bit stream に詰める。
- 変換は `PleTable::load` 内で行い、元の BF16 shard は保持しない。
- codec は `shelf/quantization/int6/int6_w32.h`（`shelf_quant_reference`）。
- BF16 / FP8 / EXL3 の checkpoint 形式判定は shard の dtype と
  `ngram_embedding.safetensors` の有無で行い、量子化形式を manifest に焼き込まない。

## サイズ

| 項目 | INT6 W32 | 参考: INT5 W32 | 参考: BF16 |
| --- | ---: | ---: | ---: |
| 行数 | 320,001,536 | 320,001,536 | 320,001,536 |
| 1 行 | 160 値 / 130 byte | 160 値 / 110 byte | 160 値 / 320 byte |
| 常駐 payload | **41,600,199,680 B** | 35,200,168,960 B | 102,400,491,520 B |
| GiB | **38.743 GiB** | 32.783 GiB | 95.367 GiB |
| bit/value | **6.5000** | 5.5000 | 16.0 |

INT6 W32 は BF16 から 56.624 GiB（59.4%）削減する。行あたりの padding は無く、
shard の 64 byte 境界配置による端数だけである。
`test_qwen4exp_load --skip-gpu` の `PLE resident` 表示も 38.743 GiB で一致した。

## 精度

全 128 shard から 2,048 行ずつ、合計 262,144 行（41,943,040 値）を抽出し、
BF16 の元値と復号値を比較した。

| 指標 | INT6 W32 | 参考: INT5 W32 | 参考: INT8 W32 |
| --- | ---: | ---: | ---: |
| relative RMS error | **2.151%** | 4.383% | 0.531% |
| SNR | **33.35 dB** | 27.16 dB | 45.46 dB |
| max abs error | 0.000740051 | 0.00151825 | 0.000181198 |
| max abs error / group amax | 0.01613 | 0.03333 | - |
| group bound 違反 | 0 / 41,943,040 | 0 / 41,943,040 | - |

- group bound は `|誤差| <= amax / 62`（scale の半分）。全 group で満たした。
- code は -32..31 を使い、極値の飽和による偏りは無い。
- FP16 scale の範囲は 1.6046e-4 .. 1.5030e-3（中央値 5.5313e-4）。

行単位の relative L2:

| 指標 | INT6 W32 | INT5 W32 |
| --- | ---: | ---: |
| p50 | **2.136%** | 4.352% |
| p95 | **2.431%** | 4.959% |
| p99 | **2.570%** | 5.241% |
| max | **2.995%** | 6.510% |

PLE 入力は 16 行を連結した 2,560 値として `key_proj` / `value_proj` に入る。
512 token 分（16 行 × 512）で投影後の誤差を測った。

| 投影 | 形式 | relative L2 p50 | p95 | max | cosine p50 | cosine min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `key_proj` [10240, 2560] | INT6 | **2.138%** | 2.534% | 3.061% | 0.9997736 | 0.9995655 |
| `key_proj` [10240, 2560] | INT5 | 4.334% | 5.114% | 6.423% | 0.9990681 | 0.9979365 |
| `value_proj` [2560, 2560] | INT6 | **2.153%** | 2.247% | 2.303% | 0.9997686 | 0.9997352 |
| `value_proj` [2560, 2560] | INT5 | 4.386% | 4.579% | 4.726% | 0.9990408 | 0.9988839 |

## 候補形式の比較

同じ評価サンプル（262,144 行、512 token）で bit 数とグループ幅を比較した。

| 形式 | table relRMS | key p50 | value p50 | bit/value | 常駐サイズ |
| --- | ---: | ---: | ---: | ---: | ---: |
| INT5 W32 | 4.383% | 4.334% | 4.386% | 5.50 | 32.783 GiB |
| INT5 W64 | 4.868% | 4.826% | 4.858% | 5.25 | 31.293 GiB |
| INT5 W16 | 3.851% | 3.784% | 3.851% | 6.00 | 35.763 GiB |
| **INT6 W32（採用）** | **2.152%** | **2.138%** | **2.153%** | 6.50 | 38.743 GiB |
| INT6 W16 | 1.891% | 1.881% | 1.892% | 7.00 | 41.723 GiB |
| INT8 W32 | 0.531% | 0.548% | 0.554% | 8.50 | 50.664 GiB |

- 5-bit の段差が支配的なため、6-bit 化は誤差をほぼ半減させる。
- W16 はグループ内の外れ値の影響を局所化するが、サイズ増に対する改善は 6-bit 化に劣る。
- INT6 W32 は +5.96 GiB で table 誤差を 4.383% から 2.151% へ下げる。

## 回転の検討（不採用）

低 bit 量子化の外れ値分散を狙い、32 値グループ単位の Hadamard 回転を INT5 W32 で検討した。
回転は PLE 側だけに閉じず、`key_proj` / `value_proj` の入力次元へ吸収できる。

```text
保存: y = R x を量子化
推論: decode した y をそのまま PLE 入力にする（逆回転しない）
重み: W' = W R^T
出力: W' y = W R^T R x = W x
```

`R` は 2560 次元のブロック対角で、1 head 160 値を 5 個の 32 値ブロックに分け、
各ブロックに正規化 Hadamard H32 を掛ける。sign は global と block ごと独立の両方、
さらに 160 column permutation も試した。評価は検索 32,768 行、評価 262,144 行、
投影 512 token で行った。推論時は逆回転せず、BF16 に丸めた `W'` を使う。

| 方式 | table relRMS | key p50 | key p95 | key max | value p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 回転なし | **4.3832%** | **4.3337%** | 5.1136% | 6.4232% | **4.3860%** |
| H32 | 4.3956% | 4.3555% | 5.2427% | 6.3471% | 4.4091% |
| Sign global seed 9 | 4.3961% | 4.3479% | 5.2095% | 7.5064% | 4.4031% |
| Sign global seed 6 | 4.3957% | 4.3564% | 5.1836% | 6.7374% | 4.3881% |
| Sign block seed 9 | 4.3961% | 4.3338% | 5.2101% | 6.4377% | 4.3910% |
| Sign block seed 4 | 4.3941% | 4.3397% | **4.9809%** | 6.3889% | 4.3845% |
| Perm seed 6 | 4.3925% | 4.3633% | 5.1319% | 6.8649% | 4.4008% |
| Perm seed 2 | 4.3947% | 4.3636% | 5.1981% | 6.7300% | 4.3951% |

- table 再構成誤差はすべての回転でわずかに悪化した。
- 投影誤差は p50 が ±0.02% 以内で、検索時の見かけの改善は評価サンプルで再現しなかった。
- `W'` の BF16 丸めだけの追加誤差は key 出力で約 0.14% であり、回転を妨げる要因ではない。

32 値グループはすでに正規分布に近く、誤差は一様量子化の段差で決まっている。
直交回転は L2 を保ったまま分布を変えるだけなので、外れ値が group 内に偏っていない
この table では利得が出ない。回転は採用しない。

## 既存 EXL3 K=5 PLE との比較

同じ Qwen3.8-Flash-Next の PLE table を持つ
`kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw` は、trellis K=5、102 byte/row
（5.1 bit/value）で PLE を保持している。同じ BF16 元値に対して table plane を比較すると、
EXL3 K=5 は relative RMS 16.451%、行 relative L2 p50 16.877% だった
（全 128 shard から 512 行ずつ、65,536 行を抽出）。
head bias を含めると 20.323% になるが、BF16 checkpoint 側に head bias が無いため
table plane のみを比較した。

INT6 W32 の 2.151% は、EXL3 K=5 の 16.451% より 7.6 倍小さい。
K=5 の EXL3 は行単位の scale を 1 本だけ持つのに対し、INT6 W32 は 32 値ごとに
scale を持つことが効いている。

## AVX512 量子化

量子化器は AVX512F / AVX512BW / AVX512VL / BMI2 が使える CPU では AVX512 経路、
それ以外は scalar 経路で動く（runtime dispatch）。AVX512 経路は 32 値の BF16 を
2 本の 512-bit vector に変換し、amax、RNE 丸め、clamp を vector で行う。6-bit code は
`vpmovdb` で byte に落とした後、BMI2 `pext` で 8 code = 48 bit を 6 byte に詰める。

CPU: AMD Ryzen 7 7800X3D（8 core / 16 thread、AVX512F/BW/VL/BMI2）、
入力 1 GiB BF16（1,048,576 行）の in-memory 量子化:

| thread | scalar | AVX512 | 倍率 |
| ---: | ---: | ---: | ---: |
| 1 | 1.56 GiB/s | 9.91 GiB/s | 6.34x |
| 8 | 8.21 GiB/s | 37.65 GiB/s | 4.58x |
| 16 | 10.44 GiB/s | 37.90 GiB/s | 3.63x |

scalar と AVX512 の packed payload は bit 一致した。

`test_qwen4exp_load --skip-gpu --model-dir models/Qwen3.8-Flash-Next-Quark-MXFP4`:

| 状態 | wall | user | sys |
| --- | ---: | ---: | ---: |
| INT6 scalar（warm） | 23.33 s | 206.5 s | 23.2 s |
| INT6 AVX512 + mmap（warm） | **19.07 s** | **24.4 s** | 26.8 s |
| INT6 AVX512 + mmap（cold） | 28.42 s | 22.7 s | 49.6 s |

この mmap 経路は BF16 shard 全体を page cache に載せ得るため、出力 38.743 GiB と
合わせて peak が 100 GiB を超える。メモリ圧を避けるため、量子化の読み込みは
shard ごとの `pread` に変更した。

## ストリーミング読み込み

- shard ごとに `SafetensorsReader` で tensor の絶対 offset だけを取得し、reader は即座に破棄する。
- 64 Ki 行（約 20 MiB）ずつ `pread` し、AVX512 で量子化する。
- 処理済み範囲を `posix_fadvise(POSIX_FADV_DONTNEED)` で page cache から落とす。
- BF16 table 全体を anonymous RAM にも page cache にも保持しない。残るのは INT6 出力
  38.743 GiB と worker あたり約 20 MiB の chunk だけである。

`test_qwen4exp_load --skip-gpu ...` の実測:

| 経路 | wall（1 回目） | wall（warm） | user | sys | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| AVX512 + mmap | 28.42 s | 19.07 s | 24.4 s | 26.8 s | 未測定（source 95.37 GiB が page cache に載り得る） |
| AVX512 + streaming | 31.89 s | 31.51 s | 24.1 s | 36.6 s | **39.24 GiB** |

streaming は page cache を毎回捨てるため warm でも disk 読み込みが残り、mmap より
約 12 s 遅い。代わりに peak RSS は出力サイズ + 約 0.5 GiB に収まる。

## 起動時間

INT6 W32 の streaming load は wall 31.5 s（1 回目 31.9 s）、peak RSS 39.24 GiB。
mmap 経路は warm 19.07 s / cold 28.42 s だが、BF16 の page cache を含めると
メモリ圧が高い。

## 判断

- INT6 W32 FP16 を採用した。6.5 bit/value、38.743 GiB で、BF16 から 59.4% 削減する。
- table 誤差は 2.151%、`key_proj` / `value_proj` 通過後も 2.14% / 2.15% で、
  INT5 W32 の約半分になる。
- 32 値 Hadamard 回転は table 誤差・投影誤差とも改善せず、不採用とした。
- end-to-end の KLD / top-1 は未測定。GPU 0,1 が別の serving で使用中で、
  TP4 に必要な 4 GPU を確保できなかった。モデル全体の品質判断には
  PLE を BF16 のまま動かした場合との logits 比較が必要である。

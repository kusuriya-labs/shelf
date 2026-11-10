# EXL3-INT8 の Qwen3.8-27B 実装と R9700 での確認

Issue #10。仕様は rocm_exl3 の `doc/exl3_int8.md`（固定 INT8 grid + mul24 codebook）。
この記録は shelf への実装の経緯と実測値である。contract は `docs/developer/quantization.md`。

## 実装の範囲

- dense の Qwen3.8-27B（`qwen3_5_text`、64 層、GDN 48 + attention 16）に限る。
- **重みは trellis のまま常駐**（4bpw、27B で約 16 GB）。INT8 の重み行列は VRAM にも CPU にも作らない。
  GEMM kernel が重みを読むたびに trellis から INT8 grid へ展開し、iu8 WMMA に渡す。
- 形式は **mul24**（`mul1` tensor の値が 0xDCD12D）。mul1 の artifact（0x83DCD12D）は読まず、
  `Status::unsupported`（再量子化を促す）で拒否する。mul24 は rocm_exl3 側の量子化器（`EXL3_MUL1_MULT=0xDCD12D` でビルド）が作る。
- decode 用 GEMM（rows <= 64）は K 方向に split し、各 split が `yprime` の別スライスへ書いて `rotate_out` が足す（決定的）。
  17〜64 行の扱いは [mid-size batch の GEMM](../20261110/exl3_int8_mid_rows.md) を参照。

## 形式の選択（mul24）

mul1 の乗数 0x83DCD12D は 32 bit 乗算 + byte 和 + 補正で、1 重みあたりの ALU 命令が多い。乗数を 24 bit
（0xDCD12D）にすると `v_mul_u32_u24` が使え、byte 和は `v_dot4_u32_u8` 1 命令になる。
さらに tile 内の並びを WMMA の iu8 B fragment（lane = t/8、行 8*(lane/16) + j、列 lane%16）に合わせたので、
復号した 8 個の int8 がそのまま B operand のレジスタになる。量子化の質は mul1 と同等
（Qwen3.5-2B の実 Hessian で proxy_err 比 0.999〜1.009、27B 全 64 層の rfn は平均 0.0151）。
検討の全体は rocm_exl3 の `research/codebook/README.md`。

## 検証（R9700、gfx1201、モデルは `qwen38-27b-exl3-4bpw-mul24`、4.0bpw、head 6bit）

- 復号: `test_exl3_int8_decode`（rocm_exl3 の NumPy 参照 = 実機復号器とビット一致）が通る。bits 1〜8 で GEMM が参照と一致。
- kernel: `test_exl3_int8_linear` 全 PASS（suh / svh は F16、trellis は n-major）。回転 + INT8 化と出力回転は CPU の float32 鏡像計算と完全一致、
  GEMM の相対誤差 約 1e-7、理想の式（float64）との相対 L2 誤差 約 0.65%（INT8 活性のノイズ）。split-K（2/3/4/8 分割）も同じ精度。
- KLD（BF16 = llama.cpp に対する、同一の 8 ケース・2632 位置）。rocm_exl3 で測定:

  | | mean KLD | top-1 一致 |
  |---|---|---|
  | 4bpw mul1（同じ量子化器、変更前） | 0.01927 | 0.9457 |
  | 4bpw mul24 fp16 | 0.01973 | 0.9472 |
  | 4bpw mul24 INT8（固定 grid + 活性 INT8） | 0.02167 | 0.9403 |

  mul24 の fp16 は mul1 に対し +0.0005（+2.4%）、INT8 化でさらに +0.0019（約 +10%）。
  rocm_exl3 は mul24 を R9700 と V620 の両方で native に復号する（V620 の decode は mul1 と同等の 22 tok/s）。
- shelf 自体の KLD: 12 位置（4 ケース × 3 位置、prefill）で BF16 に対し 0.0415。同じ位置の rocm_exl3 は INT8 0.0421、fp16 0.0416。
  shelf と rocm_exl3 INT8 の間は 0.0032。
- end-to-end: 128 token のプロンプトから greedy 64 token。生成は自然な文で、速度最適化の前後で先頭 24 token は同一。

## 実測（参考値、`docs/perf/` の正本ではない）

R9700、`shelf-compute`、bf16 KV、各 3〜5 回。main（EXL3 W16A16 を持つ `53898d0`）の上に載せた状態で測った。
同じ量子化器で作った同じモデル（4.0bpw）を、mul1（main の W16A16）と mul24（本実装）で比べる。

decode（`shelf-compute`、128 token プロンプト、63 token）:

| | decode |
|---|---|
| main の EXL3 W16A16（mul1） | 約 43.2 ms/token（23.1 tok/s） |
| EXL3-INT8（mul24、本実装） | 約 37.3 ms/token（26.8 tok/s） |

decode は W16A16 より約 14% 速い。この GPU は他の負荷の有無で decode の所要が 2 倍近くまで振れることがあり、
上は最速側の揃った値を採った。

prefill は main の W16A16 と同じ構造にした（rows >= 256）。trellis を int8 の dense 行列に展開し、hipBLASLt の
int8 GEMM（int8 x int8 -> int32、230〜310 TOPS）で計算する。`shelf-bench pp --mode forward --runs 3 --warmup 1
--device 0 --arena-gib 28`、GPU 時間の median:

| prompt token | mul1（W16A16） | mul24（dense int8） | mul24 / mul1 | 自作の融合 kernel だけ（参考） |
|---:|---:|---:|---:|---:|
| 128 | 174.8 ms | 154.6 ms | 0.88 | 152.5 ms |
| 256 | 283.8 ms | 239.7 ms | 0.84 | 288.4 ms |
| 512 | 400.2 ms | 345.1 ms | 0.86 | 548.2 ms |
| 1024 | 656.2 ms | 583.5 ms | 0.89 | 1077.9 ms |
| 2048 | 1203.8 ms | 1111.9 ms | 0.92 | 2156.8 ms |
| 4096 | 2984.2 ms（`--prefill-chunk 2048` なら 2456.1 ms） | 2100.1 ms（chunk 2048 なら 2295.9 ms） | 0.70（chunk 2048 同士で 0.93） | — |

- mul1 は 2409 行を超える 1 batch で急に遅くなる（2400 token 1424 ms、2416 token 1790 ms）。main の W16A16 は入力回転の
  領域が 80 MiB で、`rows * k * 2` がそれを超えると、down_proj（k = 17408）が dense 経路ではなく融合 kernel に落ちるため。
  4096 token の mul1 の値はその落ちた状態で、chunk 2048 に分ければ戻る。mul24 は行を分けて実行するのでこの段差が無い。
  公平な比較は chunk 2048 同士（0.93）。
- 128 token 以下は融合 kernel のまま（256 行未満）。融合 kernel だけだと 512 token 以上で W16A16 より最大 1.8 倍遅かった
  （INT8 GEMM が約 60 TOPS で頭打ち）ため、dense 経路を加えた。
- 1 つの hipBLASLt GEMM は K 全体を足すので 128 ごとの scale を掛けられない。活性の scale を**行ごとに 1 つ**
  （group scale の最大）にして codes を丸め直す。massive activation の行（1 つの group が scale energy の 50% 超）は
  丸め直さず、融合 kernel で group scale のまま計算する（`exl3_int8_flag_cap` 行まで）。
- 単体: recon、行 scale の丸め直し、int32 の GEMM、フラグ行の slot は CPU の整数計算と完全一致
  （`test_exl3_int8_linear`、フラグ行が slot を超える場合を含む）。
- 実モデルでの品質（`shelf-bench quality`、teacher-forced、融合 kernel のみの出力を基準にした KL）。比較の目安として
  「融合 kernel のまま prefill chunk を 256 にした場合」の KL（数値のノイズの床）を並べる:

  | コーパス | dense（フラグ行あり） | dense（フラグ行なし） | ノイズの床 |
  |---|---:|---:|---:|
  | corpA（KLD ケース連結、1542 位置） | 0.0149 | 0.0145 | 0.0111 |
  | corpB（mtp_perf 4 種連結、1069 位置） | 0.0111 | 0.0227 | 0.0148 |

  フラグ行を融合 kernel にすると、corpB で KL がノイズの床の水準に収まる。短い単独サンプル（json 372 位置）は
  少数の位置で KL が 10 を超え、構成を変えるだけで 0.012〜0.06 に振れる（chunk 長を変えるだけでも 0.025〜0.034）。
  BF16 に対する KLD は未測定（rocm_exl3 の INT8 活性に行 scale を入れた測定が必要）。
- 初回の hipBLASLt 呼び出し（ライブラリ初期化、mul1 も同じ）で最初の 1 回の prefill が約 0.5 s 遅れる。
  `shelf-compute` の 1 回だけの実行では融合 kernel のみの方が速い（約 480 ms 対 約 1015 ms、431 token）。
  上の表は初期化後の値。
- MTP の出力は dense prefill でも target-only と一致した（上の MTP 節の条件で確認）。

decode の速度は変わらない（約 26.6 tok/s）。

rebase 前（main に取り込む前）の実装で decode を 41.7 → 37.1 ms にした経緯:

| | decode |
|---|---|
| trellis 常駐、最適化前 | 約 41.7 ms/token |
| + decode GEMM の split-K | 約 37.4 ms/token |
| + 活性 fragment の先読み | 約 37.1 ms/token |

### GEMM の内訳（`test_exl3_int8_linear` の `EXL3_BENCH=1`、rows=1、4bit）

| 形状 (n × k) | 変更前 | split-K 後 |
|---|---|---|
| 17408 × 5120（gate / up） | 0.093 ms、477 GB/s | 0.092 ms（split 2、変化なし） |
| 5120 × 17408（down） | 0.121 ms、369 GB/s | 0.068 ms、658 GB/s |
| 12288 × 5120 | 0.059 ms、537 GB/s | 変化なし |
| 248320 × 5120（lm_head、6bit） | 1.54 ms、619 GB/s | 変化なし |

### 速度の原因切り分け（format 側の問題か）

- n が小さい線形（down_proj など）は wave が 320 しか立たず、SIMD あたり約 2.5 wave で帯域が 360 GB/s に落ちていた。
  これは形式ではなく並列度の問題で、split-K で解消した。
- gate / up（decode の GEMM 時間の約 4 割）は ALU 律速。決定的な切り分け:
  decode 計算を除いても（`state ^ (w1 >> i)`）1.00 → 1.12 Tweights/s、重みの load を除いても 0.79、両方除いても 0.86 Tweights/s
  で、GEMM の骨格だけで約 0.1 ms かかる。1 重みあたり VALU 約 6.4 命令で、R9700 の VALU 実測ピーク（約 10 T lane-op/s）の約 65%。
  DRAM の天井（約 600 GB/s）に対して 485 GB/s（約 81%）。
- 形式として削れるのは byte 和の `>>2` と 8 bit への詰め直し（約 1 命令/重み、約 10%）。codebook を 6 bit フィールド
  （`x & 0x3F3F3F3F` の byte 和 − 126）に変えれば shift が要らなくなるが、再量子化が必要で、節約は mask と相殺されるため見送った。
  表引き（LDS 64 KB）は LDS の bank 競合で ALU より遅くなる（research 済み）。

### MTP（`--mtp`）との組み合わせ

shelf の MTP 経路（K+1 行の一括 verify、GDN snapshot / compact log replay）を、mul1（main の W16A16）と
mul24 で同じ条件で比べた。R9700、greedy、64 token 生成、`tests/data/mtp_perf` の prose / code / reasoning
（繰り返しなし、約 430〜480 token）、各 1 run。出力（`GENERATED_IDS`）は全条件で target-only と一致した。
target-only は mul1 が 23.2〜23.4 tok/s、mul24 が 26.6 tok/s。

| prompt | K | mul1 tok/s | mul24 tok/s | mul1 受理 | mul24 受理 | verify/round mul1 → mul24 |
|---|---:|---:|---:|---:|---:|---|
| prose | 2 | 39.9 | 47.7 | 1.17 | 1.17 | 48.0 → 40.2 ms |
| prose | 4 | 44.8 | 46.0 | 1.86 | 1.42 | 51.3 → 42.6 ms |
| prose | 7 | 38.1 | 38.8 | 1.86 | 1.42 | 53.4 → 45.5 ms |
| code | 2 | 46.0 | 55.7 | 1.52 | 1.58 | 48.4 → 40.4 ms |
| code | 4 | 47.0 | **66.7** | 2.10 | 2.50 | 51.7 → 42.7 ms |
| code | 7 | 42.4 | 57.0 | 2.26 | 2.65 | 53.7 → 45.7 ms |
| reasoning | 2 | 38.3 | 47.9 | 1.14 | 1.21 | 48.5 → 40.4 ms |
| reasoning | 4 | 33.9 | 46.2 | 1.39 | 1.48 | 55.5 → 42.8 ms |
| reasoning | 7 | 31.5 | 39.0 | 1.39 | 1.48 | 53.7 → 45.8 ms |

- verify（8 行まで）は単一 token decode の 1.1〜1.3 倍の時間で済み、INT8 の split-K 経路（rows <= 16）に入る。
  mul24 の verify は mul1 より 15〜20% 短く、これが速度差の主因である（受理数は prompt で上下する）。
- K=7 より K=4 が速い（受理数が伸びず draft と verify の時間だけ増える）。
- 受理数は MXFP4 版の MTP（約 98%）より大きく低い。MTP block の 4bpw 量子化かプロンプトかは未切り分け。
- 1 run ずつで散布は取っていない。長い context は未測定。

## main への載せ方

- manifest と weight の読み込みは main の EXL3（trellis / suh / svh / marker の 4 plane）をそのまま使い、
  marker の値（codebook の乗数）で format を選ぶ。mul1 は W16A16、mul24 は INT8、他は load が拒否する。
- mul24 の trellis は load 時に CPU で `[nt][kt]` の並びに直す。`[kt][nt]` のままだと、キャッシュに載るベンチでは
  差が出ないが、実モデル（DRAM から流す）では約 2 割遅い。
- workspace は main の EXL3 workspace（xrot と accum）を共有する。xrot に int8 codes と scale、accum に F32 の
  回転前出力を置く。入らない行数は行を分けて実行する（lm_head の 248320 幅など）。
  split-K の分割数は accum に入る行数から決める。

## 未実施・未確認

- BF16 に対する KLD（dense prefill の行 scale）。
- 長い context、TP、MoE の expert、cb0 / mcg、DFlash2 との組み合わせ（MTP は上で測定済み）。
- shelf の teacher-forced な全位置 KLD（`--dump-logits` は sampled row だけなので 12 位置の抜き取り）。
- mul24 の実モデル断片を fixture にすること（現在の fixture は NumPy 参照復号器から再生成した期待値）。
- prefill の GEMM（MT=4、約 61 TOPS）は未最適化。
- FP8 経路の性能が変わっていないことの確認（コードは触っていないが測っていない）。

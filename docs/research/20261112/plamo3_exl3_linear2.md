# PLaMo 3 EXL3-INT8 linear の最適化 第2弾

2026-11-12〜13（JST）、R9700 / gfx1201、EXL3-INT8 4.05 bpw、BF16 KV、greedy、投機なし。開始点は `3d72e00`。
対象は QKV / O / gate-up / down / lm_head の linear。RMSNorm / attention / scheduler / stall の実装は変更していない。

## 継続投入

`plamo_closed_loop.py`、seed 17、32 入力、lengths=1/2/4/6、最大出力 320 token。最初の完了後 20 秒 warmup、90 秒窓。
before / 最終候補を各 c で2回測定する。表示は平均 ± run 間標本 SD。2回目は逆順。30秒探索と診断付き run は主表に含めない。
各 GPU 実行に UUID と flock を指定し、run ごとに解放する。全 run の値、完了数、窓の時刻は JSON に保存する。

| c | 第1弾報告 tok/s | 同条件 before tok/s | 第2弾途中 final2 tok/s | 第2弾最終 tok/s | before 比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 151.73 | 151.73 ± 0.24 | 155.28 ± 0.00 | 155.78 ± 1.45 | +2.67% |
| 4 | 541.09 | 539.92 ± 1.23 | 546.36 ± 2.96 | 554.09 ± 4.34 | +2.62% |
| 16 | 1609.39 | 1606.36 ± 5.52 | 1619.29 ± 2.44 | 1729.01 ± 2.18 | +7.64% |
| 64 | 2784.13 | 2782.01 ± 3.79 | 2815.21 ± 1.28 | 3279.04 ± 4.67 | +17.87% |
| 128 | 3151.92 | 3160.99 ± 0.93 | 3266.39 ± 9.16 | 3873.52 ± 4.89 | +22.54% |

第1弾の報告値と今回の before 再測定を区別する。採用比較は今回の before。final2 は追加指示前、prefill は丸めを変える不採用候補、rowscaleはcold悪化で不採用、最終候補はnocapである。

## 行数帯と head

head は OutputGather が選択した OUTPUT_ROWS だけを処理する。prefill 全行を投影する無駄はなかった。BF16 logits を丸めてから F32 に変換する順序を維持する。
次表は30秒の診断窓。Python / C++ の monotonic clock を照合して起動と drain を除外する。診断 overhead があるため主速度に使わない。

| c | phase | M 帯 | steps | step 平均 ms | 窓内 step 時間比 |
| ---: | --- | --- | ---: | ---: | ---: |
| 64 | before-exact-0 | 1-64 | 616 | 14.276 | 29.55% |
| 64 | before-exact-0 | 65-128 | 210 | 20.696 | 14.60% |
| 64 | before-exact-0 | 129-255 | 313 | 28.675 | 30.16% |
| 64 | before-exact-0 | 256-511 | 167 | 45.405 | 25.48% |
| 64 | before-exact-0 | 512-2048 | 1 | 65.143 | 0.22% |
| 128 | before-exact-1 | 65-128 | 149 | 25.394 | 12.75% |
| 128 | before-exact-1 | 129-255 | 210 | 32.040 | 22.67% |
| 128 | before-exact-1 | 256-511 | 354 | 49.584 | 59.14% |
| 128 | before-exact-1 | 512-2048 | 24 | 67.191 | 5.43% |
| 64 | nocap-0 | 1-64 | 733 | 14.211 | 35.02% |
| 64 | nocap-0 | 65-128 | 215 | 16.381 | 11.84% |
| 64 | nocap-0 | 129-255 | 402 | 21.997 | 29.73% |
| 64 | nocap-0 | 256-511 | 190 | 36.338 | 23.21% |
| 64 | nocap-0 | 512-2048 | 1 | 57.099 | 0.19% |
| 128 | nocap-1 | 65-128 | 196 | 20.448 | 13.50% |
| 128 | nocap-1 | 129-255 | 256 | 25.396 | 21.91% |
| 128 | nocap-1 | 256-511 | 438 | 39.698 | 58.59% |
| 128 | nocap-1 | 512-2048 | 31 | 57.480 | 6.00% |

## linear と残る時間

LINEAR_EXL3_INT8 の外側 HIP event を合計し、内包する EXL3 stage を再加算しない。表示は捕捉 step の中央値。
step の形状は実装間で1対1に対応しないため、M の中央値も併記する。event と stepwall の差を厳密な stall として扱わない。

| c | phase | M 帯 | samples | M 中央値 | linear ms | head ms | RMSNorm ms | attention ms | 有効 TOPS / peak比 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | final2 | 1-64 | 14 | 64 | 9.320 | 0.713 | 1.868 | 3.241 | 35.72 / 9.33% |
| 64 | final2 | 65-128 | 3 | 98 | 15.155 | 0.709 | 1.940 | 3.558 | 35.30 / 9.22% |
| 64 | final2 | 129-255 | 16 | 168 | 18.493 | 0.718 | 2.060 | 5.074 | 44.46 / 11.61% |
| 64 | final2 | 256-511 | 3 | 300 | 41.192 | 0.712 | 2.559 | 7.767 | 34.76 / 9.08% |
| 128 | final2 | 65-128 | 3 | 127 | 16.127 | 1.149 | 2.025 | 6.122 | 41.00 / 10.70% |
| 128 | final2 | 129-255 | 3 | 218 | 22.619 | 1.163 | 2.134 | 7.226 | 47.18 / 12.32% |
| 128 | final2 | 256-511 | 14 | 364 | 42.950 | 1.139 | 2.739 | 9.994 | 41.20 / 10.76% |
| 64 | nocap | 1-64 | 22 | 64 | 9.329 | 0.716 | 1.884 | 3.201 | 35.72 / 9.33% |
| 64 | nocap | 65-128 | 4 | 99 | 12.452 | 0.710 | 1.935 | 3.580 | 39.87 / 10.41% |
| 64 | nocap | 129-255 | 12 | 170 | 14.690 | 0.714 | 2.050 | 5.090 | 54.32 / 14.18% |
| 64 | nocap | 256-511 | 4 | 358 | 27.995 | 0.716 | 2.702 | 8.313 | 60.51 / 15.80% |
| 128 | nocap | 65-128 | 3 | 127 | 13.521 | 1.139 | 1.985 | 6.048 | 48.99 / 12.79% |
| 128 | nocap | 129-255 | 7 | 187 | 15.270 | 1.141 | 2.085 | 6.946 | 59.41 / 15.51% |
| 128 | nocap | 256-511 | 14 | 364 | 27.679 | 1.145 | 2.766 | 10.470 | 60.94 / 15.91% |
| 128 | nocap | 512-2048 | 1 | 572 | 37.475 | 1.164 | 3.455 | 14.769 | 72.94 / 19.05% |

最終候補のprojection別中央値。各中央値は独立集計であり、足して総中央値を再構成しない。

| c | M 帯 | QKV ms | O ms | gate/up ms | down ms | head ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 1-64 | 1.273 | 1.087 | 4.023 | 2.223 | 0.716 |
| 64 | 65-128 | 1.909 | 1.875 | 4.189 | 3.781 | 0.710 |
| 64 | 129-255 | 2.232 | 1.945 | 5.857 | 3.953 | 0.714 |
| 64 | 256-511 | 4.346 | 3.833 | 11.237 | 7.856 | 0.716 |
| 128 | 65-128 | 1.972 | 1.896 | 4.695 | 3.854 | 1.139 |
| 128 | 129-255 | 2.270 | 1.946 | 5.931 | 3.979 | 1.141 |
| 128 | 256-511 | 4.377 | 3.792 | 11.326 | 7.357 | 1.145 |
| 128 | 512-2048 | 5.620 | 4.428 | 16.230 | 10.033 | 1.164 |

演算量は `2*(M*2,327,838,720 + output_rows*275,251,200)`。この値を linear event 合計で割る。
公称 INT8 dense 383 TOPS を使用する。回転、復号、再量子化、flag 補正、launch 区間を含む有効率であり、WMMA busy の実測ではない。
公称 peak の出典、SGPR / VGPR / LDS / scratch / occupancy、ISA、compiler A/B は [コンパイラ解析](plamo3_exl3_linear2_compiler.md)を参照。

## 採用した実装

- 65 行以上の 4 bit は trellis を直接使う LDS GEMM、5 bit は一時展開した INT8 の LDS GEMM。QKV と gate/up の group を維持し、出力回転と BF16 丸めを融合する。
- 65〜255行および補正行は128 Kごとの整数内積とFP32 scaleの累積順を保つ。swizzle 付き入力 LDS を二重 buffer にし、128 bit の load で WMMA 2回分を運ぶ。MT=4 は B の次 group の先読みを省く。
- 256 行以上は RowScale とenergyによるflagの決定を維持する。compact用の行数制限は除去する。通常行は全Kのint32内積、補正行はgroupごとの内積/scaleを同じGEMMで処理する。compact copy、行数上限と重複GEMMを省き、全補正行を処理する。
- M=1 の 4 bit QKV / O の入力回転融合、M<=16 / N>=10240 / 全 split=1 の GEMM+out 融合、65〜255 行 head の MT4 を維持する。
- 他モデルと明示的な dense / AB override は既存 selector を使う。新しい経路は既存 workspace を再利用し、永続 weight buffer を追加しない。

実 checkpoint のbodyは96 group。65〜255行は84 groupが2 launch、12 groupが3 launchで、headが17行以上なら合計207 launch。256行以上はgroupごとのrowstat/rowfinishを加えて399 launch。
M=1 は230 launch、M=4/16 は266 launch、M=64 は291 launch。169 logical linear は維持する。
単体 micro の rotate / out の約10 µsには launch / event 区間の影響がある。169×20 µsを除去可能な固定費とは扱わない。

## 候補ごとの単体比較

合成 resident trellis、seed 17、warmup 20 回、20 回の p50 を 5 run。表示は run p50 の中央値 ± 標本 SD、単位 µs。
128 MiB の memset を GEMM / linear の event 前に置き、AB/BA を交互に測定した。個別 projection の表であり、統合 QKV 全体の時間ではない。

| 候補 | M / N×K / bit | before µs | candidate µs | 採否 |
| --- | --- | ---: | ---: | --- |
| 入力融合 Q 単体 | 1 / 3072×2560 / 4 | 35.48 ± 2.76 | 32.90 ± 0.60 | 採用（M=1/4bit） |
| 入力融合 O | 1 / 2560×3072 / 4 | 37.42 ± 1.48 | 34.38 ± 0.87 | 採用（M=1/4bit） |
| 入力融合 down | 1 / 2560×10240 / 4 | 73.32 ± 3.13 | 91.14 ± 1.42 | 不採用 |
| 入力融合 K/V 5bit | 1 / 512×2560 / 5 | 30.10 ± 2.04 | 37.78 ± 1.42 | 不採用 |
| 入力融合 M=4 | 4 / 512×2560 / 4 | 28.68 ± 1.78 | 33.28 ± 1.44 | 不採用 |
| GEMM+out gate | 1 / 10240×2560 / 4 | 71.92 ± 2.51 | 67.96 ± 0.96 | 採用 |
| GEMM+out head | 1 / 107520×2560 / 8 | 629.83 ± 0.78 | 614.63 ± 0.91 | 採用 |
| GEMM+out gate | 4 / 10240×2560 / 4 | 73.08 ± 6.27 | 68.30 ± 0.54 | 採用 |
| GEMM+out head | 16 / 107520×2560 / 8 | 696.47 ± 1.79 | 635.71 ± 0.92 | 採用 |
| shuffle load O | 1 / 2560×3072 / 4 | 35.22 ± 1.54 | 35.50 ± 0.52 | 不採用 |
| shuffle load head | 1 / 107520×2560 / 8 | 628.94 ± 1.08 | 623.54 ± 0.45 | 不採用 |

### 不採用・保留

入力融合の初期候補は全 K group を各 CTA で回転した。M=1 down、5 bit と M=4 は余分な回転と LDS により遅くなり、採用 selector から除いた。
shuffle による load 数の削減は、O が悪化し、down / head の差も小さく、実サーバーで再現する優位を確認していない。
activation fragment を s ループ内で読む stream-A も試した。M=96 gate / MT=2 は 109.56→約126.7 µs、down は 163.5→約170.7 µs となり不採用。MT=8 と 8 wave も探索範囲で決定的な優位を示さなかった。

256 行以上も exact grouped に置き換えた wide 候補は、30 秒の探索で c=64 が 2809.07 / 2815.47、c=128 が 3168.27 / 3148.20 tok/s。
丸めも変わり、主指標の反復改善と拡張品質を確認していないため採用しない。この旧 wide 候補と、後述する LDS / epilogue 融合版は別実装である。
小さい K/V の split=4 も試した。30 秒探索の c=64 は flagged 2823.20 対 split4 2828.77、c=128 は 3259.17 対 3263.63 tok/s。差は 0.2% 未満で反復改善を確定していないため split を戻した。
これら 30 秒値は探索用であり、90 秒の主表との改善率を算出しない。

hipBLASLt の zero-workspace 16 heuristic を M=320/512 の実 PLaMo shape で比較した。全候補の int32 accumulator は基準と bit 一致した。
M=320 down は 195.51→186.57 µs、M=512 down は 221.69→194.99 µs だが、形状ごとの index の違いがあり、固定 solver の継続投入改善を確認していない。採用コードには固定 index を加えなかった。
greedy head+argmax は未実装。BF16 logits の丸めと公開 logits 出力の contract を維持する必要がある。logits の公開 contract を変える argmax 融合は今回採用していない。

## large-M と compiler の候補

MT=1/2/4/8、LDS swizzle、B pipeline、launch_bounds、waves_per_eu、scheduler barrier、非一時的 load を比較した。
compiler flag は candidate の PLaMo TU に限定し、expert scheduler、schedule metric bias、memory clause、unroll-count、max-ILP を試した。全体 flag は変更しない。
旧MT4のoccupancy=8指定はVGPR spillを発生させた。新しいper-group MT4はVGPR206 / LDS16,896B / scratch0 / waves7、RowScale保持版の4bit direct MT4はVGPR171 / scratch0 / waves8。単純にoccupancyを下げるだけでは改善しない。
expert scheduler の large-M 候補は bit 不一致、max-ILP は compiler crash、unroll-count=1/2 と swizzle 除去は速度悪化により不採用。
単体の一部で非一時的 load が大きく改善しても、継続投入で同じ短縮を確認できなかった。主指標で反復改善を確定していない候補は採用しない。
全候補の before/after・run 間分散・bit 比較はコンパイラ解析の表と JSON に保存する。

RowScaleを省くexact版は90秒窓2回でc64=3317.57±0.27、c128=3992.17±13.25 tok/sだった。cold差も改善したが、M320のtop1が94.29→92.46% / 95.43→93.94%（JA/EN）となり、同じMのbeforeから1pt以上低下したため不採用。KLは0.017361/0.015814、対応位置KL差も記録した。
行数上限を維持する融合版は固定Mで全dumpがbit一致したが、cold10回のrun間差が平均5.18→6.87、中央値6→7、最大9→11となり不採用。compact bufferが不要になったため予算を除去し、行固有のenergy分類だけで全補正行を扱う。
M=1 の K-slice 入力融合も試した。c=1 は156.12/156.80 tok/s、全80 dump一致だったが、cold10回の中央値6→7、最大11→13を理由に不採用とした。平均の差だけで採用しない。

## 品質と cold

各方向18文・875位置、別集計の WikiText 521位置。c=1/64 の全 dump の SHA256 と対応位置 KL 差を確認する。

| batch | 方向 | positions | KL | top-1 | incumbent と bit一致 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | ja-en | 875 | 0.018016113 | 92.9143% | True |
| 1 | en-ja | 875 | 0.015159608 | 93.7143% | True |
| 1 | wiki | 521 | 0.050064005 | 90.2111% | True |
| 64 | ja-en | 875 | 0.017976210 | 93.0286% | True |
| 64 | en-ja | 875 | 0.014622981 | 94.0571% | True |
| 64 | wiki | 521 | 0.050514668 | 90.2111% | True |

M=192 の日英・英日各2文では、bodyが新経路を通る状態で全4 dumpのbit一致を確認した。

M=320 は一時hookで0始まり319番の行だけを採点する。headは1行、bodyは320行。最終候補の最終行は旧先頭行と全36dumpのbit一致を確認する。旧最終行との差はKL/top1と位置別KL差で判定する。
旧 flagged-row の先頭行だけを採点して RowScale の差を隠す評価を避ける。両方向875位置と対応位置 KL 差を別集計する。

| M320 variant | 方向 | positions | KL | top-1 |
| --- | --- | ---: | ---: | ---: |
| before | ja-en | 875 | 0.018430100 | 94.2857% |
| before | en-ja | 875 | 0.014958615 | 95.4286% |
| candidate | ja-en | 875 | 0.018581836 | 94.7429% |
| candidate | en-ja | 875 | 0.015530149 | 95.8857% |

旧先頭行は行数予算内なのでenergy分類どおりに補正される。新実装はその分類を全行に適用する。

| 旧先頭行 | 方向 | positions | KL | top-1 |
| --- | --- | ---: | ---: | ---: |
| before row0 | ja-en | 875 | 0.018581836 | 94.7429% |
| before row0 | en-ja | 875 | 0.015530149 | 95.8857% |

coldはserverを起動し直すc=64を各10回、c=1を各2回。全45組の出力差を比較する。

| variant | run 間差 平均 /64 | 中央値 /64 | 最大 /64 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 5.18 | 6.0 | 9 |
| final | 0.36 | 0.0 | 1 |

固定 M の bit 一致と、HTTP 到着時刻で batch 形状が変わる cold の出力差を区別する。

## 目標と残り

c=64: 3279.04 tok/s。目標4000は未達。差は720.96 tok/s、必要な追加改善は21.99%。
c=4: 554.09 tok/s。目標700は未達。差は145.91 tok/s、必要な追加改善は26.33%。

残る線形処理の帯別時間は上表のとおり。gate/up と down の寄与は JSON の projection_us に残す。
decode-only が目標を超えることと、prefill を混ぜた継続投入の到達を区別する。RMSNorm / attention はこの作業では変更しない。

## 検証と再現

関連10 ctestとPLaMo serverの5テストを対象とする。結果と exit code は JSON。skip は合格として扱わない。`--all` / rocprofv3 は実行しない。
計測 hook は `plamo_linear2_timing.patch`、拡張品質 hook は `plamo_linear2_large_quality.patch` に保存し、通常コードと build から除去する。push / PR は作成しない。
before-b/c16 は窓後の結果パイプ待ちで lock 解放が遅延したため除外し、同条件の before-c/c16 に置換した。詳細は protocol_exclusions。

```bash
export PATH=/tmp/venv-plamo/bin:/tmp/venv-so/bin:$PATH
python3 docs/research/20261112/plamo_linear_runs.py --compute build-gfx1201/plamo-linear2/shelf-compute.final-nocap --output build-gfx1201/plamo-linear2/runs --tag nocap-final-a
python3 docs/research/20261112/plamo_linear2_compiler.py --variant baseline
python3 docs/research/20261112/plamo_linear2_prefill_build.py --variant mt4-no-pipe --define PLAMO_MT=4 --define PLAMO_PIPE_B=0
python3 docs/research/20261112/plamo_linear_quality.py --bench build-gfx1201/plamo-linear2/shelf-bench.final-nocap --output build-gfx1201/plamo-linear2/quality-final-nocap
python3 docs/research/20261112/plamo_linear2_report.py --require-complete
python3 docs/research/20261112/plamo_linear2_compiler_report.py
python3 docs/research/20261112/plamo_linear2_markdown.py
```

通常 build の binary を保存してから各 runner に渡す。GPU UUID / flock / library path は runner 内で設定する。
詳細: [全結果 JSON](plamo3_exl3_linear2.json)、[compiler / ISA JSON](plamo3_exl3_linear2_compiler.json)。

## commit

実装、計測 harness、結果を小さく分けて commit する。最終 hash は Git 履歴と完了報告を参照。

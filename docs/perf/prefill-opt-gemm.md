# R9700 27B EXL3-INT8 prefill の GEMM と周辺処理

## 対象と判定方法

ベースは `feature/prefill-breakdown` の `7e6916a`。gfx1201、ROCm 7.14、Release、R9700 `GPU-a8e9ddefa2d60f55`、BF16 KV、chunk 2048。途中で `renderD130/device/hwmon/hwmon5/power1_cap` を読み、300000000 µW を確認した（測定中の連続記録ではない）。GPU を使う各実行は `flock /tmp/r9700.lock` で排他し、ビルドは `nice -n 10`、最大 16 並列。`rocprofv3` は使用しない。性能の採否は warmup 後に交互 A/B で各 5 回、中央値と幅 / 中央値を確認し、幅が 30% 以上なら FAIL とする。独立 microbenchmark の結果と実モデルの実行結果を区別する。

対象モデルは `/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24`。prompt は `sql_code_1025.psktok` の反復 32768 ID、SHA256 `3e01d5113e00df82bf116cba9b3ad8e6e851a132ddc379a1481480745bd2a6bb`。生成方法と基準診断の定義は [内訳](prefill-breakdown-r9700.md) にある。品質用 `/tmp/corpus_long.tok` は SHA256 `5b8becd6ebe6f41d104b6d5b8b39036b163fb7e2b8c13541b9280c1ac0ba0c49`。

形状別の全候補と単体試作の生ログは [data](prefill-opt-gemm-data/) に保存した。単体ベンチの一時ソースと実行ファイルは `build-gfx1201/` に置き、GPU 実行は同じ lock と device 指定で行った。候補探索の初回は `libhipblaslt.so.1` の実行時パス不足で開始前に失敗し、ROCm の library path を設定して再実行した。

代表コマンドは次のとおり。各長さ・条件は別プロセスである。

```sh
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib
export ROCM_PATH=/opt/rocm/core-7.14 ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
flock /tmp/r9700.lock ./build-gfx1201/shelf-bench pp \
  --model-dir /home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24 \
  --mode forward --tokens-file /tmp/shelf-prefill-32768.psktok \
  --prompt-tokens 2048 --prefill-chunk 2048 --kv-cache-dtype bf16 --warmup 1 --runs 5
```

単体試作は以下の形で build directory にビルドし、各実行を個別に lock した。`gemm_sweep`、`workspace_sweep`、`fused_probe`、`lds_probe` は `-L/opt/rocm/core-7.14/lib -lhipblaslt` が必要。独立 GEMM bench の初回起動で library path が欠けていたため、`rotate_sweep` も次の共通リンク設定で再ビルドした。ファイル名は上記 data のログ名と対応する。

```sh
nice -n 10 /opt/rocm/core-7.14/bin/hipcc --offload-arch=gfx1201 -O3 -Iinclude \
  build-gfx1201/rotate_sweep.hip -L/opt/rocm/core-7.14/lib \
  -Wl,-rpath,/opt/rocm/core-7.14/lib -lhipblaslt -o build-gfx1201/rotate_sweep
flock /tmp/r9700.lock ./build-gfx1201/rotate_sweep
flock /tmp/r9700.lock ./build-gfx1201/tests/test_exl3_int8_prefill_rotate
```

品質比較は上記環境変数のまま `flock /tmp/r9700.lock ./build-gfx1201/shelf-bench quality --model-dir /home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24 --tokens-file /tmp/corpus_long.tok --context 2048 --positions 64 --prefill-chunk 2048 --kv-cache-dtype bf16 --dump build-gfx1201/quality-baseline.bin` で基準を作り、各候補の環境変数を設定して `--dump` を `--compare build-gfx1201/quality-baseline.bin` に置き換えて行った。

## 現行の実行と形状

2048 token の現行基準は GPU event p50 1154.8744 ms、min 1148.7068、max 1157.1355、幅 / p50 0.73%。これはこの実験での単一プロセス 5 回の測定であり、別の日の 1143.59 ms との比較を採否判定に使わない。

同じ設定で追加した長文の現行基準は次のとおり。候補との交互比較ではないため単独の参考値である。

| prompt | p50 ms | min / max ms | 幅 / p50 |
|---:|---:|---|---:|
| 2048 | 1154.8744 | 1148.7068 / 1157.1355 | 0.73% |
| 8192 | 4898.3193 | 4861.0513 / 5286.9722 | 8.70% |
| 32768 | 24010.5723 | 24005.2148 / 24188.5254 | 0.76% |

モデルの量子化設定に記録された 64 層の `svh` / `suh` 長から、実際の 400 行列は `(N,K): 回数` が `(17408,5120):128`、`(5120,6144):64`、`(5120,17408):64`、`(10240,5120):48`、`(6144,5120):48`、`(1024,5120):32`、`(12288,5120):16`。Saber で連結された gate/up 34816×5120 と shelf の別々の 17408×5120 は同一 dispatch ではない。

## 案 1: hipBLASLt algorithm / workspace

既存 `SHELF_EXL3_INT8_TUNE` は既定 off、workspace 上限 0、候補ごとに 2 回計って最小値を選ぶ実装であり、今回求める warmup・交互・5 回 p50 による探索とは異なる。そこで別の合成 INT8 入力を使う GPU event ベンチで、7 形状、workspace 上限 0 / 64 MiB の各上位 16 候補を測った。各候補は現行 heuristic 先頭を同じ問題に対する対照として warmup、5 回交互に計測し、min / p50 / max を記録した。224 候補の INT32 出力は要素単位で現行と一致した。ここでの正しさは合成した一入力での整数積に限る。

上限 0 の各形状で最良だった候補を以下に示す。時間は候補と交互に測った対照 / 候補の p50、率は `1 - 候補 / 対照`。全行で各 5 回の幅 / p50 は 30% 未満。選択バイアスと clock 変動を含む単体探索結果であり、独立再測定と実モデルの値ではない。tile は solution 名、split-K は明示設定なしの library heuristic（solution 名は `GSUn1`、`SK0`）である。

| N / K | 対照 index | 候補 index | tile | 対照 / 候補 ms | 短縮 | 候補 workspace |
|---|---:|---:|---|---|---:|---:|
| 17408 / 5120 | 140568 | 140558 | 128×128×64 | 1.677146 / 1.615706 | 3.7% | 0 |
| 5120 / 6144 | 140567 | 140561 | 128×128×64 | 0.579115 / 0.511315 | 11.7% | 0 |
| 5120 / 17408 | 140646 | 140674 | 128×128×64 | 2.628737 / 1.690145 | 35.7% | 0 |
| 10240 / 5120 | 140562 | 140558 | 128×128×64 | 0.882433 / 0.864472 | 2.0% | 0 |
| 6144 / 5120 | 140619 | 140561 | 128×128×64 | 0.594355 / 0.520715 | 12.4% | 0 |
| 1024 / 5120 | 139963 | 139994 | 128×128×64 | 0.127559 / 0.123959 | 2.8% | 0 |
| 12288 / 5120 | 140562 | 140558 | 128×128×64 | 1.067151 / 1.044151 | 2.2% | 0 |

64 MiB を許可した heuristic 上位 16 候補も全形状で workspace 0 だった。追加で `hipblaslt_ext::getAllAlgos` の INT8 / INT32・転置条件に合う 780 algorithm を各形状に対して `matmulIsAlgoSupported` で問い合わせた結果、必要量が 1 byte 以上 64 MiB 以下の対応候補は 7 形状とも 0 件だった。個別の split-K 設定を書き換えた探索は行っていないため、それを含む全構成に有限 workspace の候補がないとは断定しない。差が比較的大きく、workspace 上限を変えた別の探索でも改善方向が一致した 3 形状（5120×17408 は 140674、5120×6144 と 6144×5120 は 140561）について、`SHELF_EXL3_INT8_PREFILL_ALGO=1` の opt-in で試す。M=2048、INT8→INT32、workspace 0、gfx1201、および hipBLASLt git revision `cd957402` のとき、実行時 heuristic の上位に該当 index がある場合にだけ選ぶ。その他は既存 heuristic に戻る。初回に探索計測は行わず、thread-local に device と固定された形状・dtype・workspace に対する algorithm を cache する。追加 VRAM は 0。`SHELF_EXL3_INT8_PREFILL_ALGO_LOG=1` で最初の選択 index を出す。すべての形状と workspace の自動 tuning や永続 cache は未実装である。

実モデルの quality 実行時ログで 3 形状について index 140561、140561、140674 が選択されたことを確認した。context 2048、64 位置、chunk 2048 の全語彙比較は `QUALITY_KL_MEAN=0`、`QUALITY_LOGIT_MAX_ABS_DELTA=0`。実モデル 2048 token の別プロセス診断では INT8 GEMM の 400 行列合計 p50 が 497.276 → 459.114 ms（7.7%）だった。交互 A/B は計測中。

## 案 3: 回転・scale・量子化

コード上、出力は同じ行の 128 列についてのみ Walsh-Hadamard 変換し、各列の `svh` を掛ける。入力の補正元は通常行では `acc * rowscale * S_FIX`、flag 行では別 GEMM の `yprime2` である。したがって N 方向に 128 列全体を保持できる自前 GEMM なら融合を設計できるが、現在の hipBLASLt にはこの変換を挿入する epilogue がない。BF16 変換は Hadamard と `svh` の後で行う必要がある。行ごとの flag、上限超過時の扱い、有限精度での演算順も維持する。

2048 行・400 行列の `sum(K)=2,899,968`、`sum(N)=3,899,392` に基づく論理 traffic の概算を示す。基準 event 時間は [内訳](prefill-breakdown-r9700.md) の別実行における診断値であり、実際の DRAM 転送量ではない。キャッシュ再利用、flag 件数、row stride、再ロードを含めない下限や仮定である。

| 段階 | 主な中間データの論理量 | 時間と見かけの比 |
|---|---|---|
| 入力回転・量子化 | BF16 入力 read 11.878 GB、INT8 codes write 5.939 GB、F32 scales write 0.186 GB。`suh` は列ごとに最低 0.0058 GB、行ごとに読めば 11.878 GB | 126.36 ms。入力・code・scale の 18.003 GB / 時間 = 142.5 GB/s |
| rowscale・flag | scales read 最低 0.186 GB（rowstat）と 0.186 GB（rowfinish の各 group を 1 回と仮定）、通常行の codes read/write 最大 11.878 GB。flag 行は codes/scales の compact copy、slots/count の read/write が別途必要 | 47.79 ms。通常行の codes 全量 read/write と scales 2 回の仮定で 12.249 GB / 時間 = 256.3 GB/s。flag、同値 scale の書込み省略を考慮しない |
| 出力回転 | INT32 または flag の F32 結果 read 31.944 GB、BF16 output write 15.972 GB、`svh` は列ごとに最低 0.0078 GB、行ごとに読めば 15.972 GB。slots と rowscale は行ごとに計 0.0066 GB | 204.73 ms。主な read/write 47.916 GB / 時間 = 234.0 GB/s。F32 出力では write が増える |

Saber の `mxfp4_quant.hip` の add+rms+quant、silu+mul+quant は、BF16 の中間丸めを明示してから FP8 quant を行う。shelf は重み固有の `suh` と 128 要素ごとの scale / flag を持つため、その融合をそのまま同等とはみなせない。残差加算・norm や複数行列との融合は、producer の共有可能性と既存の BF16 丸め位置を実モデルで確認する必要があり、今回の実装・実測対象には含めていない。

これらは実効 DRAM 帯域の測定ではない。帯域だけが律速と断定できず、現 kernel は Hadamard の各段で shared memory と barrier も使う。rowscale と回転量子化の単純融合では、全 group の scale を集計してから各 code の再丸めと、決定的な flag 順序・cap 選別を行う依存が残る。

### 出力回転の shuffle 試作

既存 kernel の 128 要素 Hadamard の最初の 5 段は同一 wave 内で閉じる。前 5 段を `__shfl_xor`、残り 2 段を従来どおり shared memory で実行する試作を比較した。加減算の順番と、flag 行の `yprime2`、通常行の INT32→scale→Hadamard→`svh`→BF16 の順は保持する。`SHELF_EXL3_INT8_PREFILL_ROTATE=1` の場合だけ dense prefill の 256 行以上で呼び、既定は off。追加入力 workspace はない。

単体ベンチは M=2048、通常行と flag 行を混在させた決定的な入力、warmup、5 回交互、GPU event。列は min / p50 / max ms、bit 不一致数は該当出力全要素の比較。

| N | 出力 | 現行 | 試作 | bit 不一致 |
|---:|---|---|---|---:|
| 1024 | BF16 | 0.047638 / 0.048119 / 0.048678 | 0.038638 / 0.038719 / 0.038998 | 0 |
| 1024 | F32 | 0.048398 / 0.048718 / 0.049438 | 0.038718 / 0.038918 / 0.039599 | 0 |
| 17408 | BF16 | 0.848521 / 0.852721 / 0.856721 | 0.727046 / 0.728926 / 0.733606 | 0 |
| 17408 | F32 | 0.997434 / 1.020233 / 1.027952 | 0.917917 / 0.923637 / 0.964516 | 0 |

全条件の幅 / p50 は 30% 未満。ただし合成データの単体結果であり、総合 A/B を満たすまでは採用候補と確定しない。これはメモリ traffic を減らす試作ではなく同期削減である。

実モデル 2048 token の各候補単独・両候補有効時の opt-in semantic timing は別プロセスで各 warmup 1・5 回。各 run の stage event を 400 行列分合計し、その 5 合計の p50 を示す（交互 A/B ではない診断値）。5 run × 3797 event 行で欠落なし。

| stage | 現行 ms | shuffle ms | algorithm ms | 両候補 ms |
|---|---:|---:|---:|---:|
| 回転量子化 | 128.101 | 129.035 | 127.875 | 128.754 |
| rowscale・flag | 48.320 | 48.856 | 47.917 | 48.069 |
| 展開 | 51.818 | 51.939 | 51.891 | 51.970 |
| INT8 GEMM | 497.276 | 495.924 | 459.114 | 470.565 |
| flag GEMM | 26.454 | 26.730 | 26.095 | 26.184 |
| 出力回転 | 207.836 | 177.660 | 206.430 | 176.837 |

両候補同時有効の 2048 token 診断初回は、出力回転の stage p50 175.830 ms、GEMM p50 485.839 ms だったが、入力回転量子化の 5 回幅 / p50 が `(356.733 - 126.117) / 127.296 = 181.2%` と 30% を超えたため、**診断全体を FAIL** とし採否根拠に使わない。原因は未特定。再測定では上表のとおり各 6 stage の幅 / p50 が 0.02〜7.45% に収まり、各 stage の p50 を採用した。

8192 token では event 上限を避けた現行 4 回＋別プロセス 1 回と、両候補有効の完全な 5 run 分を集計した。候補 5 run の内訳は欠落した 5 回診断の先頭の完全な 1 run と、欠落のない単発診断 4 run である。各 run 15188 event、stage あたり 1600 行列を確認した。両条件とも stage の幅 / p50 は 30% 未満。異なるプロセス・時点の診断であり、交互 A/B ではない。

| stage | 現行 p50 ms | 両候補 p50 ms |
|---|---:|---:|
| 回転量子化 | 514.552 | 513.202 |
| rowscale・flag | 194.178 | 192.201 |
| 展開 | 207.518 | 207.821 |
| INT8 GEMM | 2005.348 | 1921.718 |
| flag GEMM | 103.668 | 102.006 |
| 出力回転 | 837.002 | 710.219 |

長文の診断を 5 回同一プロセスで行うと、semantic timing の累積上限 65536 event pairs に達する。8192 の試行は 10404 pairs、32768 の試行は 238224 pairs が欠落したため、欠落部分を stage の p50 に使わない。32768 は 1 回でも 60752 pairs なので、1 回ずつ別プロセスで診断する必要がある。先行基準の 32768 token は [内訳データ](prefill-breakdown-r9700-data/measurement-logs.tar.gz) の `long`〜`long5` 各 60752 event を集計し、5 回の stage 合計 p50 が回転量子化 2063.630、rowscale 776.467、展開 830.869、GEMM 8031.289、flag GEMM 429.383、出力回転 3352.038 ms だった。これは別時点の基準であり、候補との交互比較ではない。

32768 token の両候補についても、欠落した 5 回診断の先頭の完全な 1 run と、欠落のない単発診断 4 run を集計した。各 run は 60752 event、stage あたり 6400 行列で、各 stage の幅 / p50 は 0.47〜9.90%。基準は前述の別時点の 5 run で、交互 A/B ではない。

| stage | 先行基準 p50 ms | 両候補 p50 ms |
|---|---:|---:|
| 回転量子化 | 2063.630 | 2093.726 |
| rowscale・flag | 776.467 | 784.921 |
| 展開 | 830.869 | 834.473 |
| INT8 GEMM | 8031.289 | 7305.391 |
| flag GEMM | 429.383 | 436.030 |
| 出力回転 | 3352.038 | 2870.062 |

出力回転の診断上の差は 30.176 ms（14.5%）。同一環境の交互実行ではないため、総合改善の判定は別途行う。専用テスト `flock /tmp/r9700.lock ./build-gfx1201/tests/test_exl3_int8_prefill_rotate` は BF16 / F32、通常行と flag 行、余白を含む出力 stride で bit 比較に成功した。


### 入力回転量子化と rowstat の融合試作

1 block が 1 行を担当し、2 個ずつ 128-K group の量子化を実行してから、同 block 内で従来と同じ順序の scale 最大値・平方和・flag 判定を行う試作を作った。元の rotate_quantize と rowstat の 2 launch を対照とし、M=2048、K=5120 の合成入力、warmup 後 5 回交互で event p50 は **0.233241 → 0.223520 ms**。min / max は現行 0.232680 / 0.235480、試作 0.221040 / 0.237640 ms、幅 / p50 はいずれも 30% 未満。codes、group scales、rowscale、flag の全 byte 比較は不一致 0 だった。

差は 1 回あたり約 0.0097 ms で、400 行列すべてに同じ差があると仮定しても約 3.9 ms の概算。K=17408 など他形状、slots の採番と rowfinish を含む全工程、実モデル logits および総合速度は未検証である。初回の単体差が小さく形状依存も未検証のため、production へ接続していない。

### flag 選別の並列 prefix-scan 試作

現在の選別は、その入力を回転・量子化した後の各 group scale から flag を決め、行順に cap まで採番する。重みだけから flag を事前確定することはできない。選別自体については、1024 thread の元実装にある 1 thread の逐次 prefix 計算を、shared memory の並列 scan に置き換えた単体試作を実行した。flag なしと 5 行に 1 行 flag（cap 超過あり）で、256 / 2048 / 4096 行の slots と count はすべて一致した。warmup と 5 回交互 GPU event、2048 行では flag なし 0.014240 → 0.010360 ms、flag あり 0.014560 → 0.010560 ms。全条件の幅 / p50 は 30% 未満。

2048 行の差は 1 launch あたり約 4 µs、仮に 400 回すべてで同じ差が残るとして約 1.6 ms で、モデル全体の測定ぶれに対して小さい概算にとどまる。rowscale 3 kernel 全体や実モデルでの効果は未検証のため、現時点では production 経路に接続していない。これは flag 判定の事前計算ではなく、判定後の採番の試行である。

## 案 2: trellis 復号と WMMA の融合

Saber の `mxfp4_gemm_atiled.hip` と `mxfp4_atiled_wm.h` は読み取りのみで参照した。MXFP4 の重みを LDS に stage して M 方向の wave で再利用し、fragment 形式の FP8 活性を使う構造だが、EXL3 trellis の復号や整数累積とは異なる。試作は shelf の既存 kernel を基に実装し、Saber からコードを複写していない。Saber は Apache-2.0。

既存の decode 型 kernel を出発点に、通常行について group ごとの F32 scale を外して K 全体を INT32 で累積する 4-bit trellis / WMMA の予備試作を作った。64 行 × 64 列の workgroup で、復号重みは波ごとに直接生成し、M 方向の複数 wave 間で LDS に共有しない。入力は合成した INT8 codes と trellis である。現行の復号展開 + hipBLASLt を同じ入力の対照とし、warmup 後に 5 回交互に GPU event で計測した。

| M / N / K | 復号 + hipBLASLt min / p50 / max ms | 試作 min / p50 / max ms | INT32 不一致 |
|---|---|---|---:|
| 2048 / 17408 / 5120 | 1.940390 / 2.017348 / 2.039307 | 4.525563 / 4.600280 / 4.839314 | 0 |

試作は p50 で 2.28 倍かかり、この方式自体は不採用。出力回転や flag 補正は含まず、これは decode 型の直接復号試作に対する結果であって、復号 ALU の実測上限でも、LDS を使う設計すべてへの結論でもない。別に M 方向の 2 wave で復号済み fragment を LDS 共有する 128×64×128 試作は、同じ形状と検証条件で INT32 不一致 0、現行 2.066823 ms に対して 5.445274 ms であり、不採用。A は LDS に置かず、double buffer と出力回転の融合は未実装。この結果から全ての LDS 方式を否定することはできない。コンパイラの `-Rpass-analysis=kernel-resource-usage` は VGPR 101、LDS 8192 bytes/block、spill 0、占有率 12 waves/SIMD と報告した。これは静的な資源推定で、実行時の占有率ではない。

## 交互 A/B：完了分

各組は別プロセスで warmup 1・計測 1 回。組ごとの順序を base→candidate、candidate→base と交互にした。5 組で候補と基準それぞれの中央値・幅 / 中央値を計算し、30% 以上は FAIL とする。

| 条件 | prompt | 現行 p50 ms | 候補 p50 ms | 短縮 | 幅 / p50 現行・候補 |
|---|---:|---:|---:|---:|---|
| 出力回転のみ | 2048 | 1152.8561 | 1122.5214 | 2.63% | 2.07% / 3.90% |
| 出力回転のみ | 8192 | 4915.5439 | 4812.5615 | 2.10% | 1.34% / 7.86% |
| 出力回転のみ | 32768 | 24211.7305 | 23595.5859 | 2.54% | 0.61% / 0.52% |
| 両候補 | 2048 | 1151.9254 | 1073.9244 | 6.77% | 2.20% / 1.90% |
| 両候補 | 8192 | 4937.9331 | 4630.9897 | 6.22% | 6.30% / 4.37% |

いずれも PASS。回転候補は stage の短縮と最終 logits の一致に加え、3 長さすべてで交互 A/B の全体短縮を確認し、opt-in の採用候補として残す。algorithm 単独と両候補は継続中。

algorithm 単独 2048 token の初回 5 組は現行 p50 1153.4523 ms、候補 1110.3031 ms だったが、現行の幅 / p50 が 32.95%（候補 12.85%）であるため **FAIL**。この短縮率を採用根拠にせず、独立した 5 組を再測定中。外れ値の原因は未特定。

## 交互 A/B の途中値

5 組の収集中であり、以下はいずれも最初の 1 組のみの GPU 時間である。warmup は各プロセス 1 回、chunk 2048。**p50 や採否判定ではない。**

| 条件 | prompt | 現行 ms | 候補 ms |
|---|---:|---:|---:|
| 出力回転のみ | 2048 | 1132.6847 | 1124.8898 |
| 出力回転のみ | 8192 | 4924.41 | 4812.56 |
| 出力回転のみ | 32768 | 24163.45 | 23595.59 |
| algorithm のみ | 2048 | 1157.68 | 1110.30 |
| 両候補 | 2048 | 1150.5142 | 1074.7476 |
| 両候補 | 8192 | 5203.56 | 4750.25 |
| 両候補 | 32768 | 23992.0625 | 22860.7754 |

## 精度と総合判定

`shelf-bench quality --tokens-file /tmp/corpus_long.tok --context 2048 --positions 64 --prefill-chunk 2048 --kv-cache-dtype bf16` で baseline dump と回転候補を比較した。vocab 248320、score start 2049、全語彙 64 位置について `QUALITY_KL_MEAN=0`、`QUALITY_LOGIT_MAX_ABS_DELTA=0`、top-1 一致 1、PPL は両者 12.132562。B0 の logits 一致はこの条件で確認した。異なる chunk 長の比較は含めない。algorithm 単独の同条件比較も KL=0、最大絶対差=0。両候補を同時に有効にした同条件比較も KL=0、最大絶対差=0。既存 `test_exl3_int8_linear` と `test_exl3_int8_tp` もそれぞれ lock 下で実行し、両方 exit 0・ALL PASS。総合 A/B は計測中。

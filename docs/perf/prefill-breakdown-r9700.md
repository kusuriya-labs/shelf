# R9700 の 27B EXL3-INT8 prefill 内訳

2026-11-08〜09 JST、測定対象 `8dcb8a6` に本コミットの opt-in 計測を追加。高速化の順位と品質評価は [提案書](../research/20261108/prefill_speedup_proposals.md) を参照する。

## 結果の読み方

- `shelf-bench pp`、TP=1、`--mode forward`、BF16 KV、page 16、指定 R9700 `GPU-a8e9ddefa2d60f55`、モデル `qwen38-27b-exl3-4bpw-mul24`。ROCm `/opt/rocm/core-7.14`、gfx1201、Release、sysfs の `renderD130/device/hwmon/hwmon5/power1_cap` は測定中に 300000000 µW と読み取った。power cap は変更していない。1 GPU 1 process、各プロセス全体を `flock /tmp/r9700.lock` で排他。起動時の重みロードと 1 回の warmup を測定から除き、5 回の p50 と max−min の p50 比を示す。5 回は同一プロセスであり独立した再起動の p50 ではない。30% 以上は FAIL として表記する。chunk 2048 と 4096 は条件ごとにまとめて順に測り、交互 A/B ではないため、差は採用判定ではなく参考値。
- `--tokens-file` は先行評価の `sql_code_1025.psktok`（SHA256 `c8b54856b1e5173953a3d2d85c90849024be919b89be03e1cb9809d377c5e33c`）を繰り返した 32768 ID、SHA256 `3e01d5113e00df82bf116cba9b3ad8e6e851a132ddc379a1481480745bd2a6bb`。現実の長文 corpus ではなく、flag 行の発生や速度は入力依存。先行の同一 ID 反復とも入力は異なる。token corpus は保存しない。
- GPU event は stream 上の開始・終了間の経過時間で、GPU が host を待つ空白も含み得る。wall は同じ区間の CPU 時刻で、sequence 作成・最初の setup sync・モデル load は含まない。壁時間と event の差は GPU idle 全体でも server overhead でもない。`pp` の `run_detail` は GPU と wall をそれぞれソート後に表示するため行同士の対応に使わない。
- `SHELF_QWEN35_PERF_TIMING=1` の診断は dispatch と 6 段階をそれぞれ event で挟む。入れ子なので親と子を二重加算しない。event ペア上限 65536、超過時は警告する。1 chunk は 3797 ペア、32K は 60752 ペアで上限未満。CSV の全行数が logical dispatch 数と内側 stage 数の和に等しいことを全診断で確認した。stage の event 範囲には GPU の待ちや library 内の複数作業も入り得るため純粋な kernel 実行時間と断定しない。`--perf-stats-output` は dispatch と自己申告の物理 launch 統計であり、kernel の名称別 trace ではない。EXL3 dense 分岐のカウンターは hipBLASLt 内部の launch を含まないため過小計数する。
- 診断の p50 は 1196.46 ms、通常の p50 は 1143.59 ms。4.6% の観測差があり、診断の各区分は診断 p50 を分母にした概算比として提示し、無計測時の厳密な分解と見なさない。別々の区分の中央値は必ずしも同じ run から得られず、総和も一致しない。

## 再現

ビルドは本 worktree のみで `nice -n 10 cmake --build build-gfx1201 --target shelf-bench -j 8`。configure は `cmake -S . -B build-gfx1201 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_CXX_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm/core-7.14 -DSHELF_ROCM_ROOT=/opt/rocm/core-7.14 -DSHELF_LIBCXX_LLVM_ROOT=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23 -DSHELF_HIPBLASLT_LIBRARY=/opt/rocm/core-7.14/lib/libhipblaslt.so`、`LD_LIBRARY_PATH` と `ROCM_PATH` は下記と同じ。トークン作成と代表測定:

```sh
python3 - <<'PY'
import struct
p = '/home/homelab1/coding-local/saber-review/eval/logs/shelf-kld/sql_code_1025.psktok'
b = open(p, 'rb').read()
n = struct.unpack_from('<Q', b, 12)[0]
t = struct.unpack_from('<%di' % n, b, 20)
t = (t * ((32768 + n - 1) // n))[:32768]
open('/tmp/shelf-prefill-32768.psktok', 'wb').write(
    b'PSKLDTOK' + struct.pack('<IQ', 1, len(t)) + struct.pack('<%di' % len(t), *t))
PY
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib
export ROCM_PATH=/opt/rocm/core-7.14 ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
flock /tmp/r9700.lock ./build-gfx1201/shelf-bench pp \
  --model-dir /home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24 \
  --mode forward --tokens-file /tmp/shelf-prefill-32768.psktok \
  --prompt-tokens 2048 --prefill-chunk 2048 --kv-cache-dtype bf16 --warmup 1 --runs 5
# 診断は別プロセス。/tmp/diag を先に作る。
SHELF_QWEN35_PERF_STATS=1 SHELF_QWEN35_PERF_TIMING=1 flock /tmp/r9700.lock \
  ./build-gfx1201/shelf-bench pp \
  --model-dir /home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-27b-exl3-4bpw-mul24 \
  --mode forward --tokens-file /tmp/shelf-prefill-32768.psktok \
  --prompt-tokens 2048 --prefill-chunk 2048 --kv-cache-dtype bf16 --warmup 1 --runs 5 \
  --perf-stats-output /tmp/diag/stats.json
```

最終ビルドは `shelf-bench` / `shelf-compute` とも成功。最終バイナリで 512 token、warmup 1、測定 1 回の opt-in semantic 診断を別に実行し、CSV は 3797 行 = logical 1397 + 内側 stage 2400、6 stage が揃い欠落警告なし。これは動作確認であり速度表の 5 回測定には混ぜない。timing 有効かつ出力先なしの CLI が exit 2 で拒否されることも確認した。

同様に prompt 長を 512〜32768、chunk を 2048 / 4096 に変更。`semantic_timing.csv` は指定 stats と同じ directory に書き込まれる。`SHELF_PP_HOST_TIMING=1` は sequence setup、各 `execute_batch` 呼び出しの CPU 時間、最後の event wait を stderr に出す。CPU 時間は GPU との重なりや暗黙の待ちを含み、足し合わせて GPU に上乗せできない。各条件の生ログ・診断 CSV と stats は [data](prefill-breakdown-r9700-data/) に置く。

## 速度一覧

| prompt | chunk 2048 p50 ms / tok/s | 幅 / p50 | chunk 4096 p50 ms / tok/s | 幅 / p50 |
|---:|---:|---:|---:|---:|
| 512 | 347.23 / 1,474.5 | 0.20% | 347.08 / 1,475.2 | 0.13% |
| 1024 | 600.45 / 1,705.4 | 0.28% | 600.55 / 1,705.1 | 0.43% |
| 2048 | 1,143.59 / 1,790.9 | 0.45% | 1,152.73 / 1,776.7 | 2.14% |
| 4096 | 2,364.45 / 1,732.3 | 1.26% | 2,207.68 / 1,855.3 | 1.66% |
| 8192 | 4,896.15 / 1,673.2 | 0.96% | 4,552.20 / 1,799.6 | 1.78% |
| 16384 | 10,535.42 / 1,555.1 | 2.25% | 9,920.40 / 1,651.5 | 0.70% |
| 32768 | 24,007.85 / 1,364.9 | 0.46% | 22,590.15 / 1,450.5 | 2.11% |

全 14 条件で 5 回の幅 / p50 は 30% 未満。512 / 1024 / 2048 は指定 chunk が prompt 長に切り詰められるため、2 列は同じ実際の chunk 長の別計測である。4096 / 8192 / 16384 / 32768 における chunk 4096 の短縮率は順に 6.6% / 7.0% / 5.8% / 5.9%。交互 A/B ではない参考値で、chunk による logits 差は別途品質 gate が必要。

## 2048 token / 1 chunk の診断

logical dispatch 1397、自己申告 physical launch 3799、うち EXL3 INT8 logical 401。dense の 400 行列について `launches += 7` は自前 HIP kernel 7 個（量子化 1、rowscale 3、展開 1、flag GEMM 1、出力回転 1）だけを数え、hipBLASLt GEMM を含めない。したがって少なくともさらに約 400 件の library 側 GEMM launch があり、全 launch は約 4199 件以上というソース上の下限で、実 kernel trace で検証した正確値ではない。サンプリング関連の dispatch は forward にも記録されるが、ベンチで生成 token を評価していない。`correctness_fallbacks=0` はこの 1 chunk の全 dispatch についての値であり、EXL3 の flag 行処理がゼロという意味ではない。flag の実数はこの instrumentation では取得していない。

| EXL3 段階 | 中央値 ms | 診断全体比 | stage event 数 / chunk | 補足 |
|---|---:|---:|---:|---|
| 活性回転・INT8 量子化 | 126.36 | 10.6% | 400 | 128 ごとの scale を生成 |
| rowscale・flag 選別 | 47.79 | 4.0% | 400 | 自前 kernel は 3 回 / 行列。最大 rows/8、実際の flagged 行数は未計数 |
| EXL3 重みを INT8 に展開 | 51.90 | 4.3% | 400 | 毎行列・毎 chunk、共有 workspace を上書き |
| hipBLASLt INT8 GEMM | 529.68 | 44.3% | 400 | event 間に library 側待ちが含まれ得る |
| flagged 行用 GEMM | 25.70 | 2.1% | 400 | 0 行でも launch、cap 基準の grid で空 tile は早期 return |
| 出力回転・scale | 204.73 | 17.1% | 400 | flag 行の結果を選択 |
| EXL3 親 dispatch | 1002.79 | 83.8% | 401 | 上記 400 行列を含む。残り 1 は最終 1 行の lm_head |

6 段階は親の内訳で、親に重ねて足さない。event 数は実 kernel 数と同じとは限らず、例えば rowscale は初期化などを伴う。全 stage 以外に親の前後処理や計測境界がある。全 flag kernel が呼ばれることはコードとイベントから確認したが、入力依存の flag 件数は未計測。

| 層種・処理 | 中央値 ms | 診断全体比 | logical dispatch 数 | 内容 |
|---|---:|---:|---:|---|
| GDN 48 層全体 | 882.82 | 73.8% | 1056 | 各層 MLP、norm、残差を含む |
| full attention 16 層全体 | 303.76 | 25.4% | 336 | 各層 MLP、norm、残差を含む |
| GDN qkv / z / out 投影 | 98.91 / 76.45 / 71.71 | 20.7% 合計 | 各 48 | EXL3 親 dispatch。a/b の BF16 投影は別 |
| GDN a / b 投影（BF16） | 4.86 | 0.4% | 96 | 48 層 × 2、a と b 個別の時間は未分離 |
| GDN conv / recurrence | 9.24 / 51.10 | 5.0% 合計 | 各 48 | recurrence は chunked scan 等を含む dispatch |
| attention q / k / v / o 投影 | 47.91 / 9.30 / 8.85 / 22.86 | 7.4% 合計 | 各 16 | EXL3 親 dispatch |
| attention RoPE / KV append / paged attention | 2.72 / 0.46 / 25.30 | 2.4% 合計 | 32 / 16 / 16 | BF16 KV、attention は 32 query 行・8 wave の既存 WMMA |
| MLP gate / up / down | 約 205.6 / 206.6 / 251.8 | 約 55.5% 合計 | 各 64 | GDN と full attention 層それぞれの線形の中央値を合算した近似 |
| MLP SwiGLU | 22.11 | 1.8% | 64 | MLP 線形とは別 |
| RMS norm / 残差 | 15.62 / 10.44 | 2.2% 合計 | 209 / 128 | 層内分を含む |
| embedding / lm_head / sampling dispatch | 0.044 / 1.589 / 0.009 | 約 0.14% 合計 | 各 1 | lm_head は最後の 1 行、embedding は重みロードを含まない |

層群と操作別の行は互いに重なる。層の分類は dispatch の順序、各層 2 回の残差、モデル config の 3 GDN + 1 attention の繰り返しと lowering の線形順から対応付けたもので、CSV 自体に layer ID はない。分類前に各層の EXL3 件数（GDN 6、attention 7）を検査した。展開は GDN 48 層合計 39.04 ms、attention 16 層合計 12.85 ms、各層の中央値は 0.801〜0.828 ms。`ctx.exl3_dense` は共通 workspace であり、`try_launch_linear_exl3_int8` の dense 分岐は毎回 `launch_exl3_int8_recon` を呼ぶ。重みのキャッシュではない。

診断 5 回には 1 回だけ RMS_NORM 区間に約 231 ms の突出があり、診断総時間 1191.1〜1463.9 ms、幅 / p50 = 22.8%。30% の FAIL 基準未満だが異常は残る。原因は特定しておらず、各区分は平均でなく中央値を使用した。通常 2048 token の 5 回は 1139.7〜1144.8 ms だった。event 区間は GPU が途中で待つ時間を含むため、この突出を RMS kernel の計算そのものとは断定できない。

## 長い context

32K / chunk 2048 の event 診断は **別プロセス 5 回**、各 warmup 1 回・測定 1 回。診断全体 p50 24576.20 ms、範囲 24527.39〜24805.69 ms（幅 1.13%）。以下は各 chunk の 16 層 `PAGED_ATTENTION` event 合計の p50。8K までの 4 chunk の合計 p50 は 372.27 ms、32K 全 16 chunk の合計 p50 は 5864.96 ms（5858.19〜5919.15 ms、幅 1.04%）。同じ 8K の処理を独立した 8K prompt として測った値ではない。

| chunk 末尾 token | 2048 | 4096 | 6144 | 8192 | 16384 | 24576 | 32768 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| attention ms / chunk | 25.65 | 70.84 | 115.49 | 160.09 | 340.87 | 524.46 | 717.83 |

各 chunk の時間は見える過去 token の増加にほぼ比例し、全体は概ね二次に増える。32K の後半 8K は同じ 2048 行でも per-chunk 524〜718 ms 程度で、最初の 8K 計 372 ms と大きく異なる。イベント計測時の結果であり、他の処理を一律に差し引いて通常の 32K event を厳密に分解するものではない。32K の stats には 45 の correctness fallback があり、最終以外の 15 chunk に各 1 回の RMS_NORM / OUTPUT_GATHER / SAMPLING と記録された。`PAGED_ATTENTION` 自体は 256 dispatch 全て optimized。各 fallback が実際に有効な計算を行ったかは別途確認が必要。


## 演算量・理論値

AMD の [R9700 公式仕様](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) は非 sparse の INT8 matrix 383 TOPS、FP8 matrix 383 TFLOPS、FP16 matrix 191 TFLOPS、640 GB/s、32 GB、300 W。**同ページには BF16 matrix の独立した公称値を確認できない**。191 TFLOPS を BF16 の公式値とは呼ばない。BF16 に FP16 と同等 throughput を仮定した数字は仮定付き参考値である。sparse の 2 倍値は使わない。

モデルの `quantization_config.json` にある 64 層の EXL3 行列の `suh` / `svh` 長から `sum(N*K)=24,326,963,200`、400 行列を数えた。行列積を multiply + add の 2 ops とすると、2K の main INT8 GEMM は `2 * 2048 * sum(N*K)` = **99.643 TOP**、383 TOPS で 260.2 ms が演算のみの理想下限。診断の GEMM 529.68 ms から model 有効値は **188.1 TOPS（公称の約 49.1%）**。flag 行の追加 GEMM、row padding、内部実行命令、scale、head、BF16 の小投影、GDN、softmax はこの数に含めない。1 回の INT8 multiply-add を 2 ops と数える。モデル全体 27B を一律に 2048 倍する計算は、embedding と 1 行だけの head を過大評価する。

causal attention の QK + PV を有効対数だけで数えると `4 * 16 layers * 24 heads * 256 dim * L*(L+1)/2`、2K で 0.825 TFLOP、32K で 211.11 TFLOP。2K の 25.30 ms では 32.6 TFLOPS、32K の p50 5864.96 ms では 36.0 TFLOPS に相当する **有効演算量**。実 kernel は tile padding、softmax、PV の高位・残差 2 回の WMMA を行うため、発行 FLOP 数でも hardware 利用率でもない。仮に BF16 peak を 191 TFLOPS とすれば計算だけの下限は各 4.32 / 1105 ms だが、これは BF16 のメーカー公称値ではなく上記仮定に依存する。実効 GEMM ceiling の独立 microbenchmark は未測定。

各段階で実測した bytes や命令数ではなく、400 行列・2048 行について、各要素を **一度だけ** 読み書きすると仮定した論理 byte 数を stage event で割ると次のとおり。キャッシュ・繰返し load・padding・内部 workspace は含まず、DRAM 実効帯域ではない。出力回転は出力をすべて BF16 と仮定した下限で、実際に F32 の出力があれば byte 数は増える。rowscale・flag、norm、recurrence などは検証済みの traffic 定義がないため未算出。

| 区分 | 仮定した論理 traffic | event で割った値 |
|---|---:|---:|
| 入力回転量子化 | BF16 入力 2 B + INT8 code 1 B × `2048*sum(K)` = 17.82 GB | 141.0 GB/s |
| INT8 GEMM | W 1 B + code 1 B + INT32 結果 4 B を各要素 1 回として 62.21 GB | 117.4 GB/s |
| 出力回転 | INT32 読み 4 B + BF16 書き 2 B × `2048*sum(N)` = 47.92 GB | 234.0 GB/s |

展開出力の論理サイズは 24.327 GB（22.66 GiB）。51.90 ms で割った出力だけの見かけの書込率は 468.7 GB/s。4-bit trellis 読みを概算で加えると約 703 GB/s だが、キャッシュ再利用、圧縮形式・padding、LDS、writeback により **DRAM 帯域の実測値ではない**。全量が DRAM に書き戻ると仮定して 640 GB/s で割ると 38.0 ms だが、それはキャッシュ上の書込みにも適用できる厳密な下限ではない。残りの段階は実際の bytes・命令数を計数していないため TOPS / 帯域や厳密な下限を捏造しない。

## host と空き時間

`SHELF_PP_HOST_TIMING=1` の別診断では、`pp` は chunk ごとに `execute_batch` → `submit_batch` / `complete_batch` を通る。`complete_batch` の `hipStreamSynchronize` は **各 chunk の完了を待ってから state を commit** する。8K / chunk 2K、warmup 1・測定 5 回の host 時間は以下（ms、各列の p50 は別 run の可能性あり）。submit 中にも GPU は進み、HIP 呼び出しが待つ可能性があるので、submit を純 CPU 時間や未隠蔽 launch 時間とは呼べない。complete は GPU 待ちと commit の合計であり、同期そのものの追加費用ではない。

| chunk 開始位置 | submit p50 | complete p50 | 主な内容 |
|---:|---:|---:|---|
| 0 | 24.19 | 1135.73 | target enqueue と GPU 完了・state commit |
| 2048 | 24.30 | 1178.84 | 同上 |
| 4096 | 24.14 | 1226.89 | 同上 |
| 6144 | 24.53 | 1258.92 | 同上 |

全 8K の GPU event p50 は 4903.62 ms、wall p50 は 4903.66 ms。最終 event wait は直前の chunk 完了待ち後なのでほぼゼロで、これを全同期の費用としてはいけない。初回 2048 host 診断の 5 回は GPU 1152.48〜1504.15 ms、幅 / p50 30.45% で **FAIL**。原因を特定できず、異常値を残した。独立した再測定では 1152〜1158 ms 付近に戻った。8K の先行 host 診断でも最初の chunk に外れ値があり、合計時間と chunk ごとの安定性を区別する。CPU の KV page 確保、GDN state 引継ぎ、lowering、launch、GPU idle をそれぞれ独立に測定する計器はなく、内訳は未確定。既存 graph の適用条件は decode のみで prefill は対象外。

同じ token ID 2048 個・BF16 KV・prefix cache off・`max_new_tokens=1`・1 process 内 warmup 1 回、続く 5 request の stdio TTFT を別々に測った。非投機 p50 **1156.66 ms**、DFlash2 有効 p50 **1451.76 ms**、差 **295.10 ms**。server の内部 prefill p50 はそれぞれ約 1155.8 / 1450.9 ms で、TTFT までの残りは 1 ms 程度。1 回の出力 ID は全試行で 5574。入力 ID を直接渡しており HTTP や tokenization は含まない。2 条件は別プロセス・交互順ではなく、その差を厳密な因果分解とは扱わない。最初の shape warmup は非投機 3028 ms、DFlash2 3114 ms で、後続より遅かった原因は未特定。DFlash2 prefill は target の 5 layer の tap を収集して `dflash2_append_target_taps` で特徴量投影と 5 層の K/V context ring を構築し、最後の token を読み戻す。`DFLASH2_ROUNDS=0` なので、この測定には speculative decode round はない。

`SHELF_DFLASH_PREFILL_TIMING=1` で target と append を GPU event で区切り、append 後に同期した別診断は、暖機後 5 回の target p50 **1160.11 ms**、特徴量投影と context append p50 **310.70 ms**。各範囲は 1157.81〜1160.86 / 310.53〜310.85 ms。診断 TTFT p50 は **1472.23 ms** で非診断 DFlash2 の 1451.76 ms より約 20.5 ms 長い。target event は 5 tap 生成と launch 待ちを含み、append event も 5 tap concat / FC / norm と 5 層の K/V 投影・norm・RoPE・ring 書込みを合わせた区間で、前後の host 待ちを含み得る。append の内側は未分離。ソースからは、この条件で `stdio_server.hip` が drafter `max_rows=8`、`max_batch_rows=8` を設定し、`dflash2_append_target_taps` が 2048 行を 8 行ずつ **256 回**に分ける。各回の feature FC 1 回と 5 層の K / V 各 1 回で計 **2816 回の BF16 GEMM 呼び出し**となる。これは実コードの loop から数えた呼び出し数であり、event で名前別 launch を数えた結果ではない。FC は 5120×25600、各 K/V は 1024×5120 なので、これらの有効積和は計約 0.752 TFLOP、append 全体で割ると約 2.42 TFLOPS。ただし concat・norm・RoPE・ring・host 空白を含むため GEMM の実効性能ではない。診断用の event 同期で通常経路は変えない。実行は生ログ archive 内の `stdio.py` を使い、前記環境で `SHELF_DFLASH_PREFILL_TIMING=1 flock /tmp/r9700.lock python3 stdio.py draft` とした。通常の 295 ms 差と診断の 310.7 ms は別実行・別計測範囲のため、差分の厳密な一致を求めない。

診断 event の親 dispatch の合計 p50 と stream 全体 p50 の差は約 8 ms だが、event 挿入に伴う誤差、dispatch 外の GPU 操作、CPU による遅延を含み、GPU の idle を測った値ではない。GPU の実際の空き時間、空きの発生位置、およびそのうち launch 待ちと同期待ちの割合は **未取得**。


既存先行記録の shelf server TTFT 1450.7 / 1451.4 ms と `pp` event 1105.6〜1121.8 ms は別 revision・異なる入力・非投機 forward と DFlash2 serving を含む異なる区間の値。約 330〜346 ms の差を、単純に chunk 同期や host launch の費用と認定できない。サーバー側 request 受理、tokenization、scheduler、KV / GDN 準備、prefill と後続処理、DFlash2 の初回処理、最初の token 出力までの同一時計の区間を同一入力で記録する必要がある。上記の同一入力・stdio 比較は DFlash2 有効化と結び付く差を示すが、先行実験の残りの差や HTTP・tokenization の寄与は未分離。

## Saber との対応

Saber は `c8a1e84` の第三者コードを読み取りのみで参照した。先行 [REPORT](../../../../../saber-review/eval/REPORT.md) は 2048 token の Saber forward wall 482.36〜482.47 ms、TTFT 約 493 ms / 4155 tok/s、32768 TTFT 約 8828〜8844 ms / 3705〜3712 tok/s。異なる量子化、FP8 KV、head、sampling、入力と範囲があるので shelf の同一 kernel との対照実験ではない。Saber の 2048 prefill kernel capture は失敗し、stage 別時間は未取得。

| 段階 | shelf の今回の観測・実装 | Saber の設計 | 差の解釈 |
|---|---|---|---|
| 重み復号 | EXL3→INT8 全行列を 51.9 ms で毎 chunk 展開 | MXFP4 の W を LDS 内で FP8 に fold | 圧縮値の表現と fold の数値契約が異なる。復号時間を比較する Saber 実測はない |
| 行列積 | INT8 hipBLASLt 529.7 ms、行単位 scale と flag 補正 | A-tiled FP8 WMMA、WM=8 は 16 wave / 512 thread、W slab 再利用と LDS double buffer variant | タイルと dtype の違いが原因との仮説はあるが、Saber の GEMM 単独時間は未取得 |
| 前後処理 | 入力回転量子化・rowscale・出力回転を別 stage | MXFP4 W4A8 の scale / fold / staging が別の設計 | 単純に shelf の stage 全てを Saber でゼロとは扱えない |
| GDN / attention | recurrence 51.1 ms、BF16 KV の paged attention 25.3 ms（2K） | 27B は FP8 KV。attention・GDN kernel 群を使用 | kernel 別 prefill 時間なし。FP8 KV と BF16 KV は品質・帯域が異なる |
| host | `pp` event と server TTFT の区間は別 | forward wall と TTFT を別計測 | エンジン外の差は分離できない |

具体的な Saber 箇所は `crates/saber-kernels/hip/mxfp4_gemm_atiled.hip`、`mxfp4_atiled_wm.h` と先行 `OPTIMIZATIONS.md`。第三者コードの build・実行は行っていない。

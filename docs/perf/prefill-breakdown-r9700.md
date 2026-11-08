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


## Saber との比較

Saber は評価コピー `saber-review/eval/repo`、27B TP=1、同じ R9700 の 300 W、ROCm 7.14。Saber は NVFP4 由来 MXFP4 重み・FP8 活性・FP8 KV、shelf は EXL3-INT8・BF16 KV である。数値は異なる精度・投影結合・計測区間を含む。Saber の既存 serving 実測は `logs/saber-paired.jsonl` の host counter で、2048 token / 1 step の forward は 482.473 / 482.361 ms、8192 token / 4 step は 1979.187 / 1980.791 ms。これは GPU kernel 時間ではない。shelf 通常 GPU event はそれぞれ 1143.59 / 4896.15 ms（5 回 p50）。要求処理や KV 状態は完全には一致しない。


### 実サーバーの kernel trace（主たる結果）

評価コピーの `saber_prof_run_save.sh` を `flock /tmp/r9700.lock timeout 300 ./saber_prof_run_save.sh` で実行。server の実行経路は通常、prefill を HIP の `hipModuleLaunchKernel`、capture 対象を `hipGraphLaunch` で発行する。`saber-hsa` の AQL 直接 replay は別の `step-capture --fn63` 試作であり、今回の serving 経路と区別する。ROCm 7.14 の `rocprofv3 --kernel-trace` は、読み取り専用 `/work` で終了時に `/work/.rocprofv3` を作ろうとして abort した。`-w /tmp` に変更し、評価ディレクトリ内の trace 出力専用 directory のみを `/output:rw` として追加したところ、外部遮断 internal network・read-only root・capability 全削除・no-new-privileges・指定 2 device のまま trace を保存できた。元の server の `/tmp` は tmpfs なので停止後には消える。登録 error 16 は今回観測していない。

512、2048、8192 token を同じ server に順に送り、2048 行の `saber_embed_bf16` と最終 `layer.63.mlp.down_proj` を起点・終点とする各 chunk を特定した。末尾 chunk は続く row gather、最終 norm、FP8 `lm_head`、argmax、device row も含める。body には DFlash2 の補助 tap 5 launch を含む。chunk 間の DFlash2 context store、入力 copy、最初の kernel 前と最後の kernel 後の待ち、最終 chunk の後処理は含まない。8192 token の最初の 3 chunk は 758、最後は 764 launch で計 3038。2048 token は 764 launch。別実行の event 計装で取得した op 名 / kernel 名の列と各 chunk の trace kernel 名を全件、順番に照合し一致した。全対象 launch は 1 queue、時刻順に重複なし。trace の CSV は出力順が時刻順ではないため開始 timestamp で sort してから照合した。

| 対象 | kernel 合計 | 範囲の開始〜終了 | kernel 間の隙間 | launch |
|---|---:|---:|---:|---:|
| 2048 token / 1 chunk | 461.676 ms | 467.112 ms | 5.437 ms / 1.16% | 764 |
| 8192 token / 4 chunk の和 | 1902.349 ms | 1924.370 ms | 22.021 ms / 1.14% | 3038 |

隙間は各 chunk の trace 範囲から同プロセス・queue の kernel 実行時間を引いた値で、profiler の影響を含み、DMA・別 process などの GPU 全体の idle を直接示さない。4 chunk 間の待ちも含まない。各 chunk は prefix 0 / 2048 / 4096 / 6144 token で、GPU 範囲 465.350 / 475.272 / 485.557 / 498.190 ms、kernel 合計 459.954 / 469.742 / 480.025 / 492.628 ms。最後の chunk だけ head を含む。

同一の prompt ID と設定による別 server process の host forward counter は、2048 で profiler なし **483.131 ms**、あり **488.126 ms**（+4.995 ms / +1.03%）、8192 で **1990.811 → 2008.719 ms**（+17.908 ms / +0.90%）。別実行の各 1 回であり、差を純粋な profiler overhead と断定しない。event 計装ありではそれぞれ **510.608 ms**（+27.477 ms / +5.69%）と **2088.908 ms**（+98.097 ms / +4.93%）。event 計装は op の直前・直後に HIP event を入れ、最後に同期してログに出すため、kernel 間の待ちと CPU の event 管理を増やし得る。event あり / なしの出力 token ID は 512 / 2048 / 8192 の各 1 回で一致した（順に 685 / 248068 / 321）。これは限定された出力一致であり品質評価ではない。

下表の合計・平均は profiler が記録した kernel 開始〜終了の和と平均。比率は上表の各 chunk 範囲の合計を分母とする。N/K は同じ順序の event op 名と graph の形状の照合による。1 chunk 1 回 / 4 chunk 1 回であり、各行を独立反復の中央値と誤認しない。

| kernel / N×K | 2048 回数 | 合計 ms | 平均 ms | 区間比 | 8192 回数 | 合計 ms | 平均 ms | 区間比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (34816×5120) | 64 | 179.371 | 2.8027 | 38.40% | 256 | 721.526 | 2.8185 | 37.49% |
| `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (5120×17408) | 64 | 91.401 | 1.4281 | 19.57% | 256 | 361.795 | 1.4133 | 18.80% |
| `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (16384×5120) | 48 | 64.006 | 1.3335 | 13.70% | 192 | 255.614 | 1.3313 | 13.28% |
| `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (5120×6144) | 64 | 32.507 | 0.5079 | 6.96% | 256 | 129.152 | 0.5045 | 6.71% |
| `silu_mul_quant_t512_tiled1` | 64 | 19.273 | 0.3011 | 4.13% | 256 | 77.322 | 0.3020 | 4.02% |
| `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (14336×5120) | 16 | 18.594 | 1.1622 | 3.98% | 64 | 74.994 | 1.1718 | 3.90% |
| `add_rms_quant_t512_tiled1` | 128 | 14.657 | 0.1145 | 3.14% | 512 | 58.473 | 0.1142 | 3.04% |
| `saber_gdn_chunk_scan_f16s_k128_v128_c64` | 48 | 10.174 | 0.2120 | 2.18% | 192 | 41.088 | 0.2140 | 2.14% |
| `saber_attn_prefill_h256_gqa6_fp8kv_qk8pv8` | 16 | 7.363 | 0.4602 | 1.58% | 64 | 92.009 | 1.4376 | 4.78% |
| `saber_gdn_conv_prep_w4_h128_hpb8_ab1` | 48 | 6.713 | 0.1398 | 1.44% | 192 | 26.956 | 0.1404 | 1.40% |
| `gdn_norm_quant_t256_tiled1` | 48 | 4.666 | 0.0972 | 1.00% | 192 | 18.704 | 0.0974 | 0.97% |
| `qk_norm_rope_wg8_h256_rot64_mi1_red5` | 16 | 3.797 | 0.2373 | 0.81% | 64 | 15.429 | 0.2411 | 0.80% |
| `attn_gate_quant_t256_tiled1` | 16 | 2.313 | 0.1445 | 0.50% | 64 | 9.094 | 0.1421 | 0.47% |
| `saber_gdn_kkt_solve_k128_c64_r3` | 48 | 2.290 | 0.0477 | 0.49% | 192 | 9.207 | 0.0480 | 0.48% |
| `dflash_bgemm_w8_m32n16_w128_u16_nt` (248320×5120) | 1 | 2.070 | 2.0698 | 0.44% | 1 | 2.059 | 2.0593 | 0.11% |
| `mxfp4_gemm_skinny_n1t_w4_u8_wperm1` (96×5120) | 48 | 1.516 | 0.0316 | 0.32% | 192 | 6.079 | 0.0317 | 0.32% |
| `kv_append_h256_b16_fp8` | 16 | 0.452 | 0.0283 | 0.10% | 64 | 1.807 | 0.0282 | 0.09% |
| `saber_argmax_bf16` | 1 | 0.316 | 0.3161 | 0.07% | 1 | 0.309 | 0.3094 | 0.02% |
| `saber_row_copy_bf16` | 5 | 0.161 | 0.0322 | 0.03% | 20 | 0.642 | 0.0321 | 0.03% |
| `saber_embed_bf16` | 1 | 0.022 | 0.0223 | 0.00% | 4 | 0.077 | 0.0192 | 0.00% |
| `saber_row_gather_bf16` | 2 | 0.007 | 0.0033 | 0.00% | 2 | 0.007 | 0.0033 | 0.00% |
| `saber_add_rms_bf16` | 1 | 0.005 | 0.0046 | 0.00% | 1 | 0.005 | 0.0047 | 0.00% |
| `saber_device_row_stage_t1024_bf16` | 1 | 0.003 | 0.0025 | 0.00% | 1 | 0.003 | 0.0025 | 0.00% |

線形 304 / 1216 回は 2048 / 8192 token で **387.395 / 1549.159 ms**、形状から `2*M*N*K` を積算した実効値は **257.5 / 257.5 TOPS 相当**。2048 の大きな形状は **253.7〜260.5 TFLOPS**。96×5120 は実 server の skinny kernel に切り替わり、48 回 1.516 ms・63.7 TFLOPS。shelf の 400 EXL3 行列の hipBLASLt は 529.68 ms / 約 208 TOPS だが、Saber の 304 回と投影結合・weight format が違う。Saber の MXFP4 fold は GEMM の中、shelf の EXL3 復号展開 51.90 ms は別である。

Saber の GDN conv / KKT / scan は 2048 で 6.713 / 2.290 / 10.174 ms、8192 で 26.956 / 9.207 / 41.088 ms。shelf の conv / recurrence は 2048 で 9.24 / 51.10 ms だが、GDN state 型と処理範囲が異なる。Saber の attention は 2048 で 7.363 ms / **112.0 TFLOPS**、8192 で 92.009 ms / **143.4 TFLOPS**。演算量は 16 層、24 query head、head 256、各 query の causal 対数で QK と PV の合計 `4*256*24*対数` を計上したモデル値。8192 の各 chunk の attention は 7.417 / 17.694 / 28.236 / 38.662 ms（111.2 / 139.8 / 146.0 / 149.3 TFLOPS）。shelf の BF16 KV は 2048 で 25.30 ms、別の 32K 診断の先頭 8192 token で 372.27 ms。FP8/BF16 KV と context・診断方式の差があるため一対一の最適化幅ではない。

2048 の層別 kernel 和は GDN 48 層 **342.486 ms**、attention 16 層 **116.607 ms**。8192 はそれぞれ **1369.288 / 529.959 ms**。各層の MLP・norm 等を含み、表の kernel 時間と重複する。8192 では attention 層の線形以外の時間が context と共に増える。

生データは `/home/homelab1/coding-local/saber-review/eval/logs/saber-prof-save-output/148364e58f22/1_kernel_trace.csv`、対応する `logs/profiler-p2048-{ops,kernels,layers,opgroups}.csv` と `logs/profiler-p8192-{ops,kernels,layers,opgroups}.csv`、`logs/profiler-p2048-summary.json` / `logs/profiler-p8192-summary.json`。`analyze_profiler_serving.py` が event 計装の `serving-*-ops.csv` と照合して生成する。event の生ログは `logs/saber-events-on-server.log`、計装差分は `patches/serving-prefill-events.patch`、計装あり / なしと profiler の条件・counter は各 `saber-events-{on,off}-requests.jsonl`、`saber-prof-save-requests.jsonl`。電力 cap の読取りと binary hash は `logs/prefill-power-cap-microwatts.txt` / `logs/prefill-binary-sha256.txt`。CPU 利用のみのビルドは network none、`nice -n 10 cargo build --offline --locked --release -j 8 -p saber-cli`。

### 2048 行の診断 capture

`flock /tmp/r9700.lock timeout 300 env PREFILL_ROWS=2048 ./saber_capture_prefill.sh` を `saber-review/eval` で実行。各実行の container は `/dev/kfd` と `renderD130` のみ、network none、read-only mount、capability 全削除、no-new-privileges。元の Saber clone は変更せず、FP8 head と body snapshot の既存評価差分を使った。`--rows 5` で graph の構築幅を既定 5 に揃える回避策を使用。body 753 op / 753 node、head 2 node、row gather 等 3 node を含む全 758 node。warmup 1、交互 2 回の event 計測で whole graph **463.643 ms**、eager **464.436 ms**。1 回の replay で eager と arena 612029440 B、state 584187904 B が byte 一致した。これは実行結果間の一致であり、BF16 reference に対する精度確認ではない。通常 server は少なくとも小さい N の GEMM に別の skinny kernel を選ぶため、下表は実 server の厳密な内訳ではない。

per-op は body の各 op を別々の graph にして event 測定し、その合計 **466.997 ms** と whole graph との差 **+3.354 ms / +0.7%** を 753 個へ等分した **4.45 µs/op** を既存ツールが引いた値。表の合計は形状ごとにまとめた op の中央値から作った値であり、各 dispatch の生 kernel 占有時間でも、観測差を正確に補正した値でもない。各 op の event は graph 間待ちを含み得る。whole は head と選択を含むのに per-op は body のみであり、補正の残差ゼロは計算で作られた値である。特に短い kernel の比率と TFLOPS に補正が与える影響に注意。合計・平均・全体比は表示値から算出し、形状別 GEMM 行は同名 kernel でも N/K で分けた。

| 区分 | kernel / N×K | 回数 | 合計 ms | 平均 µs | whole 比 |
|---|---|---:|---:|---:|---:|
| GEMM | `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (34816×5120) | 64 | 178.743 | 2792.9 | 38.6% |
| GEMM | `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (5120×17408) | 64 | 91.050 | 1422.7 | 19.6% |
| GEMM | `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (16384×5120) | 48 | 63.743 | 1328.0 | 13.7% |
| GEMM | `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (5120×6144) | 64 | 32.402 | 506.3 | 7.0% |
| GEMM | `mxfp4_gemm_atiled_tn4_lbk64_wperm1_wm8sp` (14336×5120) | 16 | 18.676 | 1167.3 | 4.0% |
| GEMM | `mxfp4_gemm_atiled_tn2_lbk128_wperm1` (96×5120) | 48 | 2.630 | 54.8 | 0.6% |
| GDN | `saber_gdn_chunk_scan_k128_v128_c64` | 48 | 10.508 | 218.9 | 2.3% |
| GDN | `saber_gdn_conv_prep_w4_h128_hpb8_ab1` | 48 | 7.347 | 153.1 | 1.6% |
| attention | `saber_attn_prefill_h256_gqa6_fp8kv_qk8pv8` | 16 | 7.491 | 468.2 | 1.6% |
| GDN | `saber_gdn_kkt_solve_k128_c64_r3` | 48 | 2.564 | 53.4 | 0.6% |
| producers | `qk_norm_rope_wg8_h256_rot64_mi1_red5` | 16 | 3.817 | 238.6 | 0.8% |
| producers | `silu_mul_quant_t512_tiled1` | 64 | 19.768 | 308.9 | 4.3% |
| producers | `add_rms_quant_t512_tiled1` | 128 | 16.270 | 127.1 | 3.5% |
| attention | `attn_gate_quant_t256_tiled1` | 16 | 2.482 | 155.1 | 0.5% |
| GDN | `gdn_norm_quant_t256_tiled1` | 48 | 5.143 | 107.1 | 1.1% |
| producers | `kv_append_h256_b16_fp8` | 16 | 0.504 | 31.5 | 0.1% |
| embed+head | `saber_embed_bf16` | 1 | 0.026 | 25.6 | 0.0% |

GEMM の 6 形状は 304 回、各形状集約値の和 387.244 ms、`2*M*N*K*回数` の和 99.740 TOP より **257.6 TOPS 相当**。大きい 5 形状は 254.5〜261.4 TFLOPS、小さい 96×5120 は 36.7 TFLOPS。均等補正分を各呼び出しに戻す概算は合計約 388.6 ms、約 256.7 TOPS 相当。Saber の総合計は展開を kernel 内に含む。一方 shelf の 400 行列は 529.68 ms / 約 208 TOPS、展開 51.90 ms は別。線形の形式・形状・演算回数は一致せず差額 142.4 ms は削減保証ではない。Saber attention の 110.1 TFLOPS はツールの causal 演算量モデルによる値で、shelf の BF16 KV attention 25.30 ms と直接の同一演算ではない。

同じ capture の層別 graph event 中央値は GDN 48 層 345.724 ms、attention 16 層 117.806 ms、全 64 層の平均 7.243 ms（範囲 7.156〜7.453 ms）。層にはそれぞれ MLP・norm・投影が含まれ、kernel 表と重複する。層リンクの合計 463.560 ms、全 graph 463.643 ms、層リンク chain の wall 466.296 ms。層の分類は 3 GDN + 1 attention の繰り返しに基づく。

前回の `--verify-width 8` による p2048 fault は、診断コマンドが幅 5 で作った graph の `verify_width` field だけを 8 に変更し、GDN conv state の allocation と kernel stride の根拠が食い違うコードに対応する。今回 512 行でも修正前の幅 8 で同じ kernel に fault、既定幅 5 では成功した。graph を最初から要求幅で構築する評価コピー向け `patches/step-capture-verify-width-state.patch` をオフラインビルドした後、幅 8 / 512 行で exit 0、arena 153410560 B と state 590086144 B は eager と graph の replay で byte 一致した（`logs/prefill-capture-512-width8-fixed.log`）。幅 5 の state は 584187904 B、幅 8 で 5898240 B 増加した。48 層・各 2 slot・10240 channel・幅差 3・BF16 2 B から計算される 48×2×10240×3×2 = 5898240 B と一致する。幅による state と stride の不整合が原因という説明を修正前後の結果が支持するが、故障アドレスを特定の state buffer に対応付けたわけではない。

生ログ・再現用 script は `/home/homelab1/coding-local/saber-review/eval/logs/prefill-capture-2048-width5.log`、`logs/prefill-capture-512-width5.log`、kernel 集計 CSV は `logs/prefill-capture-2048-kernels.csv`、各層 CSV は `logs/prefill-capture-2048-layers.csv`。計算を再生成する `extract_capture_prefill.py` とパッチは同 eval にある。

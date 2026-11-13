# EXL3-INT8 の連続 linear 統合

## 対象と条件

2026-11-13、R9700 / gfx1201、UUID `GPU-a8e9ddefa2d60f55`、開始点 `2c1000a`。対象は Qwen3.8-27B EXL3 4 bpw mul24、DFlash2 BF16 / mul24 drafter、BF16 KV、greedy である。比較は同じ GPU、各プロセスで `flock /tmp/r9700.lock` を取得し、別の GPU workload と重ねずに実施した。ROCm は `/opt/rocm/core-7.14`、clang 23、Release build。rocprofv3 と全体 regression は使っていない。

結果の全試行、比較 token、event、検査結果は [JSON](exl3_general.json) に保存する。速さの判定には通常 build を使う。一時計測 hook は [patch](exl3_stage_timing.patch) に保存し、通常 build から除いている。`docs/perf/current.md` は更新しない。

## 実行時の形と内訳

実 checkpoint を実行し、個別 linear を強制する一時 hook で入力回転・GEMM・出力回転を HIP event で測定した。各形状で最大 3 回、下表は原則として初回を除く 2 回の中央値、単位は µs。各 linear の直後に同期する診断であり、launch 間隔や clock が変わるため通常実行の stage 時間として加算しない。head など測定 1 回の形状は 1 回の値を使う。

| 実行 | M | N × K / bit | split | 入力回転 | GEMM | 出力回転 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| c=1 非投機 | 1 | 1024 × 5120 / 4 | 5 | 9.20 | 19.30 | 10.48 |
| c=1 非投機 | 1 | 17408 × 5120 / 4 | 4 | 9.70 | 98.16 | 10.48 |
| c=1 非投機 | 1 | 248320 × 5120 / 6 | 2 | 9.02 | 1548.93 | 16.22 |
| c=1 DFlash2 固定 draft | 8 | 17408 × 5120 / 4 | 4 | 9.24 | 103.26 | 12.82 |
| c=2 DFlash2 固定 draft | 16 | 17408 × 5120 / 4 | 4 | 10.54 | 107.58 | 16.02 |
| c=1 DFlash2 | 10 | 17408 × 5120 / 4 | 4 | 9.58 | 108.18 | 13.48 |
| c=1 DFlash2 | 10 | 248320 × 5120 / 6 | 2 | 9.18 | 1656.67 | 68.08 |
| c=4 非投機 | 4 | 17408 × 5120 / 4 | 4 | 9.04 | 101.48 | 11.76 |
| c=4 DFlash2 | 32 | 17408 × 5120 / 4 | 3 | 11.68 | 127.80 | 20.30 |
| c=4 DFlash2 | 38 | 17408 × 5120 / 4 | 3 | 11.94 | 141.64 | 22.46 |
| c=4 DFlash2 | 32 | 248320 × 5120 / 6 | 1 | 11.54 | 1769.69 | 148.28 |

非投機 c=2/3/5/6/7 と固定 draft の c=1/2 を追加し、M=1〜8 と 16 を捕捉した。c=4 の DFlash2 では M=1、2、10、17、29、32、38、46 を捕捉した。プロンプト処理も含むため全行数を verify とみなさない。全 shape と sample は JSON の `runtime_stage_timing` にある。

同じ GPU で `gpu-memory --stream-read --warmup 2 --iterations 5 --iters 20` を測ると read p50 は **622.94 GB/s**。合成 trellis を 256 MiB 以上巡回させた通常 build の単体計測では、M=1 / N=1024 / K=5120 / 4 bit の GEMM は 14.63 µs、trellis bytes / 時間は 179.2 GB/s（read p50 の 28.8%）。M=8 / N=17408 / K=5120 / 4 bit は 95.97 µs、464.4 GB/s（74.5%）、M=8 の head は 1638.54 µs、582.0 GB/s（93.4%）。これは転送量を trellis サイズで割った指標であり、DRAM counter の実測ではない。入力回転と出力回転は同じ M=1 / N=1024 でそれぞれ 4.18 / 4.83 µs だった。各時間は個別 event の中央値であり、合計と別区間の total を混同しない。

shape ごとの M=1〜8、16、32、40 は JSON の `micro` と `micro-extra` に 5 run 分を保存する。出力回転の 6144×5120・M=16 の基準 run は p50 間幅が中央値の 38% だったため、安定した性能値として採用しない。

## E3 — ACCEPT: 同一入力の連続 projection

同じ入力 slot の連続 `LINEAR_EXL3_INT8` から最大 3 本の 4 bit projection を検出し、入力回転、GEMM、出力回転をそれぞれ 1 launch にまとめる。異なる `suh`、scale、split と出力は projection ごとに分離する。K-split の数と加算順、BF16 中間値は変えない。形状固定の入力融合・出力融合は汎用経路では使わない。64 行超、F32 出力、列部分書き込み、非 4 bit、workspace 不足、入出力の重なり、debug override 等は個別経路に戻る。

Qwen3.8 の attention Q/K/V、GDN の QKV/Z、dense MLP の gate/up は、同じ入力を使う連続 linear なので条件を満たせば対象になる。GDN の a/b はこの checkpoint では BF16 重みで別経路。MoE の shared gate/up は同じ条件を満たす場合に対象となるが、routed expert は別の grouped dispatch と routing plan を使うため対象外。DFlash2 drafter の linear は独立した `quantized_linear` 呼び出しであり、この dispatch 統合の対象外である。drafter の独立 linear 自体は dispatch 統合されないが、共通回転・GEMM の変更は適用される。BF16 と mul24 の両 drafter で end-to-end を比較した。

合成重み、128 MiB の cache flush、1 回ごとに HIP event、20 回の中央値を 5 run、交互順。次表は run 中央値 ± 標本 SD、µs。これは統合全体の単体比較である。

| M | N の組 / K=5120 / 4 bit | 個別 | 統合 | BF16 要素の差 |
| ---: | --- | ---: | ---: | ---: |
| 1 | 12288, 1024, 1024 | 190.52 ± 5.20 | 164.64 ± 0.92 | 0 |
| 8 | 10240, 6144 | 202.10 ± 6.60 | 194.44 ± 2.07 | 0 |
| 8 | 17408, 17408 | 353.88 ± 3.32 | 335.52 ± 1.58 | 0 |
| 32 | 17408, 17408 | 394.42 ± 4.71 | 371.74 ± 0.97 | 0 |

## E1 — ACCEPT: 出力回転の列分割

既存の f4 kernel は 1 wave が 1 行全体を処理し、現行 selector では 512 行以上に使われる。M=1〜16 は既に scalar kernel の列方向 grid を使っていた。f4 の新しい variant は 1 wave が 128 列を担当し、行と列の両方向に block を分ける。split の加算順、Hadamard の演算順、scale と BF16 の丸め順を保持する。512 行以上の従来 f4 はそのまま残す。入力・出力 pointer と stride の整列を確認し、満たさなければ scalar を使う。

合成 trellis の 5 run 中央値では、M=8 の gate/up の出力回転が 6.99→5.4 µs、head は 34.38→15.4 µs。一方 N=1024・M=1 は 4.83→4.95 µs だったため、この小さい N は新経路に入れない。ユーザーが採否基準を変更した後は、end-to-end 差が小さくても bit 一致し、分散を超える再現した悪化がなければ適用する。

境界探索で F32 plane の大きさが 4 / 5 / 6 MiB の一部 shape に低速化を確認した。例えば N=5120・M=256・split=1 は 1 launch ごとの event で F32 28.40→36.64 µs、BF16 28.90→36.80 µs。N=10240・M=128 などでも同じ 5 MiB plane の一部で悪化した。5 MiB の前後 1 行の 40 条件、4 / 6 MiB の前後 1 行の 88 条件を測った。変動が大きかった 1 条件は graph event で再測定し、前後側の悪化は確認しなかった。未解明の組合せを含め、plane が 4 / 5 / 6 MiB のときは従来の scalar に戻す。同じ plane サイズでも速くなる組はあり、この fallback は保守的である。

N=5120・M=255・BF16・split=1 は run 間幅が 30% を超え、per-op と外部 graph event の再測定でも遅い run が再現したため旧経路に戻す。初回 graph 計測は default event で `invalid resource handle` となり、外部 event node に変更して再測定した。M=257、258、260、272 は F32/BF16・split=1/5 の全組が安定して短縮し、M=320、384、448、511 も短縮した。N=17408 の M=256/511 と、N=65536 / 107520 / 248320 の大行数も各測定条件で短縮したため、測定した N の大行数と大きい N の側は有効にする。未測定の中間 N の 256〜511 行は旧経路に残す。

測定した高行数の代表例。単位は µs、5 run の中央値 ± 標本 SD。per-op は 1 launch ごとの event を 20 回記録して中央値を取り、batch event は 20 launch の平均を event で測った。区間が異なるため両者の絶対値を直接比較しない。

| M | N | split | 出力 | 方法 | scalar | 列分割 | 選択 |
| ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| 256 | 5120 | 1 | F32 | per-op | 28.40 ± 0.18 | 36.64 ± 0.03 | scalar |
| 256 | 5120 | 1 | BF16 | per-op | 28.90 ± 0.36 | 36.80 ± 0.40 | scalar |
| 511 | 5120 | 1 | BF16 | per-op | 45.92 ± 0.68 | 21.26 ± 0.27 | 列分割 |
| 256 | 17408 | 1 | BF16 | per-op | 70.88 ± 2.18 | 28.72 ± 0.82 | 列分割 |
| 256 | 65536 | 1 | BF16 | batch event | 792.22 ± 0.87 | 224.08 ± 0.30 | 列分割 |
| 511 | 248320 | 2 | BF16 | batch event | 3694.76 ± 2.62 | 2132.63 ± 0.14 | 列分割 |

大行数の 20 launch 一括 event では 19 条件が run 間幅 30% 以上となり、その値を採否に使わなかった。該当条件は per-op event で再測定し、M=128・N=5120・split=5・BF16 の残った変動は graph event と各 run の preroll で再確認した。M=255・N=5120・BF16・split=1 は graph 再測定でも安定せず、前述の fallback とした。4 / 5 / 6 MiB の周辺にも 30% 以上の条件があり、旧経路に残すか graph で再確認した。初回値と再測定値の両方を JSON に保存する。

単体 test は scalar 基準と F32/BF16 を raw bit で比較する。split=1/2/5、未整列 pointer / stride、NaN・無限大・符号付きゼロ、255/256/257、511/512/513 行、plane fallback の境界と padding を含む。数値が異なる近似や split-K 数の変更は採用していない。

## E2 — ACCEPT: 小 N の行数別 tile と 5 bit の先読み省略

4 bit は N=512 で K=1024〜17408・49〜64 行、N=1024 で K=1024〜5120・49〜64 行、または K=5120 超〜8192・56〜64 行に MT=2 を使う。5 bit は N=512 で K=1024〜17408・17〜64 行、N=1024 で K=1024〜8192・17〜64 行、または K=8192 超〜17408・33〜64 行に MT=4 と次 K group の先読み省略を使う。device の active-row 指定がある補正 GEMM は従来経路を維持する。

5 run の単体中央値 ± 標本 SD は、4 bit・N=1024・K=5120・M=64 が 27.54 ± 0.10→23.18 ± 0.04 µs、5 bit・同形状・M=40 が 28.14 ± 0.13→21.00 ± 0.10 µs。広い候補（REJECT）では 4 bit・N=1024・K=17408・M=49 が 43.99 ± 0.15→53.46 ± 0.40 µs、5 bit・同 K・M=17 が 38.76 ± 0.19→42.26 ± 0.18 µs と悪化したので、その帯は上の条件から除いた。F32 accumulator の独立 kernel 比較と、split=1/2/5 等の BF16 出力ファイル 70 条件の byte 比較はすべて一致した。

`kPipe` の template 化が適用外 kernel を遅くしないか、4 bit の N=1024 / 10240 / 17408 / 5120 と 6 bit head、M=1/2/4/8 の 20 条件を 5 run ずつ交互に測った。GEMM 中央値の差はすべて 1% 以内、run 間幅はすべて 30% 未満だった。詳細は JSON の `gemm-unchanged`。

## E4 — 未採用: 小行数の回転と GEMM の融合

回転と GEMM の小行数融合は、PLaMo の固定形状に依存するため汎用 dispatch では使わない。今回、数値の丸め順を保ち、対象 shape を一般化した候補は採用していない。

Flash Next の checkpoint は約 80 GB で、指定の単一 R9700 には全体を載せられない。この checkpoint の end-to-end は実施せず、共通 kernel の単体一致と既存 grouped test を確認した。Gemma-4 / Qwen4Exp の実モデル全体の再検証も行っていない。

## End-to-end と一致性

初期 revision と最終 commit を交互に各 3 回。表示は平均 ± run 間標本 SD、単位 tok/s。c=1 は `shelf-compute` の decode 指標、c=2/4 は stdio protocol の同時 request で、warmup 後の 128 token batch の prefill と出力送受信を含む。HTTP transport は含まない。c=1 の投機は draft 7・既定 tree、c=2/4 の投機は draft 7・tree budget 0。

| 経路 | c | 初期 | 最終 | 平均比 |
| --- | ---: | ---: | ---: | ---: |
| 非投機 | 1 | 27.36 ± 0.10 | 29.55 ± 0.01 | +8.00% |
| 非投機 | 2 | 50.27 ± 1.43 | 55.00 ± 1.13 | +9.40% |
| 非投機 | 4 | 94.64 ± 0.10 | 102.25 ± 0.08 | +8.05% |
| DFlash2 BF16 | 1 | 84.98 ± 0.05 | 90.15 ± 0.09 | +6.08% |
| DFlash2 BF16 | 2 | 88.70 ± 4.10 | 93.23 ± 1.71 | +5.11% |
| DFlash2 BF16 | 4 | 128.25 ± 2.94 | 139.53 ± 6.38 | +8.79% |
| DFlash2 mul24 | 1 | 84.99 ± 0.34 | 90.46 ± 0.28 | +6.44% |
| DFlash2 mul24 | 2 | 87.45 ± 8.37 | 88.39 ± 0.92 | +1.07% |
| DFlash2 mul24 | 4 | 148.34 ± 12.22 | 152.93 ± 5.80 | +3.10% |

連続統合のみの版と、そこに列分割・tile を加えた最終版の比較。c=1 非投機は外れ値を含め追加 5 組も合算し、各 8 run とした。

| 経路 | c | 統合のみ | 全候補 | 平均比 |
| --- | ---: | ---: | ---: | ---: |
| 非投機 | 1 | 29.33 ± 0.51 | 29.33 ± 0.42 | -0.01% |
| 非投機 | 2 | 54.84 ± 0.85 | 55.41 ± 0.35 | +1.03% |
| 非投機 | 4 | 103.73 ± 0.15 | 103.71 ± 0.15 | -0.02% |
| DFlash2 BF16 | 1 | 90.07 ± 0.05 | 90.10 ± 0.10 | +0.03% |
| DFlash2 BF16 | 2 | 92.31 ± 0.12 | 95.60 ± 4.41 | +3.57% |
| DFlash2 BF16 | 4 | 138.43 ± 6.25 | 135.53 ± 0.15 | -2.09% |
| DFlash2 mul24 | 1 | 90.06 ± 0.29 | 90.25 ± 0.23 | +0.21% |
| DFlash2 mul24 | 2 | 93.97 ± 8.94 | 93.99 ± 8.91 | +0.03% |
| DFlash2 mul24 | 4 | 155.55 ± 13.70 | 150.06 ± 3.26 | -3.53% |

c=4 投機の平均は下振れしたが、いずれも統合のみの標本 SD 内で、中央値は BF16 が 135.17→135.51、mul24 が 147.88→148.44 tok/s。各 3 組で負方向が再現しておらず、更新後の採否基準では維持した。小差が測定誤差と区別できないことと、常に高速化すると証明したことは区別する。

c=64 の非投機は 32 token・prefill を含む 3 組で 310.49 ± 7.86 → 324.05 ± 2.53 tok/s。64 request すべての token が一致した。別の初回 1 組は 318.63→311.50 tok/s で、こちらも JSON に残す。

PLaMo 専用経路の短窓参考値。最初の完了後 5 秒 warmup、30 秒窓、各 2 run。短窓なので細かな差の有意性を主張しない。

| c | 初期 | 最終 |
| ---: | ---: | ---: |
| 1 | 154.32 ± 2.90 | 156.37 ± 0.00 |
| 64 | 3223.95 ± 1.30 | 3230.23 ± 8.96 |

固定 prompt 3 種、c=1・2・4、非投機と BF16 / mul24 drafter の投機で対応する生成 token ID はすべて一致した。batch の ID は固定幅 int32 で byte 比較し、c=1 の ID 列は出力 byte 列を比較した。単体の BF16 出力 70 条件もファイルを byte 比較し、PLaMo の c=1/64 dump も一致した。全結果と SHA256 は JSON に保存する。

## PLaMo とテスト

PLaMo 専用経路は残した。各 52,900,856 byte の `general-corpus-0` teacher-forced dump は、c=1 が基準・変更後とも SHA256 `15a8869e8362b2bda7ecd84a6e0dc7550e671c9cb453e758e199f4e154c35965`、c=64 がともに `9c0a9419dde4e9e8b986fabab015e4cb93e078a7b78747edfdeb7c297a5cb40f` であり、実ファイル同士も byte 一致した。これは各 batch の代表 1 case であり、全 corpus の比較を意味しない。

最終 build の関連 ctest は 10/10 成功。実 mul24 drafter / 復元 BF16 checkpoint を使う `test_drafter_quant_linear` は skip せず M=1/8/16 を実行した。この test は量子化誤差の検証であり、bit 一致の代用にはしない。PLaMo projection test は形状固定融合を無効にした複数 projection と、F32/BF16 回転、split、特殊値、未整列 pointer、padding を比較する。`test_compute_concurrency.py` は 30 passed / 0 failed、PLaMo 実 server は 10 passed / 0 failed。server 系では `PATH=/tmp/venv-so/bin:$PATH`、PLaMo では `/tmp/venv-plamo/bin` を設定した。全体 regression は行っていない。

batch 計測 harness の初回は warmup と測定に同一 request ID を使って `duplicate request_id` で失敗した。ID を分けて再実行し、失敗を `batch-initial-failure` に残した。失敗 run は速度比較に含めない。

## 再現

build と実行時 library path は次を使う。各 GPU 実行には UUID と lock を付け、個々の run ごとに解放する。

```bash
cmake -S . -B build-gfx1201 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSHELF_ROCM_ROOT=/opt/rocm/core-7.14 -DCMAKE_PREFIX_PATH=/opt/rocm/core-7.14 \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-7.14 -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DSHELF_LIBCXX_LLVM_ROOT=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23
ninja -C build-gfx1201
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
```

`exl3_micro.py` / `exl3_micro_extra.py` / `exl3_tile_micro.py`、`exl3_group_micro.hip`、`exl3_final_e2e.py`、`exl3_final_batch.py`、`exl3_batch.py`、`exl3_trace_runs.py`、`exl3_plamo_dump_final.py` は今回の実行条件と GPU 排他を示す。baseline と candidate の binary は、build 後に `build-gfx1201/exl3-general/` へ `shelf-compute.base` / `shelf-compute.final` 等として保存する。group micro は gfx1201 用 clang 23 で `libshelf_kernels_optimized.a` と HIP / libc++ に link する。PLaMo dump は既存の `/tmp/plamo-cumulative-final-quality-1/general-corpus-0.tokens` と `.baseline` を使う。

未制限の出力回転 micro は `8862fa2` に [unrestricted patch](exl3_rotate_unrestricted.patch) を適用した別 binary と、同じ harness 内の scalar 基準を使う。広い tile の探索は初期 source に [探索 patch](exl3_tile_candidate.patch) を適用する。採用 selector を含む通常 build にこれらの patch は適用しない。

hook は `88f0c2e` の clean source に patch を適用して trace binary を保存し、source を戻して通常 build を再作成する。この時点の共通回転・GEMM は初期版であり、hook は連続統合も無効にする。`SHELF_EXL3_STAGE_TRACE=1` は汎用 projection 統合を無効化し、個別 linear の内訳を同期計測する。集計は `python3 docs/research/20261113/exl3_general/exl3_report.py`。commit、結果と uncommitted 差分は Git history を参照する。

# PLaMo 3 EXL3-INT8 linear の最適化

2026-11-12、R9700 / gfx1201、ROCm 7.14、BF16 KV、greedy、投機なし。対象は 2B / 4.05 bpw / 8 bit head のみ。
RMSNorm、attention、scheduler、stall 対策は変更していない。push と PR 作成は行っていない。

## 継続投入

既存 `plamo_closed_loop.py` の seed 17・32 入力・最大出力 320 token・20 秒 warmup・90 秒窓を使用した。
各 GPU 実行を `flock /tmp/r9700.lock` と対象 UUID で保護し、run ごとに lock を解放した。
採用判定には診断なしの実行だけを使う。表示は平均 ± run 間標本標準偏差。JSON に各 run の値と完了数を保存する。

| c | before tok/s | after tok/s | 変化 | run 数 before / after |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 127.84 ± 0.64 | 151.73 ± 0.24 | +18.69% | 3 / 2 |
| 4 | 459.87 ± 1.04 | 541.09 ± 2.58 | +17.66% | 2 / 2 |
| 16 | 1343.60 ± 0.00 | 1609.39 ± 1.12 | +19.78% | 2 / 2 |
| 64 | 2522.09 ± 1.06 | 2784.13 ± 4.78 | +10.39% | 3 / 2 |
| 128 | 3022.39 ± 2.37 | 3151.92 ± 11.29 | +4.29% | 2 / 2 |

## shape 別 HIP event

M は行数、N は出力、K は入力。shape micro は seed 17 の合成 resident trellis を使う。
実 checkpoint の step event と継続投入は別に測定する。個別 projection の after は融合を含まない。
各 run は warmup 20 回・測定 20 回の p50、5 run を AB / BA で交互に測定した。
表は 5 個の p50 の中央値。単位は µs。GEMM と linear の前には 128 MiB の hipMemset を挿入した。
これは cache 置換を意図した前処理で、hardware の cache miss は計測していない。raw の cold=true はこの前処理の有無を表す。
微小 kernel の event 区間には host / launch の影響もあるため、stage の和を全 linear 時間に足し直さない。
roofline は既存測定の帯域 636 GB/s を使った weight-only 下限で、activation、partial、回転、launch を含まない。
実 step の全 linear trellis は cache 容量を超えるが、個別 shape の cache 状態は完全には再現していないため、比は参考値とする。

| M | N × K / bit | calls | rotate before / after | GEMM before / after | out before / after | linear before / after | GEMM / roof before / after | linear / roof before / after |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 × 2560 / 4 | 36 | 9.98 / 9.94 | 19.92 / 18.98 | 9.70 / 9.84 | 29.08 / 28.26 | 19.33 / 18.42 | 28.22 / 27.42 |
| 1 | 512 × 2560 / 5 | 12 | 9.98 / 9.98 | 22.16 / 20.46 | 9.76 / 9.84 | 31.22 / 29.74 | 17.20 / 15.88 | 24.24 / 23.09 |
| 1 | 2560 × 3072 / 4 | 18 | 9.90 / 9.86 | 24.92 / 24.94 | 9.72 / 9.84 | 34.44 / 34.60 | 4.03 / 4.03 | 5.57 / 5.60 |
| 1 | 2560 × 3072 / 5 | 6 | 9.92 / 9.86 | 30.78 / 30.28 | 9.80 / 9.84 | 39.90 / 39.40 | 3.98 / 3.92 | 5.16 / 5.10 |
| 1 | 2560 × 10240 / 4 | 24 | 9.96 / 9.94 | 61.86 / 61.54 | 10.66 / 10.80 | 72.74 / 72.72 | 3.00 / 2.99 | 3.53 / 3.53 |
| 1 | 3072 × 2560 / 4 | 18 | 9.94 / 9.88 | 25.52 / 25.60 | 9.64 / 9.72 | 34.60 / 34.66 | 4.13 / 4.14 | 5.60 / 5.61 |
| 1 | 3072 × 2560 / 5 | 6 | 9.96 / 9.94 | 30.38 / 29.04 | 9.64 / 9.82 | 39.38 / 38.50 | 3.93 / 3.76 | 5.10 / 4.98 |
| 1 | 10240 × 2560 / 4 | 48 | 9.92 / 9.88 | 62.72 / 62.06 | 9.48 / 9.58 | 71.04 / 70.86 | 3.04 / 3.01 | 3.45 / 3.44 |
| 1 | 107520 × 2560 / 8 | 1 | 9.90 / 9.88 | 617.97 / 617.61 | 10.62 / 9.98 | 631.77 / 629.43 | 1.43 / 1.43 | 1.46 / 1.45 |
| 64 | 512 × 2560 / 4 | 36 | 11.40 / 11.40 | 34.02 / 26.20 | 9.86 / 9.84 | 45.56 / 37.72 | 33.01 / 25.42 | 44.21 / 36.60 |
| 64 | 512 × 2560 / 5 | 12 | 11.40 / 11.40 | 30.86 / 35.32 | 9.84 / 9.74 | 42.54 / 47.10 | 23.96 / 27.42 | 33.03 / 36.57 |
| 64 | 2560 × 3072 / 4 | 18 | 11.44 / 11.44 | 42.24 / 37.00 | 12.16 / 10.72 | 58.36 / 52.60 | 6.83 / 5.98 | 9.44 / 8.51 |
| 64 | 2560 × 3072 / 5 | 6 | 11.50 / 11.48 | 43.26 / 41.52 | 12.28 / 10.78 | 59.10 / 57.12 | 5.60 / 5.37 | 7.65 / 7.39 |
| 64 | 2560 × 10240 / 4 | 24 | 15.04 / 14.96 | 94.32 / 89.64 | 12.70 / 11.18 | 112.80 / 111.50 | 4.58 / 4.35 | 5.47 / 5.41 |
| 64 | 3072 × 2560 / 4 | 18 | 11.32 / 11.32 | 43.74 / 36.68 | 32.16 / 10.60 | 79.24† / 51.34 | 7.07 / 5.93 | 12.82 / 8.30 |
| 64 | 3072 × 2560 / 5 | 6 | 11.42 / 11.40 | 43.82 / 43.48 | 31.96 / 10.76 | 59.96† / 58.50 | 5.67 / 5.63 | 7.76 / 7.57 |
| 64 | 10240 × 2560 / 4 | 48 | 10.80 / 10.72 | 98.92 / 83.62 | 19.20 / 13.62 | 121.33 / 103.64 | 4.80 / 4.06 | 5.89 / 5.03 |
| 64 | 107520 × 2560 / 8 | 1 | 11.38 / 11.38 | 849.98 / 789.16 | 96.40 / 37.16 | 974.80 / 870.76 | 1.96 / 1.82 | 2.25 / 2.01 |

† M=64・4 / 5 bit Q の baseline 単体 linear は、run p50 の最大 / 最小差が 30% を超え、安定性判定を FAIL とした。
隔離再測定と warmup 1000 回でも約 59〜79 µs の変動が残った。原因は確定していない。
この単体 linear の改善率は採用根拠に使わず、下の projection 統合列・実 step・診断なし closed-loop の再現性で判断する。
再測定の全値を `stability_followup` に残す。

## launch 統合

QKV と gate/up は SUH が異なるので、量子化した同じ入力を使い回していない。各 projection の演算と丸めを保ったまま grid をまとめた。
linear は論理 169 個のまま、1〜255 行の対象 step では物理 launch が 507→291、実 dispatch が 169→97 になる。
gfx12 の INT8 WMMA は従来から使用しており、最終経路でも維持する。M=64 の 4 bit GEMM は各 wave で weight fragment を 32 行に再利用し、行方向に 2 block を使う。
同じ N tile の行 block を隣接して配置する。cache miss は未計測なので、DRAM からの読み出しが厳密に 1 回になるとは主張しない。
次表は既存の個別 projection 列と最終の統合列を交互に測定した値。表示は 5 run p50 の中央値 ± 標本標準偏差。

| M | 対象 / bit | before µs | after µs |
| ---: | --- | ---: | ---: |
| 1 | QKV / 4 | 78.40 ± 2.58 | 39.64 ± 0.72 |
| 1 | QKV / 5 | 88.12 ± 3.20 | 45.02 ± 0.84 |
| 1 | gate/up / 4 | 134.47 ± 1.94 | 126.11 ± 1.06 |
| 64 | QKV / 4 | 145.89 ± 12.27 | 60.42 ± 0.81 |
| 64 | QKV / 5 | 148.51 ± 7.38 | 65.94 ± 1.79 |
| 64 | gate/up / 4 | 243.37 ± 2.37 | 170.95 ± 0.49 |

## 実 step の残り

既存の semantic HIP event を executor の一時 hook から使い、8 回目の decode-only step を 3 run 捕捉した。
計測用変更は `plamo_linear_timing.patch` だけに保存し、通常 build から除去した。scheduler と各演算は変更していない。
event 区間には計測 overhead がある。stepwall も同じ診断 run からの値であり、clean throughput は継続投入表だけを使う。
event 合計と stepwall の差を使って厳密な内訳にはできない。

| c | variant | 診断 run の step p50 ms | linear event ms | RMSNorm event ms | attention event ms | event 合計 ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | 7.558 | 6.443 | 1.856 | 0.349 | 10.004 |
| 1 | final | 6.834 | 4.704 | 1.874 | 0.359 | 8.288 |
| 64 | baseline | 15.649 | 12.543 | 1.858 | 1.868 | 17.910 |
| 64 | final | 12.625 | 9.184 | 1.874 | 1.945 | 14.638 |

同じ診断 run の固定長・同一段落の decode-only 窓は次のとおり。server の継続投入とは workload と計測範囲が異なる。
診断付きの補助値であり、4,000 tok/s の server 目標を達成したという判定には使わない。

| c | baseline decode tok/s | final decode tok/s |
| ---: | ---: | ---: |
| 1 | 135.82 ± 0.52 | 154.32 ± 5.55 |
| 64 | 4063.92 ± 15.76 | 5023.64 ± 28.23 |

linear の trellis は 1 step あたり 1.4529 GB。1 回ずつ読む weight-only 下限は 2.284 ms。
既存文書の約 3.16 ms は embedding を含む checkpoint 全体（約 2.007 GB）からの概算であり、ここでの linear 限定の byte 数とは異なる。
測定で得た最高値は探索した候補での到達値であり、今後の kernel も含む厳密な上限ではない。

修正版の closed-loop 観測最大は c=64 が 2787.51、c=4 が 542.91 tok/s。
c=64 の目標 4,000 tok/s は未達。平均との差は 1215.87 tok/s、さらに 43.67% のスループット改善が必要。
c=4 の目標 700 tok/s は未達。平均との差は 158.91 tok/s、さらに 29.37% のスループット改善が必要。

closed-loop は新規 prefill を含むため、decode-only の改善率から全体の速度を換算できない。
256 行以上の dense / flagged-row 経路、64 行を超える head、残る linear の復号と partial traffic は今回の採用経路で改善していない。
RMSNorm、attention、その他の演算と待ち時間も残るが、今回それらの最適化や stall の因果調査は行っていない。

## 品質

各方向 18 文・875 位置を CPU BF16 と比較し、WikiText 521 位置を別集計した。各列を同じ batch 行数に複製する既存手順であり、混在文の teacher forcing ではない。
c=1 / 64 の全 80 dump は直前の採用済み実装と byte 一致した。今回の対応位置 KL 差は数学的に 0 で、最大 logits 差・top-1 低下も 0。
位置ごとの float64 再計算には最大 4.00e-15 の丸め誤差がある。元の計算値を JSON に保存した。

| batch | 方向 | KL | top-1 | CPU BF16 との最大 logits 差 |
| ---: | --- | ---: | ---: | ---: |
| 1 | ja-en | 0.01801611 | 92.9143% | 4.750000 |
| 1 | en-ja | 0.01515961 | 93.7143% | 6.066406 |
| 64 | ja-en | 0.01797621 | 93.0286% | 5.500000 |
| 64 | en-ja | 0.01462298 | 94.0571% | 5.906250 |

c=64 の元の基準（KL 0.017130 / 0.017396）との差は、既採用 split-K 4 の累積差である。今回の変更では増減していない。
元の基準に対する paired per-position KL と文単位 bootstrap 95% CI、および incumbent との差を JSON に保存する。
case 名と dump 位置に対応する全 KL 差も `paired_positions` に格納する。

## cold と batch 形状

固定行数の kernel 比較は 1 / 4 / 16 / 17 / 32 / 33 / 48 / 49 / 64 / 70 / 128 / 192 / 255 行で BF16 出力と padding が一致した。
HTTP の cold 64 件を各 5 回起動し直し、全 10 組の run 間出力差を比較した。
以前の 5/64 は固定の再現数ではなく、今回の同条件 baseline も異なる件数になった。過去の 5 件以下を保証したという主張はしない。

| variant | run 間差の平均 /64 | 中央値 /64 | 最大 /64 |
| --- | ---: | ---: | ---: |
| baseline | 8.70 | 8.50 | 12 |
| final | 7.10 | 6.50 | 10 |

c=1 の繰り返しと baseline / final 間の出力は一致した。固定形状の数値差は増えておらず、cold の差も同条件 baseline と比較する。

## 採否と再現

採用: QKV / gate-up の独立回転を保つ launch 統合、M に応じた tile、5 bit の先読み省略、出力回転の列方向分割。split-K と BF16 中間値は維持した。
探索の初期 JSON は 20 回平均 × 5 run、採用の比較は上記の p50 × 5 run と区別する。

関連 kernel / unit と PLaMo BF16 / EXL3 / synthetic / template / startup server テストを実行した。結果の詳細は JSON に保存する。`--all` は実行していない。
詳細データ: [plamo3_exl3_linear.json](plamo3_exl3_linear.json)。

### 不採用と保留

途中の出力回転は、各 wave の列ループの刻みが 1 で、同じ列区間を複数 wave が重複して書いていた。
出力値は bit 一致していたが、列の所有者が一意になるよう刻みを wave 数に修正した。
修正前の closed-loop は `discarded_overlap_runs` に分離し、上の採用表から除外した。修正後に品質・kernel・server・速度を再検証した。

INT8 への事前復号は、既存の WMMA fragment 順に展開した cache を使って試した。
次表の cache は探索した MT / split-K の中の最速値で、baseline は production incumbent。split 数の変更も含む探索値であり、採用の精度検証を通した値ではない。

| M | 対象 | baseline GEMM µs | 最速 cache GEMM µs | MT / split |
| ---: | --- | ---: | ---: | --- |
| 1 | down | 61.66 | 106.92 | 1 / 1 |
| 64 | down | 94.66 | 109.63 | 2 / 1 |
| 64 | gate/up | 101.23 | 113.19 | 1 / 1 |
| 64 | lm_head | 831.41 | 677.16 | 1 / 1 |

4 bit 全体の展開は weight bytes が倍になり、この探索では遅くなったため不採用。
8 bit head の cache は単体では改善したが、追加 cache は約 275 MB。削減は約 0.15 ms / decode step にとどまり、
step 全体で約 1% の改善余地に相当する。head cache 単独の closed-loop / 拡張品質評価は行っておらず、採用せず保留した。
採用コードには復号済み weight cache を残していない。

direct INT8 dot4 も試した。split-K を incumbent に固定した比較では、M=1 の K/V や M=64 の head が遅くなった。
split を増やすと勝つ小形状もあるが加算順が変わり、今回の融合に重ねた品質・速度上の優位を確認していない。WMMA を維持した。

| M | 対象 | baseline µs | dot4 µs | split |
| ---: | --- | ---: | ---: | ---: |
| 1 | K/V | 20.11 | 25.75 | 2 |
| 64 | lm_head | 831.41 | 7382.08 | 1 |

split-K は 1 / 2 / 4 / 5 / 8 / 10、row tile は 1 / 2 / 4、wave は 4 / 8 を探索した。
8 wave と split の変更は形状により勝敗があり、統合後の全体速度と品質を根拠に採用できる優位を確定していない。
最終 GEMM は 4 wave と既存 split を使い、加算順を維持する。5 bit の先読み省略は register の live 範囲を縮めるために選択した。

適用行数を拡大した途中の候補では、MT=2 のままの M=192 gate が約 +11%、M=128 / 192 head が約 +12% / +15% 遅くなった。
その選択は外し、4 bit は 129 行以上を MT=4、head は 65 行以上を既存経路に戻した。生の比較は `wide_candidate_p50_us` にある。
greedy の head+argmax 融合は未実装。head の採用変更は tile と出力回転までであり、sampling を変更したとは主張しない。

### 再現手順

指定 worktree の root で実行する。baseline は開始時点 `d3a8a7b` の保存済み binary、final4 は最終コードの保存済み binary。
参照データの作成は [既存手順](plamo3_exl3.md#拡大品質評価と継続投入の測定方法) に従う。
各 runner が GPU UUID、library path、flock を設定する。CPU の build は GPU lock を取得しない。

```bash
export PATH=/tmp/venv-plamo/bin:/tmp/venv-so/bin:$PATH
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
python3 docs/research/20261112/plamo_linear_build.py
python3 docs/research/20261112/plamo_linear_scan.py --selected --output build-gfx1201/plamo-linear/final-selected.jsonl
python3 docs/research/20261112/plamo_linear_scan.py --groups --output build-gfx1201/plamo-linear/final-groups-selected.jsonl
python3 docs/research/20261112/plamo_linear_scan.py --sweep --mean20 --output build-gfx1201/plamo-linear/sweep.jsonl
python3 docs/research/20261112/plamo_linear_scan.py --extra --mean20 --output build-gfx1201/plamo-linear/extra.jsonl
flock /tmp/r9700.lock env ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 build-gfx1201/plamo-linear/micro 64 3072 2560 5 4 0 > build-gfx1201/plamo-linear/q5-repeat.jsonl
flock /tmp/r9700.lock env ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 build-gfx1201/plamo-linear/micro 64 3072 2560 5 4 0 1000 > build-gfx1201/plamo-linear/q5-preroll.jsonl
flock /tmp/r9700.lock env ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 build-gfx1201/plamo-linear/micro 64 3072 2560 4 4 0 1000 > build-gfx1201/plamo-linear/q4-preroll.jsonl
python3 docs/research/20261112/plamo_linear_quality.py --bench build-gfx1201/plamo-linear/shelf-bench.final4 --output build-gfx1201/plamo-linear/quality-final
python3 docs/research/20261112/plamo_linear_compare.py --baseline build-gfx1201/plamo-linear/shelf-compute.baseline --candidate build-gfx1201/plamo-linear/shelf-compute.final4 --output build-gfx1201/plamo-linear/runs --tag paired --concurrency 64 1 4 16 128
python3 docs/research/20261112/plamo_linear_runs.py --compute build-gfx1201/plamo-linear/shelf-compute.final4 --output build-gfx1201/plamo-linear/runs --tag final-repeat --concurrency 128 64 16 4 1
python3 docs/research/20261112/plamo_linear_cold.py --baseline build-gfx1201/plamo-linear/shelf-compute.baseline --candidate build-gfx1201/plamo-linear/shelf-compute.final4 --output build-gfx1201/plamo-linear/cold-final
python3 docs/research/20261112/plamo_linear_tests.py --output build-gfx1201/plamo-linear/server-tests-final
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/venv-plamo/bin/python docs/research/20261112/plamo_linear_positions.py
python3 docs/research/20261112/plamo_linear_report.py
```

棄却した large-M の MT=2 候補は `plamo_linear_build.py --wide-mt2` で生成できる。
`plamo_linear_scan.py --selected --rows 128 192` で比較し、その後、通常の build.py で最終 kernel を再生成する。

step event の再取得は次のように行う。計測 hook は既存の executor timing だけを有効にする。

```bash
git apply docs/research/20261112/plamo_linear_timing.patch
cmake --build build-gfx1201 --target shelf-compute -j 6
cp build-gfx1201/shelf-compute build-gfx1201/plamo-linear/shelf-compute.profile-final
git apply -R docs/research/20261112/plamo_linear_timing.patch
cmake --build build-gfx1201 --target shelf-compute shelf-bench -j 6
python3 docs/research/20261112/plamo_linear_profile.py --compute build-gfx1201/plamo-linear/shelf-compute.profile-final --output build-gfx1201/plamo-linear/profiles-final
```

kernel / unit は `focused_ctest_log` にテスト名と結果を保存した。server test は PLaMo 用環境を使い、構造化出力の全回帰は行っていない。

### commit

- `e8401e8` — 独立した projection をまとめる kernel と BF16 / padding 一致テスト。
- `61bfe35` — 対象 checkpoint の flag、実行範囲と trace を保つ dispatch、selector、developer contract。
- `1b0f262` — 再実行できる計測・比較・品質・cold・集計の harness。
- `bec6263` — 出力の列区間を各 wave が一度だけ担当する修正。
- 計測記録の commit はこの文書を含む Git history を参照。

# c=1 decode の launch 削減の統合

## 条件と実装

2026-11-14、R9700 gfx1201、ROCm 7.14、Qwen3.8-27B EXL3 4bpw mul24、BF16 KV、context 128。基準は `origin/main` の `44da8fe`。GPU 実行は `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` と `/tmp/r9700.lock` で排他した。attn、rq、boundary の実装を順に取り込み、GDN の実験は取り込まない。boundary は既定有効で、`SHELF_DISABLE_BOUNDARY_I1`、`SHELF_DISABLE_BOUNDARY_I8`、`SHELF_DISABLE_BOUNDARY_I9` で無効化できる。

## 競合と融合の接続

`cmake/targets.cmake` は attention と boundary の両ソースを登録し、`optimized_dispatch.hip` は SPLIT、SIGMOID、RESIDUAL_ADD の各 matcher を保持した。EXL3 single linear の launcher は rq の producer 引数と boundary の出力回転遅延を共存させた。`fixedcost_lucebox.hip` の GDN producer も同じ接続を使う。

最初の統合は 1,124 launch だった。rq が single linear を消費すると boundary I9 がその linear に到達できなかったため、producer からの呼び出しでも後続 boundary の窓を処理し、消費 dispatch 数を合算した。従来通り不適用時は出力回転へ戻る。接続修正は `34acae15`。

## launch と品質

perf stats の decode は baseline 1,581、統合後 1,001（−580）。`trace_names.cpp` の decode 窓は 998 / 1,001、prefill は 1,914。統合後の BetterBench 256 位置は mean KL 0、top-1 100%、最大 logit 絶対差 0。固定 code / prose / ja の各 256 token の greedy ID も baseline と一致。関連 ctest 12/12 成功。PLaMo EXL3、入力 ID `[1, 2, 3]`、128 token の baseline と統合後の greedy ID は一致。

通常の `test_compute_dflash2.py` は BF16、FP8、Octave を含め 99/99 成功した。テスト実行ごとに GPU lock を取得した。tree の投機と非投機の一致は baseline を含め完全には成立していない。

## 速度

baseline build と統合 build を交互に 5 ペアずつ実行した。非投機と DFlash2 は code prompt、256 token、最大長 1024、temperature 0。DFlash2 は BF16 drafter、drafts 7。GPU event は `shelf-bench tg --mode forward --context 128 --tokens 128 --warmup 4`。ロック取得後に load average が 8 未満であることを確認し、終了時の最大値は 4.24。途中で他の正しさ試験も GPU lock を介して交互に実行された。非投機と DFlash2 の各 10 run の生成 ID は、すべて共有非投機 baseline と一致した。

| 指標 | baseline 平均 ± 標本 SD | 統合後 平均 ± 標本 SD | 変化 |
|---|---:|---:|---:|
| 非投機生成 tok/s | 30.886 ± 0.070 | 31.598 ± 0.071 | +2.31% |
| GPU event µs/token | 32,530.948 ± 64.242 | 31,707.844 ± 55.525 | −2.53% |
| DFlash2 tok/s | 108.846 ± 0.151 | 101.242 ± 0.082 | −6.99% |

平均差は各群の標本 SD を上回る。ペア差の標本 SD は順に 0.008 tok/s、54.299 µs/token、0.093 tok/s。非投機の逆数から求めた時間差は 580 launch 当たり 1.258 µs/launch、GPU event では 1.419 µs/launch に相当する。これらは平均差の換算であり、個々の launch 費用の直接測定ではない。DFlash2 の launch 数は測っていないため、580 で割った値を DFlash2 の launch 当たり費用として解釈しない。

単独トラックの非投機改善 +1.02%、+1.05%、+0.52% の単純和は約 +2.59%、統合実測は +2.31%。ほぼ同程度だが、別の実行時刻・binary・対象条件の測定であり、厳密な加算性や差の統計的有意性は判定できない。目標の +5% には達しない。DFlash2 は明確な速度低下を示し、非投機の改善だけで統合全体を評価できない。

DFlash2 低下の単発探索では、統合 binary の I9 のみ無効が 106.03 tok/s、boundary 全無効が 109.05、rq 全無効が 100.49、attn 全無効が 101.35 だった。いずれも 44 rounds、accepted drafts 211、生成 ID は同一。これは各条件 1 回であり、boundary と低下の関係を示す手掛かりであって、kernel 内の原因の証明ではない。複数行 verify で boundary の処理費用が増えている可能性はあるが、個別区間の時間は計測していない。

boundary 単独の過去測定で GPU event −2.84% に対して生成 +1.02% だった理由は未特定。GPU event と生成速度の計測範囲は同一ではなく、sampling、host 側処理、graph、測定間変動の寄与が候補だが、この統合測定だけで寄与量は判別できない。今回の統合では両指標とも約 2.3〜2.5% 改善した。

## tree の baseline 比較

`test_compute_dflash2_tree.py` の 3 prompt を baseline build と統合 build の両方で実行した。各 GPU 子 process を個別に lock と 280 秒 timeout で包んだ。両版とも 3/3 prompt がテストの許容条件を満たした。target、chain、tree0/1/2/4、budget8/10/12/16/24 の対応する 256 token 列は、3 prompt すべてで両版間に差がなかった。各 mode の不一致位置は [launch_integration.json](launch_integration.json) に記録する。

| prompt | baseline と統合後の chain 対 target 不一致数 | 最初の位置（0 始まり） | 最初の target margin |
|---|---:|---:|---:|
| ファイル管理 | 12 | 244 | 0.010727 |
| 最長共通部分列 | 132 | 118 | 0.004416 |
| √2 の証明 | 213 | 43 | 0.029339 |

これは baseline から存在する投機と非投機の差である。テストは最初の差の margin が 0.1 以下なら許容するため、テストの PASS は投機と非投機の完全一致を意味しない。今回の入力と各 mode では、統合による出力の悪化は観測されなかった。

## 再現条件と未解決

Release / gfx1201 build は `nice -n 19 ninja -C build-gfx1201 -j8` で対象を build した。launch は `SHELF_QWEN35_PERF_STATS=1 shelf-bench tg --mode forward --context 128 --tokens 8 --warmup 1 --perf-stats-output <json>` と `trace_names.cpp` の LD_PRELOAD を併用した。品質は共有 corpus、context 128、256 位置、BF16 KV を `shelf-bench quality --compare /tmp/ln_baseline/base_ctx128_p256.psq` で比較した。各 GPU コマンドは指定 GPU と `flock /tmp/r9700.lock timeout -k 5 280` を使用した。

DFlash2 速度低下の kernel 単位の原因と、tree に既存の低 margin 差は未解決。`run_server_regression.py --all` は本作業では実行していない。TP や混在 request の実モデル評価は行っていない。

## 追試: DFlash2 の低下と行数制限

同じ R9700 とモデルで、前節の統合 binary を使い、DFlash2 code prompt、drafts 7、BF16 drafter、256 token を全部入りと各無効化で交互測定した。まず各 1 ペアで探索し、改善した候補は各 3 ペアを追加した。GPU は各 run で排他し、開始時の load average が 8 未満であることを確認した。全測定で生成 ID は同一。

| 無効化 | 全部入り tok/s | 無効 tok/s | ペア数 | 変化 |
|---|---:|---:|---:|---:|
| 全融合 | 102.147 ± 0.108 | 109.247 ± 0.116 | 3 | +6.95% |
| I1 | 101.967 ± 0.061 | 110.170 ± 0.050 | 3 | +8.05% |
| I8 | 101.863 ± 0.071 | 109.840 ± 0.101 | 3 | +7.83% |
| I9 | 101.877 ± 0.040 | 106.803 ± 0.067 | 3 | +4.84% |
| I2 | 102.17 | 101.86 | 1 | −0.30% |
| I7a | 102.19 | 101.88 | 1 | −0.30% |
| I7b | 102.12 | 101.98 | 1 | −0.14% |
| attention Q | 102.05 | 102.07 | 1 | +0.02% |
| attention KV | 102.07 | 101.92 | 1 | −0.15% |

I1 の無効化は I8・I9 も無効にし、I8 の無効化は I9 も無効にする。そのため各行は単独効果ではない。I1 を残して I8・I9 を止めても回復することから、I8／I9 のみ実行行数 1 に制限した。出力 domain の行数も検査し、I9 は複数行では出力回転を遅延しない。I1 と rq、attention は複数行でも従来どおり適用する。1 行だけに限定する変更は、この追試での明示指示に基づく。256 位置の 1 行 logits 一致だけで全入力・全行数の bit 一致が証明されたわけではないため、以下の実モデル試験を実施した。

変更前後の DFlash2 64 token、graph debug、N=10 verify graph の capture を一時的な `hipGraphGetNodes` フックで調べた。kernel node は 1,067 → 1,317（+250）、全 node は 1,073 → 1,323。これは capture された graph の node 数であり、graph replay の host launch 数ではない。`shelf-compute` は verify 窓の perf stats を出力しないため、perf stats による verify 値ではない。通常 decode の perf stats は変更後も 1,001 launch/token。

変更後は baseline build と交互に 5 ペア測定した。条件は非投機・DFlash2 とも code prompt、256 token、最大長 1024、temperature 0、DFlash2 は drafts 7。平均 ± 標本 SD は次の通り。途中の正しさ試験は GPU lock で排他されており、終了時の最大 load average は 2.94。

| 指標 | baseline | 変更後 | 変化 |
|---|---:|---:|---:|
| 非投機 tok/s | 30.918 ± 0.189 | 31.604 ± 0.038 | +2.22% |
| DFlash2 tok/s | 108.842 ± 0.096 | 108.950 ± 0.121 | +0.10% |

非投機の平均差は両群の標本 SD を超える。DFlash2 の平均差 0.108 tok/s はペア差の標本 SD 0.145 tok/s より小さく、baseline と同程度と判断する。全 20 run の生成 ID は共有 baseline と一致した。非投機 +2% 以上、DFlash2 が baseline の −1% 以内という今回の目標を満たした。

BetterBench 256 位置の mean KL 0、top-1 一致率 100%、最大 logit 絶対差 0。固定 code / prose / ja の各 256 token は baseline と一致し、通常 DFlash2 試験は BF16・FP8・Octave 合計 99/99、関連 ctest は 12/12 成功した。

修正後の tree 試験は baseline・変更後とも 3 prompt でテスト成功。target、chain、tree0/1/2/4、budget8/10/12/16/24 の対応する各 256 token 列は全 prompt で baseline と一致した。既存の chain 対 target の差は 12／132／213 token、最初の位置は 244／118／43 で変わらない。最初の margin は 0.010727／0.004416／0.029339。各 mode の不一致位置を JSON の `dflash_followup.tree` に記録した。差はテストの許容条件内だが、投機と非投機の完全一致ではなく、既存の問題として残る。

行数制限の実装は `68edead1`。DFlash2 の低下について、I8／I9 の複数行無効化で回復したことは観測したが、冗長な norm 計算など kernel 内の支配的な原因は計測していない。単発の項目別結果はばらつきの範囲を判定できず、対象外モデルや未測定の入力全般への数値同一性も証明していない。`run_server_regression.py --all` は実行していない。

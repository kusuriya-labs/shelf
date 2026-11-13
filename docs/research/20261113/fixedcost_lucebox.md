# Qwen3.8-27B の小 kernel 融合

## 条件

2026-11-13、R9700 gfx1201、ROCm 7.14、基準 `ca29aa1`。Qwen3.8-27B EXL3 mul24、BF16 KV、単一 request。全 GPU process を `/tmp/r9700.lock` と `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` で排他した。基準は指定済みの `integ-final5/build-gfx1201`、候補は本 worktree の Release build。非投機を主指標とし、DFlash2 は draft 7、BF16 と mul24 の drafter を別に測る。

## 現行 main の dispatch

`shelf-bench tg --mode forward --context 128 --tokens 4 --warmup 1` で `SHELF_QWEN35_PERF_TIMING=1` と `SHELF_QWEN35_PERF_STATS=1` を指定。HIP event 区間の合計を 4 で割る。event の挿入によって通常実行は変わり、下の値は独立した kernel 実行時間でも通常時の end-to-end 時間でもない。測定結果の集計とカウンターは [fixedcost_lucebox_dispatch.json](fixedcost_lucebox_dispatch.json)。

| dispatch | 論理回数/token | event ms/token |
|---|---:|---:|
| EXL3 linear | 401 | 28.761 |
| BF16 linear | 96 | 1.344 |
| RMSNorm | 209 | 2.327 |
| GDN convolution | 48 | 0.486 |
| SiLU | 96 | 0.890 |
| L2 normalize | 96 | 0.906 |
| scale | 48 | 0.438 |
| GDN recurrence | 48 | 1.130 |
| MUL | 64 | 0.587 |
| SwiGLU | 64 | 0.596 |
| residual add | 128 | 1.180 |
| split | 16 | 0.147 |
| RoPE | 32 | 0.326 |
| KV append | 16 | 0.155 |
| paged attention | 16 | 0.287 |
| sigmoid | 16 | 0.150 |

全体は論理 1,397、自己申告の物理 launch 1,767/token、fallback 0。以前の `docs/perf/exl3-decode-bandwidth.md` の 2,199 と異なり、現行 main では隣接する EXL3 projection の既存融合がある。EXL3 401 は論理数であり、event 区間は複数 projection をまとめたものを含む。区間総和は 39.749 ms/token。

## 案と期待値

- Q/K の L2 normalize 2 回と Q scale 1 回を統合する。48 層で最大 96 launch 削減。基準 event 区間は合計 1.344 ms/token で、これは実際の改善の上限や正確な寄与ではない。
- GDN 出力の RMSNorm、SiLU、MUL を統合する。最大 96 launch 削減。RMSNorm 全体 209 回には他の正規化も含まれ、0.890 ms の SiLU と 0.587 ms の MUL も他の演算を含む。計測上、対象 45 層で 90 launch 削減。
- conv + SiLU + Q/K normalize はさらに 48 launch 以上を削減できる可能性がある。state の更新、BF16 丸めと head ごとの同期を一つの kernel へ収める必要があるため、今回は未実装。
- residual add + RMSNorm は最大 128 launch 削減の候補。ただし residual 自体は次層でも必要であり、保存と workspace の alias を含めた対応が必要。未実装。
- attention の Q/K norm、RoPE、KV append の統合は 16 層が対象。paged KV の位置と書き込み先、Q/K の依存を確認する必要があり未実装。
- GDN の a/b BF16 linear は 96 回、1.344 ms/token。2 行列を同一 launch に収める候補はあるが、異なる重み pointer と出力を扱う新 GEMV が必要で未実装。
- MLP SwiGLU は棚ですでに単一 dispatch。gate/up の EXL3 projection も現行 main に共有入力の融合がある。重複して実装しない。
- DFlash2 draft は独自経路で、今回の単一行 target 融合は複数行 verify や drafter 自体には適用されない。draft の小 kernel と verify 前後の独立 launch 数は未計測であり、効果を推測しない。

## 実装と診断

- `af29469`: GDN recurrence 直後の RMSNorm + SiLU + MUL を 1 launch。F32 の正規化後は既存経路同様に BF16 に丸めてから重みを掛ける。1 行・対象形状・後続の EXL3 mul24・値の使用先とメモリ非重複を確認する。値の trace または layer override があるときは従来経路を使う。
- `0149494`: Q と K の L2 normalize と Q scale を 1 launch に統合。同じ 1 行の対象形状と後続 recurrence、EXL3 を確認する。両融合は `SHELF_DISABLE_LUCEBOX_FUSION=1` で無効化できる。

同じ event 診断で、Q/K 融合は 48 層、出力融合は 45 層で選択。論理数は 1,397 のまま、物理 launch は 1,581/token に減少した。event 区間総和は 37.805 ms/token。Q/K 融合区間 0.430 ms/token と基準の L2 + scale 1.344 ms/token の差は約 0.914 ms/token。出力融合の基準 48 層の連続 3 区間は 1.346 ms/token、45 層相当に比例換算すると約 1.262 ms/token、候補の融合区間は 45 回で 0.460 ms/token であり差は約 0.802 ms/token。異なる run の計測を引いた概算であり、各案の独立 A/B や通常実行の純粋な寄与ではない。

## 参照と採用しなかった案

Luce-Org/lucebox `cbc6d7c27108e984599e22670564aae16e0e7eb4`、Apache-2.0。`server/docs/HIP_PERF_PLAN.md`、`server/docs/ENGINE_COMPONENTS.md`、`server/docs/ARCHITECTURE.md`、`server/src/qwen35/qwen35_target_graph.cpp`、`glue_fusions.h`、`server/src/common/dflash_draft_graph.cpp`、`dflash_spec_decode.cpp` を read-only で参照し、コードを転記せず shelf 向けに実装した。lucebox は dense chain 等の条件で conv + SiLU と state 更新を統合し、隣接 Q/K の L2 normalize をまとめ、gated norm の SiLU + MUL を統合する。gate/up stack は型や backend の条件付き。draft は固定長 padding や sliding-window 条件に応じた graph 再利用を持つ。

同資料の launch 起因 10〜15% は別構成における約 1 万 kernel × 3〜5 µs の推定であり、本測定へ流用しない。ggml HIP graph は該当試験で -0.73%、tile と併用して +1.37%；動的な `kv_start` を固定 view に埋め込む設計が per-forward graph 再利用を妨げていた。R9700 の MMQ tile 変更 +8.3% は別形式の verify 用であり本担当の linear 範囲外。MMVQ を batch 16/23 に拡張した案は RDNA3+ で大きく悪化しており再実施しない。memoized q8 activation、KV rotation の省略、draft graph 最適化も形式と状態管理が異なるので、そのままの移植は行っていない。

## 品質とテスト

BetterBench の 29 cases、2,203 tokens から corpus を生成し、context 128、256 位置、BF16 KV、batch 1 で基準 dump と候補 dump を `shelf-bench quality --compare` で 1 回比較した。平均 KL は **0.001283664**、top-1 一致率 **98.828125%**、top-5 overlap 98.671875%、perplexity 22.6990 → 22.3631。目安の KL < 0.005 と top-1 ≥ 98% を満たす。logit 最大絶対差は 1.15759。corpus は `tools/quality_gate.py corpus` を使用し、SHA-256 は `99fe6aa90b4f52a9427361391f0dc3e17377391d67793c1c716e890d8c28799d`。詳細は [fixedcost_lucebox_quality.json](fixedcost_lucebox_quality.json)。

`shelf-compute` と `shelf-bench` の Release build に成功。関連 CPU selector test `test_qwen35_rmsnorm_selector`、`test_qwen35_l2_normalize_selector` は 2/2 通過。最初の ctest は `LD_LIBRARY_PATH` 未指定でライブラリ読み込みに失敗し、指定して再実行した。これは新しい融合自体を単体テストしたものではなく、上記の実モデル品質比較が融合を含む確認である。

`optimizations/megakernel/README.md` も設計上の参考とした。0.8B CUDA の全層 persistent kernel は、launch 削減と引き換えに resident block の上限、grid 同期、register 圧力を制約に持つ。27B の HIP 移植対象とはしていない。

## 速度測定の途中結果

[fixedcost_lucebox_bench.py](fixedcost_lucebox_bench.py) で code prompt、128 token、temperature 0 の process を交互に実行する。[fixedcost_lucebox_e2e.json](fixedcost_lucebox_e2e.json) は途中結果であり、非投機は基準 29.37 / 29.40 tok/s 2 回、候補 30.15 / 29.98 tok/s 2 回。基準と候補それぞれ 3 回以上という条件を満たしておらず、この値から平均・標本 SD や採否を確定できない。共有 GPU の待ち行列が長く、1 回はロック取得前に 300 秒 timeout となり、待機のみを延ばして再開した。

生成確認は code と prose の固定 prompt が各 256 token を完走し、確認範囲で文字化け・明白な反復・早期 EOS はなかった。日本語 prompt は未完了。[fixedcost_lucebox_greedy.json](fixedcost_lucebox_greedy.json) に取得分を保存した。

排他待ちが指示された時間の目安を大幅に超えたため、待機中の自分の実行を停止した。非投機の各 3 回と BF16 / mul24 drafter の速度比較、日本語 prompt の 256 token 生成、draft step と verify 前後の launch 内訳は未完了。DFlash2 の速度・品質や全部入りの採否は判断できない。上記 script は保存済み JSON を初期値として成功済み run を飛ばし、残りを `/tmp/fc_lucebox/bench.json` に記録する。統合前に残項目を実施する。

## 複数行対応と一致性（2026-11-13）

DFlash2 の verify では 8 行前後の target 実行が従来経路となる一方、1 行の decode では融合経路となり、数値差によって greedy の完全一致が崩れていた。両 kernel の grid に行を加え、各 block がそれぞれの行と 128 要素の group を担当する。入力・出力の row stride を個別に使い、1 行当たりの reduction、BF16 丸め、演算順序は変更していない。通常 kernel の selector にある行数上限を融合対象の判定へ持ち込まないよう、形状の判定は 1 行で行い、実際の行数で融合 kernel を launch する。重なりは実際の行数ではなく workspace の確保済み行容量全体で判定する。対象形状、後続演算、trace / override 条件、無効化環境変数は維持した。

変更前の 1 行 kernel と、変更後に 1、8、10、16、257 行をまとめて実行した結果を、各行の全要素についてビット単位で比較した。各入力行は異なる値とし、入力・出力 stride に padding を含めた。257 行を 7 行ずつの chunk に分割した場合も一致した。これらは一時的な HIP 検証であり、実モデルの end-to-end 一致とは別の確認である。

実モデルの prefill dispatch 診断では、1、8、10、16、128 行の各実行で Q/K 融合が 48 層、出力融合が 45 層だった。DFlash2 draft 7 の実 verify 4 回でも各 forward の L2 normalize dispatch は融合有効時 48、無効時 96、SiLU は 51 対 96、MUL は 19 対 64 だった。Q scale の 48 回も Q/K 融合に含まれるため、verify 1 回で Q/K が 96、出力側が 90、計 186 launch 減る。診断には `SHELF_OP_DUMP=1` を用いたため、速度測定には用いていない。非投機 decode の既存診断では 1,767 → 1,581 launch/token だった。

`shelf-bench quality` は同じ BetterBench corpus、context 128、256 位置、BF16 KV の基準 dump と候補 dump を比較した。平均 KL は **0.00189127688**、top-1 一致率は **99.21875%**、top-5 overlap は 98.125%、perplexity は 22.6990 → 23.3707。目安の KL < 0.005、top-1 ≥ 98% を満たした。corpus の SHA-256 は前節と同一である。

指定された統合テストをそれぞれ単独で GPU lock 下に実行した。

| テスト | 結果 |
|---|---|
| `test_compute_dflash2.py` | 99/99 成功。BF16 / FP8 / Octave KV で投機 serve と非投機 greedy が完全一致 |
| `test_compute_dflash2_unaligned.py` | 7/7 成功。非整列 prefill と prefix 復元を含め非投機 greedy と完全一致 |
| `test_compute_dflash2_prefix_long.py` | 8/8 成功。2048 以降の chunk 境界からの prefix 復元が cache 無効時および非投機 greedy と完全一致 |
| `test_server_dflash2_tree.py` | 成功。budget 8 の chain 一致と、budget 10 / auto の単一 request tree との一致を確認 |

全体 regression と `test_compute_dflash2_tree.py` は実行していない。

同じ code prompt、128 token、temperature 0、最大 sequence 長 1024、単一 request の `shelf-compute` を使い、基準と候補を ABAB 順に各 3 回比較した。DFlash2 は draft 7。BF16 drafter の基準 1 回が低かったため、その組を除外せず追加で 1 組測定した。各 run は別 process とし、GPU lock の取得ごとに 270 秒の timeout を設けた。生成した 128 token の列は全 20 run で一致した。

| 条件 | 基準の各 run (tok/s) | 候補の各 run (tok/s) | 平均の基準 → 候補 | 中央値の基準 → 候補 |
|---|---|---|---|---|
| 非投機 | 29.33, 28.15, 29.48 | 29.99, 30.25, 30.04 | 28.99 → 30.09（+3.82%） | 29.33 → 30.04 |
| DFlash2 BF16 | 90.74, 80.37, 90.89, 90.94 | 91.93, 92.02, 92.06, 91.98 | 88.24 → 92.00（+4.26%） | 90.82 → 92.00 |
| DFlash2 mul24 | 90.83, 90.53, 89.13 | 92.06, 91.74, 92.00 | 90.16 → 91.93（+1.96%） | 90.53 → 92.00 |

BF16 baseline の 80.37 tok/s の原因は特定できていない。全 run を含む平均はこの 1 回の影響を受けるため、中央値も示した。共有 GPU の別作業とは process ごとに排他し、baseline build と toolchain は変更していない。

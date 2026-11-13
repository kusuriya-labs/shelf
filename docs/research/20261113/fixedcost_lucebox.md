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

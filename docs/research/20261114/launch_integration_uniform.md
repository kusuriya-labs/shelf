# decode と verify で共通の launch 融合

## 条件

2026-11-14、R9700 gfx1201、ROCm 7.14、Qwen3.8-27B EXL3 mul24、BF16 KV。基準は `origin/main` 44da8fe、変更前は `c5d3f209`。`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` を指定し、各 GPU process を `flock /tmp/r9700.lock timeout -k 5 280` で実行した。`run_server_regression.py --all` は実行していない。実装 commit は `fac720f5`。

## verify の増加と融合の適用

`c5d3f209` の I8 / I9 は 1 行に限定され、複数行ではそれぞれ 125 回ずつ適用されなかった。これが前回の N=10 verify graph の kernel node 1,067 → 1,317 の増加に対応する。さらに、変更前の attention Q / KV 融合は tree metadata があると除外され、tree で 16 層 × 4 = 64 launch を残していた。既存の RoPE resolver の位置指定と KV append の row metadata を使う経路で tree 除外を外した。

変更前を同一 binary の `SHELF_DISABLE_BOUNDARY_I8=1` と `SHELF_DISABLE_FINAL_RQ=1`、tree では加えて attention Q / KV 無効で再現し、指定の `trace_names.cpp` で計測した。独立した診断 binary では `SHELF_QWEN35_PERF_STATS=1` と一時フックにより、`launch_program_run` 前後の `physical_kernel_launches` の差を計測した。結果は一致した。trace は status memset から status D2H までの窓、perf stats は forward program 内を数える。

| 区間 | 変更前 | 変更後 | 減少 |
|---|---:|---:|---:|
| 非投機 decode、1 行 | 1,001 | 999 | 2 |
| DFlash2 chain verify、8 行 | 1,252 | 1,000 | 252 |
| DFlash2 tree verify、10 行 | 1,316 | 1,000 | 316 |

実行終盤の chain 3 行、tree 5 行も変更後は 1,000。追加診断の 5 drafts + tree extra 2 による tree 8 行も 1,000。使用 drafter は block size 8（最大 7 drafts）のため、chain 10 行に必要な 9 drafts はこのモデルの対応範囲外であり未測定。N=8 / N=10 の graph capture は各 1,007 node、そのうち kernel 1,001。forward の前に batch descriptor 準備 kernel が 1 件あるので、forward の 1,000 と矛盾しない。graph replay 自体の host launch 数と混同しない。

変更前の chain 8 行で rq を全無効にすると kernel node は 1,253 → 1,394（+141）、attention 無効は 1,253 → 1,317（+64）。変更前の tree 10 行は rq 全無効で 1,317 → 1,458（+141）、attention 無効では 1,317 のままだった。変更後の tree graph には attention Q / KV の各 kernel が 16 回、SwiGLU rotate 64 回、GDN rotate 45 回、attention gate rotate 16 回存在した。I2 / I7a / I7b の削減はそれぞれ 64 / 45 / 32 件。

## 複数行 kernel の探索

代表的な合成入力、2 projection、split-K 4、100 回の HIP event 平均（µs）。未融合は residual / RMSNorm / group rotate、I9 ではさらに出力回転を含む。これは単発の kernel 比較であり、実モデル全体の時間とは異なる。

| 行数 | 旧 I8 | 未融合 I8 相当 | 新 I8 | 旧 I9 | 未融合 I9 相当 | 新 I9 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 9.567 | 14.428 | 9.520 | 12.113 | 17.714 | 12.775 |
| 8 | 23.010 | 14.380 | 8.110 | 35.578 | 17.857 | 14.564 |
| 10 | 27.435 | 14.568 | 9.259 | 42.907 | 18.845 | 17.848 |
| 16 | 35.904 | 14.767 | 11.666 | 57.359 | 19.255 | 23.197 |

旧 kernel はすでに grid で行を分けていたが、各 CTA が 5,120 要素の residual / norm、I9 ではさらに出力回転を再計算し、そのうえ正規化後の 5,120 要素すべてを生成していた。新 kernel は加算・出力回転と二乗和を一度の走査で行い、各 CTA に必要な 128 要素だけ正規化する。出力が必要な CTA は residual と norm の全要素を保存する。2 projection の H128 を CTA 内の別の 128 thread 群に割り当てて前処理を共有する。累算と shuffle の順序、BF16 丸め位置は従来に合わせた。

1 行・2 projection では CTA は 80 → 40、3 projection では 120 → 80 に減る。一方、独立した 128-thread の H128 計算群は各 projection ごとに残り、1 行を単一 CTA に集約しない。CTA 数自体は減っており、CTA 数の厳密な不変を求める解釈では並列度の条件を満たさない。上表の 1 行は I9 が単発で 0.66 µs 遅く、実モデル全体では後記の速度条件を満たした。合成入力の 1 / 8 / 10 / 16 行、2 / 3 projection で residual・norm・codes・scales は従来経路と bit 一致した。別に cooperative grid synchronization で行全体を 1 回だけ計算する案も試したが、約 104〜115 µs と遅く採用しなかった。

末尾は OUTPUT_GATHER・final RMSNorm・LM head の rotate_quantize を 1 kernel にまとめ、2 launch を削減した。gather は外部 hidden 出力なので省略せず、kernel 内で値を書き出す。入力と出力の非重複と中間値の後続使用を確認する。独立した代表入力検証では、行数 1 / 8 / 10 / 16 と非連続の出力行選択で gather 値、codes、scales が bit 一致した。

## 速度

baseline と変更後を ABAB 交互に各 5 回実行。非投機と DFlash2 は code prompt、256 token、最大長 1024、temperature 0、DFlash2 は BF16 drafter、drafts 7。実行時に load average が 8 未満であることを確認し、終了時の最大は 3.24。ほかの正しさ試験も GPU lock で排他して同じ期間に実行した。平均 ± 標本 SD。

| 指標 | baseline | 変更後 | 変化 |
|---|---:|---:|---:|
| 非投機 tok/s | 30.976 ± 0.186 | 31.724 ± 0.096 | +2.41% |
| DFlash2 tok/s | 109.012 ± 0.303 | 108.738 ± 0.214 | −0.25% |

DFlash2 のペア差の標本 SD は 0.277 tok/s で平均差 0.274 tok/s より大きく、baseline と同程度と扱う。各 run の生成 ID は全条件で共有 baseline と一致した。非投機 +2% と DFlash2 −1% 以内の目標を満たした。

## 適用条件と制限

対象の I8 / I9、rq、attention、末尾融合は、1〜16 行で同じ条件・同じ kernel を用いる。prefill、TP、他モデル、value trace、layer override、非対応 dtype、shape、aliasing や workspace、出力の利用条件では既存経路へ戻る。末尾融合の重み形状は vocab 248,320 × hidden 5,120、EXL3 6-bit trellis に限定し、`SHELF_DISABLE_FINAL_RQ=1` で無効化できる。tree attention は既存の行位置 metadata を使い、chain と同じ kernel を使う。

既存の PLaMo 固有 norm / SwiGLU producer の 1 行条件は変更していない。対応 launcher 自体が `args.rows != 1` を拒否し、kernel に行方向 index がないため、条件だけを外せない。他モデルの経路を変えないため、この作業での複数行化は対象外とした。これは将来の改修が不可能という証明ではない。prefill に相当する 16 行超の別経路や split-K の既存選択も維持した。

## 正しさ

BetterBench 256 位置で mean KL 0、top-1 100%、logit 最大絶対差 0。固定 code / prose / ja の各 256 token greedy は baseline と一致。通常 DFlash2 は BF16 / FP8 / Octave で 99/99、関連 ctest は 12/12 成功した。

`test_compute_dflash2_tree.py` は baseline・変更後とも 3 prompt でテスト成功。target、chain、tree0/1/2/4、budget8/10/12/16/24 の各 256 token 列は、対応する baseline と全 mode で一致した。chain 対 target の既存差は 12 / 132 / 213 token、先頭位置はそれぞれ 244 / 118 / 43、最初の target margin は 0.010727 / 0.004416 / 0.029339 のまま。テストの PASS は投機と非投機の完全一致を意味しない。

## 未解決

1 行の有効な H128 処理群数は保ったが、前述のとおり CTA 数は減った。CTA 数自体を減らさないという厳密な条件の下で同時に速度目標を満たす実装は、今回確認できていない。独立検証の入力は代表例であり、全入力に対する bit 一致の数学的証明ではない。tree の既存の投機・非投機差も未解決。TP、混在 request、対象外モデルでの実モデル再測定は行っていない。

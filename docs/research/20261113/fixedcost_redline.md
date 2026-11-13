# c=1 decode の HIP graph と Redline

2026-11-13、R9700 gfx1201、ROCm 7.14、BF16 KV。基準は `origin/main` の `ca29aa1`。GPU 実行は UUID `GPU-a8e9ddefa2d60f55` を指定し、各 run を `/tmp/r9700.lock` で排他した。`rocprofv3` は使用していない。数値の生データは [JSON](fixedcost_redline.json) にまとめた。

## 結論と採否

PLaMo も `qwen35` 共通 executor を使用しており、既存の `SHELF_HIP_GRAPH=ON` build で decode を capture / replay できた。新たな graph 実装を重複させず、PLaMo は **単一 request・単一 row・投機なし decode** に限定し、`SHELF_PLAMO_HIP_GRAPH=0` で従来経路を選べるようにした（`d7519a2`）。既定の CMake option は OFF のままなので、利用時は明示的に `-DSHELF_HIP_GRAPH=ON` を指定する。prefill と複数 request は既存の通常経路を使う。Qwen は既存 graph の実装を変更せずに評価した。

既存 executor は token と request descriptor、位置などを固定 staging および device batch context で更新する。attention の実行形状が変わる場合は signature が変わり、graph を再 capture する。capture の初回費用と、短い request の場合の償却は workload に依存する。

PLaMo の通常 HIP graph は速度・品質・連続実行の結果から、上記 build option を有効にする構成での採用を推奨する。Qwen の既存 graph も今回の非投機 c=1 条件では改善した。Redline の retained PM4 は今回実行できず、採用しない。`LD_PRELOAD` は opt-in だが、今回の graph は通常 HIP にフォールバックする。

## 速度

PLaMo は `plamo_closed_loop.py`、c=1、seed 17、最初の完了後 20 秒 warmup、90 秒窓、最大出力 320、基準と graph を ABABAB 交互に各 3 回。Qwen は同一 code prompt、c=1、非投機、greedy、最大 256 token の `shelf-compute` を同様に交互測定。平均 ± run 間標本 SD。

| 対象 | 基準 tok/s | graph tok/s | 変化 |
| --- | ---: | ---: | ---: |
| PLaMo 継続投入 | 153.71 ± 2.16 | 175.49 ± 0.00 | +14.17% |
| Qwen 生成 decode | 29.41 ± 0.06 | 30.68 ± 0.06 | +4.31% |

PLaMo の graph 側は 3 回とも 202 request、15,794 completion token が窓内に完了しており、同一 workload の離散的な集計結果として SD が 0 になった。個々の step のばらつきがゼロという意味ではない。benchmark の定義上 PLaMo の値には継続投入の prefill と request の切り替えも含まれる。Qwen と同一の計測範囲ではない。

## Dispatch と時間の診断

`bench tg --mode forward --context 128` の decode-only、単一 row、非 sampling。8 token のカウンタから 1 token 当たりに換算した。host は [一時 interposer](fixedcost_redline_trace.cpp) を使い、最初の token H2D enqueue から status D2H enqueue までを計測した。dispatch の解決、input upload、batch descriptor と launch 呼び出しを含み、enqueue より前の scheduler 処理を除く。16 step のうち最初の対象 1 step を除いた 15 回の中央値。HIP event は診断 run の step 中央値。trace 自体の overhead を含む。

| 対象 | 論理 dispatch / step | 物理 launch / step | host 投入 ms | GPU event step ms |
| --- | ---: | ---: | ---: | ---: |
| PLaMo | 486 | 547 | 5.808 | 6.364 |
| Qwen | 1,397 | 1,767 | 23.744 | 33.393 |

host と GPU は並行するので、この 2 区間を加算したり差を GPU の idle 時間としたりはできない。PLaMo では投入時間が step 区間に近く、実際の graph 化による大きな改善もあるため launch 処理が有力な制約と判断する。Qwen は GPU 区間が長く、graph の実測改善は小さい。host が常に GPU より先行している、または GPU が常に遊休しているという厳密な証明ではない。Qwen の既存資料にある 2,199 物理 launch は sampling を含む別の生成条件であり、この forward 診断の 1,767 と区別する。

同じ GPU で [小 kernel の連続投入](fixedcost_redline_floor.hip) も測定した。1 work-item が同じ device word を書く単純 kernel、4 回中先頭を除く 3 回の GPU event 区間中央値は、547 launch で 1.721 ms（3.15 µs/launch）、1,767 launch で 5.116 ms（2.90 µs/launch）。host と GPU の重なり、書き込みとイベント、clock の影響を含む参考値である。実モデルの kernel 時間の和や厳密な kernel 間隙間は測っておらず、この値を step 時間から差し引くことはできない。

## 品質と安定性

PLaMo は `plamo_quality.py` と既存 CPU BF16 reference を使い、各方向の held-out 18 文、c=1、teacher forcing。結果は以下。

| 方向 | 位置数 | mean KL | top-1 一致 |
| --- | ---: | ---: | ---: |
| JA→EN | 875 | 0.018016 | 92.914% |
| EN→JA | 875 | 0.015160 | 93.714% |

基準参考値はそれぞれ KL 0.0180 / 0.0152、top-1 92.9% / 93.7%。両方向で KL < 0.02、top-1 の低下 < 1 pt を判定基準とする。

Qwen は `tools/quality_gate.py corpus` の corpus SHA-256 `99fe6aa90b4f52a9427361391f0dc3e17377391d67793c1c716e890d8c28799d`、context 128、256 位置で main dump と graph を比較し、mean KL 0、top-1 一致率 100% だった。code・日本語・prose の固定 3 prompt × 256 token の greedy 生成は全 token ID が一致した。生成テキストを確認し、文字化けや明らかな反復は見られなかった。3 件とも 256 token の上限まで生成した。

通常 graph の `bench tg --mode greedy --context 0 --tokens 2000` は PLaMo と Qwen の両方が位置 2,000 まで正常終了した。試験期間の kernel journal に amdgpu / kfd の新規 fault はなかった。これは固定入力を使う連続実行・resource lifecycle の試験であり、文章品質の評価ではない。PLaMo の別試験では可視長 2,049 で既存の attention correctness 経路へ切り替わって極端に低速になったため、共有 GPU の占有を避け中断した。範囲外の長時間性能は未評価。`SHELF_PLAMO_HIP_GRAPH=0` と `SHELF_GRAPH_DEBUG=1` の組み合わせでは graph 出力なしで通常経路が完了した。2 request・2 row の prefill と decode でも graph の capture がないことを debug 出力で確認した。CPU test `test_plamo3_lowering` は 1/1 通過した。

## Redline の検証

参照は `warpfront__redline` commit `274c196359dcc4a5f76ea41e5226cf720f5f93e4`、Apache-2.0。`README.md`、`docs/INTEGRATION.md`、`docs/DISPATCH-FLOOR.md` と `crates/redline-hipgraph` の設計を確認した。`warpfront__hipfire` commit `89e32856d25d6f6e6645319f2ec2c641c46be525` の `docs/REDLINE.md` も admission・可変状態の条件を確認した。コードの転記はしていない。参照ディレクトリは read-only とし、Redline は `/tmp/fc_redline/redline` にコピーして `cargo build --release -p redline-hipgraph -j 8` で build した。

通常 graph の preload は PLaMo の HIP 554 node に対して Redline 548 node、Qwen は 1,774 対 1,768 となり、topology 不一致で HIP fallback した。実験として PLaMo の入出力コピーと error word 初期化を capture の外に移すと 548 対 548 で一致したが、PM4 の plan 構築が `Gfx12Build(UnsupportedKernelProperties(2))` で失敗し、やはり HIP fallback した。隔離コピーへの診断出力追加で特定した値 `0x0002` は HSA の `ENABLE_SGPR_DISPATCH_PTR` であり、この Redline の GFX12 lowering では未対応。サポート判定を迂回していない。コピー分離の試作は通常 build に残していない。

各 preload 実行は排他中に 100 秒 timeout、外側も 115 秒 timeout とした。`dmesg` は権限不足で読めなかったため、各回の前後に `journalctl -k` の amdgpu / kfd / HSA 記録を比較した。4 回とも差分はなく、標準エラーに HSA fault や hang はなかった。PM4 replay は一度も観測できておらず、PM4 の速度・品質・長時間安定性は未測定である。今後は dispatch pointer の正しい binding、stream 依存と lifetime を実装・検証した上で node の完全性、複数位置での出力・状態、fallback の有無、長時間の fault を再検証する必要がある。

## 再現の要点

自分の worktree で `SHELF_HIP_GRAPH=ON` を付け、共通ルールの ROCm / gfx1201 / libc++ の CMake 引数で構成し、`nice -n 19 ninja -C build-gfx1201 -j16 shelf-compute shelf-bench shelf-server` で build する。CPU 品質評価は `docs/research/20261112/plamo_quality.py` を各方向の `--prefix heldout-ja-en-` / `--prefix heldout-en-ja-` で実行する。GPU を保持した一括処理は方向ごとに分け、それぞれ 270 秒で打ち切った。

```bash
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
export LD_LIBRARY_PATH=/home/homelab1/coding-local/sLLM/.local-artifacts/phaseshift-draft-vocab/toolchain/root/usr/lib/llvm-23/lib:/opt/rocm/core-7.14/lib
export MODEL=/home/homelab1/datapool/rocm-exl3-rdna2/models/plamo-3-translate-2b-exl3-4.05bpw-mul24
nice -n 19 /opt/rocm/core-7.14/bin/hipcc --offload-arch=gfx1201 -O2 docs/research/20261113/fixedcost_redline_floor.hip -o /tmp/fc_redline_floor
flock /tmp/r9700.lock timeout 60s /tmp/fc_redline_floor 547
nice -n 19 g++ -std=c++17 -O2 -fPIC -shared -D__HIP_PLATFORM_AMD__ -I/opt/rocm/core-7.14/include docs/research/20261113/fixedcost_redline_trace.cpp -ldl -o /tmp/fc_redline_trace.so
flock /tmp/r9700.lock timeout 270s env LD_PRELOAD=/tmp/fc_redline_trace.so SHELF_PLAMO_HIP_GRAPH=0 build-gfx1201/shelf-bench tg --model-dir "$MODEL" --mode forward --context 128 --tokens 16 --warmup 1
flock /tmp/r9700.lock timeout 270s build-gfx1201/shelf-bench tg --model-dir "$MODEL" --mode greedy --context 0 --tokens 2000 --warmup 1
```

速度 run は `docs/research/20261112/plamo_closed_loop.py --concurrency 1 --warmup 20 --duration 90` で baseline / graph の server を交互に指定する。Qwen は `docs/perf/exl3-decode-bandwidth.md` の非投機 `shelf-compute` 手順を用いた。preload は試作 graph でも PM4 が拒否されたため、通常 HIP graph と PM4 の速度比較値はない。

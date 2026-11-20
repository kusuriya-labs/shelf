# EXL3-INT8 prefill A 経路の評価

2026-11-20、R9700 `GPU-a8e9ddefa2d60f55`、ROCm 10.2、libstdc++ 先行、BF16 KV。各 GPU 実行は `/tmp/r9700.lock` で排他し、`ROCR_VISIBLE_DEVICES` と `HIP_VISIBLE_DEVICES=0` を指定した。`rocprofv3` は使っていない。測定器・BF16 基準・有効語彙・cases-v1 の説明は [README](README.md) を参照。B は環境変数未設定、A は `SHELF_EXL3_MUL24_PREFILL_AB=A`、融合はさらに `SHELF_EXL3_MUL24_FUSED_DECODE=1` を指定した。測定器の実行ファイルは opt-in の融合 decode に対応するが、`ebaeef74` と `56a0911c` を適用した既定化候補そのものではない。

位置ごとの KL は BF16 基準から候補への KL、top-1 は BF16 基準と一致したかを意味する。同一の case・入力位置・基準 logits を対応づけて A−B を算出した。95% 区間は case を復元抽出する bootstrap 10 万回、seed 20261120、抽出した case 内の位置すべてを位置数で重み付けした平均の percentile 区間。8 case しかないため区間の推定も不安定であり、位置が独立という前提は置かない。短い 3 case は prefill が 256 行未満で、A の dense prefill は動かない。

集計は `tools/quality/paired_kld.py BのJSON AのJSON`（位置別 JSON はリポジトリに含めていない。`kld-vs-bf16.sh` で再生成する）。JSON は各位置の KL と top-1 一致、入力・基準・実行ファイル・dump の SHA-256 を含む。実行ファイルやモデル、基準、split が異なる結果の単純な比較に注意する。

## 既存の長文・環境依存の結果

先行記録は `investigate/rocm10-abi` の `docs/perf/rocm10-bench.md`、revision `8dea2173`。4bpw、反復 corpus の入力 8192 token の後 64 位置、BF16 KV、prefill chunk 4096、7.14 B + rocRoller を暫定基準とした。10.2 A + Tensile の KL は 0.394211、top-1 一致 93.75%、10.2 B + rocRoller は dump が 7.14 B と byte 一致した。これは BF16 との KL ではない。64 位置と反復 corpus は自然な長文全般を代表しない。同条件の 10.2 A 再実行と 7.14 A は 10.2 A と byte 一致した。一方、2048 token の c=16 では 7.14 A と 10.2 A の生成列は 15/16 一致し、入力依存の版差があった。B はこの比較範囲で 7.14 と 10.2 が一致したが、全環境の bit 一致を保証しない。

先行する 8k・c=16 の greedy 128 ID は A と 7.14 B が 1/16 要求で一致、各条件の同じ要求番号間では繰り返し一致した。先行の DFlash2 と非投機の一致確認には 40 token の短い入力が使われており、A が実際に稼働する長文での一致の証明ではない。

## 失敗モードの静的確認

`hipErrorNotSupported` は `optimized_dispatch.hip` で受け、group scale を再生成した融合 kernel へ戻す。先行の rocRoller 指定では INT32 accumulation 非対応のログと一致する fallback が観測されている。ただし全ての failure の安全性を証明するものではない。ABI 不一致や handle 初期化失敗は `hipErrorInitializationError` であり、この分岐の fallback 対象ではない。また `shelf-bench` は `libhipblaslt.so.1` を直接必要とするため、共有ライブラリ欠落時にプロセスは起動できない。候補を既定にするなら欠落を含む実装上の取り扱いの設計と実行時テストが必要。

`56a0911c` の 256 MiB preference 案は algorithm cache が algorithm 本体しか保存せず、cache hit 時 `workspace_size` は 0 のまま再呼び出しする。非ゼロ workspace の algorithm が選ばれた場合は不正または失敗のおそれがあり、実際の選択実績の有無とは別に修正と試験が必要。候補の A 分岐自体に明示的な gfx1201 限定条件は見当たらず、他 GPU を使用せず静的に確認した範囲では要求を保証できない。

## algorithm と VRAM の追加実測

4bpw、BetterBench 2048 token 入力と後続 64 位置で、10.2 A の `SHELF_EXL3_INT8_TUNE=1` と探索なしは dump SHA-256 `2dcd20ae632e4fb5056f2c44291f3dea50a09187d48c080c0ffb888a238cb06e` で byte 一致した。A + `HIPBLASLT_USE_ROCROLLER=1` は INT32 非対応のメッセージを多数出し、10.2 B + rocRoller および 7.14 B + rocRoller と dump が一致した（SHA-256 `683d0cd0e397e5d1177df70883eb11ed81301113ef842c564b72d2b60ccb7360`）。この入力では unsupported による融合 kernel の fallback と整合する。A + Tensile との KL は 0.0569332、top-1 一致 89.0625%。試験は 1 入力・各 1 回で、探索の内部で実際に異なる algorithm が選択されたかは未確認。rocRoller の診断は静かな 1 行ではない。

GPU は公称 32 GB の R9700 で、16 GiB の実機ではない。3.05bpw、8192 token、chunk 4096、octave:c5v5、単一 request、arena 15 GiB、warmup なし各 1 回で `shelf-bench pp` を実行し、対象 GPU の `renderD130/device/mem_info_vram_used` を約 50 ms ごとに読んだ。arena の commit は両経路 16,106,127,360 B、arena 内使用は両経路 13,082,464,640 B。観測ピークは B 16,543,731,712 B（15.4076 GiB、366 標本）、A 16,581,586,944 B（15.4428 GiB、310 標本）で、差は約 36.1 MiB。標本化では短い allocation を見落としうる。測定器は workspace preference 0 の build であり、最大 256 MiB の候補のピークは測っていない。実機 16 GiB の保証や断片化時の余裕の証明ではない。

同じ 8k 入力を今回の品質測定 build で再測した。7.14 B + rocRoller 基準に対し、10.2 B 明示は KL 0.128177・top-1 一致 98.4375%、10.2 A は KL 0.394211・93.75% だった。B 自体が以前の基準と一致しないため、その差を A の効果とは解釈できない。build revision、backend、明示的 B 指定時の dispatch は基準と異なる。同じ今回 build の B から A への KL は 0.228894、top-1 一致は 95.3125%（61/64）、最大 logit 絶対差 15.0484。語彙全体を対象とした比較であり、BF16 からの KL ではない。今回の A dump は保存済み 10.2 A と byte 一致した。入力 SHA-256 は `3e01d5113e00df82bf116cba9b3ad8e6e851a132ddc379a1481480745bd2a6bb`。

## BF16 基準の位置対応結果

| モデル | 経路 | 平均 KLD | BF16 top-1 一致 |
| --- | --- | ---: | ---: |
| 4bpw | B | 0.02074514 | 94.8708% |
| 4bpw | A | 0.02117609 | 94.7568% |
| 4bpw | A-fused | 0.02117609 | 94.7568% |
| 3.05bpw | B | 0.05912932 | 91.3374% |
| 3.05bpw | A | 0.05947134 | 91.1474% |
| 3.05bpw | A-fused | 0.05947134 | 91.1474% |

| モデル | ΔKLD A−B | case bootstrap 95% 区間 | KLD 比 A/B | Δtop-1 (pt) | B のみ一致 | A のみ一致 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bpw | +0.000431 | [-0.000427, +0.001359] | 1.0208 | -0.1140 | 25 | 22 |
| 3.05bpw | +0.000342 | [-0.000750, +0.002142] | 1.0058 | -0.1900 | 18 | 13 |

| case | 位置数 | 4bpw ΔKLD | 4bpw KLD 比 | 4bpw Δtop-1 (pt) | 3.05bpw ΔKLD | 3.05bpw KLD 比 | 3.05bpw Δtop-1 (pt) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| en_prose_129 | 129 | +0.000000 | 1.0000 | +0.0000 | +0.000000 | 1.0000 | +0.0000 |
| ja_prose_129 | 129 | +0.000000 | 1.0000 | +0.0000 | +0.000000 | 1.0000 | +0.0000 |
| zh_tech_257 | 257 | -0.000218 | 0.9875 | -0.3891 | +0.000103 | 1.0017 | +0.0000 |
| rust_code_257 | 257 | +0.000528 | 1.0227 | +1.1673 | +0.005414 | 1.0664 | -1.1673 |
| python_code_513 | 513 | -0.000907 | 0.9504 | -0.5848 | +0.000933 | 1.0204 | +0.5848 |
| sql_code_1025 | 1025 | +0.000673 | 1.0591 | -0.0976 | -0.000368 | 0.9906 | -0.3902 |
| arithmetic_65 | 65 | +0.000000 | 1.0000 | +0.0000 | +0.000000 | 1.0000 | +0.0000 |
| structured_257 | 257 | +0.003231 | 1.2460 | -0.3891 | -0.002409 | 0.9454 | -0.3891 |

上表の「一致」は BF16 基準の top-1 との一致であり、実際の正解ラベルとの一致ではない。参考の基準自体の chunk による揺れは KLD 0.000496、top-1 一致 99.62%。

4bpw の A と A＋融合 decode は 8/8 dump で SHA-256 が一致し、全 logits が bit 一致した。A と B のそれぞれについて、先行実行との再比較は 8/8 一致した。

3.05bpw の A と A＋融合 decode は 8/8 dump で SHA-256 が一致し、全 logits が bit 一致した。A 同士の独立再実行は 8/8 一致した。B は既存の 2 回と 8/8 一致した。

## 既定化の判断

| 提案した基準 | 結果 | 判定 |
| --- | --- | --- |
| 全体 KLD 比 ≤ 1.03 | 4bpw 1.0208、3.05bpw 1.0058 | 通過 |
| 全 case KLD 比 ≤ 1.10 | 4bpw structured 1.2460 | 未達 |
| 全体 top-1 低下 ≤ 0.3 pt | 4bpw 0.1140 pt、3.05bpw 0.1900 pt | 通過 |
| 非投機と DFlash2 の出力一致 | A が稼働する長文での 128 ID 一致は未確認 | 未確認 |
| ABI・fallback・gfx1201 制限 | 初期化失敗・ライブラリ欠落、gfx gate、256 MiB cache に課題 | 未達 |
| 16 GiB の資源余裕 | 32 GB カードで制限付き測定のみ、256 MiB 案は未測定 | 未確認 |

**推奨は既定化を見送り、現行の opt-in を維持する。** case 別の閾値未達、8k の入力依存差、長文 DFlash2 一致と fallback・資源の未確認が残る。提示閾値は判断案であり品質保証の一般基準ではない。既定化が妥当という条件を満たさないため `perf/a-path-default` は作成せず、変更候補の必須テスト・5 指標は新規実行していない。再検討時は BF16 の長文基準、実用途の品質、長文での非投機/DFlash2 128 ID、障害注入、gfx gate、workspace cache と 16 GiB 実機の測定が必要。

追記: 8k の BF16 基準を作成した追加評価では、A のみが BF16 から離れるという解釈は支持されなかった。[長文の追加評価](long-context/README.md) を参照。ここでの見送りは、BF16 基準なしでの暫定判断として読む。新たな既定化候補の品質・fallback・必須試験は別途評価する。

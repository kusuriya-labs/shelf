# 長文の BF16 基準による追加評価

2026-11-20。4bpw EXL3-INT8、R9700 `GPU-a8e9ddefa2d60f55`、ROCm 10.2、BF16 KV、各 GPU 実行で `/tmp/r9700.lock` を取得。KL は FP32 logits の有効語彙 248,077 ID について FP64 で `KL(基準 || 対象)`、top-1 は基準との一致。A/B は既定化候補導入前の品質測定 build（`db3c0e93` より前に構築された opt-in 融合対応 build）であり、候補 build の評価ではない。BF16 の KV は FP16、chunk 64、shelf は BF16 KV、prefill chunk 4096 であり、量子化だけの差ではない。

| 入力 | 比較 | 平均 KL | top-1 一致 |
| --- | --- | ---: | ---: |
| 4,097 token・514 位置 | BF16 → B | 0.00458417 | 98.4436% |
| 同上 | BF16 → A | 0.00457655 | 99.0272% |
| 同上 | B → A | 0.00109676 | 99.0272% |
| 8k・64 位置 | BF16 → B | 0.74349736 | 87.5000% |
| 同上 | BF16 → A | 0.49130727 | 89.0625% |
| 同上 | B → A | 0.22889365 | 95.3125% |

4k の BF16→候補 KL の差 A−B は −0.00000762、位置を独立に復元抽出した bootstrap 95% 区間は [−0.000932, +0.000711]。同一 case 内の位置は相関するため、この区間は独立位置の仮定を含む参考値であり、2 case から母集団への一般化はできない。natural 257 位置では BF16→B 0.00002174、BF16→A 0.00002720、B→A 0.00000290、code 257 位置ではそれぞれ 0.00914661、0.00912590、0.00219062。4,097 token を 4096+1 行で prefill し、BF16 基準の入力位置 3840〜4096 を選択した。最後の 1 位置を decode とした別実行の A−B は −0.00000763 で、全体結論は変わらなかった。

8k は過去の反復入力（PSKLDTOK SHA-256 `3e01d5113e00df82bf116cba9b3ad8e6e851a132ddc379a1481480745bd2a6bb`）の input index 8192〜8255 を比較。A−B は −0.252190、位置 bootstrap 95% 区間 [−0.811882, +0.126003]。64 位置は一続きで相関し、区間は独立という仮定の参考値。両経路が BF16 から大きく離れており、この一入力では A のみの長文劣化という仮説を支持せず、B の方が平均 KL は大きい。反復 corpus・1 入力・64 位置に限るため、一般の長文品質や異なる algorithm の品質を保証しない。

8k BF16 基準は V620×2、`/tmp/v620.lock` と `/tmp/v620-76a08.lock` を保持した実行で作成。llama.cpp `bc52a12b38941b0a690ade65fbc5749715224e30`、ROCm 7.14、BF16 GGUF、layer split、FP16 KV、batch/ubatch 64。元ライブラリが消失していたため同 revision を再ビルドし、元 runner を使用した。確認として 4k natural case を同条件で再作成したところ、元 BF16 raw logits と SHA-256 `9863adee77b49de71a5be5e56067dbd62ad59cb7e857c1d6d94a52e0e579122d` が一致した。8k raw logits の SHA-256 は `0ca617bcf3601b5215a14603cfa85bd118c413be301b47039ea2b93eb89d7d9d`。保存先は `~/datapool/qwen38-kld-20260918/results/shelf-repeat-8256-bf16-20261120/`、入力は同 directory の `cases-shelf-repeat-8256-v1.json`。位置別値・dump の hash は [4k](4k-metrics.json) と [8k](8k-metrics.json) に記録する。

structured 257 では A−B の KL 差の総和 0.830295 のうち先頭 16 位置は +1.361516、後続位置は負の差で相殺される。位置 6 と 9 はそれぞれ +0.548941 と +0.494157。A の dense 経路を `SHELF_EXL3_INT8_DENSE=0` で無効にした追加測定の BF16 KL は 0.01313568、top-1 98.0545% で、B の 0.01313568 / 98.0545% に近く、A の 0.01636640 / 97.6654% より低かった。dense 経路の寄与と整合するが、row-scale、flagged-row 補正、hipBLASLt のどれが主因かは分離できていない。

256 MiB 案の cache は初回 heuristic の `workspaceSize` を cache entry に保存せず、hit 時 0 に初期化される。したがって非ゼロ workspace を必要とする algorithm の cache hit で、確保されない workspace を渡す可能性がある。元の workspace 0 build による過去の品質測定から、この failure が実際に発生したとは言えない。候補ではサイズを cache し、cache hit と device 切替を検証する必要がある。

## 更新した採用判断

BF16 基準を得た結果、8k の B→A の差を A 固有の長文劣化として既定化見送りの根拠にすることはできない。4k・8k の測定範囲では A は B と同程度か、8k の平均 KL は B より小さかった。ただし 8k は両経路とも BF16 との KL が大きく、自然な長文へ一般化できない。実験 branch `perf/a-path-default` の candidate commit `3fcd0797` では、A が稼働する 2048 token code 入力で通常生成と DFlash2 が 128 ID のうち 1 ID 不一致となった。B では一致し、A の両実行と B の両実行は再現した。tree を無効にし逐次 GDN、draft 1 にしても同じ位置で不一致だった。ユーザー要件の一致を満たさないため、長文 KLD ではなくこの確認された不一致と共有ライブラリ欠落 fallback の未対応を理由に、既定化・統合は引き続き保留とする。候補側の `docs/perf/a-path-default-candidate.md` に再現条件と性能・試験結果を記録する。

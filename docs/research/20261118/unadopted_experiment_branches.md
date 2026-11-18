# 不採用・記録のみの実験 branch の概要

2026-11-18 に、main へ採用しなかった実験 branch 14 本を削除した。この文書は各 branch の目的・結論・最終 commit だけを残す。詳細の文書・測定ログ・コードは、削除前に作成した bundle（`shelf-branch-archive/unadopted-20261118.bundle`、リポジトリ外）にある。復元は次のとおり。

```sh
git fetch <bundle> 'refs/heads/*:refs/heads/*'
```

数値は各 branch の文書に書かれた条件での値で、現在の build に対する保証ではない。

| branch（最終 commit） | 目的 | 結論 |
|---|---|---|
| `feature/dspine`（c532e6fa、28 commit） | `tencent/Qwen3.8-27B-DSpine-b16` の drafter を shelf に実装し、greedy speculative decode に接続 | 実装は動作し、greedy ID は全条件で一致。R9700、draft 7 で code 67.60 tok/s、prose 35.39 tok/s、日本語 37.18 tok/s（各 1 回、non-spec との比較は未記録）。DFlash2 / MTP と排他。PR は出さない方針で main には入れていない |
| `feature/launch-gdn`（7ff1df09） | GDN の launch 削減（I6 gate 同一アドレス許可、I3 a/b BF16 linear、I4 conv+SiLU+Q/K norm） | 3 項目同時で decode 1,581 → 1,431 launch。ただし KL 0.00111、top-1 一致 98.44% で品質基準を満たさず、寄与の特定まで既定無効とした。main へは入れていない |
| `feature/launch-attn`（2f4f58ae） | attention 前処理の launch 融合の測定記録 | Q 側と K/V 側の融合で 1,581 → 1,517 launch（−64）。コードは別 branch 経由で main に反映済み、この branch は測定記録のみ |
| `exp/kv-octave-fakequant`（a32d3066） | KV cache の Octave / exllamav3 / FP8 方式を fake-quant で比較する実験 | 比較用の実験コード（`kv_fakequant.hip` など）のみ。採用方針は出していない |
| `feature/fp8kv-direct`（4c986246） | FP8 KV decode の復号方式（LUT / hw cvt / arith / packed）の比較 | 全方式の復号結果は一致するが、加算順を保つ hw / arith は LUT より遅く、ビット一致のまま速くなる方式は無かった。packed は 190.7 対 175.8 GB/s（batch 1、64K split）だが加算順が変わるため不採用 |
| `feature/lm-head-proxy`（869af958） | EXL3 lm_head の INT2 候補生成 PoC | PLaMo 3 Translate 2B / Qwen3.8-27B とも、c=1 の改善余地が小さく production へ組み込む価値は低い。containment の検証は不十分（各 1 入力） |
| `feature/prefill-opt-kernels`（b016fafa） | 27B EXL3-INT8 prefill の内訳計測と kernel 最適化の試行 | 内訳計測（32K で診断 p50 1196 ms、通常 1143.6 ms）と、BF16 prefill attention の LDS stage 等の試行結果を記録。採用に至った変更なし |
| `feature/prefill-opt-sched`（cd04ff61、89 commit）/ `feature/dflash2-kv-diff`（00f36aae、90 commit） | prefill の chunk・並列スケジュール・DFlash2 prefill の K/V 差の調査 | 4 並列 prefill / decode の出力差、chunk 差、flag 介入の位置別 KL などを調査・記録した長い調査 branch。DFlash2 prefill の context 投影を opt-in でまとめる変更は main に入っている |
| `feature/server-dflash2-mismatch`（ffcbbc33） | server での DFlash2 と greedy の生成一致性 | 4 本同時で不一致（DFlash2 4 本 0/8、非投機 4 本でも 6/8 一致）。1・2 本同時は一致。原因分析を文書化 |
| `feature/startup-queue-warmup`（63d224d9） | 起動時の prefill 測定（queue / warmup） | 起動ごとの初回・2 回目・3 回目の prefill 時間の参考値のみ |
| `feature/shared-expert-fusion`（7417a02e） | MoE の shared expert 融合の調査 | routed（MXFP4）と shared（BF16）が同一形式でなく、現行 grouped kernel に shared を追加できない。実行順序の記録のみ |
| `feature/gemm-weight-ptr`（eec4f028） | EXL3 mul24 GEMM の重み pointer hoist の残り候補の記録 | hoist 本体は main に採用済み（`0f1ac309`）。非一時（non-temporal）load は単体で −2.4% だが実 graph で +4 ms となり不採用、原因は未特定 |
| `feature/persistent-ffn`（c8b8597a） | FFN persistent kernel（megakernel）の調査 | 実重みでの harness を作れず未完了。megakernel の採否は未判断（decode の megakernel / persistent の上限見積もりは約 3〜3.5 ms） |

## 削除せず残した branch

main への反映が commit 単位では確認できなかったため、削除していない。

- `feature/exl3-int8` と `backup/exl3-int8-mul24-20261105`（内容は main に別の形で入っている）
- `feature/exl3-int8-mid-rows`、`feature/dflash2-policy`、`feature/spec-draft-cut`、`feature/dflash2-exl3-tap`、`feature/drafter-quant`
- `feature/fixedcost-strix`、`feature/launch-boundary`、`feature/launch-rq`、`feature/batch-decode-final`、`exp/integ-gemm-dflash`

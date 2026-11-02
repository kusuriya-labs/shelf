# EAGLE-4 Phase C Gate 記録

- ID: EAGLE4-PHASE-C-GATE
- 実施日: 2026-10-22
- 判定: REJECT → 動的 sizing 不採用（既定は静的、chain が既定）。


### Phase C Gate の判定（2026-10-22）

C1/C2 の実装（`runtime/eagle4_cost.*`、`--eagle4-tree-dynamic`）に対し、
TP2・既定プロンプト・128トークン・5-run p50・chain/静的/動的を交互に測定:

| arm | p50 tok/s | spread | vs chain | MEAN_ACCEPTED |
| --- | --- | --- | --- | --- |
| chain（既定） | 65.76 | 1.8% | 基準 | 2.342 |
| tree W8/D4（静的最良） | 60.39 | 1.7% | −8.2% | 2.267 |
| dynamic（`--eagle4-tree-dynamic 1`） | 57.69 | 0.3% | −12.3% | 2.267 |

動的モードは freeze 後の選択が全 run で同一（受容数の分散ゼロ = 決定が決定論的）
だが、warmup 8 round が静的境界（depth 6）で走る分と選択結果が静的最良を
下回り、chain とも静的最良とも負ける。
**判定: C は不採用。既定は静的設定のまま（tree はオプトイン）、chain が既定。**
並列度スイープ（1〜8・高並列）の完全版は Phase D（batch 化）を前提とするため
保留とするが、単一並列で両 baseline に負けるため採否は覆らない。

---


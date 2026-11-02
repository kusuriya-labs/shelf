# EAGLE-4 Phase B Gate 記録

- ID: EAGLE4-PHASE-B-GATE
- 実施日: 2026-10-22
- 判定: 正しさ PASS / 性能 FAIL → 既定は chain（tree はオプトイン）。depth-first 修正後も不変。


### Phase B Gate の判定（2026-10-22）

**正しさ（PASS）**: TP1 で width{2,4,8} × depth{2,4,8} の有効6組合せ
（width ≤ depth+1 の構造制約により3組は起動不可）が参照とトークン完全一致。
TP2 で chain と tree W2/W4/W8 がともに TP2 vanilla と完全一致した
（TP1 と TP2 は投機なしでも greedy が1箇所で食い違う性質のため、
parity は同一 TP 構成内で比較）。parity ランは128トークン生成を含み、
GDN state 復元・KV relocation・compact 行指定を通時的に検証している。

**性能（FAIL、5-run p50・TP2・raw 128トークン・chain と交互に測定）**:

| arm | decode tok/s p50 | run 間 spread | MEAN_ACCEPTED |
| --- | --- | --- | --- |
| chain（drafts 7・既定） | 57.45 | 0.6% | 1.953 |
| tree W2 | 45.00（−21.7%） | 3.2% | 1.292 |
| tree W4 | 42.89（−25.3%） | 1.5% | 1.397 |
| tree W8 | 40.85（−28.9%） | 0.0% | 1.438 |

**判定: tree が全幅で chain に負けるため、既定は chain のまま**（Gate の
事前ルールどおり。tree は `--eagle4-tree-width` を指定した場合のみ有効）。

原因: 現実装の展開方針は frontier 全行を毎 step で広げる breadth-first で、
budget = width のため step1 で予算が埋まり深さが1止まりの木になる。
深さが出ず受入数が chain の線形 drafts（1.953/round）に届かず、
verify 行数の増分と relocation・親疎通のコストだけが残る。
深さを成立させるには best ノードを1つずつ展開する depth-first 方針と
(width, depth) を分離した budget 配分（children/step ≈ (width−1)/depth）
が必要で、Phase C のコスト表（C1 は (d,w) 毎の計測を前提）は
この修正を先に含むこととする。

**depth-first 展開への修正後の再測定（2026-10-22、TP2・既定プロンプト・
128トークン・5-run p50・chain と交互・model-host 経由、GPU2/3）**:
best ノードを1つずつ降りる展開（children/step = (width−1)/depth）と
compact log 行数 = max(drafts+1, width) の OOB 修正を反映した結果:

| arm | p50 tok/s | spread | vs chain | MEAN_ACCEPTED |
| --- | --- | --- | --- | --- |
| chain（drafts 7） | 65.08 | 1.9% | 基準 | 2.342 |
| tree W8/D4 | 60.47 | 2.0% | −7.1% | 2.267 |
| tree W13/D7 | 51.88 | 1.6% | −20.3% | 2.488 |
| tree W15/D7 | 50.99 | 0.6% | −21.7% | 2.476 |

受容数は W13/W15 で chain を上回るが（2.488/2.476 > 2.342）、
draft ステップの同期と verify 行の増分コストが上回り、
**全幅で chain が勝つため判定は変わらず「既定は chain」**。

**TP2 検証（model-host 2台構成、2026-10-22 実測）**: TP2 vanilla・
chain・tree W2D2・tree W8D4 がすべて TP2 vanilla とトークン完全一致
（chain MEAN_ACCEPTED=3.400、tree W8D4=3.600）。MODEL_LOAD_MS≈0.6 秒。

---

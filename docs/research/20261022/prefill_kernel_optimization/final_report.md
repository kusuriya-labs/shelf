# Prefill 個別最適化 — 最終報告（2026-10-22 時点）

対象: Qwen3.8-27B FP8 / R9700 ×2 / TP=2 / PP=2048。カーネル融合・dispatch 削減・
HIP Graph・TP 方式変更・量子化変更は一切不使用。
数値はすべて本セッションの実測。実験の全履歴は [experiments.md](experiments.md)、
ベースラインは [baseline.md](baseline.md)、選定計画は [opportunity.md](opportunity.md)。

## 1. ベースライン

- 構成: R9700（gfx1201）×4 のうち GPU0+1 を TP2 使用、PEX89048 Gen5 スイッチ +
  各カード upstream **Gen5 x8**、HipPeer(p2p)、HIP 7.15.26333 / clang 23、
  model revision `017b9c7a…`、FP8 E4M3（weight 128×128 ブロック、act トークン毎128）
- 663ms の再現（A-1）: `phaseshift-compute --tp 2 … /tmp/pp2048.txt …`
  → **PREFILL_MS = 663.4 / 669.9**（ゴール値と一致）、`GENERATED_IDS=47410`（全実験で一致）
- 主指標（A/B 判定）: `phaseshift-bench pp --tp 2 --prompt-tokens 2048
  --arena-gib 16 --warmup 1 --runs 5` の `wall_ms_median`
  → 初期基準 **657.75 ms**（ばらつき 0.3%）。以後は interleaved ペア差で判定。
- 区間: PREFILL_MS = prefill step 開始〜同期完了（ロード・decode 除外）。
  `PHASESHIFT_OP_DUMP=1` の実行時間は不使用。

## 2. Kernel breakdown（最新 trace: G9 適用後、rocprofv3 kernel trace・4306 dispatch・両 GPU）

- rank0: kernel 合算 518.4 / host gap 52.7 / span 571.1 ms、
  rank1: 515.9 / 107.5 / 623.5 ms（プロファイラ付加込み。clean wall = 551.09）
- 活動の和集合 534.6 ms、**両 GPU 非活動 88.9 ms**（peer 待ち・host・リンク待ち）

| family | duration | share | 実効値 | 律速分類 |
|---|---:|---:|---|---|
| gemm_fp8 prefill2d wide | 280.6 ms | 54.1% | 当初 383.8 から −103（G3/G8/G9） | ② 計算/pass |
| tp_copy_bf16 | 95.2 ms | 18.4% | 29.1 GB/s = Gen5 x8 理論の 92%（飽和） | **① 帯域・削減不能** |
| gdn_recurrence_wmma_lossy | 53.9 ms | 10.4% | チャンク 14.4µs 線形、sync 6/chunk | ② 同期/pass |
| PagedAttention prefill | 15.3 ms | 3.0% | 105 TFLOP/s | ② |
| fused_resid_rmsnorm_quantize | 15.0 ms | 2.9% | 中間の global 往復 2 回（EXP-M1 対象） | ①/② |
| tp_add_peers_bf16 | 14.9 ms | 2.9% | ローカル | ② |
| swiglu / quantize×2 / gdn sub×3 | 各 5〜11.5 | 計 9.5% | — | ② ほか |
| lm_head（n=1） | 2.1 ms | 0.4% | 635 GB/s = DRAM 直下 | ①・削減不能 |

- critical path: wall ≈ rank0 chain（kernel 合算 + host gap）。
  **rank0 の gap 52.7 ms の内訳（相関分解）**: peer 待ち（rank1 動作中）15.1 ms (29%) +
  **両者アイドル 37.6 ms (71%) = host 提出・セグメント協調 = ④（スコープ外）**。

## 3. 最適化結果（実験台帳）

### 採用（ACCEPT）

| ID | 内容 | E2E ペア差 | 主な根拠 |
|---|---|---:|---|
| C2-b | GEMM wide 幾何のみ（128×128, ww×wa） | **−26.83 ms** | 288/0、gate/up −6.1%、3/3 ペア |
| G3 | scale を K128 毎へ遅延 fold（数式等価） | **−68.63 ms** | マイクロ −14.4%、VGPR 184→216（4→3 blk）でも E2E 増益、288/0・121/121 |
| G6 | s_as の世代分離 ping-pong（正しさ修正） | ±0（中立） | 世代間 race の構造を排除、LDS 512B→1KB |
| G8 | 4 行同時計算（Weight 共有 4、4 系 acc、K 順序保存で byte 等価） | **−2.19 ms** | E2E 5/5 ペア、VGPR216 不変、GENERATED_IDS 一致 |
| G9 | **CTA 実行順 swizzle（G=4、参照の kGroupM 同型）** | **−12.25 ms** | 往復スイープで G4 両巡最速、working set ≈4.6MB < L2 8MB、3/3 ペア |
| S1 | **SWIGLU+quantize 融合（ユーザー解禁の唯一の融合・命令数基準で採用）** | 未確定（E2E ノイズ −0.92ms・符号3:2） | **ISA 776→503 命令（−35%）**、dispatch 2→1、byte-exact、required 122/122。wall 効果は未確証のまま構造的利点で採用（ユーザー判定基準） |

### 棄却（REJECT、全件 experiments.md に記録）

C1（as 直読み 3 変種: VGPR 増で +5.6〜+51%）、C2-a（staging wide: **+13.3%**、uniform
2-way bank conflict は動的証明）、G1-a（gdn kWv 64→32: +7.3% — 有効 warp 半減が
occupancy 増を上回る）、G1-b（gdn syncA/C 削減: **+5.4%** — v_exp_f32 +60 が
barrier 2 本分を上回る）、G4（hoist バンドル: ±0 — コンパイラ CSE 済み）、
G5（acc 2 本交互: **+3.1%** — 同一 dst の accumulator forwarding が重要）、
G7（2 行 Weight 共有: ±0）、G10（ws CTA-uniform: +0.25% ノイズ内）、
A1（TP 通信 stream 分離: **+37.8ms 劣化** — overlap 理論を実測で否定）、
A1b（原因分解: peer_write 常時 false を実測、NonBlocking −4ms、残る +28.9ms は構造そのもの）、
M1（resid_rmsnorm レジスタ保持化: 読減 3.9MB/dispatch・byte-exactだが E2E ノイズ内、
gfx1201 の NaN 符号決定論の知見を副産物に獲得）、
G1-d（gdn チャンク頭事前読み: +0.8% 劣化 — ロード待ちは非支配と実測）。
棄却差分は `/tmp/g1a_kWvP.patch` `/tmp/g1b_syncAC.patch` `/tmp/g4_hoist.patch`
`/tmp/g5_acc2.patch` `/tmp/g10_ws_uniform.patch` に保存。

### ISA / Resource 差分（採用分）

- G3: scale 適用 fma/mul を K32 毎→K128 毎へ約 1/4。VGPR 184→216（alloc216 →
  6/SIMD、3 blk、Scr 0）、LDS 512B→1KB（G6）。
- G9: リソース変化なし（ブロック毎の除算 2 回のみ）。grid 形状・dispatch 数・
  オペレータ間依存は不変。

### 正しさ・回帰

focused 288/0（毎実験）、required **121/121**（G3・G8・G9 時点）、
`GENERATED_IDS=47410` 全一致。gdn 系 focused / spec_history / exact_history は
REJECT 時も全 PASS（性能のみが棄却理由）。

## 4. 最終性能

| 指標 | Before | After |
|---|---:|---:|
| pp wall ペア中央値（主系列） | 660.47 | **551.09** |
| compute PREFILL_MS | 663.4 | **552.5** |
| Throughput（pp 換算） | 3,100.8 tok/s | **3,716 tok/s** |
| **163ms 削減達成率** | — | **109.38 / 163 = 67.1%（残り 51.09 ms）** |
| 500ms 到達 | — | 未達（差 51.09 ms） |
| 回帰テスト | — | required 121/121、focused 288/0、トークン gate 一致 |

Phase F: 660.47 → 633.64（C2-b）→ 565.53（G3）→ 563.34（G8）→ **551.09（G9、以降 S1 採用も wall はノイズ内）**。

**実験総数 17 件: 採用 6（C2-b / G3 / G6 正しさ / G8 / G9 / S1）・棄却 11。**
スコープ内候補はすべて消化済み（gdn 3 件・GEMM 6 件・misc 2 件・TP overlap 2 件）。

## 5. 残存ボトルネックと現行制約下の上限

**削減不能と証明済み（スコープ外）**

- tp_copy 95.2 ms: PCIe Gen5 x8 飽和（bytes 削減 = TP 方式変更で禁止）
- 両者アイドル 37.6 ms（rank0 gap の 71%）: **CPU 非関与を実測**（run_rank_rank_step は
  rank 毎 1 回、submit 36ms 一括投入、complete 516.6ms 待機、CPU 同期 API 0 回、
  transport CPU 計 6.8ms/141+129 回）。主因は **rank ずれ（通信依存の待ち合わせ）**。
  **通信 stream 分離（案 A1）は EXP-A1 で反証済み**: overlay ON で +37.8ms 劣化
 （3/3 ペア）— 次計算は加算結果に真に依存し、単一 stream の方が最速。イベント
 往復の追加だけが乗る。残る削減経路は kernel 短縮への連動のみ
- lm_head 2.1 ms: DRAM 直下 ①
- peer 待ち 15.1 ms: kernel の縮小に連動して縮む独立レバーなし

**未最適化 kernel と残見積（根拠付き）**

- misc family 合計 ≈60 ms: EXP-M1（中間 global 往復 2 回の排除、見込み −3〜5）＋
  その他 small（−2 級）
- gdn 53.9 ms: G1-d（チャンク先読み、−1〜2）が唯一の未試レバー。
  G1-a / G1-b は反証済み
- GEMM 280.6 ms: G4・G5・G7・G10 で hoist・チェーン・uniform は枯渇。残るは
  VGPR 216→192（4 blk 復帰、G3 の実績から効果は低確度）、128-bit ロード、
  geometry 64×128（いずれも大規模・未着手）

```text
現行制約下の推定上限 = 551.09 − (misc 5 + gdn 2 + 低確度 GEMM 0〜10) ≒ 534〜544 ms
→ 500 ms には約 34〜44 ms の不足
→ 到達には現行スコープの追加レバー（要承認）が必須
```

**次に取り組むべき課題（優先度順）**

1. EXP-M1（misc の中間再読排除）の採否確定
2. gdn G1-d（チャンク先読み）
3. 有望候補の消化後、停止条件に該当するかを判断し、不足分を数値で締める
4. スコープ外候補（要承認）: tp_copy と計算の overlap、④ の 37.6 ms を生む
   dispatch / handshake の削減

**保留中の保管物**: `/tmp/g1a_kWvP.patch` `/tmp/g1b_syncAC.patch` `/tmp/g4_hoist.patch`
`/tmp/g5_acc2.patch` `/tmp/g10_ws_uniform.patch`、`/tmp/g6_bin` `/tmp/g9_bin`（A/B 用
バイナリ、再起動で消える）。`docs/perf/current.md` への本計測の反映は未実施
（FP8 E2E 節が「再計測未実施」のまま。反映はマージ時に判断）。

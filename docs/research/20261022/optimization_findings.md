# 最適化知見（2026-10-22、Qwen3.8-27B FP8 Prefill / GEMM / TP ミッションより）

再利用を目的とした汎用知見の抽出。個別実験の経緯は
`prefill_kernel_optimization/experiments.md`、総括は同 `final_report.md` を参照。

## gfx1201 の資源と occupancy

- Occupancy は **1536 VGPR/SIMD、granule 24、上限 16 wave/SIMD**:
  `waves/SIMD = min(16, floor(1536 / (ceil(VGPR/24) × 24)))`。
  レガシーの 47,104 dwords / 8 粒度モデルは誤り（旧「184 の崖」は存在せず、192 まで
  8/SIMD）。LDS（131,072B/WGP）は別枠で判断する。
- **occupancy 減が速度に効かないことがある**: GEMM wide は VGPR184(4blk)→216(3blk) に
  なっても増速した（G3, −14.4%）。occupancy 段階の変化だけで成否を予測しない。
- VGPR は段境界で評価する（例: 3blk 維持は VGPR≤240、8/SIMD は ≤192）。

## 計算法・命令の性質

- **同一 dst の WMMA アキュムレータ連鎖はハードウェア forwarding で安価**:
  2 本へ分割すると forwarding が効かず遅くなる（G5, +3.1%。指摘ベースの仮説も
  実測で反証）。
- **コンパイラの CSE/LICM は既に働き過ぎている**: ソース側のアドレス整形・
  ロード重複除去・スカラー化は実測でゼロ効果（G4/G7/G10）。ISA を見てから手を出す。
- **scale の適用は K128（weight 128×128 ブロック）単位へ畳める**: 数式等価・
  バイト等価（K 順序保存時）・tolerance 内 → 原則4 の事前承認範囲。
  GEMM の scale fma を K32→K128 で約 1/4 に（G3, E2E −68.6ms、本ミッション最大）。
- **gfx1201 の NaN 符号は add の左オペランド優先**（p+m=+qNaN、m+p=−qNaN、実測）。
  完全展開でコンパイラが `a+b` を交換 canonical 化するため、byte-exact 必須の
  reduce は `v_add_f32` の inline asm でソース順を固定する（M1 の経験）。
- BF16 往復（`__float2bfloat16` → bit 同一の戻し）はロスフリー。融合でも
  「丸め1回」の位置を直列と同じに保てばビット一致。

## キャッシュと実行順

- **CTA 実行順の swizzle は L2 working set を律速させる**: 出力 128 列固定で
  行ブロック G 個を同一 weight 列で連続実行（G=4 が最適: act 2.6MB + inflight ≈4.6MB
  < L2 8MB）。G8（G=1 既定比）E2E −12.25ms。G を 1,2,4,8 の**往復2巡**で測る。
- working set の当たりは `(G×act_tile + inflight weight) ≒ L2` の素朴計算で事前に当たる
  （G=8 は 7.2MB で逼迫 → 測定で劣化）。

## wall と kernel の対応（最重要の構造知見）

- **critical path 上の kernel だけが wall に変換される**: GEMM 系（chain ~55%）の
  改善はすべて転換（C2-b/G3/G8/G9）。misc family（swiglu・resid_rmsnorm 等）は
  **kernel で −34% しても wall はノイズ内**（rank ずれの待ちに吸収）— S1/M1 で2回確認。
  kernel benchmark の改善だけで採否を決めないこと（E-2 と同じ帰結を family 選定にも適用）。
- TP2 の待ちは 2 種類: peer 待ち（rank ずれ、kernel 縮小に連動）と両者アイドル
  （host/協調）。実測では CPU は一括投入（36ms）+ 完了待機（516ms）のみで
  **per-segment の CPU barrier・同期 API は不使用** → ④ の CPU 仮説は実測で棄却。
- **単一 stream 内の直列（計算→送信→wait→add）は最速形**: 通信用 stream への分離は
  依存を解かずイベント往復（130 集合点 ×2 段）だけ増やして **+37.8ms**（A1 反証）。
  重ね合わせは **チャンク粒度（A2/span）** しかないが、参照自身の TP2/2K 実測は
  −6.1%（負）で期待値は低い。

## 供給線と物理制約

- PEX89048 + 各カード upstream **Gen5 x8 = 31.5GB/s**。tp_copy 実測 29.1GB/s = 92%
  **飽和** → bytes または TP 方式（両方禁止）以外にレバーなし。
- 参照（libr4d/radiance）も push がラインレート飽和を測定済み（「codec タックスは物理」）。
- launch 数削減は 4µs floor / backpressure で無意味（参照 W17・step-graph と同様）。

## 計測規律（本ミッションで効いたもの）

- A/B は **同一バイナリ・env 切替**（G9/S1/G8/A1 で実施）が最も綺麗。
- 絶対値は同セッション内でも GPU 状態で ±7%、GPU 個体差 ±2%。**ペア差のみ**を信用する。
- ペア差がばらつき帯（±1ms 級）に入ったら**追加ペアではなく判定基準を見直す**
  （S1 はユーザー指示で ISA 命令数を採否基準に切り替えて決着）。
- 同一 args でも別インボケーション間で p50 が +14% ずれた例あり（継続負荷後の clock）。

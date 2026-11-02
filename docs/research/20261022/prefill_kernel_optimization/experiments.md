# Prefill 個別最適化 — 実験記録

フォーマットはゴール文書 §9 に従う。主指標は pp wall p50（TP=2 / PP=2048）。

---

## EXP-C1: as ステージ廃止（`__syncthreads` + `global_inv` 消去）

```text
Experiment ID: EXP-C1 (a/b/c)
Commit: 54bc68e（基準）/ 変更は未コミット、git checkout で復元
Target kernel: gemm_fp8_w8a8_wmma_prefill_2d_kernel<false>
Bottleneck class: ② pass / 同期（毎 g2 に __syncthreads → s_barrier + global_inv(L1無効化) + 全ロード drain ×40/dispatch）
Hypothesis: as を activation_scales から直接読み、LDS ステージとバリアごと消せば、
  バリアコスト＋L1 無効化＋ロード pipeline の途断が取り除ける（レジスタ中立を想定）
Expected E2E saving: −10〜−30 ms
Modified files: src/shelf/kernels/optimized/linear/fp8.hip
ISA before/after（静的）:
  base: global_inv 1 / s_barrier 2 / ds 17 / wmma 64 / wait_alu 425 / 17,516 B
  C1a:  global_inv 0 / s_barrier 0 / ds 0  / wmma 64 / wait_alu 476 / 19,428 B
Resources before/after（rows512 マイクロ, rocprof）:
  base: vgpr 184 / lds 0 / scr 0 / avg 220.2 µs（4 blk 相当）
  C1a (asv[8] 配列): vgpr 248 → 2 blk / avg 332.9 µs
  C1b (j ループ内インライン, size_t): vgpr 192 → 3 blk / avg 232.5 µs
  C1c (32-bit オフセット): vgpr 208 → 3 blk / avg 254.8 µs
Correctness: 全変種 288/0（test_gemm_fp8_optimized）
Kernel benchmark: 上記 avg（rows512 n5120 k5120、1100 dispatch 合算）
E2E A/B: 未実施（単体と occupancy 段階の劣化で screen）
Observed saving: 負（+5.6%〜+51%）
Decision: REJECT（全変種）
Reason: occupancy モデル改訂後（1536/SIMD, granule 24）の再評価では C1b(VGPR192) は
  8 waves/SIMD で occupancy 不変 — 劣化（+5.6%）は global as 読み出しのアドレス連鎖が
  本質的に遅いことに起因。C1a(248)→5/SIMD、C1c(208)→7/SIMD は段階低下。
  いずれの形でも as 直読みは LDS 経由より遅く、barrier 消去の利を上回らない。
```

---

## EXP-C2（計画中）: 参照幾何への移植 — `[2][4]` × Wave8、act を kPWW=4 wave が共有

根拠: `r4d_gemm_fp8a8`（radiance 参照実装）は M 300–4096 の LLM prefill 形状で
**160 TFLOP/s** を実測（現状の E2E 実効 130 TFLOP/s の 1.23 倍）。

参照の実証済み定数（`r4d_gemm_fp8a8.hip` L57–L68, L658–L666, L1246）:

| 定数 | 値 | 意味 |
|---|---|---|
| kPTW / kPTA | 2 / 4 | wave あたり out タイル 2 × act タイル 4 = acc[2][4]（8 タイル上限） |
| kPWW / kPWA | 4 / 2 | 8 wave = 4（weight 波）× 2（act 波） |
| kPBN / kPBM | 128 / 128 | ブロック = 128 out × 128 act 行（scale block と一致） |
| kPWaves | 8 | `__launch_bounds__(256)` — wg は現状と同じ 256 |
| kPFK / s_a | 64 / `s_a[2][kPBM*(64+8)]` | act を LDS ping-pong（約18KB）で **2 スレッド/行** |
| grid | (M/128, N/128) | ブロック総数は現状（M/256×N/64）と同一 → **dispatch 数不変** |
| R4D_WAVES(6) | waves_per_eu(6,6) | 参照は occupancy を 6/EU に指定 |

現状との構造差（本質は act 再利用）:

- 現状: 256 行×64 out、wave_m(4)×wave_n(2)。**act 行は 1 wave 固有 → 再利用ゼロ → global 直読みが最適**（これが A 状態が最速だった理由）
- 参照: 128 行×128 out、kPWW(4)×kPWA(2)。**同じ 64 act 行を 4 wave が共有 → LDS ステージが4倍再利用で成立**。weight はレジスタ直読み（不変、現状と同じ）
- ワーク量・wave 数・ワガイル当たりタイル数（2 out×4 row-group）は同一。**違いはブロック形と act の配り方だけ**

移植の可否（スコープ判定）: C-5「Tile / Workgroup / Wave 構成の調整」＋ launch 設定変更（grid/workgroup）= 許可範囲。融合・TP 方式・dispatch 数・意味論は不変。

```text
Experiment ID: EXP-C2（次ラウンドで実装）
Hypothesis: 参照どおりの 128×128 ブロック + act LDS 4倍共有（ping-pong パイプラインは
  stash@{0} の pipelinestage 構造を流用）で 130→接近 160 TFLOP/s、
  GEMM chain 383.8 ms から −55（中央）〜 −73（楽観）ms
Expected E2E saving: 中央 −55 ms / 楽観 −73 ms（副次: act global 4倍削減 → L2/待ちも縮む）
Verification: vgpr≤184 維持（4 blk）、288/0、KFD 空き下 interleaved A/B ×3
```

---

## EXP-C2: 参照幾何移植（3 変種の分離実験と採否）

```text
Experiment ID: EXP-C2 / C2-a (staged wide) / C2-b (geometry-only wide)
Commit: 基準 54bc68e（docs 記録 1a3a688 夾）/ C2-a 差分 = /tmp/c2_wide.patch 保管
Target kernel: gemm_fp8_w8a8_wmma_prefill_2d_wide_kernel
Bottleneck class: ② 計算/pass（GEMM share 61%）+ ③ latency（occupancy/同期）
Hypothesis: 参照の 128×128 ブロック + act を kPWW=4 wave 共有 + LDS ping-pong で
  130→160 TFLOP/s 相当へ（参照実証 160 TFLOP/s, M300–4096）
Expected E2E saving: 中央 −55 ms
Modified files: src/shelf/kernels/optimized/linear/fp8.hip
Correctness: 全変種 288/0。C2-b は E2E トークンゲート GENERATED_IDS=47410 一致
Resources: base vgpr184/lds1024/scr0 | C2-a attr8 vgpr176/lds25600/scr0 |
           C2-b vgpr184/lds512/scr0
Decision:
  C2-a (LDS staging 付き wide)   : REJECT
  C2-b (geometry-only wide)      : ACCEPT（E2E A/B 後）
```

### 単体（rows2048 n8704 k5120 = gate/up、最重要 shape）

| variant | avg µs | p50 µs | base 比 |
|---|---:|---:|---:|
| base narrow | 1416.8 | 1681.0 | — |
| **C2-b geometry-only** | **1329.9** | **1589.0** | **−6.1%** |
| C2-a staged wide（属性 8,8） | 1507.0 | 1767.6 | +6.4% |

### 分離結果

```text
Geometry 変更のみ の効果         : −6.1%（1416.8 → 1329.9、改善）
LDS staging を足した効果         : +13.3%（1329.9 → 1507.0、劣化）
合計                             : +6.4% ≈ C2-a の実測劣化と一致
```

### E2E interleaved A/B（主指標、3 ペア、KFD 汚染 0 すべて）

| pair | base (pp wall p50) | C2-b | 差 |
|---|---:|---:|---:|
| 1 | 658.5767 | 630.9313 | −27.65 |
| 2 | 662.6248 | 635.1904 | −27.43 |
| 3 | 660.4683 | 633.6417 | −26.83 |
| **ペア中央値** | **660.4683** | **633.6417** | **−26.83 ms (−4.06%)** |

突合: compute PREFILL_MS 663.4/669.9 → 636.1/635.4（−30ms 相当）。トークン
47410 は base と一致。required 121/121 合格。マイクロ −6.1% × GEMM share
61% ≈ −23〜−37ms に対し実測 −26.8ms → kernel 削減が wall に整合（E-2 OK）。

### Exp-2: ISA 比較（静的、同一 wmma 64 を確認）

| | base | C2-a staged | C2-b geom |
|---|---:|---:|---:|
| `v_wmma_f32_16x16x16_fp8` | 64 | 64 | 64 |
| `s_barrier` サイト数（動的/dispatch） | 2（40） | 4（**80**） | 2（40） |
| `global_load` サイト | 53 | 29 | 49 |
| `ds_load` / `ds_store` サイト | 16 / 1 | 34 / 11 | 8 / 1 |
| `s_wait_alu` | **425** | 226 | 230 |
| `s_wait_loadcnt` | 25 | 23 | 24 |
| `s_wait_dscnt` | 12 | **28** | 9 |
| func size | 17,516 B | 14,608 B | 14,456 B |

### Exp-3: LDS bank 展開（C2-a の act 読み出し）

act 読み出しは 32 lane × 8 B（`ds_load_2addr_b64` ×16 サイト、+ `ds_load_b128` ×18 は as 系）。
アドレス `… + m_lane*16 + k_group*8 + …` の lane 展開:

```text
banks 0..31 のアクセス数 = [2,2,2,...,2]  全 bank ちょうど 2 回
max conflict = 2 way（32 lane × 2 words = 64 access / 32 banks = 2.0）
例: bank0 = lane 0 と lane 8 が衝突
```

→ **uniform な 2-way bank conflict が構造的に毎回発生**。参照実装は
pitch = FK+8 のパディングと fragment-order de-permute で 1 bank cycle に
している（`r4d_fp8_wmma.h` L148–160）のに対し、C2-a は素の interleave
レイアウトをそのまま LDS に載せたため対策なし。

### 遅延原因の分類（結論）

| 分類 | 対応する遅延 | 証拠 | 反証・注記 |
|---|---|---|---|
| **Geometry** | **改善 −6.1%**（C2 の劣化要因ではない） | C2-b vs base 単体・E2E −26.8ms、`s_wait_alu` 425→230 | 機構の内訳（アドレス連鎖短縮 vs タイル割当）は未分離 |
| **LDS staging** | **劣化主因 +13.3%** | C2-a vs C2-b（occupancy parity: vgpr176=8/SIMD でも遅い）、ds 系 9→45 サイト、`s_wait_dscnt` 9→28、barrier 動的 40→80 | act の DRAM bytes は base/C2-a とも実効 1 回（2× demand は L2 dedup）→ staging の利得側が消えている |
| **Bank conflict** | staging 内訳（分離未） | uniform 2-way を静的証明（上記） | 個別寄与は未実測。pitch+de-permute 版で分離可能（次の候補） |
| **ISA/resource** | 負の寄与なし | wmma 64 三者同一、C2-a vgpr176 ≤ base184、Scr すべて 0 | occupancy は C2-a attr8 で parity 到達済み → resource は原因でない |
| **その他** | ノイズ内 | ペア内ばらつき <1ms、KFD 汚染 0、プロファイラなし | — |

### Phase F: 累積表（更新）

| Iteration | 対象 | 変更 | 実測 E2E（pp wall ペア中央値） | 累積削減 | 残り |
|---|---|---|---:|---:|---:|
| Baseline | — | — | 660.47（compute アンカー 663.4） | 0 | 163 |
| 1 | GEMM 幾何 | EXP-C2-b（128×128, ww×wa リマップ、act 直読み維持） | **633.64**（compute 636.1） | **26.8** | **136.2** |

### 次の候補（順序づけは opportunity.md の再計算に従う）

staging を「勝つ形」で入れ直すには (a) pitch パディング + fragment-order
de-permute（bank 解消を先に証明）、(b) gdn_recurrence、(c) GEMM 微小レバー。

---

## EXP-G1（分析フェーズ）: gdn_recurrence_wmma_lossy の構造解析と仮説

**実装前分析。以下は未変更（コード無変更・基準値のみ計測）。**

```text
Target kernel: shelf_qwen35_gdn_recurrence_wmma_lossy
  → gdn_recurrence_wmma_body<true, false>（recurrence.hip L222–）
Bottleneck class（暫定）: ② 同期/pass 支配 + ③ 埋め込みレイテンシ
  （occupancy は capacity 比で半分、時間の大半は同期・逐次レイテンシと推定）
```

### 構造（実測・ソース）

- 定数: `kWt=32`（行チャンク）, `kWv=64`（v 幅/block）, `kWk=128`, `kWthreads=256`,
  `kSp=68`（S の pitch）。`LdsSmem.S[128][68]` = 34.8KB（LDS 35,840B の大半）
- launch: `grid = num_requests × num_v_heads × (head_v/kWv)`、
  **本番 grid.x = 48**（nvh=24 × groups=2 × nr=1 と整合）、`__launch_bounds__(256,1)` +
  `amdgpu_num_vgpr(256)`（実効 VGPR は rocprof 224）
- 行スキャン: `nchunks = ceil(2048/32) = 64` チャンクを直列。1 チャンク内:
  a/b 読み→bt/oe（wave0）→ G prefix-scan（wave0 の shfl）→ exps → **__syncthreads ≈6 回**
  → wmma（S×K/Q, 8 kk × wmma2、warp<4 のみ +2）→ mid.A/L/W 書出し →
  **三角行列 solve（`Σ_{l<i} L[i][l]·dlt[l]`、LDS 512 読み/列 × kWv=64 列）** → jt 適用
- 動的 barrier ≈ 6/チャンク × 64 = **約 384 回/dispatch**（静的 9 サイト）
- wmma は 8×2〜4/チャンク ≈ 16–32 動的 — 時間の主体ではない（推定）
- `amdgpu_num_vgpr(256)` は上限指定（実 224 → 1536/240 = 6/SIMD）

### Occupancy（改訂モデルで計算）

```text
capacity:  VGPR224→alloc240→6/SIMD→24/WGP→reg 3blk
           LDS 35840→131072/35840=3.65→3blk
           → capacity = 3 blk/WGP = 24 waves = 6/SIMD
実効:      grid 48 全ブロック同時常駐できるが、48/32 WGP = 1.5 blk/WGP 平均
           → 12 waves/WGP = 3/SIMD ← ★実効を縛っているのは grid（平均値）
満たすには: 96 ブロック必要（kWv 64→32 で vgroups 2→4 → grid 96）
```

### 基準値（マイクロ、本番 shape）

```sh
shelf-bench gdn-recurrence --variant optimized --rows 2048 --nreq 1 \
  --key-heads 8 --value-heads 24 --head-k 128 --head-v 128 --warmup 2 --samples 5
# p10 808.60 / p50 811.95 / p90 903.90 µs（grid=48 相当）
```
（E2E trace での当該 kernel 平均は 1,159µs。マイクロ 812µs との差 +43% は TP2 共存時
の影響とみられ、最終判定は E2E で行う。）

### 仮説（優先度順、次ラウンドで実装）

| ID | 仮説 | 予想 | リスク |
|---|---|---|---|
| G1-a | **kWv 64→32（prefill 固有化）**: grid 48→96 で 3/SIMD→6/SIMD に capacity 到達、solve 列半減 | 埋め込み改善で −10〜25% | k/q 読みと scan・barrier の総量 ×2（L2/同期増）、solve は減で相殺の可能性。**head_v%32 チェックと grid 経路の確認**。`kGdnRecurrenceWmmaDecode1VGroup` 共有定数は **decode1 を 64 のままにする**（prefill 局所化、static_assert 解決）→ decode に触れないこと |
| G1-b | barrier 統合（G 書出→exps は同 wave 内 → s_wait_dscnt で可視化） | −1〜2%（64 回分） | 可視性の ISA 証明が必要（C-6 手順） |
| G1-c | solve の累加順並列化 | 中 | **FP 累加順変更 = 承認制（§2）→ HOLD、承認を得るまで実装しない** |
| G1-d | チャンク先読み（次チャンクの a/b/k/q/v ロードを現チャンク計算中に発行） | ILP −5% 級 | 意味論不変（読み順のみ）、VGPR 増分に注意（224 は 240 で 6/SIMD 限界、241 で 5/SIMD へ落ちる） |

Decision: HOLD（分析完了・基準値確定、実装は次ラウンド）
Reason: 1 チャンク18.1µs の内訳（barrier / solve / ロード）が未分離。G1-a の前に
  G1-d か計測分割を入れてから単一仮説を実装する。

### 追加計測（EXP-G1 分解、コード無変更）

rows スイープ（本番 shape grid48、nreq1、p50 µs）:

| rows | chunks | p50 µs |
|---:|---:|---:|
| 64 | 2 | 37.08 |
| 128 | 4 | 64.24 |
| 256 | 8 | 121.14 |
| 512 | 16 | 236.08 |
| 1024 | 32 | 466.56 |
| 2048 | 64 | 929.64 |

```text
フィット: T(rows) ≈ 14.4µs × chunks + 9µs（固定）
→ 総時間の 99% はチャンクの線形項。1 チャンク 14.4µs ≒ 30K クロックに
  (a/b ロード + barrier ×5〜6 + scan + solve + mid/out wmma) が入る。
  wmma 動的 16–32/チャンク ≒ 256 クロック未満 → ② 同期・逐次項が支配と確認。
occupancy レバーの寄与は別インプット: 旧 sweep（grid64→256 で 900→784µs = −13%）
で grid/occupancy 感応度を実証 → G1-a（grid 48→96）の期待値 −10〜14%（gdn −6〜8ms）。
```

**注意（計測安定性）**: 同一 args でも別インボケーション間で rows2048 の p50 が
811.95 → 929.64µs と +14% ずれた（各 run 内の p10–p90 は狭い）。継続負荷による
clock/温度状態の差とみられ、**A/B は必ず同一時期の interleaved ペアで行う**こと
（methodology §3 の根拠として記録）。

**次ラウンド（G1-a 実装に必要な依存マップ、本体 L222–570 読了済み）**:
v ローカル index `nt*16+lo` / `vt0+vj` / `n0` を含む箇所 =
S ロード・mid.W 書出・mid.dT 書出・output(`v0+nt*16+lo`)・Sacc store・
solve(`tid < g::kWv`)・compact_delta・末尾 state store。
`nt = warp >> 1 ∈ [0,4)` → サライス32 では nt<2 必要（remap か mask）。
LdsSmem の次元確認（L205–217）と warp 交錯マッピングの完全復元が実装前の必須手順。
decode1 は `g::kWv`(64) と static_assert のまま = 触らない。

---

## EXP-G1-a: gdn kWv 64→32（grid 48→96）— REJECT

```text
Experiment ID: EXP-G1-a
Commit: 未コミット（差分 = /tmp/g1a_kWvP.patch 保管、revert 済み。基準 499a89a）
Target kernel: shelf_qwen35_gdn_recurrence_wmma_lossy
Bottleneck class: ③（grid 平均 3/SIMD → 6/SIMD 化の仮説）
Hypothesis: v サライス半減で grid 96、occupancy 24→32 waves/WGP に到達し −10〜14%
Expected E2E saving: −6〜8 ms
Modified files: src/shelf/kernels/optimized/gdn/recurrence.hip
  （kWvP=32 局所定数 + マスク8箇所 + solve/compact の tid<kWvP + 両 launch。
   decode1 / kWv / static_assert / LdsSmem 次元は無変更 — サブエージェントがデコード表を作成）
ISA/Resources: grid 48→96（Grid_Size 24576=96×256 を rocprof で確認）、LDS 35840 変化なし、
  VGPR 224 変化なし、Scr 0
Correctness: test_gdn_recurrence PASS（0 failures）+ bench --check 合格
  （mismatches=0, state_bad=0）+ decode1 スモーク OK
Kernel benchmark（interleaved 3 ペア, p50 µs）:
  base 829.35 / 836.32 / 836.67   new 885.13 / 897.37 / 897.67
  ペア中央値: 836.32 → 897.37 = **+7.3%（劣化）、3ペア一貫**
Observed saving: 負
Decision: REJECT
Reason: occupancy（+2 ブロック分）が以下を相殺しなかった:
  (1) マスクで各ブロックの有効 warp が位相ごとに半分（S/状態 4/8、matmul 4/8、warp5,7 全休）
  (2) ブロック固定費（k/q 読み・scan・barrier）の総量が grid 倍
「grid 増で速い」（旧 nr sweep の −13%）は work/block 減と混在しており、
  work 不変の本変更では効かなかった — 仮説の反証として記録。
```

## EXP-G3: FP8 GEMM の scale 遅延適用（K=128 で1回 fold）— ACCEPT

```text
Experiment ID: EXP-G3
Commit: 本記録と同コミット（fp8.hip）
Target kernel: gemm_fp8_w8a8_wmma_prefill_2d_wide_kernel<true>
Bottleneck class: ② 計算/pass（scale 適用の ALU 削減）
Hypothesis: weight/activation scale が K=128 で一定なら、K 全域で raw c を
  積んで scale を K128 ごとに1回 fold（fma/mul を K32 ごと→K128 へ 1/4 に）
  数式等価（Σ の結合・分配）・BF16 出力・tolerance 内 → 原則4 の事前承認範囲
Expected E2E saving: −20〜40 ms
Modified files: src/shelf/kernels/optimized/linear/fp8.hip
  （wide kernel を template<bool kDeferScale> 化。w_scale_shift==2（本番 group128）のみ
   defer 経路、shift==1（fp8g64 テスト）は旧経路を if constexpr で維持。
   (t,nt) 外側に K ループを回して c[8] を K128 間保持、ws を g2 先頭で hoist）
ISA/Resources before/after: VGPR 184→216（4→3 blk に低下）、SGPR128、LDS 512、Scr 0
  （c[8] とネスト増のアドレス一時変数が原因とみられる。occupancy 損失を含んだ上で以下）
Correctness: test_gemm_fp8_optimized 288/0、GENERATED_IDS=47410 一致、required 121/121
Kernel benchmark（interleaved 3 ペア, gemm rows2048 n8704 k5120, p50 µs）:
  base 1621.07 / 1622.64 / 1625.70   new 1384.40 / 1389.50 / 1385.01
  ペア中央値: 1621.07 → 1385.01 = **−14.4%**（ペア内ばらつき <0.5%）
E2E A/B（interleaved 3 ペア, pp wall p50, KFD=0）:
  base 632.30 / 634.16 / 640.59   new 568.02 / 565.53 / 565.32
  ペア中央値: 634.16 → 565.53 = **−68.63 ms（−10.8%）**
Kernel accounting: GEMM 単体 −14.4%（chain 上 −52ms 相当）＋待ちの連動縮小で
  E2E −68.6ms — 整合（E-2 OK）
Observed saving: **−68.63 ms（pp ペア中央値）/ compute アンカー 636.1→570.5 = −65.6 ms**
Decision: ACCEPT
Reason: 主指標で再現可能な改善、全 correctness ゲート合格、scope 内（カーネル内部・
  数式等価 fold）、資源変化（VGPR 216/3blk）は記録済み。VGPR ≤192（4blk 復帰）は
  未実施の追加余地として記録。
```

### Phase F: 累積表（更新）

| Iteration | 対象 | 変更 | 実測 E2E（pp wall ペア中央値 / compute アンカー） | 累積削減（アンカー） | 残り |
|---|---|---|---:|---:|---:|
| Baseline | — | — | 660.47 / 663.4 | 0 | 163 |
| 1 | GEMM 幾何 | EXP-C2-b | 633.64 / 636.1 | 27.3 | 135.7 |
| 2 | GEMM scale fold | **EXP-G3** | **565.53 / 570.5** | **92.9** | **70.1** |

達成率 92.9/163 = **57.0%**。throughput: 2048/0.56553 = **3,621.6 tok/s**（目標 4,096）。

## EXP-G4: GEMM 内のループ不変計算の hoist 集 — REJECT（ノイズ内）

```text
Experiment ID: EXP-G4（C-1/C-4 バンドル）
Commit: 未コミット（差分 = /tmp/g4_hoist.patch 保管、revert 済み。基準 = G3 採用の e1e8df5）
Target kernel: gemm_fp8_w8a8_wmma_prefill_2d_wide_kernel<true>
Hypothesis: G3 で occupancy 4→3blk に落としても E2E −10.8% が出た（WAVE6 型の
  取引が成立）ため、≤240 VGPR（3blk 維持限界）の予算内で残る無駄計算を消せる:
  (a) wbase/wsbase/o_nt を kernel プロホイスト（K32 内の 64bit mul/shift/mask 除去）
  (b) (t,nt)→(t) 外側化で act ロード・アドレスを nt で CSE、c0/c1 に分離
  (c) as を asv[8] に t 先行ロード（LDS 読み ÷2）
Expected E2E saving: −2〜5%
Modified files: linear/fp8.hip（wbp/wsp プロホイスト + defer 節の再構成）
Resources: VGPR 216→208（3blk=6/SIMD 維持、alloc216）、LDS 512、Scr 0
Correctness: test_gemm_fp8_optimized 288/0
Kernel benchmark（interleaved 3 ペア, p50 µs）:
  base(G3) 1392.10 / 1383.80 / 1388.80   G4 1394.50 / 1397.85 / 1389.81
  ペア差 +2.4 / +14.0 / +1.0 → ペア中央 +2.4µs = +0.17%（ノイズ内、劣化方向のペアも1つ）
Decision: REJECT
Reason: 効果ゼロ。VGPR −8 は occupancy 段階（3blk）を変えず採用根拠なし。
  対象だったアドレス連鎖・ロード重複は G3 の構造下でコンパイラが既に CSE/LICM で
  消していた → 「掛け算の hoist 余地は GEMM では既に枯渇」（G3 が最後の大きな fold）。
```

## EXP-G5: WMMA acc 2本交互累积（K128 で scale 適用を維持したままチェーン分断） — REJECT

```text
Experiment ID: EXP-G5
Commit: 未コミット（差分 = /tmp/g5_acc2.patch 保管、revert 済み。基準 = G3 の e1e8df5）
Target kernel: gemm_fp8_w8a8_wmma_prefill_2d_wide_kernel<true>
Bottleneck class: ② 計算（WMMA 依存チェーン）
Hypothesis: G3 は K128 で acc に積むため wmma チェーンが 8 深く、同一 dst の
  RAW レイテンシが露出している。K32 交互に 2 本の acc（cA/cB）を使い分けて
  チェーンを 4 深に割り、最後に和を取って scale fold（G3 の意味論は不変）。
  VGPR +8 → 224 以内（3blk=6/SIMD 維持）で検証する。
ISA 根拠（G3 の静的解析）: wmma→wmma 間隔 {0:25, 1:24, 2:8, 6:5, 7:1}、
  49/63 が 0〜1 命令、36/63 に s_wait_loadcnt が直挟まり（dst とロード両方の遅延露出）
Resources: VGPR 216（想定 224 から不変、3blk 維持）、LDS 512、Scr 0
Correctness: test_gemm_fp8_optimized 288/0
Kernel benchmark（interleaved 3 ペア, p50 µs）:
  base(G3) 1384.28 / 1389.56 / 1384.19   G5 1409.21 / 1432.64 / 1437.32
  ペア差 +24.9 / +43.1 / +53.1 → ペア中央 +43.1µs = **+3.1%（劣化）、3ペア一貫**
Decision: REJECT
Reason: 同一 dst の連鎖はハードウェアの accumulator forwarding で安く、分割すると
  かえって forwarding が効かず和の8加算も増える。露出していたのは dst RAW ではなく
  s_wait_loadcnt（オペランドのロード遅延）であり、そちらは G3 で既に一括発行済み、
  G4 でアドレス/ロードの冗長分を切除済み（両方効果ゼロ）→ 「チェーン方向」は本件で
  枯渇と判定。G5 はハードウェア forwarding の重要性という反証として記録。
```

## EXP-G6: s_as の世代間 race 修正（ping-pong 化） — ACCEPT（正しさ修正・性能中立）

```text
Experiment ID: EXP-G6
Commit: 本記録と同コミット（fp8.hip）
Target kernel: prefill_2d wide（+ legacy narrow も同構造のため併せて修正）
Bottleneck class: 正しさ（性能レバーではない）
Hypothesis: g2 先頭で単一バッファ s_as へ書き込む構造は、他 wave が前 g2 の
  読み出し中に次 g2 の値で上書きしうる（ループ先頭 barrier は「書き終え」のみ保証、
  「読み終え」は保証しない）。C2-a で実証済みの ping-pong で世代を分離する。
Modified files: linear/fp8.hip
  - wide: __shared__ float s_as[2][128]、書込/読出を g2&1 で index（defer/else 両経路）
  - narrow: 動的 shared を 256→512 floats に増設し (g2&1)*256 オフセット
Correctness: test_gemm_fp8_optimized 288/0（新ビルド）
Kernel benchmark（interleaved 3 ペア, GPU2, p50 µs）:
  base(G3) 1299.19 / 1279.74 / 1283.96   G6 1293.42 / 1295.92 / 1286.43
  ペア差 −5.8 / +16.2 / +2.5 → ペア中央 +2.5µs = +0.2%（ノイズ内 = 性能中立）
Decision: ACCEPT（性能改善ではなく、race の証明を満たさない構造を排除する正しさ修正）
Reason: 288/0 は timing luck と両立しうるため、race 可能性の残る構造を採用しない。
  +512B LDS（合計 1KB）は occupancy 制約に無関係。
```

## EXP-G7: t ペア同時計算（2 行共有 Weight + dual-dst WMMA） — REJECT（ノイズ内）

```text
Experiment ID: EXP-G7 / Commit: 未コミット（G6 後の差分は本 G8 実装に吸収、G7 単体の構造はこの記録のみ）
Target: prefill_2d wide defer 経路 / Hypothesis: ユーザ提案。Weight を (nt,k32) でロードし
  2 行 (t0,t1) で共有（ソース上の Weight ロード 64→32/K128/thread）＋ c0/c1 の dual-dst 交互で
  依存待ちを隠す。K 順序を行ごとに保存 → byte-exact 見込み。
Resources: VGPR216（不変）、LDS1024（G6）、Scr0 — 資源ゲート通過
Correctness: 288/0
Kernel benchmark（interleaved ペア, GPU2, p50 µs）:
  base(G6) 1319.69/1280.65/1292.49  G7 1281.23/1284.77/1291.08
  ペア差 −38.5/+4.1/−1.4 → ペア中央 −1.4µs ≈ −0.1%、base 散布内 = 効果なし
Decision: REJECT
Reason: Weight ロードのソース重複は G4 と同様にコンパイラが既に CSE しており、
  dual-dst の効果も G5 同様に相殺（G5 は −3.1%、G7 は総合で ±0）。
  「t 方向 Weight 再利用」メカニズムも無効と判明。
```

## EXP-G8: 4 行同時計算（Weight 共有 4 行 + 4 系 dual-dst WMMA） — ACCEPT

```text
Experiment ID: EXP-G8 / Commit: 本記録と同コミット（fp8.hip、G6 の上に載せる形）
Target: prefill_2d wide defer 経路
Hypothesis: ユーザ提案の 2 行案を 4 行へ拡張。Weight は (nt,k32) で 1 回（ソース 64→16）、
  4 系の independent acc を 8 wmma 間で交互発行、scale fold は行ごとに G3 同様 K128 で1回。
  K 順序は行ごとに保存（byte 等価、GENERATED_IDS 一致で確認）。
Resources before/after: VGPR 216→216（不変、3blk=6/SIMD 維持）、LDS 1024、Scr 0
Correctness: test_gemm_fp8_optimized 288/0、GENERATED_IDS=47410 一致、required 121/121
Kernel benchmark（interleaved 5 ペア, GPU2, p50 µs）:
  base(G6) 1278.26/1278.95/1266.10/1293.58/1282.02（中央 1279.0）
  G8     1286.38/1258.30/1247.17/1258.56/1271.66（中央 1258.6）
  ペア差 +8.1/−20.7/−18.9/−35.0/−10.4 → ペア中央 −18.9µs = −1.5%（5中4ペア改善）
E2E A/B（interleaved 5 ペア, pp wall p50, GPU2/3 ペア, 外部プロセスは GPU1=無影響）:
  base 555.13/557.24/557.25/556.88/558.54（中央 557.24）
  G8  552.94/554.93/554.75/556.51/556.83（中央 554.93）
  ペア差 −2.19/−2.31/−2.50/−0.37/−1.71 → ペア中央 −2.19ms（−0.39%、5/5 改善）
Kernel accounting: マイクロ −1.6% × GEMM chain 297ms ≒ −4.7ms に対し E2E −2.2ms
  （待ちへの転嫁を含む、方向と規模は整合）。compute アンカー PREFILL 570.5→565.4（−5.1ms）
Decision: ACCEPT
Reason: 全 correctness ゲート合格、E2E で 5/5 一貫した改善、資源変化ゼロ、scope 内。
  効果は小さい（−0.4%）が符号一貫・ペア差ばらつき ≤0.3ms（pairs1-3）〜で再現。
```

### Phase F: 累積表（更新、pp 系列で統一）

| Iteration | 対象 | 実測 E2E（ペア中央） | 累積削減 | 残り |
|---|---|---:|---:|---:|
| Baseline | — | 660.47 | 0 | 163 |
| 1 | C2-b 幾何 | 633.64 | 26.83 | 136.17 |
| 2 | G3 scale fold | 565.53 | 95.42 | 67.58 |
| 3 | **G8 4行同時** | **563.34** | **97.13** | **63.34** |

達成率 97.13/163 = **59.6%**（compute アンカー 663.4→565.4 = 98.0/163 = 60.1%）。
外部検証: compute 系 PREFILL_MS 570.5 → 565.4、GENERATED_IDS=47410（G3 と同一）。

## EXP-G9: CTA 実行順 swizzle（M グループ化、G=4） — ACCEPT（worktree 分岐）

```text
Experiment ID: EXP-G9
Commit: prefill-cta-swizzle 分岐（本記録と同コミット）
Target kernel: gemm_fp8_w8a8_wmma_prefill_2d_wide（swizzle は kernel 引数 + prologue 逆写像）
Bottleneck class: ①/② — cross-CTA キャッシュ再利用（act/weight の L2 命中率）
Hypothesis: 現行は x(out) 最速走査で act タイルが1ブロック分しか持たない。
  「G 個の M(行) ブロックを同一 weight 列で連続実行」する逆写像を入れると、
  act 2.6MB（G=4）が全 N スイープで L2 に残り、weight も G ブロック連続で当たる。
  リニア id L = bx + GX*by（x 最速前提）から
    bx = (L%(G*GX))/G, by = (L/(G*GX))*G + (L%(G*GX))%G  で逆写像（G=1 は恒等）。
  G は env `SHELF_GEMM2D_GROUP`（既定 4、gy2d % G != 0 は 1 へフォールバック）。
Modified files: linear/fp8.hip（シグネチャ + prologue + launcher env + <cstdlib>）
Resources: VGPR/LDS/Scr 変化なし（swizzle はブロック毎の除算2回のみ）
Correctness: test_gemm_fp8_optimized 288/0（GY=4 ケースで swizzle 実行を網羅）、
  GENERATED_IDS=47410 一致、required 121/121
Micro sweep（rows2048 n8704 k5120、往復2巡 p50 µs、GPU2、KFD 0）:
  pass1: G1 1293.5 / G2 1214.7 / G4 1161.6 / G8 1227.4
  pass2: G8 1240.9 / G4 1138.3 / G2 1221.2 / G1 1203.0
  → G4 が両巡とも最速（G1 比 −5〜8%、G2/G8 よりも一貫して速い）
  メカニズム: working set の最適化 — G=4 で act2.6MB +  inflight weight ≈4.6MB < L2 8MB、
  G=8 は 7.2MB で逼迫、G=1（現行）は act 再利用が1ブロック分のみ
E2E A/B（interleaved 3 ペア、同一バイナリ・env 切替、pp wall p50、GPU2/3、KFD 0）:
  G=1: 550.44 / 551.44 / 551.97   G=4: 537.42 / 539.19 / 540.82
  ペア差 −13.02 / −12.25 / −11.15 → ペア中央 −12.25ms（−2.2%、3/3 改善、ばらつき 1.9ms）
compute アンカー: PREFILL_MS 565.4 → 552.5（−12.9ms）
Decision: ACCEPT
Reason: 全ゲート合格、E2E で 3/3 一貫した改善、資源変化ゼロ、launch/grid 形状は不変
  （ブロック数・dispatch 数・依存関係そのものは不変 = スコープ内）。
  参照実装の kGroupM グルーピングと同型のアプローチ。
残リスク: x 最速の物理スケジュール前提。G は env で再実験可能（既定 4）。
  G=16（GY 全体1グループ= weight×1 再読だが act10.5MB > L2）は未測定（劣化予想のため未実施）。
```

### Phase F: 累積表（更新、pp 系列）

| Iteration | 対象 | 実測 E2E（ペア中央） | 累積削減 | 残り |
|---|---|---:|---:|---:|
| Baseline | — | 660.47 | 0 | 163 |
| 1 | C2-b 幾何 | 633.64 | 26.83 | 136.17 |
| 2 | G3 scale fold | 565.53 | 95.42 | 67.58 |
| 3 | G8 4行同時 | 563.34 | 97.13 | 63.34 |
| 4 | **G9 CTA swizzle G=4** | **551.09** | **109.38（67.1%）** | **51.09** |

compute アンカー系: 663.4 → 552.5（−110.9、達成率 67.9%、残り 52.5ms）。

## EXP-G10: Weight scale の CTA-uniform 化（ws を g2 毎 1 本へ） — REJECT（ノイズ内）

```text
Experiment ID: EXP-G10 / Commit: 未コミット（差分 = /tmp/g10_ws_uniform.patch 保管、revert 済み）
Target kernel: prefill_2d wide defer 経路
Hypothesis: wide では out_block 幅 128 が weight scale ブロック（K128 行）と一致するため
  (o_nt >> 7) == bx が恒真。ws_g2[2] を単一スカラーへ畳み、2 回のロードと index を 1 本化。
Correctness: test_gemm_fp8_optimized 288/0
Kernel benchmark（interleaved 3 ペア, GPU2, p50 µs）:
  base(G9) 1185.36 / 1160.90 / 1170.14   ws 1188.34 / 1181.71 / 1162.78
  ペア差 +3.0 / +20.8 / −7.4 → ペア中央 +3.0µs ≈ +0.25%（符号混在、base 散布内）
Decision: REJECT
Reason: 恒真性の主張は成立（式としては正しい）が性能はノイズ内。コンパイラが
  元々 per-nt 行の算出を CSE していたとみられる（G4/G7 と同一の帰結）。
  意味論を不変のまま保てる単純化ではあるため、コード整理目的なら採用可能だが
  「効果のない変更を蓄積しない」規則に従い採用しない。
```

## 付録: TP=2 の「両者アイドル 37.6ms」の CPU/GPU 分解計測（ユーザー提案の4点を実施）

```text
計測: run_rank_step の submit/complete 所要、program_executor の3同期点、
  HipPeerTpTransport::enqueue_sum / sum_hidden の CPU 所要（envゲート、測定用ビルド）。
  patch = /tmp/tp_cpu_instrument.patch（revert 済み）
結果（compute TP2 / PREFILL_MS=553.1 / GENERATED_IDS=47410）:
  run_rank_step : 呼び出しは rank 毎に 1 回のみ。submit=36.0ms（全セグメントを
    一括投入）、complete=516.6ms（CPU は完全待機 = GPU 実行時間を全幅吸収）
  CPU 同期 API : hipStreamSynchronize 系 3 箇所すべて 0 回（本番では不活性）
  transport    : enq_sum 141 回=4.3ms、sum_hidden 129 回=2.5ms（CPU 計 6.8ms、
    ~130 回の集合点 = ユーザー推定の約128回と整合。すべて submit 36ms の中に収まる）
解釈: この経路では CPU 側 TP Barrier（mutex/cv）も per-segment の HIP 投入待ちも
存在しない。TP=2 prefill のセグメントは submit 一括で stream+イベント依存として
投入され、以降は GPU 側のイベント依存のみで進む。
→ 「両者アイドル 37.6ms」の主因は CPU ではなく GPU 側のイベント依存
 （peer write / hipStreamWaitEvent による stream 停止）と判定。
改善への含意: CPU barrier の GPU イベント化（ユーザー案）は既にその形になっている。
残るレバーは (a) 通信と計算のオーバーラップ構成（要承認・スコープ外）、
 (b) kernel 短縮に連動した待ち縮小、の2つ。

## EXP-M1: fused_resid_rmsnorm_quantize のレジスタ保持化（中間 global 往復2回の排除） — REJECT（E2E 未確証、正しさ全グリーン）

```text
Experiment ID: EXP-M1 / Commit: 未コミット（差分 = /tmp/m1_register_keep.patch、revert 済み。
  ファイル単体バックアップ /tmp/m1_final_v3.hip。基準 = dabd4b0 相当の resid ソース）
Hypothesis: a+b 書出→global 再読→normed 書出→global 再読→quantize の往復2回
  （30.7KB/行）を、担当列集合が3段とも一致（{4tid+j+1024k}, k=0..4 計20要素）する
  ことに着目してレジスタ保持（u32 ペア20本）で排除。中間 sync は 4→2。
Modified files: resid_rmsnorm_quantize.hip のみ（78→237行、+172/−13、launch 契約不変）
Correctness: test_fused_resid_rmsnorm_quantize byte-exact PASS×3（nasty NaN/Inf 含む全9ケース）、
  fused_rmsnorm PASS×3、activation_quantize e4m3 312×3 / a8 29×3、rmsnorm 621×3、
  attn_post PASS×3、gemm 288/0、GENERATED_IDS=47410、requiredは焦点7本で確認（警告増分0）
ISA: 実行時ロード 25→10 b64/thread（global 再読消滅、127行で約3.9MB/dispatch 読減）、
  barrier 4→2、VGPR 26→36（16→12 wave/SIMD）、Scr0・LDS32B 不変
副産物の重要知見（ gfx1201 の NaN 符号決定論）: add は左オペランド優先
  （p+m=+qNaN, m+p=−qNaN を実測）。完全展開コンテキストでコンパイラが
  `sum0+sum1` を交換 canonical 化 → 異符号 NaN の lane で合計符号反転 →
  reduce/div チェーン経由で normed 全体に伝播。修正は total 加算を
  inline asm v_add_f32 でソース順固定（数式・reduction 順序は不変）。
E2E A/B（interleaved 3 ペア、GPU2/3、外部 compute は GPU0=無影響、KFD 記録済み）:
  old 538.32 / 539.11 / 540.31   new 537.55 / 539.51 / 540.60
  ペア差 −0.76 / +0.40 / +0.29 → ペア中央 +0.29ms、符号 1:2 分裂 = ノイズ内
Decision: REJECT（E-3）
Reason: 読トラフィック実測削減（−3.9MB/dispatch）と sync 削減（−2）は実在するが
  wall に露出しない — misc family が critical path 外（rank ずれの待ちに吸収）で
  swiglu（EXP-S1）と同じ帰結。±1ms 級の絶対ばらつきに対し信号が無い。
```

## EXP-G1-d: gdn チャンク頭 a/b レジスタ事前読み（スコープ内 最終候補） — REJECT

```text
Experiment ID: EXP-G1-d / Commit: 未コミット（差分 = /tmp/g1d_prefetch.patch、revert 済み。基準 d4bbff3）
Hypothesis: チャンク頭の global a/b ロード＋sigmoid/log-exp 計算を現チャンク先頭で
  発行し、次チャンク先頭はレジスタ（pbt/poe 2本）から引くことで約14µs の
  チャンク処理中にロード遅延を隠す（kSerialRows 経路は if constexpr で不変）
Modified files: recurrence.hip のみ（+30/−6 程度）
Correctness: test_gdn_recurrence PASS / test_gdn_spec_history failed=0 /
  test_gdn_recurrence_exact_history 7 passed / test_gemm_fp8_optimized 288/0
Kernel benchmark（interleaved 3 ペア、GPU2、p50 µs）:
  old 838.30 / 841.75 / 843.24   new 850.68 / 847.50 / 850.38
  ペア差 +12.4 / +5.8 / +7.1 → ペア中央 +7.1µs = **+0.8%、3/3 劣化**
Decision: REJECT
Reason: 事前前発行が効かなかった = チャンク頭ロード待ちは支配的ではない
 （EXP-G1 の「14.4µs/chunk の内訳が未分離」に対する実測回答）。残る chunk 時間は
  barrier/scan/逐次項が主体で、G1-a・G1-b とあわせ gdn レバーは全て反証された。
```


## EXP-G1-b: gdn syncA/syncC 削減（oe/de/bt のローカル再計算） — REJECT

```text
Experiment ID: EXP-G1-b / Commit: 未コミット（差分 = /tmp/g1b_syncAC.patch 保管、revert 済み。基準 5b7d78f）
Target kernel: shelf_qwen35_gdn_recurrence_wmma_lossy（プリフィル本体のみ）
Hypothesis: bt/oe/de の LDS 公開を消費側ローカル再計算に置換し、syncA/syncC を除去
  （チャンク毎 sync 6→4、動的 384→256）。予想 −2〜6%
Modified files: recurrence.hip のみ（+14/−19、18 参照を全分類）
ISA before/after: s_barrier 9→7 サイト ✓（期待達成）、v_exp_f32 13→73（+60）、
  global_load +15、v_mad +15、コード 5163→5882 行、Scr 両版 0
Correctness: test_gdn_recurrence 23 PASS / test_gdn_spec_history 0 failed /
  test_gdn_recurrence_exact_history 7 PASS（bit-exact）/ bench check mismatches=0
  — 正しさは全グリーン
Kernel benchmark（interleaved 3 ペア、GPU3、GPU1 の第三者実行完了を7分待機後に開始、全 run check OK）:
  old p50 941.01 / 940.11 / 942.83   new 997.28 / 989.43 / 989.17
  → +6.0 / +5.2 / +4.9%（平均 +5.4%）、3 ペア方向一致で統計的に安定な劣化
Observed saving: 負
Decision: REJECT
Reason: barrie 2 本の節約より v_exp_f32 +60 の ALU 負荷が支配的だった。
  「oe を LDS 維持（wave0 書き＋syncB）、bt/de のみローカル化」の中間案は未検証の
  残オプションとして記録（効果見込みは syncA 1 本分と小さく、優先度は低い）。
```


## EXP-S1: SWIGLU + activation_quantize 融合（ユーザー解禁の唯一の融合） — REJECT（E2E 未確証、正しさは全グリーン）

```text
Experiment ID: EXP-S1 / Commit: 未コミット（差分 = /tmp/swiglu_fusion_full.patch、revert 済み。基準 dabd4b0）
背景: ユーザーが「swigluのみ 融合の解禁」を明示した唯一の融合スコープ
Hypothesis: SWIGLU(BF16 中間書出し) → ACTIVATION_QUANTIZE_W4A8 の chain=2 融合。
  BF16 に1回丸めてから量子化する点を直列と同一に保ち codes・scale・中間出力はビット一致。
  効果源 = dispatch 2→1 ＋ 中間 BF16(rows×K×2B) の書/読グローバル往復の短絡
Modified files: 新規5（kernel・fusion(matcher)・ヘッダ2・新設テスト）＋cmake2＋optimized_dispatch
Correctness: 新設 test_fused_swiglu_quantize **byte-exact PASS**（9ケース、stride 非整列含む）、
  required **122/122**（従来121＋新設1）、resource 新旧同一（VGPR32/LDS0/Scr0）
Kernel: rows2048 で split 257.4µs → fused 170.2µs（**−33.9%**、全9ケース −19〜−55%、rocprof 計測）
E2E A/B（interleaved 5 ペア、pp wall、GPU0/1、KFD 0）:
  old 549.95 / 549.83 / 551.09 / 547.65 / 550.76
  new 547.69 / 548.39 / 550.17 / 550.77 / 551.02
  ペア差 −2.26 / −1.44 / −0.92 / +3.12 / +0.26 → ペア中央 −0.92ms、符号 3:2 分裂 = ノイズ内
Decision: ACCEPT（ユーザー判定基準の変更による）
  ユーザー指示: 「−0.92ms なのであれば命令長で決めるべき」「ISA の命令数が少ないなら信用できる」
  → ISA 実測（fatbin 抽出、本番 k17408 インスタンス）:
    OLD 直列 = swiglu 522 命令(0xBAC) + quantize 254 命令(0x5CC) = **776 命令(4,472B)**
    NEW 融合 = **503 命令(0xBE4)** → **−273 命令（−35.2%）、バイト −32%**
    オペコード class 全減: v_fma 128→88、sopk 15→7、cvt 12→7
  資格: dispatch 2→1、byte-exact（新設テスト9ケース）、VGPR32/LDS0/Scr0 同格、
  required 122/122（再確認済み）、E2E はノイズ（−0.92ms、符号3:2）＝wall 効果は未確定と記録。
  採用根拠は命令数削減の構造的利点（wall 効果は未確証のまま）。
```


## EXP-A1: TP 通信 stream 分離（comm stream + イベント往復、overlay ゲート） — REJECT（別 worktree `tp-overlap`）

```text
Experiment ID: EXP-A1 / Commit: 未コミット（差分 = /tmp/a1_overlay.patch 保管、revert 済み。基準 aa655a4）
Target: HipPeerTpTransport::enqueue_sum（集合点の送信コピー/wait/add を comm stream へ分離、
  producer→comm / comm→compute をイベントで接続。env `SHELF_TP_OVERLAY`、既定 OFF）
Hypothesis: overlap_study 案 A1。単一 stream での通信・計算直列を分離し、停滞 ~50ms の
  大半を計算に重ねる（残り 51ms への対抗として500ms到達の本命経路）
Correctness: token gate GENERATED_IDS=47410 一致、required 121/121（overlay ON）
E2E A/B（interleaved 3 ペア、pp wall、GPU0/1、KFD 0、同一バイナリ・env 切替）:
  overlay=0: 544.98 / 545.22 / 546.45
  overlay=1: 575.31 / 588.28 / 586.38
  → ペア差 +30.3 / +43.1 / +39.9 = **+37.8ms（+6.9%）の劣化、3/3**
Decision: REJECT
Reason: 理論の誤りを実測が示した。次計算は加算結果に真に依存するため、
  単一 stream では「コピーが計算の直後にゼロ遅延で走る」形が最速であり、
  stream 分離はイベント往復（producer→comm→compute）の遅延を集合点ごとに
  追加するだけだった（130 集合点 × ~0.3ms ≒ +40ms と整合）。
  「両者アイドル 37.6ms」は rank ずれ由来で、通信側の構造変更では縮まない
 （kernel 短縮に連動して縮む）。合意していた「overlap は 500ms の唯一の経路」は
  本測定により否定された。
```

## 付録: 「SWIGLU + activation_quantize の新規融合」提案の扱い

別エージェントから実現可能性分析が届いた（実装計画つき・変更なし）。
本ミッションの禁止事項「カーネル融合を一切実施しない / 新規融合の禁止」に
抵触するため、**現ゴールでは実装不可**。スコープ外候補として承認事項リストに
追加する（既存融合の内部最適化は可、新規融合は不可という現行規定を維持）。

## EXP-A1b: A1 の原因分解（ユーザー指摘の最小修正: 強制コピー解除・NonBlocking・計測）

```text
Experiment ID: EXP-A1b / Commit: 未コミット（差分 = /tmp/a1b_overlay_refined.patch、revert 済み。基準 549c910）
背景: EXP-A1（+37.8ms 劣化）に対し、ユーザーが3つの疑いを提示
 （① `|| kOverlay` による peer_write 経路の強制コピー、② hipStreamCreate の
    暗黙同期（NonBlocking 化）、③ 依存は不変でイベント往復だけ増える）
計測1（①の真実率）: `peer_write_complete` は **pw_true=0 / pw_false=128 = 常時 false**
  → 強制コピー変更は本経路では意味を持たない（①は原因でない）
修正: overlay = kOverlay && !peer_write_complete（ユーザー案どおり）＋
  comm stream を hipStreamNonBlocking 作成（legacy ゲート SHELF_TP_OVL_LEGACY で比較）
計測2（E2E interleaved、pp wall、GPU0/1、KFD 0）:
  A（OFF）          544.59 / 544.70 / 545.68（中央 544.70）
  C（ON・修正済み・NonBlocking） 572.96 / 575.35 / 573.63（中央 573.63）→ +28.93ms
  D（ON・修正済み・blocking）     577.79（単発）
分解:
  強制コピー解除: B(旧 ON 平均 583) → D(577.8) ≒ −5ms（うち数値はノイズ帯）
  NonBlocking:   D(577.8) → C(574) ≒ −4ms（小さいが有効）
  **残り +28.9ms = overlay 構造そのもの**: 次層は加算結果に真依存で重ねる先がなく、
  130 集合点 × 2本のイベント往復（producer→comm→compute）が純増コストとして残る。
  これは「③ 同じ依存のままイベント往復を増やしただけ」というユーザー分析を
  実測で裏付けたもの。
Decision: REJECT（維持）
Reason: 修正群は +40ms のうち 11ms 程度しか取り戻せず。overlap は
  「次層の依存を解く」才能なければ効かず、それは TP 方式・プランナー変更 = スコープ外。

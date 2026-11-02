# Prefill 個別最適化 — Phase B 機会分析

基準: [baseline.md](baseline.md)。主指標 = pp wall p50 **657.75 ms**（ゴール表記 663 ms = compute PREFILL_MS）。目標削減 **≥163 ms**。

前提となる構造: wall ≈ 両 rank の活動和集合（643.4 ms）+ 非活動区間（≈93 ms）。rank0 の kernel 合算 628.4 ms / rank1 636.5 ms。rank の chain は barrier で同期しているため、片側の kernel 削減は待ち時間に波及する。

## B-1 / B-2. Kernel family 選定表と律速分類

FLOP・bytes は safetensors の実形状と TP2 分割（Q 全量 / KV 分割 / 行並列 K 分割）から静的算定。実効値は Agent 1 の duration 合計で除す。

| Kernel family | duration (rank0) | n | share | 実効値 | VGPR | LDS | Occupancy（主拘束） | 律速分類 | 改善候補 |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| gemm_fp8 prefill2d | 383.8 ms | 400 | 61.1% | **130 TFLOP/s**（49.8 TFLOP/0.384s）。weight 転送 11.9 GB → 31 GB/s（DRAM の 5%） | 184 | 0(動的1024) | 4 blk（register、32/64 wave = 50%） | **② 計算/pass 支配**（peak 約160 に対し 81%） | 冗長アドレス削減、C-2 の命令効率、C-5。occupancy は 184→177 では段階変わらず（122 以下で 5 blk、非現実的） |
| tp_copy_bf16 | 92.4 ms | 140 | 14.7% | **29.1 GB/s**（2.68 GB/92.4ms）。リンク上限 = **PCIe Gen5 x8 = 31.5 GB/s の 92%（飽和）**（PEX89048 + 各カード upstream x8。baseline A-1） | 8 | 0 | 8 blk（wave cap、100%） | **① 帯域律速（PCIe x8、飽和）** | bytes 削減は TP 方式変更で禁止。**実効帯域の余地は 8%（0〜−5 ms）のみ** → 中央見積はほぼ削減不能として扱う |
| tp_add_peers_bf16 | 14.8 ms | 140 | 2.4% | 181 GB/s（ローカル読取） | 8 | 0 | 8 blk（100%） | ② L3 | 小。先に tp_copy |
| gdn_recurrence_wmma_lossy | 55.6 ms | 48 | 8.9% | wmma f16 + **barrier 9 拠点/呼び出し**（動的 432 回） | 224 | 35,840 | **3 blk（register=LDS 同率、24/64 wave = 75%）** | **② 同期/pass 支配**（bytes は 26 GB/s 程度で帯域律速ではない） | LDS 35.8 KB → 4 blk へ、barrier 管理の簡素化、C-6 |
| PagedAttention prefill | 15.7 ms | 16 | 2.5% | **105 TFLOP/s**（1.65 TFLOP）。KV は L3 内 | 176 | 29,696 | 4 blk（register=LDS 同率、50%） | ② 計算支配（効率は高い） | 余地小 |
| fused_resid_rmsnorm_quantize | 14.9 ms | 127 | 2.4% | 264 GB/s（3.94 GB） | 32 | 512 | 8 blk（100%） | ② L3/pass | 冗長 pass の有無確認 |
| swiglu_bf16 | 11.1 ms | 64 | 1.8% | 414 GB/s（4.6 GB） | 32 | 0 | 8 blk（100%） | ② L3/pass | 小 |
| copyBuffer/fill（trace 内） | 7.3 ms | 736 | 1.2% | — | 16 | 0 | 100% | ④ host/コピー | 融合・dispatch 削減は禁止 → 記録のみ |
| activation_quantize | 6.2 ms | 112 | 1.0% | 〜300 GB/s | 32 | 0 | 100% | ② | 小 |
| fused_gdn_conv_silu / post / qk_prepare | 12.7 ms | 144 | 2.0% | — | 24–40 | 0–512 | 8 blk（100%） | ② | 小 |
| paged kv/act 関連（lds8192） | 5.5 ms | 96 | 0.9% | — | 72 | 8,192 | 要算定 | ②/③ | 小 |
| lm_head（gemm_bf16_exact_rows、n=1） | 2.0 ms | 1 | 0.3% | **635 GB/s**（FP8 权 1.27 GB） | 24 | 0 | 100% | **① DRAM 境界（636 直下）** | bytes 不可 → 削減不能 |
| rope / rmsnorm / split / attn_post_quant 等 | 〜20 ms | — | 3% | — | — | — | — | ② | 小 |
| （host gap・同期待ち） | rank0 61.4 / rank1 107.8 ms | — | wall の 93 ms 相当が両 GPU 非活動 | — | — | — | — | ④＋同期 | **kernel 削減に連動して縮む**が、提出間隔そのものは禁止事項 |

share の注記: 上表 share は GPU 上 kernel 合算（628.4 ms）に対する比。E2E wall への寄与は両 rank が並列・同期実行のため 1:1 ではない（union 643.4 ms が wall の実体）。

## B-3. Occupancy 表（モデル改訂）

採用モデル（gfx1100 / gfx1201 共通、提供モデル）:

```text
waves_per_SIMD = min(16, floor(1536 / (ceil(VGPR / 24) * 24)))
  - 1536 VGPRs per SIMD、wave32 allocation granule 24、cap 16 waves/SIMD
  - LDS / thread / workgroup / barrier 制約はこの式に含めない（LDS は別建てで判定）
budget_waves / WGP = 4 * waves_per_SIMD              （gfx1201: WGP = 4 SIMD）
blocks / WGP = min(budget_waves / (wg/32), 131072 / LDS, 64 / (wg/32))   （wg = 256 → 8 waves/block）
```

旧式（47,104 dwords、granule 8）との差分:

- **granule 24 のため VGPR 185–192 は 8 waves/SIMD のまま** — 旧式の「184 で cliff」は誤り。EXP-C1b(VGPR192) は occupancy 不変で、劣化は純粋な性能差
- 6 waves/SIMD の境界 = **VGPR 240**（ceil(240/24)×24=240 → 1536/240=6、241 で 264→5）。参照実装の `R4D_WAVES(6)` と整合
- 1536 = 1472（47,104 dwords = 1472×32、予約 64 除く）+64 で、8-granule 時と 184 の結果が一致する点は検算済み

| kernel | VGPR | alloc(24) | waves/SIMD | LDS 制約 (blocks) | blocks/WGP | 拘束 |
|---|---:|---:|---:|---:|---:|---|
| gemm prefill2d（現行 narrow、LDS 1KB） | 184 | 192 | 8 | 128 | **4**（32 waves） | register |
| **gemm prefill2d wide（C2 設計値）** | ≤240 | ≤240 | ≥6 | 16.5KB → 7.75 | **3**（24 waves = 6/SIMD） | register（参照 `R4D_WAVES(6)` と同点） |
| tp_copy / tp_add | 8 | 24 | 16 | ∞ | **8**（64 waves） | wave 上限 |
| gdn_recurrence | 224 | 240 | 6 | 35.8KB → 3.65 | **3** | register（LDS が同率に近い） |
| attention prefill | 176 | 192 | 8 | 29.7KB → 4.4 | **4** | register=LDS 同率 |
| resid_rmsnorm_quant / swiglu / quantize 等 | 24–40 | 24–48 | 16 | ≥256 | **8** | wave 上限 |
| lm_head exact_rows | 24 | 24 | 16 | ∞ | 8 | wave（帯域律速） |

レジスタ低減の解釈も改訂: 185→192 では occupancy 不変（低減は unroll / spill 余地としてのみ評価）。216→192（7→8）、240→192（6→8）のように **granule 境界をまたぐ**低減でのみ段階が動く。

## B-4. 改善可能量の試算と 163 ms 計画

Amdahl: `saving = chain 時間 × (1 − 1/speedup)`。wall は rank chain（kernel 630 ms + 待ち 93 ms）で説明するため、削減は「対象 kernel の chain 削減」と「待ちへの波及（93×削減率 程度）」で見積もる。

| 候補 | 現状 | 仮説 | 中央見積削減 | 楽観 | 根拠 |
|---|---:|---|---:|---:|---|
| **C2: 参照幾何（geometry-only で採用済み）** | 383.8 ms / 130 TF | 幾何のみ移植（128×128 + ww×wa、act 直読み維持）。staging 付与は +13.3% で REJECT | **−26.8（実測）** | −30 | EXP-C2-b: E2E 3ペア interleaved 660.47→633.64、トークンゲート一致、required 121/121。分離実験で Geometry=−6.1%/LDS staging=+13.3% を実測分離（experiments.md） |
| GEMM 微小レバー（C-1/C-2/C-4） | 383.8 ms | 命令・アドレス効率 +6〜9% | −18 | −30 | `s_wait_alu 425`・加算 444 の冗長。C2 と重複しうるため合算時は減点 |
| 同期待ちの連動縮小 | 93 ms | kernel 縮小に比例 | −7 | −14 | 待ちは chain・リンク待ちの関数 |
| gdn_recurrence（LDS/barrier/C-6） | 55.6 ms | 同期と pass を 10–20% 短縮 | −8 | −15 | barrier 9 拠点、LDS 35.8 KB |
| norms/quantize/swiglu/attention 合計 | 〜65 ms | 各 −5〜15% | −6 | −12 | ② だが share 小 |
| tp_copy（TLP 効率の残余地のみ） | 92.4 ms | 92% → 97%（x8 上限） | −2 | −5 | in-situ 29.1 / 理論 31.5 GB/s。**残 85 ms 強は飽和＝削減不能** |
| tp_add | 14.8 ms | 小 | −2 | −4 | |
| **合計（C2-b 実測 26.8 を反映、重複織り込み）** | | | **≈ −70** | **≈ −110** | |

**必須判断（C2-b 採用後）**: 達成 **−26.8 ms**（663 → 636.1、compute アンカー）。残りの根拠付き中央見積は従来項目の **−43 ms**（GEMM 微小 −18 / 待ち連動 −7 / gdn −8 / 小 family −6 / tp_copy −2 / add −2）で、合計 **−70 ms（→ 約 593 ms）**、楽観 **−110 ms（→ 553 ms）**。**163 ms には中央で 93 ms、楽観でも 53 ms の不足** — tp_copy 飽和・lm_head①・④ host は削減不能として残る。追加の寄与源（staging の再検討は bank 対策込みで、gdn、GEMM ISA）を検証してから最終判定。

埋め方の残り筋（C〜F で検証してから上方修正する）:

1. GEMM の ISA 差分で冗長が予想を超える（アドレス再計算・wait 連鎖の除去が −30 に近づくか）
2. GEMM の C-5（launch 設定・unroll・pipeline depth）が 130 TF を押し上げる余地の実測
3. 待ち 93 ms の内訳（同期 vs host 提出 vs リンク待ち）を trace gap で切り分け、④ と link 待ちの実額を確定
4. tp_copy の TLP 効率（92→97%）が E2E で検証できるか（−2〜5 ms、優先度低）

**kernel 個別最適化だけでは削減できない時間（スコープ外として記録）**:

- **tp_copy の転送 bytes（2.68 GB/rank、92 ms）の 9 割強 — PCIe Gen5 x8（31.5 GB/s）で飽和。** bytes 削減は TP 方式変更・集合通信統合の禁止事項に抵触
- lm_head の DRAM bytes（1.27 GB、636 GB/s 直下の①）
- リンク待ちを含む同期待ち（kernel 削減に一部連動するが、link 飽和分は固定）
- dispatch 提出間隔・HIP Graph・kernel 境界削減（④、禁止）
- 融合（本ゴールの冒頭で禁止）

## 改善優先順位（推定 wall 削減量順）

1. ~~C2: 参照幾何移植~~ → **C2-b で達成 −26.8 ms（ACCEPT、staging 部分は REJECT）**
2. gdn_recurrence（−8 中央）— C-6/C-3
3. 同期待ちの連動（−7、副産物）
4. 小家族合計（−6）
5. GEMM 微小レバー（−18、C2 と重複考慮の上で）
6. tp_copy（−2、飽和のため余地ほぼ無し）

C1（as 直読み/barrier 消去）は全変種で VGPR cliff を超えられず REJECT（[experiments.md](experiments.md)）。

---

## 付録（2026-10-22 G9 適用後の再計算）

採用累積: pp 系列 660.47 → **551.09**（−109.38、達成率 67.1%、残り 51.09）。

最新 trace（G9 後）の share と残レバー:

| 優先 | 対象 | share | 残レバー | 根拠付き見込み |
|---|---|---:|---|---:|
| 1 | misc（fused_rmsnorm 15.0・swiglu 11.5・quantize×2 11.8・gdn sub 12.8・attn post） | 〜60 ms | EXP-M1: 中間 global 往復2回の排除（C-4「不必要な中間書き込み削減」） | −3〜5、他 small −2 |
| 2 | gdn_recurrence | 53.9 ms | G1-d チャンク先読みのみ（G1-a/G1-b は反証済み） | −1〜2 |
| 3 | gemm_fp8 wide | 280.6 ms | VGPR216→192（低確度）、128-bit、geometry 64×128（大規模） | 0〜−10（不確実） |
| — | tp_copy | 95.2 ms | なし（x8 飽和・bytes=TP方式） | 0 |
| — | ④ 両者アイドル（rank0 gap の 71%） | 37.6 ms | なし（fusion/dispatch/HIP Graph 禁止） | 0 |

```text
上限見積 = 551.09 − (5 + 2 + 10) ≒ 534 ms（現実的中央 ≒ 540）
→ 500ms には 34〜40ms 不足。スコープ内では到達困難と数値で確定。
```
